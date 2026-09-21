"""Tests du dataset : déterminisme, absence de fuite, formules de métriques."""
import pandas as pd
import pytest

from llmfw.data.generate import generate_dataset, normalize_text
from llmfw.data.split import drop_near_duplicates, group_stratified_split, leakage_audit
from llmfw.evaluation.metrics import compute_binary_metrics


@pytest.fixture(scope="module")
def small_df():
    return generate_dataset(seed=7, size_scale=0.15)


def test_generation_is_deterministic():
    a = generate_dataset(seed=1, size_scale=0.05)
    b = generate_dataset(seed=1, size_scale=0.05)
    pd.testing.assert_frame_equal(a, b)


def test_all_categories_present_and_labels_consistent(small_df):
    expected = {"benign", "prompt_injection", "jailbreak", "pii_exfiltration",
                "secret_exfiltration", "rag_prompt_injection"}
    assert set(small_df["category"]) == expected
    assert set(small_df.loc[small_df["category"] == "benign", "label"]) == {0}
    assert set(small_df.loc[small_df["category"] != "benign", "label"]) == {1}


def test_no_duplicates_after_normalisation(small_df):
    keys = small_df["text"].map(normalize_text)
    assert keys.is_unique


def test_split_has_no_leakage(small_df):
    split_df = group_stratified_split(small_df, seed=7)
    final_df, _ = drop_near_duplicates(split_df, threshold=0.90)
    audit = leakage_audit(final_df)
    assert all(v == 0 for v in audit.values()), audit
    for split in ("train", "validation", "test"):
        assert (final_df["split"] == split).any()
        # chaque catégorie doit être présente dans chaque split
        assert final_df[final_df["split"] == split]["category"].nunique() == 6


def test_metrics_formulas():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    y_score = [0.9, 0.8, 0.7, 0.4, 0.1, 0.2, 0.1, 0.3, 0.2, 0.6]
    m = compute_binary_metrics(y_true, y_pred, y_score)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["fpr"] == pytest.approx(1 / 6)  # FP / (FP + TN)
    assert m["fnr"] == pytest.approx(1 / 4)  # FN / (FN + TP)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
