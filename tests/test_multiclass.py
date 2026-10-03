"""Test de bout en bout des classifieurs multiclasses (TF-IDF + LR / XGBoost) sur un mini-dataset."""
import pytest

from llmfw.config import Settings
from llmfw.data.build import build
from llmfw.training.multiclass import LABELS, MODEL_NAMES, train_and_evaluate


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_multiclass_model_end_to_end(tmp_path, name):
    data_dir = tmp_path / "data"
    build(data_dir, seed=3, scale=0.2, version="test")
    settings = Settings(data_dir=data_dir, results_dir=tmp_path / "results",
                        models_dir=tmp_path / "models", random_seed=3)
    report = train_and_evaluate(name, settings)
    m = report["test_metrics"]
    assert m["labels"] == sorted(LABELS)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["accuracy"] > 0.5  # nettement au-dessus du hasard (1/6 ~ 0.17)
    assert (tmp_path / "models" / f"{name}_multiclass.joblib").exists()
    assert (tmp_path / "results" / "metrics" / f"{name}_multiclass.json").exists()
    assert (tmp_path / "results" / "confusion_matrices" / f"{name}_multiclass.png").exists()
