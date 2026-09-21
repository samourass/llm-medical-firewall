"""Test de bout en bout rapide des classifieurs classiques sur un mini-dataset."""
import pytest

from llmfw.config import Settings
from llmfw.data.build import build
from llmfw.training.classical import MODEL_NAMES, train_and_evaluate


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_classical_model_end_to_end(tmp_path, name):
    data_dir = tmp_path / "data"
    build(data_dir, seed=3, scale=0.2, version="test")
    settings = Settings(data_dir=data_dir, results_dir=tmp_path / "results",
                        models_dir=tmp_path / "models", random_seed=3)
    report = train_and_evaluate(name, settings)
    m = report["test_metrics"]
    assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == m["n_samples"]
    assert 0.0 <= m["fpr"] <= 1.0 and 0.0 <= m["fnr"] <= 1.0
    assert m["roc_auc"] > 0.8  # bien au-dessus du hasard (0.5)
    assert (tmp_path / "models" / f"{name}.joblib").exists()
    assert (tmp_path / "results" / "confusion_matrices" / f"{name}.png").exists()
    assert (tmp_path / "results" / "metrics" / f"{name}.json").exists()
