"""Entraînement + évaluation MULTICLASSE (6 catégories) : TF-IDF -> Logistic Regression / XGBoost.

Contrairement à `training/classical.py` (classification binaire benign/attack, étapes 3-5),
ce module classe directement dans les 6 catégories de la colonne `category` du dataset :
benign, prompt_injection, jailbreak, rag_prompt_injection, pii_exfiltration, secret_exfiltration.
C'est la sortie attendue par `models/unified.classify(text)` et `firewall/api.inspect_input(text)`.

Random Forest n'est PAS ré-implémenté ici (hors du périmètre resserré du document fourni :
Regex, TF-IDF+LogisticRegression, TF-IDF+XGBoost, DistilBERT).

Protocole identique à `classical.py` : petite grille d'hyperparamètres sélectionnée sur
VALIDATION (F1 macro), puis évaluation unique sur TEST. Seed fixe, aucune métrique inventée.

Usage (depuis la racine du projet) :
    python -m llmfw.training.multiclass --model logistic_regression
    python -m llmfw.training.multiclass --model all
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from llmfw.config import Settings, get_settings
from llmfw.evaluation.metrics import compute_multiclass_metrics, measure_latency, process_rss_mb
from llmfw.evaluation.plots import plot_confusion_matrix_multiclass

MODEL_NAMES = ["logistic_regression", "xgboost"]
DISPLAY = {"logistic_regression": "Logistic Regression (multiclass)", "xgboost": "XGBoost (multiclass)"}
LABELS = ["benign", "jailbreak", "pii_exfiltration", "prompt_injection",
          "rag_prompt_injection", "secret_exfiltration"]


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)


def param_grid(name: str) -> list[dict]:
    grids = {
        "logistic_regression": {"C": [0.1, 1.0, 10.0]},
        "xgboost": {"n_estimators": [200, 400], "max_depth": [4, 6]},
    }[name]
    keys = list(grids)
    return [dict(zip(keys, values)) for values in itertools.product(*grids.values())]


def make_classifier(name: str, params: dict, seed: int, n_classes: int):
    if name == "logistic_regression":
        # scikit-learn >= 1.5 : LogisticRegression choisit automatiquement une stratégie
        # multinomiale pour les problèmes multiclasses (le paramètre multi_class est retiré).
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, **params)
    if name == "xgboost":
        return XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss",
                             learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=seed, **params)
    raise ValueError(name)


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / split / f"{split}.csv")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def train_and_evaluate(name: str, settings: Settings) -> dict:
    seed = settings.random_seed
    train = load_split(settings.data_dir, "train")
    val = load_split(settings.data_dir, "validation")
    test = load_split(settings.data_dir, "test")

    encoder = LabelEncoder()
    encoder.fit(LABELS)
    y_train = encoder.transform(train["category"])

    search = []
    best = None
    search_start = time.perf_counter()
    for params in param_grid(name):
        pipe = Pipeline([("tfidf", make_tfidf()),
                         ("clf", make_classifier(name, params, seed, len(LABELS)))])
        t0 = time.perf_counter()
        pipe.fit(train["text"], y_train)
        fit_s = time.perf_counter() - t0
        val_proba = pipe.predict_proba(val["text"])
        val_pred = encoder.inverse_transform(np.argmax(val_proba, axis=1))
        m = compute_multiclass_metrics(val["category"], val_pred, val_proba, encoder.classes_.tolist())
        search.append({"params": params, "fit_time_s": fit_s, "val_f1_macro": m["f1_macro"],
                       "val_accuracy": m["accuracy"]})
        if best is None or m["f1_macro"] > best["val_f1_macro"]:
            best = {"params": params, "pipe": pipe, "val_f1_macro": m["f1_macro"], "fit_time_s": fit_s,
                    "val_metrics": m}
    search_time_s = time.perf_counter() - search_start

    pipe: Pipeline = best["pipe"]

    test_proba = pipe.predict_proba(test["text"])
    test_pred = encoder.inverse_transform(np.argmax(test_proba, axis=1))
    test_metrics = compute_multiclass_metrics(test["category"], test_pred, test_proba, encoder.classes_.tolist())

    latency = measure_latency(lambda batch: pipe.predict_proba(batch), test["text"].tolist())
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.models_dir / f"{name}_multiclass.joblib"
    joblib.dump({"pipeline": pipe, "labels": encoder.classes_.tolist()}, model_path)
    resources = {
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
        "process_rss_mb_after_inference": process_rss_mb(),
    }

    res = settings.results_dir
    title = DISPLAY[name]
    plot_confusion_matrix_multiclass(test_metrics["confusion_matrix"], encoder.classes_.tolist(),
                                     f"{title} — test", res / "confusion_matrices" / f"{name}_multiclass.png")
    (res / "predictions").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(res / "predictions" / f"{name}_multiclass_test.npz",
                        y_true=test["category"].to_numpy(), y_pred=test_pred, y_proba=test_proba,
                        labels=np.array(encoder.classes_))

    report = {
        "model": name,
        "display_name": title,
        "family": "classical_ml_multiclass",
        "dataset_version": settings.dataset_version,
        "seed": seed,
        "labels": encoder.classes_.tolist(),
        "best_params": best["params"],
        "hyperparameter_search": search,
        "train_time_s_best_config": best["fit_time_s"],
        "search_time_s_total": search_time_s,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "validation_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "latency": latency,
        "resources": resources,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "xgboost": xgboost.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform(),
                        "cpu_count": __import__("os").cpu_count()},
    }
    metrics_dir = res / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{name}_multiclass.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def summary_line(r: dict) -> str:
    m = r["test_metrics"]
    return (f"{r['display_name']:<32} acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
            f"f1_weighted={m['f1_weighted']:.4f} best={r['best_params']} "
            f"lat={r['latency']['latency_single_ms_median']:.2f}ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES + ["all"], default="all")
    args = parser.parse_args()
    settings = get_settings()
    names = MODEL_NAMES if args.model == "all" else [args.model]
    for name in names:
        print(f"=== {name} (multiclass) ===", flush=True)
        report = train_and_evaluate(name, settings)
        print(summary_line(report), flush=True)


if __name__ == "__main__":
    main()
