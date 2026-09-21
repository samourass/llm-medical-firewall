"""Entraînement + évaluation des classifieurs classiques : TF-IDF -> classifieur.

Protocole : petite grille d'hyperparamètres, sélection sur VALIDATION (F1), puis évaluation
unique sur TEST. Rien du test n'est utilisé pour choisir la configuration.

Usage (depuis la racine du projet) :
    python -m llmfw.training.classical --model logistic_regression
    python -m llmfw.training.classical --model all
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from llmfw.config import Settings, get_settings
from llmfw.evaluation.metrics import (
    breakdown_by_group,
    compute_binary_metrics,
    measure_latency,
    process_rss_mb,
)
from llmfw.evaluation.plots import plot_confusion_matrix, plot_precision_recall, plot_roc

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
DISPLAY = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
           "xgboost": "XGBoost"}


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)


def param_grid(name: str) -> list[dict]:
    grids = {
        "logistic_regression": {"C": [0.1, 1.0, 10.0]},
        "random_forest": {"n_estimators": [100, 300], "max_depth": [None, 40]},
        "xgboost": {"n_estimators": [200, 400], "max_depth": [4, 6]},
    }[name]
    keys = list(grids)
    return [dict(zip(keys, values)) for values in itertools.product(*grids.values())]


def make_classifier(name: str, params: dict, seed: int, y_train: np.ndarray):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **params)
    if name == "xgboost":
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", learning_rate=0.1,
                             tree_method="hist", scale_pos_weight=neg / pos, n_jobs=-1,
                             random_state=seed, **params)
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
    y_train = train["label"].to_numpy()

    # --- 1) Recherche d'hyperparamètres sur la validation -----------------------------------
    search = []
    best = None
    search_start = time.perf_counter()
    for params in param_grid(name):
        pipe = Pipeline([("tfidf", make_tfidf()),
                         ("clf", make_classifier(name, params, seed, y_train))])
        t0 = time.perf_counter()
        pipe.fit(train["text"], y_train)
        fit_s = time.perf_counter() - t0
        val_score = pipe.predict_proba(val["text"])[:, 1]
        val_pred = (val_score >= 0.5).astype(int)
        m = compute_binary_metrics(val["label"], val_pred, val_score)
        search.append({"params": params, "fit_time_s": fit_s,
                       "val_f1": m["f1"], "val_roc_auc": m["roc_auc"], "val_fpr": m["fpr"],
                       "val_fnr": m["fnr"]})
        if best is None or m["f1"] > best["val_f1"]:
            best = {"params": params, "pipe": pipe, "val_f1": m["f1"], "fit_time_s": fit_s,
                    "val_metrics": m}
    search_time_s = time.perf_counter() - search_start

    pipe: Pipeline = best["pipe"]

    # --- 2) Évaluation unique sur le TEST --------------------------------------------------
    test_score = pipe.predict_proba(test["text"])[:, 1]
    test_pred = (test_score >= 0.5).astype(int)
    test_metrics = compute_binary_metrics(test["label"], test_pred, test_score)
    breakdown = breakdown_by_group(test, test_pred)

    # --- 3) Latence, mémoire, taille --------------------------------------------------------
    latency = measure_latency(lambda batch: pipe.predict_proba(batch), test["text"].tolist())
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.models_dir / f"{name}.joblib"
    joblib.dump(pipe, model_path)
    resources = {
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
        "process_rss_mb_after_inference": process_rss_mb(),
    }

    # --- 4) Artefacts (figures, prédictions) ----------------------------------------------
    res = settings.results_dir
    title = DISPLAY[name]
    plot_confusion_matrix(test["label"], test_pred, f"{title} — test", res / "confusion_matrices" / f"{name}.png")
    plot_roc(test["label"], test_score, f"{title} — ROC (test)", res / "roc" / f"{name}.png")
    plot_precision_recall(test["label"], test_score, f"{title} — PR (test)",
                          res / "precision_recall" / f"{name}.png")
    (res / "predictions").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(res / "predictions" / f"{name}_test.npz",
                        y_true=test["label"].to_numpy(), y_score=test_score)

    report = {
        "model": name,
        "display_name": title,
        "family": "classical_ml",
        "dataset_version": settings.dataset_version,
        "seed": seed,
        "threshold": 0.5,
        "best_params": best["params"],
        "hyperparameter_search": search,
        "train_time_s_best_config": best["fit_time_s"],
        "search_time_s_total": search_time_s,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "validation_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
        "test_breakdown": breakdown,
        "latency": latency,
        "resources": resources,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "xgboost": xgboost.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "platform": platform.platform(),
                        "cpu_count": __import__("os").cpu_count()},
    }
    metrics_dir = res / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{name}.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    return report


def summary_line(r: dict) -> str:
    m = r["test_metrics"]
    return (f"{r['display_name']:<20} acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} "
            f"FPR={m['fpr']:.4f} FNR={m['fnr']:.4f} "
            f"lat={r['latency']['latency_single_ms_median']:.2f}ms best={r['best_params']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES + ["all"], default="all")
    args = parser.parse_args()
    settings = get_settings()
    names = MODEL_NAMES if args.model == "all" else [args.model]
    for name in names:
        print(f"=== {name} ===", flush=True)
        report = train_and_evaluate(name, settings)
        print(summary_line(report), flush=True)


if __name__ == "__main__":
    main()
