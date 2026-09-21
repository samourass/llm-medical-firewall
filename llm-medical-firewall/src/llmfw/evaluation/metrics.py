"""Métriques de classification binaire (0 = benign, 1 = attack), calculées réellement."""
from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                           y_score: Sequence[float]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,  # FPR = FP / (FP + TN)
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,  # FNR = FN / (FN + TP)
    }


def breakdown_by_group(df: pd.DataFrame, y_pred: Sequence[int]) -> dict:
    """Attaques : taux de détection par catégorie. Bénin : taux de faux positifs par sous-type."""
    df = df.assign(_pred=np.asarray(y_pred))
    out: dict = {"attack_detection_rate": {}, "benign_false_positive_rate": {}}
    for cat, g in df[df["label"] == 1].groupby("category"):
        out["attack_detection_rate"][cat] = {"n": int(len(g)), "rate": float(g["_pred"].mean())}
    for sub, g in df[df["label"] == 0].groupby("subtype"):
        out["benign_false_positive_rate"][sub] = {"n": int(len(g)), "rate": float(g["_pred"].mean())}
    return out


def measure_latency(predict_fn: Callable[[list[str]], object], texts: Sequence[str],
                    n_single: int = 200, warmup: int = 5) -> dict:
    """Latence d'inférence (ms) : requêtes unitaires (cas d'usage firewall) + débit en lot."""
    texts = list(texts)
    single = texts[:n_single]
    for t in single[:warmup]:
        predict_fn([t])
    times = []
    for t in single:
        start = time.perf_counter()
        predict_fn([t])
        times.append((time.perf_counter() - start) * 1000.0)
    start = time.perf_counter()
    predict_fn(texts)
    batch_total = (time.perf_counter() - start) * 1000.0
    return {
        "latency_single_ms_mean": float(np.mean(times)),
        "latency_single_ms_median": float(np.median(times)),
        "latency_single_ms_p95": float(np.percentile(times, 95)),
        "latency_batch_ms_per_sample": float(batch_total / len(texts)),
        "latency_n_single": len(single),
    }


def process_rss_mb() -> float:
    """Mémoire résidente du processus (Mo). Mesure approximative, multiplateforme."""
    return psutil.Process().memory_info().rss / (1024 * 1024)
