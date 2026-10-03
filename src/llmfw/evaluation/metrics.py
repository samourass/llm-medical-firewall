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
    log_loss,
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


def compute_multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                               y_proba: np.ndarray | None, labels: Sequence[str]) -> dict:
    """Métriques multiclasses (6 catégories), calculées réellement. `y_proba` : shape (n, len(labels)),
    colonnes dans l'ordre de `labels`."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    per_class = {}
    for i, lab in enumerate(labels):
        yt = (y_true == lab).astype(int)
        yp = (y_pred == lab).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        per_class[lab] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int((y_true == lab).sum()),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    ll = None
    if y_proba is not None:
        try:
            y_true_idx = np.array([list(labels).index(v) for v in y_true])
            ll = float(log_loss(y_true_idx, y_proba, labels=list(range(len(labels)))))
        except Exception:
            ll = None
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "log_loss": ll,
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
        "per_class": per_class,
    }


def compute_security_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                             benign_label: str = "benign") -> dict:
    """Attack Detection Rate (ADR), False Positive Rate (FPR), False Negative Rate (FNR).

    Définitions (formulation binaire attack-vs-benign, appliquée à une sortie multiclasse) :
      - Une requête est "détectée comme attaque" si le label PRÉDIT (argmax du classifieur,
        AVANT tout seuil de confiance appliqué par le firewall opérationnel) est différent de
        "benign" — peu importe si la catégorie exacte prédite est correcte. C'est une mesure de
        la capacité de détection brute du classifieur, pas de la politique de blocage du
        firewall (qui applique en plus un seuil de confiance FIREWALL_THRESHOLD, voir
        docs/input_firewall.md — le taux de blocage opérationnel peut donc être inférieur à
        l'ADR calculé ici).
      - ADR = Requêtes d'attaque détectées / Total des requêtes d'attaque × 100
      - FPR = Requêtes bénignes classées à tort comme attaque / Total des requêtes bénignes × 100
      - FNR = Requêtes d'attaque classées comme bénignes / Total des requêtes d'attaque × 100
        (FNR = 100 - ADR, par construction : les deux sont calculés sur la même partition)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_attack_true = y_true != benign_label
    is_attack_pred = y_pred != benign_label

    n_attacks = int(is_attack_true.sum())
    n_benign = int((~is_attack_true).sum())
    detected = int((is_attack_true & is_attack_pred).sum())
    false_positives = int((~is_attack_true & is_attack_pred).sum())
    false_negatives = int((is_attack_true & ~is_attack_pred).sum())

    return {
        "n_attacks": n_attacks,
        "n_benign": n_benign,
        "attack_detection_rate_pct": (100.0 * detected / n_attacks) if n_attacks else None,
        "false_positive_rate_pct": (100.0 * false_positives / n_benign) if n_benign else None,
        "false_negative_rate_pct": (100.0 * false_negatives / n_attacks) if n_attacks else None,
        "detected_attacks": detected,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "definition": (
            "ADR = attacks with predicted label != 'benign' / total attacks x 100 "
            "(raw classifier output, before any confidence threshold). "
            "FPR = benign requests with predicted label != 'benign' / total benign x 100. "
            "FNR = attacks with predicted label == 'benign' / total attacks x 100."
        ),
    }


def latency_stats_from_samples(latencies_ms: Sequence[float]) -> dict:
    """min/max/mean/median/p95, calculés sur des latences RÉELLEMENT mesurées (une par requête,
    inférence unitaire). Ne jamais appeler avec une liste vide."""
    arr = np.asarray(latencies_ms, dtype="float64")
    return {
        "n": int(len(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def process_rss_mb() -> float:
    """Mémoire résidente du processus (Mo). Mesure approximative, multiplateforme."""
    return psutil.Process().memory_info().rss / (1024 * 1024)
