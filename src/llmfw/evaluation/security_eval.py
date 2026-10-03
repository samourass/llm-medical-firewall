"""Évaluation scientifique du firewall d'ENTRÉE (section 1-6, 9-11 du document de cadrage).

Définitions utilisées (données une seule fois ici, réutilisées partout) :

- **Détecté** (pour l'Attack Detection Rate) : une requête d'attaque (catégorie vraie != benign)
  est comptée "détectée" si le modèle prédit une catégorie != `benign`, QUELLE QUE SOIT la
  catégorie exacte prédite. C'est une mesure large de "le firewall a réagi", distincte de
  l'exactitude de la catégorie (donnée séparément par les métriques multiclasses per-class).
- **FPR** (False Positive Rate) : proportion de requêtes BENIGN prédites comme une catégorie
  != benign (n'importe laquelle).
- **FNR** (False Negative Rate) : proportion de requêtes d'ATTAQUE (toute catégorie != benign)
  prédites comme `benign`. FNR = 100 - ADR par construction (même numérateur/dénominateur,
  vus depuis l'angle opposé) — reporté séparément pour respecter le document de cadrage.
- **Log-loss n'est PAS calculé** ici : ce n'est pas une métrique demandée par le document de
  cadrage (section 2), et la "confiance" retournée par le détecteur Regex n'est pas une
  probabilité calibrée — y appliquer log-loss serait une métrique mal formulée. Seules les
  métriques listées explicitement en section 2 sont calculées.
- **Latence** : mesurée par requête individuelle (`UnifiedClassifier.classify`, un seul texte
  à la fois, pas de traitement par lot), après 3 appels de "warm-up" non chronométrés pour
  exclure le temps de chargement du modèle (chargement paresseux, voir `models/unified.py`) du
  calcul de la latence d'inférence.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from llmfw.config import Settings, get_settings
from llmfw.evaluation.security_test_set import build_security_test_set
from llmfw.models.unified import BACKENDS, UnifiedClassifier

LABELS = ["benign", "jailbreak", "pii_exfiltration", "prompt_injection",
          "rag_prompt_injection", "secret_exfiltration"]
N_WARMUP = 3


def load_or_build_security_test_set(settings: Settings) -> pd.DataFrame:
    path = settings.data_dir / "security_eval" / "security_test_set.csv"
    if path.exists():
        return pd.read_csv(path)
    df = build_security_test_set()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _latency_stats(latencies_ms: list[float]) -> dict:
    arr = np.asarray(latencies_ms, dtype="float64")
    return {
        "min_ms": float(arr.min()), "max_ms": float(arr.max()), "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95)),
        "n_requests": int(len(arr)),
    }


def evaluate_backend(backend: str, df: pd.DataFrame, settings: Settings) -> dict:
    clf = UnifiedClassifier(backend=backend, settings=settings)

    # Warm-up (charge le modèle, exclu de la latence mesurée) ; capture les erreurs de
    # disponibilité (modèle non entraîné, dépendance absente, réseau bloqué) sans planter.
    try:
        for i in range(min(N_WARMUP, len(df))):
            clf.classify(df["text"].iloc[i])
    except Exception as exc:  # noqa: BLE001 - on veut capturer TOUTE cause d'indisponibilité
        return {"backend": backend, "status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}

    preds, confidences, latencies = [], [], []
    for text in df["text"]:
        t0 = time.perf_counter()
        result = clf.classify(text)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds.append(result["label"])
        confidences.append(result["confidence"])

    y_true = df["category"].to_numpy()
    y_pred = np.array(preds)

    accuracy = float((y_true == y_pred).mean())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)

    is_attack_true = y_true != "benign"
    is_attack_pred = y_pred != "benign"
    n_benign = int((~is_attack_true).sum())
    n_attack = int(is_attack_true.sum())
    false_positives = int(((~is_attack_true) & is_attack_pred).sum())
    false_negatives = int((is_attack_true & (~is_attack_pred)).sum())
    detected_attacks = int((is_attack_true & is_attack_pred).sum())

    fpr = 100.0 * false_positives / n_benign if n_benign else None
    fnr = 100.0 * false_negatives / n_attack if n_attack else None
    adr = 100.0 * detected_attacks / n_attack if n_attack else None

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    per_class = {
        lab: {"precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])}
        for i, lab in enumerate(LABELS)
    }

    return {
        "backend": backend, "status": "ok",
        "n_samples": int(len(df)), "n_benign": n_benign, "n_attack": n_attack,
        "accuracy": accuracy,
        "precision_macro": float(precision_macro), "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted), "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "false_positive_rate_pct": fpr, "false_negative_rate_pct": fnr,
        "attack_detection_rate_pct": adr,
        "confusion_matrix": cm.tolist(), "labels": LABELS,
        "per_class": per_class,
        "latency": _latency_stats(latencies),
        "predictions": pd.DataFrame({"text": df["text"], "true_category": y_true,
                                     "predicted_label": y_pred, "confidence": confidences,
                                     "latency_ms": latencies}),
    }


def evaluate_all_backends(settings: Settings | None = None,
                          backends: list[str] | None = None) -> dict[str, dict]:
    settings = settings or get_settings()
    df = load_or_build_security_test_set(settings)
    backends = backends or BACKENDS
    return {b: evaluate_backend(b, df, settings) for b in backends}
