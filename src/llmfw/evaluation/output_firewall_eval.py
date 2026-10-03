"""Évaluation scientifique du firewall de SORTIE (section 8 du document de cadrage), indépendante
du firewall d'entrée.

Définitions :
- **Détecté** : `should_flag=True` (une fuite attendue) ET l'action retournée est `REDACT` ou
  `BLOCK` (les deux comptent comme "la fuite a été interceptée", elles diffèrent seulement dans
  la façon dont la réponse est traitée ensuite).
- **Faux positif** : `should_flag=False` (réponse bénigne) mais l'action retournée n'est PAS
  `ALLOW`.
- **Faux négatif** : `should_flag=True` mais l'action retournée est `ALLOW`.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from llmfw.config import Settings, get_settings
from llmfw.evaluation.output_firewall_test_set import as_records
from llmfw.firewall.output import OutputFirewall

N_WARMUP = 2


def evaluate_output_firewall(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    records = as_records()
    of = OutputFirewall(settings)

    for r in records[:N_WARMUP]:
        of.inspect(r["text"])

    rows = []
    latencies = []
    for r in records:
        t0 = time.perf_counter()
        result = of.inspect(r["text"])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        rows.append({**r, "action": result["action"], "findings": ",".join(result["findings"]),
                    "latency_ms": latency_ms})

    df = pd.DataFrame(rows)
    should_flag = df["should_flag"]
    flagged = df["action"] != "ALLOW"

    true_positives = int((should_flag & flagged).sum())
    false_negatives = int((should_flag & ~flagged).sum())
    false_positives = int((~should_flag & flagged).sum())
    true_negatives = int((~should_flag & ~flagged).sum())

    n_leak = int(should_flag.sum())
    n_benign = int((~should_flag).sum())

    detection_rate_pct = 100.0 * true_positives / n_leak if n_leak else None
    false_positive_rate_pct = 100.0 * false_positives / n_benign if n_benign else None
    false_negative_rate_pct = 100.0 * false_negatives / n_leak if n_leak else None

    arr = np.asarray(latencies)
    latency = {"min_ms": float(arr.min()), "max_ms": float(arr.max()), "mean_ms": float(arr.mean()),
              "median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95)),
              "n_requests": int(len(arr))}

    by_category = (
        df[should_flag].groupby("expected_category")
        .apply(lambda g: pd.Series({"n": len(g), "n_detected": int((g["action"] != "ALLOW").sum())}))
        .reset_index()
    )
    by_category["detection_rate_pct"] = 100.0 * by_category["n_detected"] / by_category["n"]

    return {
        "n_cases": len(records), "n_leak_cases": n_leak, "n_benign_cases": n_benign,
        "true_positives": true_positives, "false_negatives": false_negatives,
        "false_positives": false_positives, "true_negatives": true_negatives,
        "detection_rate_pct": detection_rate_pct,
        "false_positive_rate_pct": false_positive_rate_pct,
        "false_negative_rate_pct": false_negative_rate_pct,
        "latency": latency,
        "by_category": by_category.to_dict(orient="records"),
        "predictions": df,
    }
