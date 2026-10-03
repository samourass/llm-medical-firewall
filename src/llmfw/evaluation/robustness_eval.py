"""Évaluation de robustesse (section 7) : comment la détection change sous perturbation.

Pour les lignes d'attaque (catégorie != benign) : "correct" = le modèle prédit une catégorie
!= benign (même définition de "détecté" que dans `security_eval.py`).
Pour les lignes benign : "correct" = le modèle prédit `benign` (pas de faux positif).
"""
from __future__ import annotations

import pandas as pd

from llmfw.config import Settings, get_settings
from llmfw.evaluation.robustness import build_robustness_dataset
from llmfw.models.unified import BACKENDS, UnifiedClassifier


def evaluate_robustness(settings: Settings | None = None, backends: list[str] | None = None) -> dict:
    settings = settings or get_settings()
    backends = backends or BACKENDS
    rows = build_robustness_dataset()
    df = pd.DataFrame(rows)

    per_backend = {}
    for backend in backends:
        clf = UnifiedClassifier(backend=backend, settings=settings)
        try:
            clf.classify(df["text"].iloc[0])
        except Exception as exc:  # noqa: BLE001
            per_backend[backend] = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
            continue

        preds = [clf.classify(t)["label"] for t in df["text"]]
        d = df.copy()
        d["predicted_label"] = preds
        d["is_attack"] = d["category"] != "benign"
        d["correct"] = ((d["is_attack"] & (d["predicted_label"] != "benign")) |
                        (~d["is_attack"] & (d["predicted_label"] == "benign")))

        by_variation = (
            d.groupby("variation")["correct"].agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "accuracy", "count": "n"})
        )
        by_variation["accuracy_pct"] = (by_variation["accuracy"] * 100).round(2)

        overall_accuracy = float(d["correct"].mean())
        baseline_accuracy = float(d[d["variation"] == "original"]["correct"].mean())

        per_backend[backend] = {
            "status": "ok",
            "overall_accuracy": overall_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "by_variation": by_variation[["variation", "n", "accuracy_pct"]].to_dict(orient="records"),
            "predictions": d,
        }

    return per_backend
