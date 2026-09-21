"""Split 70/15/15 stratifié, disjoint par gabarit, avec filtrage des quasi-doublons.

Pourquoi par gabarit ? Deux textes issus du même gabarit ne diffèrent que par quelques
mots : les mettre l'un en train et l'autre en test gonflerait artificiellement les scores.
Chaque gabarit (`template_id`) est donc affecté à UN SEUL split. La stratification se fait
par sous-type (`subtype`) : chaque catégorie est représentée dans les trois splits.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from llmfw.data.generate import normalize_text

SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def group_stratified_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Ajoute une colonne `split`. Affectation gloutonne des groupes (gabarits)."""
    rng = random.Random(seed)
    df = df.copy()
    assignment: dict[str, str] = {}

    for subtype, sub in df.groupby("subtype"):
        sizes = sub.groupby("template_id").size().to_dict()
        groups = sorted(sizes)
        rng.shuffle(groups)
        total = sum(sizes.values())
        target = {s: RATIOS[s] * total for s in SPLITS}
        current = {s: 0 for s in SPLITS}
        for g in groups:
            # split le plus "en retard" par rapport à sa cible
            best = max(SPLITS, key=lambda s: (target[s] - current[s]) / target[s])
            assignment[g] = best
            current[best] += sizes[g]

    df["split"] = df["template_id"].map(assignment)
    return df


def drop_near_duplicates(df: pd.DataFrame, threshold: float = 0.90) -> tuple[pd.DataFrame, dict]:
    """Supprime de validation/test les exemples trop proches d'un exemple de train.

    Similarité = cosinus TF-IDF (mots 1-2 grammes). Le train n'est jamais modifié.
    """
    train = df[df["split"] == "train"]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x_train = vec.fit_transform(train["text"].map(normalize_text))

    keep = pd.Series(True, index=df.index)
    report: dict = {"threshold": threshold, "removed": {}, "max_similarity": {}}
    for split in ("validation", "test"):
        part = df[df["split"] == split]
        x = vec.transform(part["text"].map(normalize_text))
        sims = (x @ x_train.T).max(axis=1).toarray().ravel()
        too_close = sims >= threshold
        keep.loc[part.index[too_close]] = False
        report["removed"][split] = int(too_close.sum())
        report["max_similarity"][split] = {
            "mean": round(float(sims.mean()), 4),
            "p95": round(float(np.percentile(sims, 95)), 4),
            "max_before_filter": round(float(sims.max()), 4),
            "max_after_filter": round(float(sims[~too_close].max()), 4) if (~too_close).any() else 0.0,
        }
    return df[keep].reset_index(drop=True), report


def leakage_audit(df: pd.DataFrame) -> dict:
    """Vérifie l'absence de doublons exacts et de gabarits partagés entre splits."""
    df = df.assign(_key=df["text"].map(normalize_text))
    by_split = {s: set(df.loc[df["split"] == s, "_key"]) for s in SPLITS}
    tmpl = {s: set(df.loc[df["split"] == s, "template_id"]) for s in SPLITS}
    return {
        "exact_duplicates_train_test": len(by_split["train"] & by_split["test"]),
        "exact_duplicates_train_validation": len(by_split["train"] & by_split["validation"]),
        "exact_duplicates_validation_test": len(by_split["validation"] & by_split["test"]),
        "shared_templates_train_test": len(tmpl["train"] & tmpl["test"]),
        "shared_templates_train_validation": len(tmpl["train"] & tmpl["validation"]),
        "shared_templates_validation_test": len(tmpl["validation"] & tmpl["test"]),
    }
