"""Génération déterministe du dataset synthétique défensif."""
from __future__ import annotations

import math
import random
import re
import string

import pandas as pd

from llmfw.data import templates as T

_FORMATTER = string.Formatter()


def normalize_text(text: str) -> str:
    """Clé de déduplication : minuscules, sans ponctuation, espaces réduits."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fields(template: str) -> list[str]:
    return [f for _, f, _, _ in _FORMATTER.parse(template) if f]


def _fill(template: str, rng: random.Random) -> str:
    fields = list(dict.fromkeys(_fields(template)))
    for _ in range(20):  # retry jusqu'à respecter les paires "distinctes"
        values = {f: rng.choice(T.SLOTS[f]) for f in fields}
        if all(not (a in values and b in values and values[a] == values[b])
               for a, b in T.DISTINCT_PAIRS):
            return template.format(**values)
    return template.format(**values)


def _decorate(text: str, rng: random.Random) -> str:
    text = rng.choice(T.PREFIXES) + text + rng.choice(T.SUFFIXES)
    if rng.random() < 0.08:
        text = text.lower()
    return text.strip()


def generate_dataset(seed: int = 42, size_scale: float = 1.0) -> pd.DataFrame:
    """Retourne un DataFrame [text, label, category, subtype, template_id].

    `size_scale` permet de réduire/augmenter le volume (tests rapides : 0.1).
    Les doublons (après normalisation) sont supprimés à la génération.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()

    for category, subtype, templates, wrap, target in T.SPECS:
        per_template = max(1, math.ceil(target * size_scale / len(templates)))
        for idx, template in enumerate(templates):
            template_id = f"{subtype}:{idx:02d}"
            produced = 0
            for _ in range(per_template * 12):  # tentatives max
                if produced >= per_template:
                    break
                text = _fill(template, rng)
                if wrap:
                    text = _decorate(text, rng)
                key = normalize_text(text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "text": text,
                    "label": T.CATEGORY_LABEL[category],
                    "category": category,
                    "subtype": subtype,
                    "template_id": template_id,
                })
                produced += 1

    df = pd.DataFrame(rows)
    # mélange déterministe
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def category_counts(df: pd.DataFrame) -> dict[str, int]:
    return df["category"].value_counts().to_dict()

