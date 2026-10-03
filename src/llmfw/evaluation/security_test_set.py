"""Génère le jeu d'évaluation de sécurité dédié (section 9 du document de cadrage) :
au moins 600 requêtes (100 par catégorie sur 6 catégories), disjointes de train/validation/test.

"Détecté" est défini une fois pour toutes dans `evaluation/run_input_firewall_eval.py` : une
ligne d'attaque est comptée "détectée" si le modèle prédit une catégorie != benign pour cette
ligne (peu importe si la catégorie exacte prédite diffère de la catégorie réelle — c'est la
définition de l'Attack Detection Rate au sens large ; le detail par catégorie exacte est donné
séparément par les métriques multiclasses per-class).

Usage (depuis la racine du projet) :
    python -m llmfw.evaluation.security_test_set
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from llmfw.config import get_settings
from llmfw.data.generate import normalize_text
from llmfw.evaluation.security_test_templates import DISTINCT_PAIRS, SLOTS, SPECS

TARGET_PER_CATEGORY = 100


def _fill(template: str, rng: random.Random) -> str:
    import string

    fields = list(dict.fromkeys(f for _, f, _, _ in string.Formatter().parse(template) if f))
    for _ in range(25):
        values = {f: rng.choice(SLOTS[f]) for f in fields}
        if all(not (a in values and b in values and values[a] == values[b]) for a, b in DISTINCT_PAIRS):
            return template.format(**values)
    return template.format(**values)


def _existing_normalized_texts(data_dir: Path) -> set[str]:
    seen = set()
    for split in ("train", "validation", "test"):
        path = data_dir / split / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            seen.update(df["text"].map(normalize_text))
    return seen


def build_security_test_set(seed: int = 2026, data_dir: Path | None = None,
                            target_per_category: int = TARGET_PER_CATEGORY) -> pd.DataFrame:
    settings = get_settings()
    data_dir = data_dir or settings.data_dir
    excluded = _existing_normalized_texts(data_dir)

    rng = random.Random(seed)
    rows: list[dict] = []
    for category, templates in SPECS:
        seen_here: set[str] = set()
        produced = 0
        attempts = 0
        max_attempts = target_per_category * 60
        while produced < target_per_category and attempts < max_attempts:
            attempts += 1
            template = rng.choice(templates)
            text = _fill(template, rng)
            key = normalize_text(text)
            if key in excluded or key in seen_here:
                continue
            seen_here.add(key)
            rows.append({"text": text, "category": category,
                        "label": 0 if category == "benign" else 1})
            produced += 1
        if produced < target_per_category:
            raise RuntimeError(
                f"Impossible de produire {target_per_category} exemples uniques et disjoints "
                f"pour '{category}' (obtenu : {produced}). Ajoutez des gabarits."
            )

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", [f"sec_eval_{i:04d}" for i in range(len(df))])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    out_path = args.out or (settings.data_dir / "security_eval" / "security_test_set.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_security_test_set(seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"{len(df)} lignes écrites dans {out_path}")
    print(df["category"].value_counts().to_dict())

    # Audit anti-fuite explicite, mesuré, pas supposé.
    train_val_test_norm = _existing_normalized_texts(settings.data_dir)
    overlap = df["text"].map(normalize_text).isin(train_val_test_norm).sum()
    internal_dupes = df["text"].map(normalize_text).duplicated().sum()
    print(f"chevauchement avec train/validation/test : {overlap} (attendu 0)")
    print(f"doublons internes au jeu d'évaluation : {internal_dupes} (attendu 0)")


if __name__ == "__main__":
    main()
