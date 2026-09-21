"""Construit le dataset : génération -> split -> audit -> CSV + fiche dataset.

Usage (depuis la racine du projet) :
    python -m llmfw.data.build
    python -m llmfw.data.build --scale 0.1 --out-dir data_tmp   # petit essai
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llmfw.config import get_settings
from llmfw.data.generate import generate_dataset
from llmfw.data.split import SPLITS, drop_near_duplicates, group_stratified_split, leakage_audit

COLUMNS = ["text", "label", "category", "subtype", "template_id"]


def build(out_dir: Path, seed: int, scale: float, version: str) -> dict:
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    for d in (raw_dir, processed_dir, *(out_dir / s for s in SPLITS)):
        d.mkdir(parents=True, exist_ok=True)

    full = generate_dataset(seed=seed, size_scale=scale)
    full.to_csv(raw_dir / "synthetic_full.csv", index=False)

    split_df = group_stratified_split(full, seed=seed)
    final_df, near_dup_report = drop_near_duplicates(split_df, threshold=0.90)
    audit = leakage_audit(final_df)

    final_df.to_csv(processed_dir / "dataset.csv", index=False)
    for s in SPLITS:
        final_df[final_df["split"] == s][COLUMNS].to_csv(out_dir / s / f"{s}.csv", index=False)

    n = len(final_df)
    card = {
        "dataset_version": version,
        "seed": seed,
        "language": "en",
        "source": "synthetic (templates in src/llmfw/data/templates.py) - no public dataset used yet",
        "generated_rows_before_filter": int(len(full)),
        "final_rows": int(n),
        "split_method": "stratified by subtype, disjoint by template_id (70/15/15 target)",
        "split_sizes": {s: int((final_df["split"] == s).sum()) for s in SPLITS},
        "split_ratios_actual": {s: round(float((final_df["split"] == s).mean()), 4) for s in SPLITS},
        "binary_label": "0 = benign, 1 = attack (any of the 5 attack categories)",
        "category_counts": {k: int(v) for k, v in final_df["category"].value_counts().items()},
        "category_counts_per_split": {
            s: {k: int(v) for k, v in final_df[final_df["split"] == s]["category"].value_counts().items()}
            for s in SPLITS
        },
        "attack_ratio_per_split": {
            s: round(float(final_df[final_df["split"] == s]["label"].mean()), 4) for s in SPLITS
        },
        "near_duplicate_filter": near_dup_report,
        "leakage_audit": audit,
        "limitations": [
            "Synthetic template data: scores will be optimistic compared with real-world attacks.",
            "Splits are disjoint by template, so test measures generalisation to unseen templates, "
            "but slot vocabulary (drugs, conditions, canary strings) is shared across splits.",
            "English only.",
        ],
    }
    (processed_dir / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the synthetic defensive dataset.")
    parser.add_argument("--out-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--scale", type=float, default=1.0, help="volume multiplier (1.0 = ~6000 rows)")
    args = parser.parse_args()

    card = build(args.out_dir, args.seed, args.scale, settings.dataset_version)
    print(json.dumps({k: card[k] for k in (
        "final_rows", "split_sizes", "split_ratios_actual", "category_counts",
        "attack_ratio_per_split", "leakage_audit")}, indent=2))
    print("near-duplicate filter:", json.dumps(card["near_duplicate_filter"]))
    print(f"Dataset written to: {args.out_dir}")


if __name__ == "__main__":
    main()
