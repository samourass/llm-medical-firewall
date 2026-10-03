"""Génère les figures (section 12) et les fichiers de résultats (section 13) de l'évaluation
de sécurité, à partir des résultats RÉELLEMENT mesurés par `security_eval.py`,
`output_firewall_eval.py` et `robustness_eval.py`. Aucune valeur n'est inventée : un modèle
`skipped` (ex. DistilBERT non entraîné) est simplement absent des graphiques comparatifs plutôt
que représenté par une valeur par défaut.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from llmfw.evaluation.plots import plot_confusion_matrix_multiclass

DISPLAY_NAME = {"regex": "Regex", "logistic_regression": "Logistic Regression",
                "xgboost": "XGBoost", "distilbert": "DistilBERT"}


def _ok_backends(results: dict) -> list[str]:
    return [b for b, r in results.items() if r.get("status") == "ok"]


def _bar_chart(labels: list[str], values: list[float], title: str, ylabel: str, path: Path,
              color: str = "#3B6FA0") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values + [1]) * 1.15 if values else 1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}",
               ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_figures(input_results: dict, output_results: dict, robustness_results: dict,
                     figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    backends = _ok_backends(input_results)
    display = [DISPLAY_NAME.get(b, b) for b in backends]

    # 1-4, 7-9 : comparaisons entre modèles
    metrics_to_plot = [
        ("f1_model_comparison.png", "f1_macro", "F1 macro — comparaison des modèles"),
        ("recall_model_comparison.png", "recall_macro", "Recall macro — comparaison des modèles"),
        ("precision_model_comparison.png", "precision_macro", "Precision macro — comparaison des modèles"),
        ("attack_detection_rate.png", "attack_detection_rate_pct", "Attack Detection Rate (%)"),
        ("false_positive_rate.png", "false_positive_rate_pct", "False Positive Rate (%)"),
        ("false_negative_rate.png", "false_negative_rate_pct", "False Negative Rate (%)"),
    ]
    for filename, key, title in metrics_to_plot:
        values = [input_results[b][key] for b in backends]
        path = figures_dir / filename
        _bar_chart(display, values, title, key, path)
        written.append(str(path))

    # 4bis : latence (médiane), échelle log utile vu l'écart regex/ML
    lat_values = [input_results[b]["latency"]["median_ms"] for b in backends]
    path = figures_dir / "latency_model_comparison.png"
    _bar_chart(display, lat_values, "Latence médiane par requête (ms)", "ms", path, color="#A0523B")
    written.append(str(path))

    # 5 : matrice de confusion par modèle
    for b in backends:
        r = input_results[b]
        path = figures_dir / f"confusion_matrix_{b}.png"
        plot_confusion_matrix_multiclass(r["confusion_matrix"], r["labels"],
                                         f"{DISPLAY_NAME.get(b, b)} — matrice de confusion "
                                         f"(jeu d'évaluation de sécurité, n={r['n_samples']})",
                                         path)
        written.append(str(path))

    # 6 : F1 par catégorie, tous modèles regroupés
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = input_results[backends[0]]["labels"] if backends else []
    width = 0.8 / max(len(backends), 1)
    x = range(len(labels))
    for i, b in enumerate(backends):
        f1s = [input_results[b]["per_class"][lab]["f1"] for lab in labels]
        ax.bar([xi + i * width for xi in x], f1s, width=width, label=DISPLAY_NAME.get(b, b))
    ax.set_xticks([xi + width * (len(backends) - 1) / 2 for xi in x])
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("F1")
    ax.set_title("F1 par catégorie et par modèle (jeu d'évaluation de sécurité)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / "per_class_f1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # 10 : robustesse — accuracy par variation, tous modèles regroupés
    rb_backends = [b for b, r in robustness_results.items() if r.get("status") == "ok"]
    if rb_backends:
        variations = [row["variation"] for row in robustness_results[rb_backends[0]]["by_variation"]]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        width = 0.8 / max(len(rb_backends), 1)
        x = range(len(variations))
        for i, b in enumerate(rb_backends):
            acc = [row["accuracy_pct"] for row in robustness_results[b]["by_variation"]]
            ax.bar([xi + i * width for xi in x], acc, width=width, label=DISPLAY_NAME.get(b, b))
        ax.set_xticks([xi + width * (len(rb_backends) - 1) / 2 for xi in x])
        ax.set_xticklabels(variations, rotation=30, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Performance de robustesse par type de variation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = figures_dir / "robustness_performance.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    return written


def write_results_files(input_results: dict, output_results: dict, robustness_results: dict,
                        results_dir: Path) -> dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # model_metrics.csv
    rows = []
    for b, r in input_results.items():
        if r["status"] != "ok":
            rows.append({"model": b, "status": "skipped", "reason": r["reason"]})
            continue
        rows.append({
            "model": b, "status": "ok", "n_samples": r["n_samples"], "accuracy": r["accuracy"],
            "precision_macro": r["precision_macro"], "recall_macro": r["recall_macro"],
            "f1_macro": r["f1_macro"], "precision_weighted": r["precision_weighted"],
            "recall_weighted": r["recall_weighted"], "f1_weighted": r["f1_weighted"],
            "false_positive_rate_pct": r["false_positive_rate_pct"],
            "false_negative_rate_pct": r["false_negative_rate_pct"],
            "attack_detection_rate_pct": r["attack_detection_rate_pct"],
            "latency_median_ms": r["latency"]["median_ms"], "latency_p95_ms": r["latency"]["p95_ms"],
        })
    df = pd.DataFrame(rows)
    path = results_dir / "model_metrics.csv"
    df.to_csv(path, index=False)
    paths["model_metrics"] = str(path)

    # class_metrics.csv
    rows = []
    for b, r in input_results.items():
        if r["status"] != "ok":
            continue
        for lab, pc in r["per_class"].items():
            rows.append({"model": b, "category": lab, **pc})
    path = results_dir / "class_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    paths["class_metrics"] = str(path)

    # latency_metrics.csv
    rows = []
    for b, r in input_results.items():
        if r["status"] != "ok":
            continue
        rows.append({"model": b, **r["latency"]})
    path = results_dir / "latency_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    paths["latency_metrics"] = str(path)

    # security_metrics.csv (ADR / FPR / FNR summary)
    rows = []
    for b, r in input_results.items():
        if r["status"] != "ok":
            rows.append({"model": b, "status": "skipped"})
            continue
        rows.append({"model": b, "status": "ok", "n_benign": r["n_benign"], "n_attack": r["n_attack"],
                    "attack_detection_rate_pct": r["attack_detection_rate_pct"],
                    "false_positive_rate_pct": r["false_positive_rate_pct"],
                    "false_negative_rate_pct": r["false_negative_rate_pct"]})
    path = results_dir / "security_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    paths["security_metrics"] = str(path)

    # robustness_metrics.csv
    rows = []
    for b, r in robustness_results.items():
        if r.get("status") != "ok":
            continue
        for row in r["by_variation"]:
            rows.append({"model": b, **row})
    path = results_dir / "robustness_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    paths["robustness_metrics"] = str(path)

    # output_firewall_metrics.csv
    of_row = {k: v for k, v in output_results.items() if k not in ("predictions", "by_category")}
    df_of = pd.DataFrame(output_results["by_category"])
    df_of["overall_detection_rate_pct"] = of_row["detection_rate_pct"]
    df_of["overall_false_positive_rate_pct"] = of_row["false_positive_rate_pct"]
    path = results_dir / "output_firewall_metrics.csv"
    df_of.to_csv(path, index=False)
    paths["output_firewall_metrics"] = str(path)

    # evaluation_results.json (tout, agrégé)
    def _clean(r):
        return {k: v for k, v in r.items() if k != "predictions"}

    aggregate = {
        "input_firewall": {b: _clean(r) for b, r in input_results.items()},
        "output_firewall": {k: v for k, v in output_results.items() if k != "predictions"},
        "robustness": {b: {k: v for k, v in r.items() if k != "predictions"}
                      for b, r in robustness_results.items()},
    }
    path = results_dir / "evaluation_results.json"
    path.write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    paths["evaluation_results"] = str(path)

    # prédictions détaillées, utile pour audit / annexes
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for b, r in input_results.items():
        if r["status"] == "ok":
            r["predictions"].to_csv(pred_dir / f"security_eval_{b}.csv", index=False)
    output_results["predictions"].to_csv(pred_dir / "output_firewall_eval.csv", index=False)

    return paths
