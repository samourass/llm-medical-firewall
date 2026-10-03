#!/usr/bin/env python3
"""Exécute l'évaluation de sécurité complète du LLM Firewall (section 15 du document de
cadrage) :

    1. Load test data           (jeu de sécurité 600 lignes, jeu de robustesse, jeu de sortie)
    2. Evaluate Regex
    3. Evaluate Logistic Regression
    4. Evaluate XGBoost
    5. Evaluate DistilBERT       (marqué "skipped" avec la raison exacte si indisponible)
    6. Evaluate Output Firewall
    7. Calculate metrics
    8. Generate plots            (results/figures/)
    9. Save results              (results/*.csv, results/evaluation_results.json)
    10. Generate report          (docs/security_evaluation.md)

Usage (depuis la racine du projet) :
    python evaluation/run_full_evaluation.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llmfw.config import get_settings  # noqa: E402
from llmfw.evaluation.output_firewall_eval import evaluate_output_firewall  # noqa: E402
from llmfw.evaluation.report_artifacts import generate_figures, write_results_files  # noqa: E402
from llmfw.evaluation.report_writer import render_report  # noqa: E402
from llmfw.evaluation.robustness_eval import evaluate_robustness  # noqa: E402
from llmfw.evaluation.security_eval import evaluate_all_backends, load_or_build_security_test_set  # noqa: E402


def main() -> None:
    t_start = time.perf_counter()
    settings = get_settings()

    print("[1/10] Loading / building security test set...", flush=True)
    df = load_or_build_security_test_set(settings)
    print(f"       {len(df)} requests, {df['category'].value_counts().to_dict()}", flush=True)

    input_results = {}
    for i, backend in enumerate(["regex", "logistic_regression", "xgboost", "distilbert"], start=2):
        print(f"[{i}/10] Evaluating input firewall backend: {backend}...", flush=True)
        result = evaluate_all_backends(settings=settings, backends=[backend])[backend]
        input_results[backend] = result
        if result["status"] == "ok":
            print(f"       accuracy={result['accuracy']:.4f} f1_macro={result['f1_macro']:.4f} "
                 f"ADR={result['attack_detection_rate_pct']:.2f}% FPR={result['false_positive_rate_pct']:.2f}% "
                 f"FNR={result['false_negative_rate_pct']:.2f}% latency_median={result['latency']['median_ms']:.3f}ms",
                 flush=True)
        else:
            print(f"       SKIPPED: {result['reason']}", flush=True)

    print("[6/10] Evaluating output firewall...", flush=True)
    output_results = evaluate_output_firewall(settings=settings)
    print(f"       detection_rate={output_results['detection_rate_pct']:.2f}% "
         f"FPR={output_results['false_positive_rate_pct']:.2f}% "
         f"FNR={output_results['false_negative_rate_pct']:.2f}%", flush=True)

    print("[6bis/10] Evaluating robustness...", flush=True)
    robustness_results = evaluate_robustness(settings=settings,
                                             backends=["regex", "logistic_regression", "xgboost", "distilbert"])
    for b, r in robustness_results.items():
        if r.get("status") == "ok":
            print(f"       {b}: overall_accuracy={r['overall_accuracy']*100:.2f}% "
                 f"baseline={r['baseline_accuracy']*100:.2f}%", flush=True)
        else:
            print(f"       {b}: SKIPPED ({r['reason']})", flush=True)

    print("[7/10] Metrics already calculated as part of evaluation.", flush=True)

    print("[8/10] Generating figures...", flush=True)
    figures = generate_figures(input_results, output_results, robustness_results,
                               settings.results_dir / "figures")
    print(f"       {len(figures)} figures written to results/figures/", flush=True)

    print("[9/10] Saving results files...", flush=True)
    paths = write_results_files(input_results, output_results, robustness_results, settings.results_dir)
    for name, p in paths.items():
        print(f"       {name}: {p}", flush=True)

    print("[10/10] Generating docs/security_evaluation.md...", flush=True)
    render_report(input_results, output_results, robustness_results, len(df),
                  PROJECT_ROOT / "docs" / "security_evaluation.md")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.1f}s.", flush=True)
    n_ok = sum(1 for r in input_results.values() if r["status"] == "ok")
    n_skipped = len(input_results) - n_ok
    print(f"Input firewall backends evaluated: {n_ok} ok, {n_skipped} skipped.", flush=True)


if __name__ == "__main__":
    main()
