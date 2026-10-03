from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from llmfw.ui.common import BACKEND_LABELS, load_csv_if_exists, require_settings, results_missing_notice  # noqa: E402

st.set_page_config(page_title="Security Evaluation", page_icon="🔒", layout="wide")
st.title("🔒 Security Evaluation")
st.caption(
    "ADR = requêtes d'attaque avec label prédit ≠ benign / total attaques × 100 (avant tout "
    "seuil de confiance opérationnel). FPR/FNR définis dans docs/input_firewall.md. Jeu "
    "d'évaluation : 600 requêtes (100/catégorie), voir docs/security_evaluation.md."
)

settings = require_settings()

security_metrics = load_csv_if_exists(str(settings.results_dir / "security_metrics.csv"))
if security_metrics is None:
    results_missing_notice("Les métriques de sécurité", "python evaluation\\run_full_evaluation.py")
    st.stop()

evaluated = security_metrics[security_metrics["status"] == "ok"].copy()
skipped = security_metrics[security_metrics["status"] != "ok"].copy()
evaluated["model_label"] = evaluated["model"].map(BACKEND_LABELS)

st.subheader("Attack Detection Rate / FPR / FNR par modèle")
st.dataframe(
    evaluated[["model_label", "n_benign", "n_attack", "attack_detection_rate_pct",
              "false_positive_rate_pct", "false_negative_rate_pct"]]
    .rename(columns={"model_label": "Model", "n_benign": "N benign", "n_attack": "N attack",
                     "attack_detection_rate_pct": "ADR (%)", "false_positive_rate_pct": "FPR (%)",
                     "false_negative_rate_pct": "FNR (%)"})
    .set_index("Model").style.format("{:.2f}", subset=["ADR (%)", "FPR (%)", "FNR (%)"]),
    width='stretch',
)
if not skipped.empty:
    for _, row in skipped.iterrows():
        st.warning(f"⚠️ **{BACKEND_LABELS.get(row['model'], row['model'])}** non évalué : {row.get('reason', 'raison inconnue')}")

c1, c2 = st.columns(2)
with c1:
    st.bar_chart(evaluated.set_index("model_label")[["attack_detection_rate_pct"]]
                .rename(columns={"attack_detection_rate_pct": "ADR (%)"}))
with c2:
    st.bar_chart(evaluated.set_index("model_label")[["false_positive_rate_pct", "false_negative_rate_pct"]]
                .rename(columns={"false_positive_rate_pct": "FPR (%)", "false_negative_rate_pct": "FNR (%)"}))

latency_metrics = load_csv_if_exists(str(settings.results_dir / "latency_metrics.csv"))
if latency_metrics is not None:
    st.subheader("Latence (ms)")
    lat = latency_metrics.copy()
    lat["model_label"] = lat["model"].map(BACKEND_LABELS)
    st.dataframe(lat.set_index("model_label")[["min_ms", "mean_ms", "median_ms", "p95_ms", "max_ms", "n_requests"]]
                .style.format({"min_ms": "{:.3f}", "mean_ms": "{:.3f}", "median_ms": "{:.3f}",
                              "p95_ms": "{:.3f}", "max_ms": "{:.3f}"}), width='stretch')

st.subheader("Résultats par catégorie d'attaque")
class_metrics = load_csv_if_exists(str(settings.results_dir / "class_metrics.csv"))
if class_metrics is not None:
    model_choice = st.selectbox("Modèle", evaluated["model"].tolist(),
                                format_func=lambda b: BACKEND_LABELS[b], key="sec_eval_model")
    sub = class_metrics[(class_metrics["model"] == model_choice) & (class_metrics["category"] != "benign")]
    st.dataframe(sub.set_index("category")[["precision", "recall", "f1", "support"]]
                .style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}),
                width='stretch')
else:
    results_missing_notice("Les résultats par catégorie", "python evaluation\\run_full_evaluation.py")

st.subheader("Robustesse")
robustness = load_csv_if_exists(str(settings.results_dir / "robustness_metrics.csv"))
if robustness is not None:
    robustness = robustness.copy()
    robustness["model_label"] = robustness["model"].map(BACKEND_LABELS)
    pivot = robustness.pivot_table(index="variation", columns="model_label", values="accuracy_pct")
    st.dataframe(pivot.style.format("{:.2f}"), width='stretch')
    st.bar_chart(pivot)
    st.caption("Accuracy (%) par type de variation (casse, espaces, ponctuation, paraphrase, "
              "instruction indirecte, roleplay, multilingue, obfuscation, benign-looking). "
              "Voir docs/security_evaluation.md pour le détail et les limites.")
else:
    results_missing_notice("Les résultats de robustesse", "python evaluation\\run_full_evaluation.py")

figures_dir = settings.results_dir / "figures"
with st.expander("Figures supplémentaires"):
    for name in ("attack_detection_rate.png", "false_positive_rate.png", "false_negative_rate.png",
                "robustness_performance.png"):
        p = figures_dir / name
        if p.exists():
            st.image(str(p), caption=name, width='stretch')
