from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from llmfw.ui.common import BACKEND_LABELS, load_csv_if_exists, require_settings, results_missing_notice  # noqa: E402

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
st.title("📊 Model Evaluation")
st.caption(
    "Métriques mesurées sur le jeu d'évaluation de sécurité (600 requêtes, 100/catégorie), "
    "voir `docs/security_evaluation.md`. Rien ici n'est inventé : si un fichier de résultats "
    "est absent, un message l'indique explicitement au lieu d'afficher des chiffres."
)

settings = require_settings()

model_metrics = load_csv_if_exists(str(settings.results_dir / "model_metrics.csv"))
if model_metrics is None:
    results_missing_notice("Les résultats d'évaluation des modèles",
                           "python evaluation\\run_full_evaluation.py")
    st.stop()

evaluated = model_metrics[model_metrics["status"] == "ok"].copy()
skipped = model_metrics[model_metrics["status"] != "ok"].copy()

evaluated["model_label"] = evaluated["model"].map(BACKEND_LABELS)

st.subheader("Tableau comparatif")
display_cols = ["model_label", "n_samples", "accuracy", "precision_macro", "recall_macro",
                "f1_macro", "false_positive_rate_pct", "false_negative_rate_pct", "latency_median_ms"]
st.dataframe(
    evaluated[display_cols].rename(columns={
        "model_label": "Model", "n_samples": "N", "accuracy": "Accuracy",
        "precision_macro": "Precision (macro)", "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)", "false_positive_rate_pct": "FPR (%)",
        "false_negative_rate_pct": "FNR (%)", "latency_median_ms": "Latence médiane (ms)",
    }).set_index("Model").style.format({
        "Accuracy": "{:.4f}", "Precision (macro)": "{:.4f}", "Recall (macro)": "{:.4f}",
        "F1 (macro)": "{:.4f}", "FPR (%)": "{:.2f}", "FNR (%)": "{:.2f}",
        "Latence médiane (ms)": "{:.3f}",
    }),
    width='stretch',
)

if not skipped.empty:
    for _, row in skipped.iterrows():
        st.warning(f"⚠️ **{BACKEND_LABELS.get(row['model'], row['model'])}** non évalué : {row['reason']}")

st.subheader("Comparaisons graphiques")
c1, c2 = st.columns(2)
with c1:
    st.bar_chart(evaluated.set_index("model_label")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
                .rename(columns={"accuracy": "Accuracy", "precision_macro": "Precision",
                                 "recall_macro": "Recall", "f1_macro": "F1"}))
with c2:
    st.bar_chart(evaluated.set_index("model_label")[["false_positive_rate_pct", "false_negative_rate_pct"]]
                .rename(columns={"false_positive_rate_pct": "FPR (%)", "false_negative_rate_pct": "FNR (%)"}))

st.bar_chart(evaluated.set_index("model_label")[["latency_median_ms"]]
            .rename(columns={"latency_median_ms": "Latence médiane (ms)"}))

st.subheader("Résultats par catégorie")
class_metrics = load_csv_if_exists(str(settings.results_dir / "class_metrics.csv"))
if class_metrics is not None:
    model_choice = st.selectbox("Modèle", evaluated["model"].tolist(),
                                format_func=lambda b: BACKEND_LABELS[b])
    sub = class_metrics[class_metrics["model"] == model_choice].set_index("category")
    st.dataframe(sub[["precision", "recall", "f1", "support"]].style.format(
        {"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}), width='stretch')
    st.bar_chart(sub[["precision", "recall", "f1"]])
else:
    results_missing_notice("Les résultats par catégorie", "python evaluation\\run_full_evaluation.py")

st.subheader("Matrices de confusion")
figures_dir = settings.results_dir / "figures"
cols = st.columns(len(evaluated))
for col, (_, row) in zip(cols, evaluated.iterrows()):
    fig_path = figures_dir / f"confusion_matrix_{row['model']}.png"
    with col:
        st.caption(BACKEND_LABELS[row["model"]])
        if fig_path.exists():
            st.image(str(fig_path), width='stretch')
        else:
            st.caption("figure indisponible")
