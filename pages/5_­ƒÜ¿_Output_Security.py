from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from llmfw.ui.common import (  # noqa: E402
    OUTPUT_SAFE_EXAMPLES,
    decision_badge,
    get_output_firewall,
    load_csv_if_exists,
    require_settings,
    results_missing_notice,
)

st.set_page_config(page_title="Output Security", page_icon="🚨", layout="wide")
st.title("🚨 Output Security — Firewall de sortie")
st.caption(
    "Démontre la détection de PII, de secrets, la fuite du prompt système, le caviardage "
    "(REDACT) et le blocage (BLOCK) d'une réponse générée par le LLM avant qu'elle n'atteigne "
    "l'utilisateur. Secrets synthétiques uniquement — jamais de vraies données."
)

settings = require_settings()
output_firewall = get_output_firewall(settings)

col_left, col_right = st.columns([2, 1])
with col_right:
    example = st.selectbox("Exemple prédéfini", ["—"] + list(OUTPUT_SAFE_EXAMPLES.keys()))
with col_left:
    default_text = OUTPUT_SAFE_EXAMPLES.get(example, "") if example != "—" else ""
    text = st.text_area("Réponse LLM (simulée) à inspecter", value=default_text, height=100)
    run = st.button("Inspecter", type="primary")

if run:
    if not text or not text.strip():
        st.warning("⚠️ Veuillez saisir un texte avant d'inspecter.")
        st.stop()

    result = output_firewall.inspect(text)

    st.divider()
    st.markdown(f"**Action** : {decision_badge(result['action'])}")
    c1, c2 = st.columns(2)
    c1.metric("Constatations", len(result["findings"]))
    c2.metric("Latence", f"{result['latency_ms']:.3f} ms")
    if result["findings"]:
        st.markdown(f"**Catégories détectées** : `{'`, `'.join(result['findings'])}`")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Réponse originale**")
        st.code(result["original"])
    with col_b:
        st.markdown("**Réponse envoyée à l'utilisateur**")
        st.code(result["sanitized"])
    st.caption(f"Raison : {result['reason']}")

st.divider()
st.subheader("Résultats mesurés (jeu de test dédié au firewall de sortie)")
of_metrics = load_csv_if_exists(str(settings.results_dir / "output_firewall_metrics.csv"))
if of_metrics is not None:
    st.dataframe(
        of_metrics.rename(columns={
            "expected_category": "Catégorie attendue", "n": "N", "n_detected": "Détectées",
            "detection_rate_pct": "Taux de détection (%)",
            "overall_detection_rate_pct": "Taux global (%)",
            "overall_false_positive_rate_pct": "FPR global (%)",
        }).set_index("Catégorie attendue"),
        width='stretch',
    )
    st.caption(
        "Échantillon volontairement petit (voir docs/security_evaluation.md) : ces taux sont "
        "une tendance, pas une estimation statistiquement précise."
    )
else:
    results_missing_notice("Les résultats du firewall de sortie", "python evaluation\\run_full_evaluation.py")
