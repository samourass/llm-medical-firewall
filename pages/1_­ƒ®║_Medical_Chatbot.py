from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from llmfw.ui.common import (  # noqa: E402
    BACKEND_LABELS,
    SAFE_EXAMPLES,
    check_backend_availability,
    decision_badge,
    get_firewall,
    get_llm,
    get_output_firewall,
    get_rag,
    require_settings,
    safe_generate,
)

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")
st.title("🩺 Medical Chatbot")
st.caption("USER → INPUT FIREWALL → RAG → LLM → OUTPUT FIREWALL → USER")

settings = require_settings()
availability = check_backend_availability(settings)
available_backends = [b for b in ("logistic_regression", "xgboost", "regex", "distilbert")
                      if availability[b]["available"]]

if not available_backends:
    st.error("Aucun backend de firewall d'entrée n'est disponible. Entraînez au moins un "
             "modèle (`.\\scripts\\03_train_multiclass.ps1`) avant de continuer.")
    st.stop()

col_left, col_right = st.columns([2, 1])
with col_right:
    backend = st.selectbox("Modèle du firewall d'entrée", available_backends,
                           format_func=lambda b: BACKEND_LABELS[b])
    example = st.selectbox("Exemple prédéfini (optionnel)", ["—"] + list(SAFE_EXAMPLES.keys()))

with col_left:
    default_text = SAFE_EXAMPLES.get(example, "") if example != "—" else ""
    user_input = st.text_area("Votre question médicale", value=default_text, height=100,
                              placeholder="Ex. What are common symptoms of influenza?")
    submitted = st.button("Envoyer", type="primary")

if submitted:
    if not user_input or not user_input.strip():
        st.warning("⚠️ Veuillez saisir une question avant d'envoyer.")
        st.stop()

    t0 = time.perf_counter()

    firewall = get_firewall(backend, settings)
    with st.status("1. Input Firewall…", expanded=True) as status:
        input_result = firewall.inspect_input(user_input)
        st.write(decision_badge("ALLOWED" if input_result["allowed"] else "BLOCKED"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Catégorie détectée", input_result["label"])
        c2.metric("Confiance", f"{input_result['confidence']:.2%}")
        c3.metric("Modèle", input_result["model"])
        c4.metric("Latence", f"{input_result['latency_ms']:.2f} ms")
        st.caption(f"Raison : {input_result['reason']}")
        status.update(label="1. Input Firewall — terminé", state="complete")

    if not input_result["allowed"]:
        st.error(
            "🔴 **Requête bloquée par le firewall d'entrée.** Elle n'a pas été envoyée au RAG "
            "ni au LLM, conformément à la politique de sécurité du pipeline."
        )
        st.stop()

    with st.status("2. RAG — récupération du contexte médical…", expanded=True) as status:
        rag, rag_error = get_rag(settings)
        if rag_error:
            st.error(f"⚠️ RAG indisponible : {rag_error}")
            status.update(label="2. RAG — échec", state="error")
            st.stop()
        retrieved = rag.retrieve(user_input)
        if retrieved:
            for chunk in retrieved:
                with st.expander(f"📄 {chunk.source} (score {chunk.score:.3f})"):
                    st.write(chunk.text)
        else:
            st.info("Aucun document pertinent trouvé pour cette question.")
        status.update(label="2. RAG — terminé", state="complete")

    with st.status("3. LLM — génération de la réponse…", expanded=True) as status:
        from llmfw.rag.pipeline import RAGPipeline

        context = RAGPipeline.build_context(retrieved)
        llm, llm_error = get_llm(settings)
        if llm_error:
            st.error(f"⚠️ LLM indisponible : {llm_error}")
            status.update(label="3. LLM — échec", state="error")
            st.stop()
        raw_response, gen_error = safe_generate(llm, user_input, context)
        if gen_error:
            st.error(f"⚠️ {gen_error}")
            status.update(label="3. LLM — échec", state="error")
            st.stop()
        st.caption(f"Fournisseur : {llm.name}")
        status.update(label="3. LLM — terminé", state="complete")

    with st.status("4. Output Firewall…", expanded=True) as status:
        output_firewall = get_output_firewall(settings)
        output_result = output_firewall.inspect(raw_response)
        st.write(decision_badge(output_result["action"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Action", output_result["action"])
        c2.metric("Constatations", len(output_result["findings"]))
        c3.metric("Latence", f"{output_result['latency_ms']:.2f} ms")
        if output_result["findings"]:
            st.caption(f"Catégories détectées : {', '.join(output_result['findings'])}")
        status.update(label="4. Output Firewall — terminé", state="complete")

    total_latency_ms = (time.perf_counter() - t0) * 1000.0

    st.divider()
    st.subheader("Réponse finale")
    st.markdown(output_result["sanitized"])
    st.caption(f"Latence totale (pipeline complet, mesurée dans cette session) : "
              f"{total_latency_ms:.2f} ms")
