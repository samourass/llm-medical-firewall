"""Point d'entrée de l'application Streamlit — page d'accueil.

Lancement (dossier : racine du projet, celle qui contient ce fichier) :
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Rend `llmfw` importable sans installation du package (streamlit run exécute ce fichier
# directement, en dehors du mécanisme pythonpath de pytest/pyproject.toml).
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from llmfw.ui.common import BACKEND_LABELS, check_backend_availability, require_settings  # noqa: E402

st.set_page_config(page_title="LLM Firewall — Medical Chatbot", page_icon="🛡️", layout="wide")

st.title("🛡️ LLM Firewall — Sécurisation d'un chatbot médical")
st.caption(
    "Conception et évaluation d'un LLM Firewall intelligent pour la sécurisation d'un chatbot "
    "médical basé sur un LLM et un système RAG — Projet de Fin d'Études"
)

st.markdown(
    """
### Architecture

```
USER → INPUT FIREWALL → RAG → LLM → OUTPUT FIREWALL → USER
```

Utilisez le menu à gauche pour naviguer :

| Page | Contenu |
|---|---|
| 🩺 Medical Chatbot | Poser une question médicale et suivre chaque étape du pipeline |
| 🧪 Security Testing | Tester le firewall d'entrée sur un texte de votre choix |
| 📊 Model Evaluation | Métriques mesurées des 4 détecteurs (Accuracy, Precision, Recall, F1, FPR, FNR, latence) |
| 🔒 Security Evaluation | Attack Detection Rate, FPR, FNR, latence, résultats par catégorie d'attaque |
| 🚨 Output Security | Démonstration du firewall de sortie (PII, secrets, fuite de prompt système) |
"""
)

st.divider()
st.subheader("État du système")

settings = require_settings()
availability = check_backend_availability(settings)

cols = st.columns(4)
for col, backend in zip(cols, ("regex", "logistic_regression", "xgboost", "distilbert")):
    info = availability[backend]
    with col:
        if info["available"]:
            st.success(f"✅ {BACKEND_LABELS[backend]}")
        else:
            st.error(f"❌ {BACKEND_LABELS[backend]}")
        st.caption(info["reason"])

st.caption(
    "DistilBERT est marqué indisponible tant qu'il n'a pas été entraîné localement "
    "(`.\\scripts\\04_train_distilbert.ps1` — nécessite un accès réseau à huggingface.co). "
    "Voir `docs/machine_learning.md` pour le détail."
)

with st.expander("Configuration active (.env)"):
    st.json({
        "LLM_PROVIDER": settings.llm_provider,
        "FIREWALL_MODEL": settings.firewall_model,
        "FIREWALL_THRESHOLD": settings.firewall_threshold,
        "EMBEDDING_PROVIDER": settings.embedding_provider,
        "VECTOR_DB": settings.vector_db,
        "OUTPUT_FIREWALL_BLOCK_THRESHOLD": settings.output_firewall_block_threshold,
    })
