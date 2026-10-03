# Dossier d'exécution : la racine du projet (llm-medical-firewall\)
# Construit l'index FAISS à partir de data/medical/ (nettoyage -> chunking -> embeddings ->
# FAISS). EMBEDDING_PROVIDER=auto (par défaut) utilise sentence-transformers si disponible,
# sinon se rabat automatiquement sur TF-IDF (voir .env.example / docs/rag.md).
$ErrorActionPreference = "Stop"
.\.venv\Scripts\python.exe -c "from llmfw.rag.pipeline import RAGPipeline; from llmfw.config import get_settings; import json; print(json.dumps(RAGPipeline(get_settings()).build_index(), indent=2))"
