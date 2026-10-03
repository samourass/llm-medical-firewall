"""Configuration centrale du projet (Pydantic Settings + fichier .env).

Tous les paramètres importants sont modifiables via `.env` sans toucher au code.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (utilisé à partir de l'étape 12) ---
    llm_provider: str = "mock"  # mock | ollama
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"

    # --- RAG ---
    vector_db: str = "faiss"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "auto" (défaut) : essaie sentence_transformers, bascule automatiquement sur "tfidf" si le
    # modèle ne peut pas être téléchargé/chargé (ex. environnement sans accès à huggingface.co) ;
    # "sentence_transformers" ou "tfidf" pour forcer explicitement un backend.
    embedding_provider: str = "auto"
    rag_chunk_size_chars: int = 500
    rag_chunk_overlap_chars: int = 80
    rag_top_k: int = 3
    medical_docs_dir: Path = PROJECT_ROOT / "data" / "medical"

    # --- Firewall (entrée) ---
    # xgboost par défaut : fonctionne "out of the box" sans entraîner DistilBERT. Ce n'est pas
    # le modèle le plus précis mesuré (logistic_regression fait mieux, voir .env.example et
    # docs/machine_learning.md) — changez FIREWALL_MODEL dans .env si besoin.
    firewall_model: str = "xgboost"
    firewall_threshold: float = 0.70

    # --- Firewall (sortie) ---
    # Nombre de constatations (findings) distinctes dans une même réponse au-delà duquel la
    # réponse est bloquée entièrement plutôt que caviardée (REDACT).
    output_firewall_block_threshold: int = 3

    # --- Données / expériences ---
    random_seed: int = 42
    dataset_version: str = "v1"

    # --- Chemins ---
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"
    models_dir: Path = PROJECT_ROOT / "models"
    logs_dir: Path = PROJECT_ROOT / "results" / "logs"


def get_settings() -> Settings:
    return Settings()
