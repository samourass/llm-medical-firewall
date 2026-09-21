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

    # --- RAG (utilisé à partir de l'étape 11) ---
    vector_db: str = "faiss"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Firewall (utilisé à partir de l'étape 13) ---
    firewall_model: str = "distilbert"
    firewall_threshold: float = 0.80

    # --- Données / expériences ---
    random_seed: int = 42
    dataset_version: str = "v1"

    # --- Chemins ---
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"
    models_dir: Path = PROJECT_ROOT / "models"


def get_settings() -> Settings:
    return Settings()
