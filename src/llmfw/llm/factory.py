"""Sélectionne le fournisseur LLM selon `LLM_PROVIDER` (`.env`)."""
from __future__ import annotations

from llmfw.config import Settings, get_settings
from llmfw.llm.base import LLMProvider
from llmfw.llm.mock import MockLLM
from llmfw.llm.ollama import OllamaLLM


def get_llm(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    if settings.llm_provider == "mock":
        return MockLLM()
    if settings.llm_provider == "ollama":
        return OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_base_url)
    raise ValueError(f"LLM_PROVIDER inconnu : {settings.llm_provider!r} (attendu : mock | ollama)")
