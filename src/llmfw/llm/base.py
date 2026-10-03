"""Interface commune des fournisseurs LLM."""
from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    name: str

    def generate(self, user_input: str, context: str) -> str:
        """Génère une réponse à partir de la question utilisateur et du contexte RAG (déjà
        construit par `RAGPipeline.build_context`, éventuellement vide)."""
        ...
