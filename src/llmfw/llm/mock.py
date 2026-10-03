"""LLM factice, déterministe (aucun appel réseau, aucun aléa) : utile pour les tests
automatisés et pour faire fonctionner le pipeline complet sans dépendance externe.

Ne génère pas de texte libre : construit une réponse à partir du contexte RAG fourni, de façon
purement mécanique (extraction des 2 premières phrases de chaque passage retenu). Ce n'est pas
un LLM réel — c'est un composant de test/démo, documenté comme tel.
"""
from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class MockLLM:
    name = "mock"

    def generate(self, user_input: str, context: str) -> str:
        if not context.strip():
            return (
                "I don't have specific information about that in my medical knowledge base. "
                "Please consult a qualified healthcare professional. "
                "(mock LLM response — for demonstration/testing only, not medical advice.)"
            )

        # Un extrait court et déterministe de chaque passage retenu par le RAG.
        excerpts = []
        for block in context.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            sentences = _SENTENCE_SPLIT.split(block)
            excerpts.append(" ".join(sentences[:2]).strip())

        body = " ".join(excerpts)
        return (
            f"Based on the retrieved medical information: {body} "
            "(mock LLM response — for demonstration/testing only, not medical advice; "
            "consult a qualified healthcare professional for personal guidance.)"
        )
