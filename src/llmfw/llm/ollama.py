"""Fournisseur LLM basé sur Ollama (serveur local, API HTTP compatible).

Nécessite une instance Ollama en cours d'exécution (`OLLAMA_BASE_URL`, défaut
`http://localhost:11434`) avec le modèle `OLLAMA_MODEL` (défaut `llama3.2:3b`) déjà présent
(`ollama pull llama3.2:3b`). Aucune clé API n'est utilisée ou stockée en dur.

Non exécuté/mesuré dans le bac à sable de développement (pas de serveur Ollama disponible dans
cet environnement) — voir docs/llm.md.
"""
from __future__ import annotations

import requests

_SYSTEM_PROMPT = (
    "You are a medical information assistant for a demonstration chatbot. Answer only using "
    "the provided context. If the context does not contain the answer, say you don't have "
    "enough information and recommend consulting a healthcare professional. Never provide a "
    "diagnosis; provide general educational information only."
)


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str, base_url: str, timeout_s: float = 60.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, user_input: str, context: str) -> str:
        prompt = (
            f"{_SYSTEM_PROMPT}\n\nContext:\n{context or '(no relevant context found)'}\n\n"
            f"Question: {user_input}\n\nAnswer:"
        )
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Impossible de joindre Ollama sur {self.base_url} (modèle '{self.model}') : {exc}. "
                "Vérifiez qu'Ollama tourne (`ollama serve`) et que le modèle est bien tiré "
                f"(`ollama pull {self.model}`)."
            ) from exc
        return response.json().get("response", "").strip()
