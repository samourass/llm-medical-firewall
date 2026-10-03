# LLM Abstraction

## Interface (`llm/base.py`)

```python
class LLMProvider(Protocol):
    name: str
    def generate(self, user_input: str, context: str) -> str: ...
```

Sélection via `LLM_PROVIDER` (`.env`) et `llm/factory.py::get_llm(settings)`.

## Mock LLM (`llm/mock.py`) — `LLM_PROVIDER=mock` (défaut)

- Aucun appel réseau, aucun aléa : **déterministe** (même entrée -> même sortie), voir
  `tests/test_llm.py::test_mock_llm_is_deterministic`.
- Ne génère pas de texte libre : construit une réponse mécaniquement à partir des 2 premières
  phrases de chaque passage retenu par le RAG. Toujours suffixée d'un avertissement explicite
  (« mock LLM response — for demonstration/testing only, not medical advice »).
- Si le RAG ne retourne aucun contexte, répond qu'il n'a pas assez d'information et recommande
  de consulter un professionnel de santé, plutôt que d'inventer une réponse.
- Utilisé pour tous les tests automatisés (RAG + firewalls + pipeline complet), y compris ceux
  de ce dépôt (`tests/test_pipeline_integration.py`).

## Ollama (`llm/ollama.py`) — `LLM_PROVIDER=ollama`

- Client HTTP vers une instance Ollama locale (`OLLAMA_BASE_URL`, défaut
  `http://localhost:11434`), modèle `OLLAMA_MODEL` (défaut `llama3.2:3b`).
- Aucune clé API : Ollama tourne en local, pas d'API payante.
- Prompt système imposant : répondre uniquement à partir du contexte fourni, ne jamais donner de
  diagnostic, recommander un professionnel de santé si l'information manque.
- En cas d'échec de connexion, lève une `RuntimeError` explicite (message avec la commande
  `ollama pull` à lancer) plutôt que d'échouer silencieusement ou de renvoyer une réponse vide.
- **Non exécuté/mesuré dans ce bac à sable** : aucun serveur Ollama n'y est disponible. À tester
  chez vous avec Ollama installé et `ollama pull llama3.2:3b` déjà lancé.

## Configuration (`.env`, voir `.env.example`)

| Variable | Défaut | Rôle |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` \| `ollama` |
| `OLLAMA_MODEL` | `llama3.2:3b` | Modèle Ollama à utiliser |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL du serveur Ollama local |
