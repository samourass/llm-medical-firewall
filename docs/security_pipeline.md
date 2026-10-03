# Security Pipeline — vue d'ensemble

## Architecture complète (cette phase)

```text
USER
  |
  v
INPUT FIREWALL     (docs/input_firewall.md)   — protège le RAG/LLM contre la requête utilisateur
  |  bloqué ? -> réponse de blocage, RAG/LLM jamais appelés
  v
RAG                (docs/rag.md)               — récupère le contexte médical pertinent
  |
  v
LLM                (docs/llm.md)                — génère une réponse à partir du contexte
  |
  v
OUTPUT FIREWALL    (docs/output_firewall.md)   — protège l'utilisateur contre la réponse du LLM
  |  fuite détectée ? -> caviardage (REDACT) ou blocage (BLOCK) de la réponse
  v
USER
```

Implémenté dans `src/llmfw/pipeline.py::ChatPipeline.chat(user_input)` :

```python
def chat(self, user_input: str) -> dict:
    input_result = self.firewall.inspect_input(user_input)
    if not input_result["allowed"]:
        return security_block_response(input_result)   # RAG/LLM jamais appelés

    retrieved = self.rag.retrieve(user_input)
    context = RAGPipeline.build_context(retrieved)
    raw_response = self.llm.generate(user_input, context)
    output_result = self.output_firewall.inspect(raw_response)
    return {...}
```

## Input protection vs Output protection

| | Input Firewall | Output Firewall |
|---|---|---|
| Inspecte | La requête de l'utilisateur | La réponse générée par le LLM |
| Objectif | Empêcher une requête malveillante d'atteindre le RAG/LLM (contournement d'instruction, jailbreak, exfiltration) | Empêcher une fuite de sortir vers l'utilisateur (PII, secrets, fuite du prompt système), même si la requête d'origine était bénigne |
| Détecteurs | Regex + 1 modèle ML au choix (`logistic_regression` \| `xgboost` \| `distilbert`), 6 catégories | Regex uniquement (motifs par catégorie de fuite) |
| Décision | `allowed: true/false` (binaire) | `action: ALLOW \| REDACT \| BLOCK` (3 niveaux) |
| Sur blocage | La requête ne va jamais au RAG ni au LLM | La réponse est soit caviardée (segments remplacés), soit entièrement remplacée par un message générique |

Les deux sont nécessaires et complémentaires : une requête bénigne peut tout de même produire
une réponse contenant une fuite (le LLM invente ou répète une donnée sensible présente dans le
contexte ou dans ses propres paramètres) — c'est le scénario que testent
`tests/test_pipeline_integration.py::test_output_secret_leakage_caught_by_output_firewall` et
`test_output_pii_leakage_caught_by_output_firewall`, avec un LLM factice qui simule une fuite.

## Journalisation de sécurité (`logging_utils.py`)

Chaque appel à `chat()` écrit un événement JSON dans `results/logs/security_events.jsonl` :
horodatage, `request_id`, classification d'entrée (label/confiance/décision/modèle/latence),
décision de sortie (action/catégories de constatations/modèle/latence), latence totale.

**Ce qui n'est jamais journalisé** : le texte brut de la requête (seulement sa longueur et un
préfixe de hash SHA-256, pour corrélation sans reconstruction), le texte brut de la réponse du
LLM, et les valeurs exactes détectées par le firewall de sortie (seulement leurs catégories).
Voir `logging_utils.py` pour le détail exact et la justification.

## Tests d'intégration (`tests/test_pipeline_integration.py`, 8 scénarios demandés)

| # | Scénario | Résultat mesuré |
|---|---|---|
| 1 | Question médicale bénigne | Traverse tout le pipeline, `output_firewall.action == ALLOW` |
| 2 | Prompt injection | Bloqué par le firewall d'entrée, RAG/LLM jamais appelés |
| 3 | Jailbreak | Bloqué par le firewall d'entrée |
| 3bis | Jailbreak paraphrasé (limite connue) | **Laissé passer** — faux négatif mesuré et documenté, pas caché (voir `docs/input_firewall.md`) |
| 4 | RAG prompt injection | Bloqué par le firewall d'entrée |
| 5 | PII exfiltration | Bloqué par le garde-fou Regex du firewall d'entrée |
| 6 | Secret exfiltration | Bloqué par le garde-fou Regex du firewall d'entrée |
| 7 | Fuite de secret en sortie | Requête bénigne autorisée, réponse caviardée/bloquée par le firewall de sortie |
| 8 | Fuite de PII en sortie | Requête bénigne autorisée, réponse caviardée par le firewall de sortie |

52/52 tests pytest passent au total dans ce dépôt à la fin de cette phase (voir README pour le
détail par fichier).

## Limites de cette phase (ce qui n'est pas fait)

- Pas d'API FastAPI ni de dashboard (prévus dans le plan en 10 phases antérieur, hors périmètre
  du document de cadrage de cette phase).
- DistilBERT (firewall d'entrée) et `sentence-transformers` (RAG) non exécutés dans ce bac à
  sable (réseau `huggingface.co` bloqué) — le pipeline reste fonctionnel et testé grâce aux
  replis prévus (backend ML `logistic_regression`/`xgboost` pour le firewall, TF-IDF pour le
  RAG), mais les performances réelles avec les modèles visés par défaut restent à mesurer chez
  vous.
- Ollama non testé (pas de serveur disponible dans ce bac à sable).
- Pas de calibration formelle du seuil du firewall d'entrée ni du seuil de blocage du firewall
  de sortie (voir limites détaillées dans `docs/input_firewall.md` et `docs/output_firewall.md`).
