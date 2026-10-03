# Output Firewall — protection en SORTIE

Protège l'utilisateur en inspectant la réponse **générée par le LLM** avant qu'elle ne lui soit
retournée. Voir `docs/input_firewall.md` pour la protection symétrique en entrée, et
`docs/security_pipeline.md` pour la vue d'ensemble Input vs Output.

## Ce qu'il détecte (`firewall/output.py`)

| Catégorie | Exemple de motif |
|---|---|
| `email` | adresse email |
| `phone_number` | numéro de téléphone |
| `pii_ssn` | numéro type SSN (`123-45-6789`) |
| `api_key` | `TEST_API_KEY=...`, `sk-...`, `AKIA...`, `ghp_...` |
| `credential` | `TEST_SECRET=...`, `password: ...`, `secret key: ...` |
| `synthetic_test_secret` | secrets synthétiques de test (`TEST-123456789`, `MEDICAL-DEMO-SECRET`) |
| `system_prompt_leak` | tournures typiques d'une fuite du prompt système (« you are an AI... », « I was instructed to... ») |

Secrets synthétiques de test utilisés pour la validation (jamais de vrais identifiants) :
```
TEST_API_KEY=TEST-123456789
TEST_SECRET=MEDICAL-DEMO-SECRET
```

## Résolution des chevauchements

Les motifs les plus spécifiques (clé API, secret synthétique) sont prioritaires sur les motifs
génériques (téléphone) quand ils se chevauchent — sans cette résolution, une suite de chiffres
à l'intérieur d'une clé API (`TEST_API_KEY=TEST-123456789`) serait comptée à tort à la fois
comme `api_key` ET comme `phone_number`, ce qui a été observé pendant le développement (voir
`_non_overlapping_matches` dans `firewall/output.py`) et corrigé avant la livraison de cette
phase.

## Actions — `ALLOW` | `REDACT` | `BLOCK`

1. **`BLOCK`** si une catégorie non caviardable est détectée (`system_prompt_leak` : on ne peut
   pas retirer proprement une fuite du prompt système sans risquer de laisser passer le reste),
   ou si le nombre de constatations distinctes atteint `OUTPUT_FIREWALL_BLOCK_THRESHOLD`
   (`.env`, défaut `3`) — trop de fuites dans une même réponse, on ne prend pas le risque de
   caviarder partiellement.
2. **`REDACT`** sinon, si au moins une constatation caviardable a été trouvée : chaque segment
   correspondant est remplacé par `[REDACTED]`, le reste de la réponse est conservé intact.
3. **`ALLOW`** si rien n'est détecté.

## Exemple exact du document de cadrage — mesuré, pas inventé

```python
>>> from llmfw.firewall.output import OutputFirewall
>>> OutputFirewall().inspect("The API key is TEST_API_KEY=TEST-123456789")
{'action': 'REDACT', 'sanitized': 'The API key is [REDACTED]', ...}
```

## API — `OutputFirewall.inspect(response_text)`

```python
{
  "action": "REDACT",
  "original": "The API key is TEST_API_KEY=TEST-123456789",
  "sanitized": "The API key is [REDACTED]",
  "findings": ["api_key"],
  "reason": "redacted categories: ['api_key']",
  "model": "output_firewall_regex",
  "latency_ms": 0.05
}
```

## Limites connues

- Détection par Regex uniquement (pas de modèle ML dédié à la sortie dans cette phase) : les
  motifs sont génériques mais peuvent manquer une fuite formulée de façon inhabituelle, ou
  caviarder à tort un texte médical légitime ressemblant à un motif (peu probable sur les motifs
  actuels, mais pas prouvé formellement — aucun test de faux positifs à grande échelle n'a été
  mené sur la sortie, contrairement au firewall d'entrée).
- Le seuil de blocage par nombre de constatations (`OUTPUT_FIREWALL_BLOCK_THRESHOLD=3`) est un
  choix simple, non calibré sur un jeu de données de sortie dédié (aucun jeu de test de réponses
  LLM fuitées n'existe dans ce projet — seulement les secrets synthétiques de test).
