# Threat Model

## Système protégé

Un chatbot médical basé sur un LLM et un système RAG (documents médicaux synthétiques/publics).
Le système répond à des questions médicales générales ; il n'est pas destiné au diagnostic.

## Acteurs

- **Utilisateur légitime** : pose des questions médicales générales.
- **Attaquant côté entrée** : interagit avec le chatbot via l'interface utilisateur (aucun accès
  système, aucun accès aux poids du modèle, aucun accès à l'infrastructure).
- Le firewall ne modélise pas un attaquant ayant un accès direct au serveur, à la base
  vectorielle, ou capable de modifier les documents médicaux eux-mêmes (empoisonnement de
  corpus) — ce dernier point est un axe futur (voir `docs/limitations.md`).

## Surfaces et catégories d'attaque couvertes

| Catégorie | Surface | Objectif de l'attaquant |
|---|---|---|
| `prompt_injection` | Entrée utilisateur | Faire ignorer les instructions/contraintes du système au LLM |
| `jailbreak` | Entrée utilisateur | Contourner les garde-fous de sécurité/éthique du LLM |
| `rag_prompt_injection` | Entrée utilisateur (imite un marqueur de document/contexte) | Faire croire au pipeline qu'une instruction provient d'un document de confiance |
| `pii_exfiltration` | Entrée utilisateur | Faire révéler des informations personnelles (patients, tiers) |
| `secret_exfiltration` | Entrée utilisateur | Faire révéler des secrets applicatifs (prompt système, clés, identifiants) |
| Fuite en sortie (email, téléphone, PII, clé API, jeton, identifiants, prompt système) | Réponse du LLM | Le LLM répète ou invente une donnée sensible dans sa réponse, même pour une requête bénigne |

Le firewall d'entrée couvre les 5 premières catégories ; le firewall de sortie couvre la
dernière — les deux sont nécessaires (voir `docs/security_pipeline.md`, section « Input vs
Output »), car une requête bénigne peut tout de même produire une réponse fuyante.

## Ce que le système NE modélise PAS (hors périmètre)

- Attaques sur l'infrastructure (réseau, serveur, base de données, dépendances compromises).
- Empoisonnement du corpus RAG par un attaquant ayant un accès en écriture à `data/medical/`.
- Attaques par déni de service (volumétrie de requêtes).
- Attaques adverses sur les poids du modèle (extraction de modèle, inversion de gradient).
- Attaques nécessitant plusieurs tours de conversation pour construire un contexte malveillant
  (le firewall d'entrée évalué dans ce projet inspecte chaque requête indépendamment — voir
  `docs/limitations.md`).

## Hypothèses de confiance

- Les documents dans `data/medical/` sont considérés fiables (contrôlés par l'équipe du
  projet), pas par un tiers non vérifié.
- Le firewall de sortie fait confiance à la sortie du LLM comme source de texte à inspecter,
  mais pas à son contenu — c'est précisément ce qu'il inspecte.
- Aucune donnée patient réelle n'est utilisée nulle part dans ce projet (dataset synthétique,
  documents médicaux publics/synthétiques, secrets de test synthétiques).

## Lien avec les mesures effectuées

Les résultats de `docs/security_evaluation.md` quantifient dans quelle mesure les détecteurs
évalués (Regex, Logistic Regression, XGBoost — DistilBERT non mesuré, voir
`docs/machine_learning.md`) couvrent effectivement les 5 catégories d'entrée de ce modèle de
menace, y compris sous variation (`docs/security_evaluation.md`, section Robustness), et dans
quelle mesure le firewall de sortie détecte les 6 types de fuite listés ci-dessus.
