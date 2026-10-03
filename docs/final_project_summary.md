# Final Project Summary

> Ce document synthétise l'ensemble du projet. Chaque chiffre cité provient d'une exécution
> mesurée dans les environnements de développement utilisés pour ce projet (voir les fichiers
> `results/` et `docs/` liés) — aucune valeur n'est estimée ou supposée.

## 1. Project objective

Concevoir et évaluer un **LLM Firewall bidirectionnel** (entrée + sortie) pour sécuriser un
chatbot médical basé sur un LLM et un système RAG, puis mesurer scientifiquement sa capacité de
détection, sa robustesse, son comportement en faux positifs, et sa latence.

## 2. Architecture

```
USER → INPUT FIREWALL → RAG → LLM → OUTPUT FIREWALL → USER
```

Détail complet : `docs/architecture.md`, `docs/security_pipeline.md`. Orchestré par
`src/llmfw/pipeline.py::ChatPipeline`, exposé dans l'interface Streamlit (`app.py`, `pages/`).

## 3. Dataset

- **6 566 exemples** synthétiques (entraînement du firewall d'entrée), 6 classes, split
  70/15/15 stratifié sans fuite (audité par gabarit) — `docs/dataset.md`.
- **Jeu de sécurité dédié** : 600 requêtes (100/catégorie), gabarits distincts du jeu
  d'entraînement, 0 chevauchement vérifié — `docs/security_evaluation.md`.
- **6 documents médicaux** synthétiques/publics pour le RAG (`data/medical/`) —
  aucune donnée patient réelle nulle part dans le projet.

## 4. Models

| Modèle | Statut | Accuracy | F1 macro | ADR | FPR | FNR | Latence médiane |
|---|---|---|---|---|---|---|---|
| Regex | mesuré | 0,2883 | 0,2217 | 21,2 % | 0,00 % | 78,8 % | 0,076 ms |
| Logistic Regression | mesuré | 0,9900 | 0,9900 | 99,0 % | 0,00 % | 1,0 % | 0,458 ms |
| XGBoost | mesuré | 0,8000 | 0,7929 | 90,2 % | 0,00 % | 9,8 % | 1,081 ms |
| DistilBERT | **jamais mesuré** (réseau huggingface.co bloqué) | — | — | — | — | — | — |

Détail par catégorie et matrices de confusion : `docs/security_evaluation.md`,
`docs/machine_learning.md`.

## 5. Input protection

Regex (garde-fou secret/PII, FPR=0%) → backend ML configurable (`FIREWALL_MODEL`, défaut
`xgboost` — voir `.env.example` pour la justification de ce choix et sa limite connue) → décision
`allowed: true/false`, seuil `FIREWALL_THRESHOLD` (défaut `0.70`). Détail : `docs/input_firewall.md`.

## 6. RAG

Documents → nettoyage → découpage par phrases → embeddings (sentence-transformers visé, repli
TF-IDF automatique et mesuré ici) → FAISS (`IndexFlatIP`, cosinus exact) → contexte attribué à
sa source. Détail, limite mesurée du repli TF-IDF : `docs/rag.md`.

## 7. LLM

Interface commune (`llm/base.py`), deux implémentations : `MockLLM` (déterministe, sans réseau,
utilisé par tous les tests automatisés) et `OllamaLLM` (implémenté, **jamais testé** — aucun
serveur Ollama disponible dans les environnements de développement). Détail : `docs/llm.md`.

## 8. Output protection

Regex par catégorie (email, téléphone, PII, clé API, jeton, identifiants, secret synthétique,
fuite de prompt système), résolution des chevauchements par priorité, actions ALLOW/REDACT/BLOCK.
Mesuré sur 20 cas : détection 100 %, FPR 0 %, FNR 0 % (échantillon petit, voir
`docs/limitations.md`). Détail : `docs/output_firewall.md`.

## 9. Evaluation methodology

Jeu de sécurité dédié (600 requêtes), métriques sklearn (accuracy, precision/recall/F1
macro+weighted, matrice de confusion), ADR/FPR/FNR définis explicitement (formulation
binaire attack-vs-benign appliquée à une sortie multiclasse — voir `docs/security_evaluation.md`
section « Evaluation Metrics »), latence mesurée requête par requête (min/max/mean/median/p95),
robustesse sur 11 types de variation, firewall de sortie évalué indépendamment. Script
reproductible unique : `evaluation/run_full_evaluation.py`, seed fixe (42).

## 10. Measured results

Voir la section 4 ci-dessus pour le tableau modèle-par-modèle. Constats mesurés les plus nets :

- **Logistic Regression domine nettement** Regex et XGBoost sur ce jeu de 600 requêtes
  (f1_macro 0,99 vs 0,22 et 0,79) — c'est une observation, pas une généralisation garantie à
  d'autres distributions d'attaques.
- **Regex a un FPR nul mais un ADR faible** (21,2 %) : fiable pour ne jamais bloquer à tort du
  texte bénin, insuffisant seul comme détecteur généraliste.
- **Tous les modèles se dégradent sur le multilingue (français) et l'obfuscation par
  substitution de caractères** (voir `docs/security_evaluation.md`, section Robustness) — la
  dégradation la plus nette mesurée dans ce projet.
- **Firewall de sortie** : reproduit exactement l'exemple du document de cadrage
  (`TEST_API_KEY=TEST-123456789` → `The API key is [REDACTED]`), testé et mesuré.
- **60 tests pytest passaient à l'issue de la phase d'évaluation ; 71 passent à l'issue de la
  finalisation** (ajout des tests de l'application Streamlit).

## 11. Limitations

Résumé (détail complet et actions pour lever chaque limite : `docs/limitations.md`) :
DistilBERT jamais entraîné/évalué (réseau bloqué), RAG en repli TF-IDF dans cet environnement,
faux négatifs mesurés sur reformulations de jailbreak, échantillons de robustesse/sortie
volontairement petits, pas de calibration formelle du seuil, Ollama non testé, décision du
firewall d'entrée par requête isolée (pas de contexte multi-tours).

## 12. Future work

- Entraîner et évaluer DistilBERT sur une machine avec accès réseau à huggingface.co.
- Valider `sentence-transformers` pour le RAG et comparer objectivement au repli TF-IDF mesuré
  ici.
- Calibrer `FIREWALL_THRESHOLD` par une étude ROC/precision-recall dédiée plutôt qu'une valeur
  par défaut.
- Étendre le jeu de robustesse et le jeu de test du firewall de sortie (actuellement 132 et 20
  cas) pour des estimations plus précises.
- Tester `OllamaLLM` avec un serveur réel.
- API FastAPI et conteneurisation Docker si le projet est amené à être déployé (hors périmètre
  des documents de cadrage fournis pour ce projet).

## 13. Conclusion

Les quatre composants du pipeline (firewall d'entrée, RAG, LLM, firewall de sortie) sont
implémentés, intégrés, et couverts par 71 tests automatisés qui passent tous dans cet
environnement. Trois des quatre détecteurs d'entrée prévus ont des métriques réelles et
mesurées sur un jeu de sécurité dédié de 600 requêtes ; le quatrième (DistilBERT) reste
techniquement implémenté mais non validé, faute d'accès réseau dans les environnements de
développement utilisés — ce n'est pas dissimulé, c'est documenté à chaque endroit pertinent
(`docs/machine_learning.md`, `docs/security_evaluation.md`, `docs/limitations.md`, ce document,
`README.md`, `PROJECT_STATE.md`). Le projet est démontrable de bout en bout via l'interface
Streamlit (`app.py`) avec les backends effectivement entraînés (Regex, Logistic Regression,
XGBoost).
