# PROJECT_STATE.md

> Lire ce fichier AVANT toute modification pour connaître l'état réel du projet sans avoir à le
> redécouvrir. Mis à jour à chaque phase depuis sa création.

## Dernière mise à jour

**Phase de finalisation** : interface Streamlit (5 pages), documentation académique complète
(16 fichiers `docs/` + ce fichier), revue de sécurité et de qualité du code, README réécrit,
71/71 tests pytest. Projet considéré démontrable et reproductible de bout en bout avec les
backends effectivement entraînés (Regex, Logistic Regression, XGBoost).

## Phases complétées

1. **Détection (firewall d'entrée)** — dataset 6 classes, Regex, Logistic Regression et
   XGBoost multiclasses, API unifiée `classify()`.
2. **Intégration RAG + LLM + firewalls** — documents médicaux, pipeline RAG (FAISS), LLM
   (Mock + Ollama), firewall de sortie, `ChatPipeline.chat()`, journalisation sécurité.
3. **Évaluation scientifique** — jeu de sécurité dédié (600 requêtes), métriques sklearn,
   ADR/FPR/FNR, latence, robustesse (132 requêtes/11 variations), évaluation indépendante du
   firewall de sortie, rapport généré automatiquement (`docs/security_evaluation.md`).
4. **Finalisation** (cette phase) — Streamlit, documentation académique, revue de sécurité et
   de performance, README professionnel, ce fichier.

## Architecture actuelle

```
USER → INPUT FIREWALL (Regex + ML) → RAG (FAISS) → LLM (Mock/Ollama) → OUTPUT FIREWALL (Regex) → USER
```
Orchestré par `src/llmfw/pipeline.py::ChatPipeline`, exposé dans `app.py` + `pages/` (Streamlit).
Détail : `docs/architecture.md`, `docs/security_pipeline.md`.

## État par composant

| Composant | Fichier(s) | État |
|---|---|---|
| Dataset (6 classes) | `data/{train,validation,test}/`, `src/llmfw/data/` | ✅ 6 566 ex., mesuré |
| Regex / Security Rules (entrée) | `src/llmfw/detection/regex_rules.py` | ✅ mesuré, FPR=0% (test split et jeu de sécurité dédié) |
| LR/XGBoost binaires (historique) | `src/llmfw/training/classical.py` | ✅ mesuré (phase 1 initiale) |
| LR/XGBoost multiclasses | `src/llmfw/training/multiclass.py` | ✅ entraînés, mesurés |
| DistilBERT | `src/llmfw/training/distilbert.py` | ⚠️ implémenté, **jamais entraîné** (réseau huggingface.co bloqué — HTTP 403 confirmé) |
| API unifiée `classify()` | `src/llmfw/models/unified.py` | ✅ testé, 4 backends |
| Firewall d'entrée | `src/llmfw/firewall/api.py` | ✅ testé |
| Documents médicaux | `data/medical/` (6 fichiers) | ✅ |
| RAG (cleaning→chunking→embeddings→FAISS) | `src/llmfw/rag/` | ✅ testé ; embeddings = TF-IDF (repli automatique, sentence-transformers jamais exécuté ici) |
| LLM (Mock + Ollama) | `src/llmfw/llm/` | ✅ Mock testé ; ⚠️ Ollama jamais testé (pas de serveur disponible) |
| Firewall de sortie | `src/llmfw/firewall/output.py` | ✅ testé, ALLOW/REDACT/BLOCK |
| Pipeline complet `chat()` | `src/llmfw/pipeline.py` | ✅ testé |
| Journalisation sécurité | `src/llmfw/logging_utils.py` | ✅ |
| Jeu d'évaluation de sécurité (600 req.) | `data/security_eval/` | ✅ généré, 0 fuite/doublon vérifié |
| Jeu de robustesse (132 req., 11 variations) | `src/llmfw/evaluation/robustness.py` | ✅ mesuré (3 backends) |
| Jeu d'évaluation du firewall de sortie (20 cas) | `src/llmfw/evaluation/output_firewall_test_set.py` | ✅ mesuré (100% détection, 0% FPR/FNR) |
| Orchestrateur d'évaluation reproductible | `evaluation/run_full_evaluation.py` | ✅ exécuté, ~60 s |
| Résultats (CSV/JSON/figures) | `results/*.csv`, `results/evaluation_results.json`, `results/figures/` (12 PNG) | ✅ générés, mesurés |
| Rapport scientifique | `docs/security_evaluation.md` | ✅ généré automatiquement, 15 sections |
| **Interface Streamlit (5 pages)** | `app.py`, `pages/` | ✅ testée (AppTest, 0 exception, scénarios clés vérifiés) |
| Tests | `tests/` (13 fichiers) | ✅ **71/71 passent** |
| Documentation académique | `docs/` (13 fichiers) + ce fichier | ✅ |
| API FastAPI, Docker, MLflow | — | ⏳ pas commencé (hors périmètre des documents de cadrage fournis) |

## Dépendances installées (bac à sable de développement, versions exactes mesurées)

```
numpy 2.4.4          pandas 3.0.2           scikit-learn 1.8.0     xgboost 3.4.1
joblib 1.5.3          pydantic-settings 2.15.0  pytest 9.1.1        psutil 7.2.2
faiss-cpu 1.15.1       requests 2.33.1        streamlit 1.64.0
torch 2.14.0+cu130      transformers 5.17.0     datasets 5.0.1       accelerate 1.15.0
sentence-transformers 6.1.0
```
`requirements-ml.txt` (base) + `requirements-app.txt` (Streamlit) + `requirements-dl.txt`
(optionnel, torch/transformers/sentence-transformers — installé ici pour permettre les tests
d'intégration RAG/DistilBERT, mais DistilBERT et sentence-transformers restent inutilisables
sans accès réseau à huggingface.co).

## Modèles entraînés (présents dans `models/`, non versionnés — voir `.gitignore`)

- `models/logistic_regression_multiclass.joblib`, `models/xgboost_multiclass.joblib` (6 classes)
- `models/rag/` (index FAISS + `tfidf_embedder.joblib`, backend TF-IDF)
- `models/distilbert/` : **absent** (jamais entraîné dans cet environnement)

## Statut des tests

**71/71 tests pytest passent** (~75-130 s selon l'environnement) :
`.\scripts\run_tests.ps1` ou `pytest tests/ -v`. Répartition : dataset, Regex, LR/XGBoost
(binaire historique + multiclasse), API unifiée, firewall d'entrée, RAG (6 tests indépendants),
LLM (Mock + factory), firewall de sortie, pipeline d'intégration (8 scénarios + 1 limite
documentée), évaluation de sécurité, application Streamlit (6 pages + 5 interactions).

## Statut de l'évaluation

`evaluation/run_full_evaluation.py` exécuté avec succès, ~60 s. Résultats mesurés (jeu de
sécurité, 600 requêtes) :

| Model | Accuracy | F1 macro | ADR | FPR | FNR | Latence médiane |
|---|---|---|---|---|---|---|
| Regex | 0,2883 | 0,2217 | 21,2% | 0,00% | 78,8% | 0,076 ms |
| Logistic Regression | 0,9900 | 0,9900 | 99,0% | 0,00% | 1,0% | 0,458 ms |
| XGBoost | 0,8000 | 0,7929 | 90,2% | 0,00% | 9,8% | 1,081 ms |
| DistilBERT | non mesuré | — | — | — | — | — |

Robustesse (accuracy globale) : Regex 45,45%, XGBoost 53,79%, Logistic Regression 86,36% — chute
nette sur le multilingue et l'obfuscation par caractères pour les 3 modèles. Firewall de sortie
(20 cas) : détection 100%, FPR 0%, FNR 0%. Détail complet : `docs/security_evaluation.md`.

## Limitations connues (détail et actions pour les lever : `docs/limitations.md`)

1. DistilBERT jamais entraîné/évalué (réseau bloqué).
2. RAG en repli TF-IDF, pas sentence-transformers (même cause).
3. Faux négatifs mesurés sur reformulations de jailbreak sous le seuil `FIREWALL_THRESHOLD`.
4. Regex : FPR nul mais ADR faible (21,2%) sur phrasing neuf — pas un détecteur généraliste.
5. Échantillons de robustesse (132) et de firewall de sortie (20) volontairement petits.
6. Ollama jamais testé (pas de serveur disponible).
7. Pas de calibration formelle du seuil ni des probabilités inter-modèles.
8. Décision du firewall d'entrée par requête isolée (pas de contexte multi-tours).

## Configuration par défaut (`.env.example`)

`FIREWALL_MODEL=xgboost`, `FIREWALL_THRESHOLD=0.70` (choisis pour fonctionner sans entraîner
DistilBERT — **pas le backend le plus précis mesuré** : `logistic_regression` obtient un
f1_macro de 0,99 contre 0,80 pour xgboost, voir `docs/machine_learning.md`).
`EMBEDDING_PROVIDER=auto` (repli TF-IDF automatique). `LLM_PROVIDER=mock`.

## Prochaines étapes suggérées (non commencées)

1. Entraîner DistilBERT et valider sentence-transformers sur une machine avec accès réseau à
   huggingface.co, puis relancer `.\scripts\07_run_evaluation.ps1`.
2. Tester `OllamaLLM` avec un serveur Ollama réel.
3. Calibrer `FIREWALL_THRESHOLD` (étude precision/recall par seuil) pour réduire le faux négatif
   documenté sur les reformulations de jailbreak.
4. Étendre le jeu de robustesse et le jeu de test du firewall de sortie pour des estimations
   plus précises (actuellement 132 et 20 cas).
5. Renforcer la généralisation au multilingue et à l'obfuscation par caractères (chute la plus
   nette mesurée dans `docs/security_evaluation.md`, section Robustness).
6. API FastAPI, conteneurisation Docker, MLflow — hors périmètre des documents de cadrage
   fournis pour ce projet à ce jour.

## Commandes de référence (PowerShell, depuis la racine du projet)

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt -r requirements-app.txt
.\scripts\01_build_dataset.ps1
.\scripts\03_train_multiclass.ps1
.\scripts\05_build_rag_index.ps1
.\scripts\run_tests.ps1
.\scripts\07_run_evaluation.ps1
.\.venv\Scripts\python.exe -m streamlit run app.py
```

S�quence complète et commentée : `docs/reproducibility.md`.
