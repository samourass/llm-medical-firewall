# Architecture

## Objectif

Concevoir et évaluer un **LLM Firewall** intelligent qui protège un chatbot médical
(LLM + RAG + documents médicaux) à la fois en **entrée** (requête utilisateur) et en **sortie**
(réponse du LLM). Cette phase du projet couvre uniquement le **firewall d'entrée** : détecter
si une requête utilisateur est bénigne ou relève de l'une des cinq familles d'attaques ciblées.
Le RAG, le LLM et le firewall de sortie ne sont **pas encore implémentés** (hors périmètre de
cette phase, volontairement).

## Architecture cible (vue d'ensemble)

```text
USER
  |
  v
INPUT FIREWALL   <-- implémenté dans cette phase
  |
  v
RAG              <-- pas encore implémenté
  |
  v
LLM              <-- pas encore implémenté
  |
  v
OUTPUT FIREWALL  <-- pas encore implémenté
  |
  v
USER
```

## Input Firewall — architecture interne

```text
                      +----------------------+
   texte utilisateur  |   RegexRules         |  toujours actif (rapide, explicable,
   --------------->   |   (detection/        |  zéro dépendance ML)
                      |    regex_rules.py)   |
                      +----------+-----------+
                                 |
                    label in {secret_exfiltration, pii_exfiltration} ?
                          |  oui                  |  non
                          v                       v
                     BLOCK (garde-fou)   +--------------------------+
                                         |  UnifiedClassifier        |
                                         |  (models/unified.py)      |
                                         |  backend configurable :   |
                                         |  regex | logistic_regr.   |
                                         |  | xgboost | distilbert   |
                                         +-------------+-------------+
                                                       |
                                     label != benign ET confidence >= seuil ?
                                           |  oui                |  non
                                           v                     v
                                        BLOCK                 ALLOW
```

Le garde-fou Regex court-circuite les modèles ML pour les deux catégories de fuite
(`secret_exfiltration`, `pii_exfiltration`) : peu coûteux, explicable, et ne dépend d'aucun
modèle entraîné. Le reste de la décision (prompt_injection, jailbreak, rag_prompt_injection,
faux négatifs de la regex) passe par le backend ML configuré (`FIREWALL_MODEL` dans `.env`,
seuil `FIREWALL_THRESHOLD`). Voir `docs/firewall.md` pour le détail de la politique de décision.

## Détecteurs comparés

Quatre approches, chacune exposant la même interface `classify(text) -> {label, confidence,
model, latency_ms}` (voir `docs/machine_learning.md`) :

| Détecteur | Fichier | Entraînement requis |
|---|---|---|
| A. Regex / règles de sécurité | `src/llmfw/detection/regex_rules.py` | non |
| B. TF-IDF + Logistic Regression | `src/llmfw/training/multiclass.py` | oui (`scripts/03_train_multiclass.ps1`) |
| C. TF-IDF + XGBoost | `src/llmfw/training/multiclass.py` | oui (`scripts/03_train_multiclass.ps1`) |
| D. DistilBERT | `src/llmfw/training/distilbert.py` | oui (`scripts/04_train_distilbert.ps1`) |

Random Forest existe déjà dans le dépôt (étapes 1-5, classification binaire) mais n'est **pas**
étendu au multiclasse : le document de cadrage de cette phase limite volontairement le
périmètre à Regex / Logistic Regression / XGBoost / DistilBERT.

## Structure du dépôt (mise à jour de cette phase)

```text
llm-medical-firewall/
├── data/{raw,processed,train,validation,test}/   # inchangé
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── machine_learning.md
│   └── firewall.md
├── models/                        # .joblib (classiques) + distilbert/ (HF), non versionnés
├── results/                        # métriques JSON, figures, prédictions
├── scripts/
│   ├── setup.ps1, 01_build_dataset.ps1, 02_train_classical.ps1, run_tests.ps1  # existants
│   ├── 03_train_multiclass.ps1     # nouveau : LR + XGBoost, 6 classes
│   └── 04_train_distilbert.ps1     # nouveau : DistilBERT, 6 classes
├── src/llmfw/
│   ├── config.py                   # inchangé (Pydantic Settings + .env)
│   ├── data/                       # inchangé (génération, split, audit)
│   ├── detection/
│   │   └── regex_rules.py          # nouveau : détecteur A
│   ├── training/
│   │   ├── classical.py            # inchangé (binaire, étapes 3-5 historiques)
│   │   ├── multiclass.py           # nouveau : détecteurs B et C (6 classes)
│   │   └── distilbert.py           # nouveau : détecteur D (6 classes)
│   ├── evaluation/                 # metrics.py / plots.py étendus (fonctions multiclasses ajoutées)
│   ├── models/
│   │   └── unified.py              # nouveau : classify(text) unifié, 4 backends
│   └── firewall/
│       └── api.py                  # nouveau : Firewall.inspect_input(text)
├── tests/                          # 5 fichiers existants/nouveaux, 25 tests
├── requirements-ml.txt             # inchangé
├── requirements-dl.txt             # nouveau : torch/transformers/datasets/accelerate (optionnel)
├── .env.example, pyproject.toml, README.md
```

## Ce qui n'est pas fait dans cette phase

- RAG (base vectorielle, retrieval)
- Intégration LLM (Ollama / API)
- Firewall de sortie
- API FastAPI / dashboard / Docker / MLflow (prévus dans le plan en 10 phases antérieur, mais
  hors du périmètre resserré du document fourni pour cette étape)
