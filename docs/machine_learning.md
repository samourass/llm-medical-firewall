# Machine Learning — détecteurs comparés

Quatre approches, comparées sur la tâche de classification **multiclasse à 6 catégories**
(`benign`, `prompt_injection`, `jailbreak`, `rag_prompt_injection`, `pii_exfiltration`,
`secret_exfiltration`). Protocole : sélection d'hyperparamètres sur **validation**, une seule
évaluation finale sur **test**, seed fixe (42), aucune métrique inventée.

Toutes les métriques ci-dessous sont **mesurées** dans un bac à sable Linux, CPU, un seul cœur
utilisé par process (voir `environment` dans chaque `results/metrics/*.json`). Elles ne sont
pas transférables telles quelles à votre machine (surtout les latences) : relancez les scripts
localement pour obtenir vos propres chiffres.

## A. Regex / Security Rules (`src/llmfw/detection/regex_rules.py`)

Aucun entraînement. Motifs génériques par famille (contournement d'instruction, jailbreak,
injection RAG, PII, secrets), volontairement **non calés sur les gabarits exacts du dataset**
pour éviter le sur-ajustement (consigne explicite du document de cadrage).

Mesuré sur le vrai split de test (1 051 exemples) :

| Catégorie | Recall exact | Détecté comme une attaque (n'importe laquelle) |
|---|---|---|
| prompt_injection | 100,0 % | 100,0 % |
| secret_exfiltration | 60,0 % | 60,0 % |
| rag_prompt_injection | 6,4 % | 32,1 % |
| pii_exfiltration | 12,9 % | 12,9 % |
| jailbreak | 2,9 % | 2,9 % |

**Taux de faux positifs sur les 361 exemples benign : 0,0 %.**

Interprétation : la regex est un filtre précis (aucun faux positif mesuré) mais à faible rappel
sur les familles les plus paraphrasées (jailbreak, PII, RAG injection) — attendu pour des règles
génériques. Elle sert de garde-fou rapide (voir `docs/firewall.md`), pas de détecteur principal.

## B. TF-IDF + Logistic Regression

Meilleurs hyperparamètres (sélection validation) : `C=10.0`.

| Métrique (test, 1 051 ex.) | Valeur |
|---|---|
| Accuracy | 0,9382 |
| F1 macro | 0,9202 |
| F1 weighted | 0,9296 |
| Latence médiane (requête unique) | 0,42 ms |
| Taille du modèle | 0,27 Mo |

Par catégorie (precision / recall / F1) :

| Catégorie | P | R | F1 | support |
|---|---|---|---|---|
| benign | 0,9116 | 1,0000 | 0,9538 | 361 |
| jailbreak | 1,0000 | 1,0000 | 1,0000 | 140 |
| pii_exfiltration | 1,0000 | 1,0000 | 1,0000 | 140 |
| prompt_injection | 0,8485 | 1,0000 | 0,9180 | 140 |
| rag_prompt_injection | 0,9655 | 1,0000 | 0,9825 | 140 |
| secret_exfiltration | 1,0000 | 0,5000 | 0,6667 | 130 |

Point faible net : `secret_exfiltration` (recall 50 %) — à examiner en priorité si ce modèle
est retenu pour le firewall (voir limites du dataset dans `docs/dataset.md`, seulement 4 gabarits
par catégorie en test).

## C. TF-IDF + XGBoost

Meilleurs hyperparamètres (sélection validation) : `n_estimators=400`, `max_depth=4`.

| Métrique (test, 1 051 ex.) | Valeur |
|---|---|
| Accuracy | 0,8402 |
| F1 macro | 0,8033 |
| F1 weighted | 0,8310 |
| Latence médiane (requête unique) | 1,14 ms |
| Taille du modèle | 2,32 Mo |

Par catégorie :

| Catégorie | P | R | F1 | support |
|---|---|---|---|---|
| benign | 0,8856 | 0,9861 | 0,9332 | 361 |
| jailbreak | 1,0000 | 0,4071 | 0,5787 | 140 |
| pii_exfiltration | 1,0000 | 0,9143 | 0,9552 | 140 |
| prompt_injection | 0,7562 | 0,8643 | 0,8067 | 140 |
| rag_prompt_injection | 1,0000 | 0,6500 | 0,7879 | 140 |
| secret_exfiltration | 0,6103 | 1,0000 | 0,7580 | 130 |

Sur ce dataset et cette grille d'hyperparamètres, XGBoost multiclasse est **nettement en
dessous** de la Logistic Regression (f1_macro 0,80 vs 0,92), notamment sur `jailbreak`
(recall 41 %) et `rag_prompt_injection` (recall 65 %). La grille d'hyperparamètres testée est
volontairement petite (2×2) : un `RandomizedSearchCV` plus large pourrait réduire cet écart,
mais rien n'indique dans ces mesures que XGBoost soit le meilleur choix multiclasse ici.

## D. DistilBERT

**Script implémenté (`src/llmfw/training/distilbert.py`), détection CUDA automatique, mais
non exécuté dans ce bac à sable** : le téléchargement des poids pré-entraînés
`distilbert-base-uncased` nécessite un accès réseau à `huggingface.co`, qui est bloqué par la
liste blanche réseau de cet environnement (échec `HfHubHTTPError` observé lors du test). Le
code a été vérifié jusqu'à ce point (imports, préparation du dataset Hugging Face,
tokenisation) mais l'entraînement et les métriques associées **n'ont pas pu être mesurés ici**.

À exécuter sur votre machine (accès internet normal requis) :

```powershell
# Dossier : llm-medical-firewall\
.\scripts\04_train_distilbert.ps1 -Epochs 1 -MaxTrainSamples 500   # run rapide, pour vérifier que tout fonctionne
.\scripts\04_train_distilbert.ps1                                    # run complet (3 epochs, dataset complet)
```

Aucune métrique DistilBERT n'est donc rapportée ici : ne pas se fier à un chiffre qui ne serait
pas dans `results/metrics/distilbert.json` généré par une exécution réelle.

## Comparaison — ce qu'on peut conclure à ce stade

| Détecteur | F1 macro (test) | Latence médiane | Entraînement requis | Statut |
|---|---|---|---|---|
| Regex | n/a (pas un classifieur probabiliste) | < 0,1 ms | non | mesuré |
| Logistic Regression | 0,9202 | 0,42 ms | oui | mesuré |
| XGBoost | 0,8033 | 1,14 ms | oui | mesuré |
| DistilBERT | — | — | oui | **non mesuré ici** (réseau bloqué) |

Sur les seules mesures disponibles, **TF-IDF + Logistic Regression est le meilleur compromis
précision/latence** des détecteurs comparés jusqu'ici. Ceci n'est pas une conclusion définitive :
DistilBERT reste à évaluer, et la petitesse du test (4 gabarits/catégorie) limite la robustesse
de ces chiffres (voir `docs/dataset.md`).
