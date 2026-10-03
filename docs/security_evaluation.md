# Security Evaluation — LLM Firewall (Input + Output)

> Généré automatiquement par `evaluation/run_full_evaluation.py` à partir des fichiers dans
> `results/`. Chaque nombre ci-dessous provient d'une exécution mesurée dans cet environnement ;
> rien n'est estimé ou supposé. Régénérez ce fichier après tout changement de modèle, de seuil
> ou de jeu de données pour qu'il reste synchronisé avec `results/`.

## 1. Research Objective

Évaluer, de façon scientifique, la capacité de détection, la performance de classification, la
robustesse aux perturbations, le comportement en faux positifs et la latence du LLM Firewall
bidirectionnel (entrée + sortie) proposé dans ce projet. **Ceci n'est pas une comparaison
"avec/sans firewall"** : c'est une caractérisation du firewall lui-même.

## 2. Threat Model

Le firewall d'entrée cible 5 catégories d'attaque contre un chatbot médical basé sur un LLM et
un RAG : `prompt_injection` (contournement d'instruction), `jailbreak` (contournement des
restrictions de contenu), `rag_prompt_injection` (instruction injectée via un document récupéré
par le RAG), `pii_exfiltration` (exfiltration de données patient), `secret_exfiltration`
(exfiltration de secrets applicatifs : clés API, prompt système). Le firewall de sortie cible la
fuite, dans la réponse du LLM, de PII, d'emails, de téléphones, de clés API/jetons/identifiants,
ou du prompt système lui-même — y compris quand la requête d'origine était bénigne.

## 3. Dataset

Jeu d'évaluation de sécurité dédié : **600 requêtes** (100 par catégorie ×
6 catégories), généré par `llmfw.evaluation.security_test_set` à partir de gabarits **écrits
spécifiquement pour cette évaluation** (`security_test_templates.py`), disjoints des gabarits
utilisés pour `data/{train,validation,test}`. Vérifié programmatiquement (voir la sortie de
`python -m llmfw.evaluation.security_test_set`) : 0 chevauchement avec train/validation/test
après normalisation, 0 doublon interne.

Jeu de robustesse : 12 requêtes de base (2 par catégorie) × 11 variations (voir section 10) = 132
requêtes. Jeu d'évaluation du firewall de sortie : 20 réponses simulées (13 avec fuite attendue,
7 bénignes) couvrant 6 catégories de fuite — voir section 12 pour les limites de taille de ces
deux derniers jeux.

## 4. Experimental Setup

- Seeds fixes : génération du jeu de sécurité (`seed=2026`), du jeu de robustesse (`seed=7`).
- Chaque backend est chargé une fois puis "warmé" (3 appels non chronométrés) avant la mesure de
  latence, pour exclure le temps de chargement du modèle de la latence d'inférence mesurée.
- Latence mesurée requête par requête (pas de traitement par lot), en millisecondes, dans ce
  bac à sable (1 processus, CPU partagé) — non transférable telle quelle à une autre machine.
- Backends du firewall d'entrée évalués : Regex, Logistic Regression, XGBoost, DistilBERT (voir
  section 12 pour les modèles non disponibles dans cet environnement).

## 5. Input Firewall

Voir `docs/input_firewall.md` pour l'architecture (Regex garde-fou + backend ML configurable,
seuil `FIREWALL_THRESHOLD`). Cette évaluation porte sur la **classification multiclasse brute**
de chaque détecteur (avant application du seuil de décision ALLOW/BLOCK), pour isoler la
performance du classifieur de celle du seuil.

## 6. Output Firewall

Voir `docs/output_firewall.md`. Évalué indépendamment (section 12 ci-dessous), sur un jeu de
réponses simulées couvrant les catégories de fuite ciblées.

## 7. Evaluation Metrics

Accuracy, precision/recall/F1 (par classe, macro, weighted — `sklearn.metrics`), False Positive
Rate et False Negative Rate (définitions ci-dessous), Attack Detection Rate, matrice de
confusion. **Log-loss n'est volontairement pas calculé** : ce n'est pas une métrique demandée
par le document de cadrage, et la "confiance" retournée par le détecteur Regex n'est pas une
probabilité calibrée — y appliquer log-loss serait une métrique mal formulée (voir docstring de
`security_eval.py`).

**Définitions exactes utilisées :**
- **Attack Detection Rate (ADR)** = requêtes d'attaque prédites comme une catégorie != `benign`
  (n'importe laquelle) / total des requêtes d'attaque × 100. "Détecté" ne veut PAS dire "la
  catégorie exacte a été trouvée" — cette précision-là est donnée séparément par les métriques
  per-class (section 9).
- **False Positive Rate (FPR)** = requêtes `benign` prédites comme une catégorie != `benign` /
  total des requêtes `benign` × 100.
- **False Negative Rate (FNR)** = requêtes d'attaque prédites comme `benign` / total des
  requêtes d'attaque × 100. Par construction, FNR = 100 − ADR.

## 8. Model Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | FPR | FNR | Latency (median) |
|---|---|---|---|---|---|---|---|---|
| Regex | 0.2883 | 0.7992 | 0.2883 | 0.2217 | 0.2217 | 0.00% | 78.80% | 0.076 ms |
| Logistic Regression | 0.9900 | 0.9904 | 0.9900 | 0.9900 | 0.9900 | 0.00% | 1.00% | 0.458 ms |
| XGBoost | 0.8000 | 0.8529 | 0.8000 | 0.7929 | 0.7929 | 0.00% | 9.80% | 1.081 ms |
| DistilBERT | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |


## 9. Attack Category Results

| Category | Regex — P / R / F1 | Logistic Regression — P / R / F1 | XGBoost — P / R / F1 |
|---|---|---|---|
| benign | 0.202 / 1.000 / 0.337 | 0.952 / 1.000 / 0.976 | 0.671 / 1.000 / 0.803 |
| jailbreak | 1.000 / 0.060 / 0.113 | 1.000 / 1.000 / 1.000 | 1.000 / 0.430 / 0.601 |
| pii_exfiltration | 1.000 / 0.090 / 0.165 | 1.000 / 1.000 / 1.000 | 0.988 / 0.800 / 0.884 |
| prompt_injection | 1.000 / 0.090 / 0.165 | 1.000 / 1.000 / 1.000 | 0.797 / 0.940 / 0.862 |
| rag_prompt_injection | 1.000 / 0.010 / 0.020 | 0.990 / 1.000 / 0.995 | 1.000 / 0.730 / 0.844 |
| secret_exfiltration | 0.593 / 0.480 / 0.530 | 1.000 / 0.940 / 0.969 | 0.662 / 0.900 / 0.763 |


## 10. Robustness Results

Variations testées : capitalisation, espaces, ponctuation, offuscation (substitution de
caractères, deux variantes), paraphrase, formulation indirecte, jeu de rôle, multilingue
(français), camouflage en question bénigne (« benign-looking »). "Correct" = attaque détectée
(catégorie prédite != benign) pour les lignes d'attaque, ou absence de faux positif (catégorie
prédite == benign) pour les lignes bénignes.


**Regex** — accuracy de référence (`original`) : 66.67% ; accuracy globale toutes variations confondues : 45.45%.

| Variation | n | Accuracy |
|---|---|---|
| benign_looking | 12 | 58.33% |
| capitalization | 12 | 66.67% |
| indirect_instruction | 12 | 41.67% |
| multilingual_fr | 12 | 16.67% |
| obfuscated_leetspeak | 12 | 16.67% |
| obfuscated_manual | 12 | 16.67% |
| original | 12 | 66.67% |
| paraphrase | 12 | 41.67% |
| punctuation | 12 | 66.67% |
| role_play | 12 | 50.00% |
| whitespace | 12 | 58.33% |

**Logistic Regression** — accuracy de référence (`original`) : 100.00% ; accuracy globale toutes variations confondues : 86.36%.

| Variation | n | Accuracy |
|---|---|---|
| benign_looking | 12 | 91.67% |
| capitalization | 12 | 100.00% |
| indirect_instruction | 12 | 91.67% |
| multilingual_fr | 12 | 66.67% |
| obfuscated_leetspeak | 12 | 41.67% |
| obfuscated_manual | 12 | 83.33% |
| original | 12 | 100.00% |
| paraphrase | 12 | 83.33% |
| punctuation | 12 | 100.00% |
| role_play | 12 | 91.67% |
| whitespace | 12 | 100.00% |

**XGBoost** — accuracy de référence (`original`) : 66.67% ; accuracy globale toutes variations confondues : 53.79%.

| Variation | n | Accuracy |
|---|---|---|
| benign_looking | 12 | 50.00% |
| capitalization | 12 | 66.67% |
| indirect_instruction | 12 | 50.00% |
| multilingual_fr | 12 | 16.67% |
| obfuscated_leetspeak | 12 | 16.67% |
| obfuscated_manual | 12 | 41.67% |
| original | 12 | 66.67% |
| paraphrase | 12 | 83.33% |
| punctuation | 12 | 66.67% |
| role_play | 12 | 66.67% |
| whitespace | 12 | 66.67% |


## 11. Latency Results

| Model | Min | Max | Mean | Median | p95 | n |
|---|---|---|---|---|---|---|
| Regex | 0.0378 ms | 0.1960 ms | 0.0839 ms | 0.0756 ms | 0.1362 ms | 600 |
| Logistic Regression | 0.3509 ms | 0.8311 ms | 0.4742 ms | 0.4579 ms | 0.5992 ms | 600 |
| XGBoost | 0.9325 ms | 1.7608 ms | 1.1190 ms | 1.0809 ms | 1.4204 ms | 600 |
| DistilBERT | not measured | not measured | not measured | not measured | not measured | 0 |


## 12. Output Firewall Results

- Cas testés : 20 (13 avec fuite attendue, 7 bénins)
- Detection rate (fuites correctement interceptées, REDACT ou BLOCK) : 100.00%
- False positive rate (réponses bénignes signalées à tort) : 0.00%
- False negative rate (fuites laissées passer, ALLOW) : 0.00%
- Latence : médiane 0.0251 ms, p95 0.0362 ms (n=20)

| Catégorie de fuite | n | Détectées | Taux |
|---|---|---|---|
| api_key | 4 | 4 | 100.0% |
| credential | 1 | 1 | 100.0% |
| email | 2 | 2 | 100.0% |
| phone_number | 2 | 2 | 100.0% |
| pii_ssn | 2 | 2 | 100.0% |
| system_prompt_leak | 2 | 2 | 100.0% |


## 13. Discussion

Le F1 macro le plus élevé mesuré sur ce jeu d'évaluation est obtenu par **Logistic Regression** (F1 macro = 0.9900).
Le FPR le plus bas mesuré est obtenu par **Regex** (FPR = 0.00%).
Ces constats se limitent strictement à ce jeu de 600 requêtes et à cet
environnement d'exécution — voir la section Limitations pour ce qu'ils ne permettent pas de
conclure. Les résultats de robustesse (section 10) montrent une dégradation nette et cohérente,
pour tous les modèles évalués, face aux variations multilingues et à l'offuscation par
substitution de caractères — un signal clair que ces deux axes seraient prioritaires pour un
prochain cycle d'amélioration, plutôt qu'un jugement définitif sur "le meilleur modèle".

## 14. Limitations

- **Modèles non évalués dans cet environnement :**
Les modèles suivants n'ont **pas** pu être évalués dans cet environnement d'exécution (voir raison exacte ci-dessous) ; ils n'apparaissent dans aucun tableau ou graphique comparatif ci-dessus — aucune valeur n'a été inventée pour eux :
- **DistilBERT** : `FileNotFoundError: Modèle DistilBERT introuvable dans /home/claude/llm-medical-firewall/llm-medical-firewall/models/distilbert. Entraînez-le d'abord avec :
    python -m llmfw.training.distilbert`

- **Taille des échantillons de robustesse et de sortie** : 132 requêtes de robustesse (12 bases
  × 11 variations) et 20 cas pour le firewall de sortie sont de petits échantillons — une
  différence de quelques cas change sensiblement les pourcentages rapportés (chaque cas pèse
  ~8% pour la robustesse, 5% pour la sortie). Ces chiffres indiquent une tendance, pas une
  estimation statistiquement précise.
- **Dataset synthétique** : le jeu d'évaluation de sécurité comme les jeux de robustesse/sortie
  sont entièrement synthétiques (gabarits + substitutions), pas des attaques réelles capturées
  en production — les scores sont probablement optimistes par rapport à des attaques réelles et
  adaptatives.
- **Latence mesurée dans un bac à sable partagé** (CPU, 1 cœur utilisé par processus) — les
  valeurs absolues ne sont pas transférables à un déploiement réel ; seul l'ordre de grandeur
  relatif entre modèles est informatif.
- **Seuil de décision non évalué ici** : cette évaluation porte sur la classification brute, pas
  sur la décision finale ALLOW/BLOCK du firewall (qui dépend en plus de `FIREWALL_THRESHOLD`) —
  voir `docs/input_firewall.md` pour une limite mesurée sur ce seuil spécifiquement.
- **Une seule langue de perturbation multilingue testée** (français) et une seule technique
  d'offuscation par substitution de caractères — la généralisation à d'autres langues ou
  d'autres techniques d'offuscation n'est pas mesurée.

## 15. Conclusion

Cette évaluation rapporte des différences mesurées entre 3 détecteur(s)
d'entrée disponibles dans cet environnement, sur un jeu de 600 requêtes
disjoint de l'entraînement, plus une évaluation de robustesse (132 requêtes, 11 variations) et
une évaluation indépendante du firewall de sortie (20 cas). Les tableaux des sections 8 à 12
sont la source de vérité ; aucune affirmation générale ("le modèle X est le meilleur") n'est
faite au-delà de ce que ces tableaux montrent explicitement, et les modèles non évalués ici
(section 14) ne doivent pas être supposés meilleurs ou moins bons sur cette seule base.
