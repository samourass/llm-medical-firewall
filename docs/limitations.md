# Limitations

Vue consolidée des limites déjà documentées, dispersées dans plusieurs fichiers `docs/`. Ce
fichier ne répète pas le détail chiffré (voir les liens) mais liste chaque limite avec son
impact et où elle est mesurée.

## 1. DistilBERT n'a jamais été entraîné ni évalué

**Cause** : `huggingface.co` est bloqué (HTTP 403 confirmé) dans les environnements de
développement utilisés pour ce projet. `training/distilbert.py` est un script complet,
vérifié jusqu'au point de téléchargement des poids, mais jamais exécuté de bout en bout.
**Impact** : sur les 4 détecteurs prévus par le document de cadrage, 3 seulement (Regex,
Logistic Regression, XGBoost) ont des métriques réelles. Toute ligne « DistilBERT » dans les
tableaux de résultats est marquée `not measured` / `skipped`, jamais une valeur inventée.
**Voir** : `docs/machine_learning.md`, `docs/security_evaluation.md`.
**Action pour lever cette limite** : `.\scripts\04_train_distilbert.ps1` sur une machine avec
accès réseau à huggingface.co, puis relancer `python evaluation\run_full_evaluation.py`.

## 2. Le RAG utilise TF-IDF, pas des embeddings sémantiques, dans cet environnement

**Cause** : même blocage réseau que ci-dessus — `EMBEDDING_PROVIDER=auto` bascule
automatiquement sur TF-IDF quand `sentence-transformers/all-MiniLM-L6-v2` ne peut pas être
téléchargé.
**Impact mesuré** : sensibilité au choix exact des mots (ex. « common flu symptoms » remonte un
document non pertinent en tête, alors qu'un choix de mots différent remonte le bon document).
**Voir** : `docs/rag.md`.
**Action** : `pip install -r requirements-dl.txt` sur une machine avec accès réseau, puis
`EMBEDDING_PROVIDER=sentence_transformers` (ou laisser `auto`) et reconstruire l'index.

## 3. Faux négatifs mesurés sur des reformulations de jailbreak/prompt injection

**Cause** : le seuil `FIREWALL_THRESHOLD` (0.70 par défaut) et les modèles ML n'ont pas été
calibrés sur des paraphrases éloignées des gabarits du dataset d'entraînement.
**Impact mesuré** : `tests/test_pipeline_integration.py::test_paraphrased_jailbreak_may_evade_threshold_known_limitation`
documente un cas réel de contournement. Les résultats de robustesse
(`docs/security_evaluation.md`, section Robustness) confirment une dégradation nette sur les
variations multilingues et l'offuscation par substitution de caractères, pour les 3 détecteurs
mesurés.
**Action** : envisager un seuil plus bas (au prix de plus de faux positifs sur du texte bénin,
compromis non calibré dans ce projet), ou l'ajout d'exemples de robustesse à l'entraînement.

## 4. Regex a un rappel faible sur un jeu de phrasing neuf, malgré un FPR nul

**Mesuré** (jeu de sécurité de 600 requêtes) : ADR Regex = 21,2 % contre 99,0 % pour Logistic
Regression, mais FPR = 0,00 % pour les deux — Regex ne bloque jamais un texte bénin, mais rate
la majorité des attaques reformulées différemment des motifs codés en dur.
**Impact** : Regex ne doit pas être utilisé seul comme détecteur principal ; il sert de
garde-fou ciblé (secret/PII) dans `Firewall.inspect_input`, pas de classifieur généraliste.
**Voir** : `docs/security_evaluation.md`, `docs/input_firewall.md`.

## 5. Échantillons de robustesse et de firewall de sortie volontairement petits

132 requêtes de robustesse (12 par variation), 20 cas de firewall de sortie. Suffisant pour
détecter une tendance nette (ex. la chute sur le multilingue), pas pour une estimation
statistiquement précise (chaque cas de sortie pèse 5 % du taux mesuré).
**Voir** : `docs/security_evaluation.md`, `docs/output_firewall.md`.

## 6. Firewall d'entrée : décision par requête isolée, pas par conversation

Chaque appel à `inspect_input` évalue un texte indépendamment des messages précédents. Une
attaque construite progressivement sur plusieurs tours de conversation (jamais testée ici)
n'est pas modélisée. Voir `docs/threat_model.md`, section « ce qui n'est pas modélisé ».

## 7. Ollama non testé

`llm/ollama.py` est implémenté (gestion d'erreur de connexion explicite) mais aucun serveur
Ollama n'a été disponible dans les environnements de développement de ce projet. Seul `MockLLM`
a été exercé par les tests automatisés. Voir `docs/llm.md`.

## 8. Pas de calibration formelle des probabilités entre modèles

Les scores de confiance de Logistic Regression et XGBoost ne sont pas nécessairement
comparables entre eux (pas de Platt scaling / isotonic regression appliqué). Le seuil unique
`FIREWALL_THRESHOLD` est donc une approximation, pas un seuil calibré par modèle.

## 9. Mesures de latence : bac à sable partagé, CPU, un seul cœur utilisé par process

Les latences absolues (`docs/security_evaluation.md`, section Latency) ne sont pas
transférables telles quelles à une autre machine. L'ORDRE relatif entre modèles (Regex <
Logistic Regression < XGBoost, mesuré) est plus robuste que les valeurs absolues.

## 10. Random Forest n'a pas été étendu au multiclasse

Random Forest existe dans le dépôt (étapes 1-5 historiques, classification binaire) mais n'a
volontairement pas été ré-entraîné en 6 classes : le document de cadrage de la phase de
détection limitait explicitement le périmètre à Regex/Logistic Regression/XGBoost/DistilBERT.
Voir `docs/machine_learning.md`.

## 11. Dataset synthétique : biais de gabarits

Le dataset d'entraînement et le jeu de sécurité sont générés par gabarits (voir
`docs/dataset.md`). Les modèles peuvent apprendre des régularités propres aux gabarits plutôt
qu'une compréhension générale des attaques — la section Robustesse de
`docs/security_evaluation.md` est une première mesure de cet effet, pas une preuve d'absence.

## 12. Ce qui reste hors périmètre du projet (jamais implémenté)

API FastAPI, déploiement Docker/MLflow (prévus dans un plan en 10 phases antérieur, explicitement
exclus des documents de cadrage effectivement fournis pour ce projet). Voir
`docs/final_project_summary.md`, section « Future work ».
