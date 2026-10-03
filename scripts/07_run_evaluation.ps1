# Dossier d'exécution : la racine du projet (llm-medical-firewall\)
# Lance l'évaluation de sécurité complète (jeu de 600 requêtes, robustesse, firewall de sortie),
# régénère results/*.csv, results/figures/*.png et docs/security_evaluation.md.
# Nécessite que les modèles multiclasses soient déjà entraînés (scripts\03_train_multiclass.ps1)
# et, si vous voulez inclure DistilBERT, scripts\04_train_distilbert.ps1 lancé au préalable.
$ErrorActionPreference = "Stop"
.\.venv\Scripts\python.exe evaluation\run_full_evaluation.py
