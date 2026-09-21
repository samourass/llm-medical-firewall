# Dossier d'exécution : la racine du projet (llm-medical-firewall\)
# Étapes 3 à 5 : Logistic Regression, Random Forest, XGBoost
# Usage : .\scripts\02_train_classical.ps1            (les trois modèles)
#         .\scripts\02_train_classical.ps1 xgboost    (un seul modèle)
param([string]$Model = "all")
$ErrorActionPreference = "Stop"
.\.venv\Scripts\python.exe -m llmfw.training.classical --model $Model
