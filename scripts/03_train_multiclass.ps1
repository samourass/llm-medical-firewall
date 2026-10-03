# Dossier d'exécution : la racine du projet (llm-medical-firewall\)
# Entraîne les classifieurs TF-IDF en 6 classes (benign, prompt_injection, jailbreak,
# rag_prompt_injection, pii_exfiltration, secret_exfiltration), requis par le firewall unifié.
# Usage : .\scripts\03_train_multiclass.ps1                    (les deux modèles)
#         .\scripts\03_train_multiclass.ps1 -Model xgboost     (un seul modèle)
param([string]$Model = "all")
$ErrorActionPreference = "Stop"
.\.venv\Scripts\python.exe -m llmfw.training.multiclass --model $Model
