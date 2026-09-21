# Dossier d'exécution : la racine du projet (llm-medical-firewall\)
# Étape 2 : génération + split + audit anti-fuite du dataset
$ErrorActionPreference = "Stop"
.\.venv\Scripts\python.exe -m llmfw.data.build
