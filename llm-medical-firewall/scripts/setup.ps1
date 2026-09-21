# Dossier d'exécution : la racine du projet (llm-medical-firewall\)
# Usage : .\scripts\setup.ps1
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m pip install -e .

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
}
Write-Host "Environnement prêt. Activez-le avec : .\.venv\Scripts\Activate.ps1"
