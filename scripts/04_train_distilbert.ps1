# Dossier d'exécution : la racine du projet (llm-medical-firewall\)
# Installe torch/transformers/datasets (poids ~qq centaines de Mo, non installés par défaut)
# puis entraîne DistilBERT en 6 classes. CUDA est détecté automatiquement (GPU si présent,
# sinon CPU). Nécessite un accès réseau à https://huggingface.co pour télécharger les poids
# pré-entraînés "distilbert-base-uncased" au premier lancement.
# Usage : .\scripts\04_train_distilbert.ps1                              (3 epochs, dataset complet)
#         .\scripts\04_train_distilbert.ps1 -Epochs 1 -MaxTrainSamples 500  (run rapide / sanity check)
param(
    [int]$Epochs = 3,
    [int]$BatchSize = 16,
    [int]$MaxTrainSamples = 0
)
$ErrorActionPreference = "Stop"

.\.venv\Scripts\python.exe -m pip install -r requirements-dl.txt

if ($MaxTrainSamples -gt 0) {
    .\.venv\Scripts\python.exe -m llmfw.training.distilbert --epochs $Epochs --batch-size $BatchSize --max-train-samples $MaxTrainSamples
} else {
    .\.venv\Scripts\python.exe -m llmfw.training.distilbert --epochs $Epochs --batch-size $BatchSize
}
