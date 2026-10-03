# Reproducibility — séquence complète (Windows PowerShell)

Toutes les commandes s'exécutent depuis la **racine du projet** (le dossier qui contient
`src\`, `tests\`, `app.py`), sauf mention contraire. Graine aléatoire fixe : `RANDOM_SEED=42`
(`.env` / `.env.example`).

## 1. Créer l'environnement virtuel

```powershell
# Dossier : racine du projet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Installer les dépendances

```powershell
# Dossier : racine du projet
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
# Optionnel (DistilBERT + embeddings sentence-transformers, dépendances lourdes) :
.\.venv\Scripts\python.exe -m pip install -r requirements-dl.txt
```

## 3. Générer le dataset

```powershell
# Dossier : racine du projet
.\scripts\01_build_dataset.ps1
```
Produit `data/{raw,processed,train,validation,test}/` (6 566 exemples, 6 classes, split
70/15/15 stratifié et sans fuite — voir `docs/dataset.md`).

## 4. Valider le dataset

```powershell
# Dossier : racine du projet
.\.venv\Scripts\python.exe -m pytest tests\test_dataset.py -v
```

## 5. Entraîner Logistic Regression (+ XGBoost, même script)

```powershell
# Dossier : racine du projet
.\scripts\03_train_multiclass.ps1 -Model logistic_regression
```

## 6. Entraîner XGBoost

```powershell
# Dossier : racine du projet
.\scripts\03_train_multiclass.ps1 -Model xgboost
# (ou .\scripts\03_train_multiclass.ps1 sans -Model pour entraîner les deux d'un coup)
```

## 7. Entraîner DistilBERT

```powershell
# Dossier : racine du projet
# Nécessite requirements-dl.txt installé et un accès réseau à huggingface.co (non disponible
# dans les environnements de développement de ce projet — voir docs/limitations.md).
.\scripts\04_train_distilbert.ps1 -Epochs 1 -MaxTrainSamples 500   # run rapide de vérification
.\scripts\04_train_distilbert.ps1                                    # run complet (3 epochs)
```

## 8. Construire l'index FAISS (RAG)

```powershell
# Dossier : racine du projet
.\scripts\05_build_rag_index.ps1
```

## 9. Lancer les tests

```powershell
# Dossier : racine du projet
.\scripts\run_tests.ps1
```
Attendu (mesuré dans cet environnement) : voir `docs/final_project_summary.md`, section
« Mesures ».

## 10. Lancer l'évaluation complète

```powershell
# Dossier : racine du projet
.\scripts\07_run_evaluation.ps1
```
Régénère `results/*.csv`, `results/evaluation_results.json`, `results/figures/*.png`, et
`docs/security_evaluation.md`.

## 11. Lancer Streamlit

```powershell
# Dossier : racine du projet
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Ouvre le tableau de bord sur `http://localhost:8501`. Utilisez le menu latéral pour naviguer
entre les 5 pages (voir `README.md`, section « Streamlit »).

## Récapitulatif — une seule commande par étape

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt -r requirements-app.txt
.\scripts\01_build_dataset.ps1
.\scripts\03_train_multiclass.ps1
.\scripts\05_build_rag_index.ps1
.\scripts\run_tests.ps1
.\scripts\07_run_evaluation.ps1
.\.venv\Scripts\python.exe -m streamlit run app.py
```
