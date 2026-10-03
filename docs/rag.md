# RAG (Retrieval-Augmented Generation)

## Pipeline

```text
Documents (data/medical/*.md)
  -> Cleaning        (rag/cleaning.py : retire la ponctuation Markdown structurelle, normalise les espaces)
  -> Chunking         (rag/chunking.py : découpe par phrases, blocs <= RAG_CHUNK_SIZE_CHARS, recouvrement RAG_CHUNK_OVERLAP_CHARS)
  -> Embeddings        (rag/embeddings.py : sentence-transformers OU repli TF-IDF, voir ci-dessous)
  -> FAISS               (rag/index.py : IndexFlatIP sur vecteurs normalisés = similarité cosinus exacte)
  -> Similarity Search     (RAGPipeline.retrieve(query, k))
  -> Context                (RAGPipeline.build_context(chunks) : passages attribués à leur source)
  -> LLM
```

Chaque étape est un module indépendant, testable seul (`tests/test_rag.py`, 6 tests, aucune
dépendance au firewall ni au LLM).

## Documents (`data/medical/`)

6 documents Markdown, 100 % synthétiques/publics (information de santé générale, aucune donnée
patient réelle) : `seasonal_flu.md`, `hypertension.md`, `type2_diabetes.md`,
`cold_vs_allergies.md`, `medication_safety.md`, `clinic_appointments.md`. Chacun porte un
en-tête explicite « SYNTHETIC DEMONSTRATION DOCUMENT ».

## Embeddings — deux backends, sélection automatique

`EMBEDDING_PROVIDER` (`.env`) :

- **`sentence_transformers`** (le backend visé par défaut) : `sentence-transformers/all-MiniLM-L6-v2`,
  embeddings sémantiques réels. Nécessite `requirements-dl.txt` et un accès réseau à
  `huggingface.co` au premier chargement (mis en cache ensuite).
- **`tfidf`** : repli 100 % local, aucune dépendance réseau. TF-IDF (bi-grammes) + normalisation
  L2.
- **`auto`** (défaut) : essaie `sentence_transformers`, bascule automatiquement sur `tfidf` en
  cas d'échec (import ou réseau), avec un avertissement journalisé — **le pipeline RAG ne casse
  jamais silencieusement pour une question de réseau.**

### Mesuré dans ce bac à sable

`huggingface.co` est bloqué par la liste blanche réseau de cet environnement (même limitation
que pour DistilBERT, voir `docs/machine_learning.md`). `EMBEDDING_PROVIDER=auto` est donc
**retombé sur TF-IDF** ici — c'est un choix mesuré et documenté, pas un bug caché. Exemple
mesuré (`scripts/05_build_rag_index.ps1`, 6 documents, 29 chunks) :

```json
{"n_documents": 6, "n_chunks": 29, "embedding_backend": "tfidf", "embedding_dim": 1367}
```

Conséquence connue : la recherche TF-IDF ne capture pas les paraphrases/synonymes. Exemple
mesuré : la requête « What are common flu symptoms? » remonte en tête un passage sur les
rendez-vous de clinique (score 0,1043) avant le document sur la grippe (score 0,1034) — un
choix de mots différent (« flu symptoms fever cough ») remonte correctement le bon document en
tête (score 0,159). Avec `sentence-transformers` (backend visé par défaut, à valider chez vous),
cette sensibilité au choix exact des mots serait nettement réduite.

## Index FAISS

`IndexFlatIP` (produit scalaire = cosinus, car les vecteurs sont normalisés), recherche exacte
(adaptée à un petit corpus de démonstration, pas à un ANN à grande échelle). Persisté dans
`models/rag/` (`faiss.index`, `chunks.jsonl`, `meta.json`, + `tfidf_embedder.joblib` si le
backend TF-IDF a été utilisé, pour ré-encoder les requêtes avec le même vocabulaire).

## Construire l'index

```powershell
# Dossier : llm-medical-firewall\
.\scripts\05_build_rag_index.ps1
```

## Utilisation programmatique

```python
from llmfw.rag.pipeline import RAGPipeline
from llmfw.config import get_settings

rag = RAGPipeline(get_settings())
rag.build_index()                                   # ou rag.load() si déjà construit
chunks = rag.retrieve("What are common flu symptoms?", k=3)
context = RAGPipeline.build_context(chunks)          # texte attribué, prêt pour le LLM
```
