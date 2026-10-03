"""Fournisseurs d'embeddings pour le RAG.

Deux backends, derrière une interface commune `encode(list[str]) -> np.ndarray (n, dim)` :

- `SentenceTransformerEmbedder` : embeddings sémantiques réels via
  `sentence-transformers/all-MiniLM-L6-v2` (ou un autre modèle configuré). C'est le backend
  demandé par défaut. Nécessite de télécharger le modèle depuis huggingface.co au premier
  chargement (mis en cache localement ensuite).
- `TfidfEmbedder` : repli 100% local (aucun réseau requis), TF-IDF + normalisation L2. Moins
  performant sémantiquement (pas de synonymes/paraphrase) mais garantit que le RAG reste
  testable dans un environnement sans accès réseau à Hugging Face.

`get_embedder(settings)` choisit le backend selon `EMBEDDING_PROVIDER` :
- "sentence_transformers" : force ce backend (lève une erreur explicite si indisponible).
- "tfidf" : force ce backend.
- "auto" (défaut) : essaie sentence_transformers, bascule sur tfidf en cas d'échec (import,
  téléchargement, ou tout autre problème de chargement) et journalise un avertissement clair —
  ne fait JAMAIS échouer silencieusement le pipeline RAG pour une question de réseau.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger("llmfw.rag.embeddings")


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")


class TfidfEmbedder:
    """Embedder de repli, sans dépendance réseau. Doit être `fit()` sur le corpus avant `encode`."""

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self._fitted = False
        self.dim = 0

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True
        self.dim = len(self._vectorizer.vocabulary_)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit(corpus) doit être appelé avant encode().")
        matrix = self._vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def get_embedder(settings, corpus_for_tfidf_fit: list[str] | None = None) -> Embedder:
    provider = settings.embedding_provider

    def _make_tfidf() -> TfidfEmbedder:
        emb = TfidfEmbedder()
        emb.fit(corpus_for_tfidf_fit or [])
        return emb

    if provider == "tfidf":
        return _make_tfidf()

    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)

    if provider == "auto":
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:  # noqa: BLE001 - repli volontairement large (réseau, import, etc.)
            logger.warning(
                "EMBEDDING_PROVIDER=auto : impossible de charger '%s' (%s: %s) — repli sur "
                "l'embedder TF-IDF local (aucun accès réseau requis).",
                settings.embedding_model, type(exc).__name__, exc,
            )
            return _make_tfidf()

    raise ValueError(f"embedding_provider inconnu : {provider!r}")
