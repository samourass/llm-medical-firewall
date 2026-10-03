"""Pipeline RAG complet : Documents -> Cleaning -> Chunking -> Embeddings -> FAISS ->
Similarity Search -> Context.

Usage :
    from llmfw.rag.pipeline import RAGPipeline
    from llmfw.config import get_settings

    rag = RAGPipeline(get_settings())
    rag.build_index()                       # une fois (ou charge un index existant avec load())
    chunks = rag.retrieve("What are common flu symptoms?", k=3)
    context = rag.build_context(chunks)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from llmfw.config import Settings, get_settings
from llmfw.rag.chunking import chunk_documents
from llmfw.rag.cleaning import clean_text
from llmfw.rag.embeddings import Embedder, SentenceTransformerEmbedder, TfidfEmbedder, get_embedder
from llmfw.rag.index import FaissIndex
from llmfw.rag.loader import load_documents


@dataclass
class RetrievedChunk:
    text: str
    source: str
    doc_id: str
    score: float


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._index: FaissIndex | None = None
        self._embedder: Embedder | None = None

    # --- construction de l'index ---

    def build_index(self, save: bool = True) -> dict:
        """Charge les documents, nettoie, découpe, embed, indexe. Retourne des statistiques
        (nombre de documents/chunks, backend d'embedding utilisé) — mesuré, rien d'inventé."""
        settings = self.settings
        raw_docs = load_documents(settings.medical_docs_dir)
        cleaned_docs = [
            type(d)(doc_id=d.doc_id, source=d.source, text=clean_text(d.text)) for d in raw_docs
        ]
        chunks = chunk_documents(cleaned_docs, settings.rag_chunk_size_chars, settings.rag_chunk_overlap_chars)
        if not chunks:
            raise RuntimeError("Aucun chunk produit à partir des documents médicaux.")

        chunk_texts = [c.text for c in chunks]
        embedder = get_embedder(settings, corpus_for_tfidf_fit=chunk_texts)
        vectors = embedder.encode(chunk_texts)

        index = FaissIndex(dim=vectors.shape[1])
        index.add(vectors, chunks)

        self._embedder = embedder
        self._index = index

        stats = {
            "n_documents": len(raw_docs),
            "n_chunks": len(chunks),
            "embedding_backend": embedder.name,
            "embedding_dim": vectors.shape[1],
        }
        if save:
            self._save(embedder)
            stats["saved_to"] = str(self._index_dir())
        return stats

    # --- persistance ---

    def _index_dir(self) -> Path:
        return self.settings.models_dir / "rag"

    def _save(self, embedder: Embedder) -> None:
        out_dir = self._index_dir()
        embedding_model = getattr(embedder, "model_name", None)
        self._index.save(out_dir, embedder_name=embedder.name, embedding_model=embedding_model)
        if isinstance(embedder, TfidfEmbedder):
            joblib.dump(embedder, out_dir / "tfidf_embedder.joblib")

    def load(self) -> None:
        out_dir = self._index_dir()
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        self._index = FaissIndex.load(out_dir)
        if meta["embedder"] == "tfidf":
            self._embedder = joblib.load(out_dir / "tfidf_embedder.joblib")
        else:
            self._embedder = SentenceTransformerEmbedder(meta["embedding_model"])

    def _ensure_ready(self) -> None:
        if self._index is None or self._embedder is None:
            if (self._index_dir() / "meta.json").exists():
                self.load()
            else:
                self.build_index()

    # --- recherche ---

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self._ensure_ready()
        k = k or self.settings.rag_top_k
        query_vector = self._embedder.encode([query])[0]
        results = self._index.search(query_vector, k=k)
        return [RetrievedChunk(text=c.text, source=c.source, doc_id=c.doc_id, score=score)
                for c, score in results]

    @staticmethod
    def build_context(chunks: list[RetrievedChunk]) -> str:
        """Concatène les passages récupérés en un contexte attribué à sa source, prêt à être
        injecté dans le prompt du LLM."""
        if not chunks:
            return ""
        parts = [f"[Source: {c.source}] {c.text}" for c in chunks]
        return "\n\n".join(parts)
