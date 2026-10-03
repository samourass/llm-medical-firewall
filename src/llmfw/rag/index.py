"""Index FAISS pour la recherche de similarité sur les embeddings des chunks.

Les vecteurs sont supposés déjà normalisés (norme L2 = 1) : un index `IndexFlatIP` (produit
scalaire) équivaut alors à une similarité cosinus, avec une recherche exacte (pas d'ANN
approximatif) adaptée à un petit corpus de démonstration.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from llmfw.rag.chunking import Chunk


class FaissIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == self.dim
        self.index.add(np.ascontiguousarray(embeddings.astype("float32")))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        query_vector = np.ascontiguousarray(query_vector.astype("float32")).reshape(1, -1)
        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, out_dir: Path, embedder_name: str, embedding_model: str | None) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "faiss.index"))
        with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        meta = {"dim": self.dim, "n_chunks": len(self.chunks), "embedder": embedder_name,
                "embedding_model": embedding_model}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path) -> "FaissIndex":
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        obj = cls(dim=meta["dim"])
        obj.index = faiss.read_index(str(out_dir / "faiss.index"))
        obj.chunks = []
        with (out_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                obj.chunks.append(Chunk(**d))
        return obj
