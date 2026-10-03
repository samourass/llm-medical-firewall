"""Découpage (chunking) d'un texte nettoyé en passages de taille bornée, avec recouvrement.

Approche : découpe en phrases (regex simple), puis regroupe les phrases par blocs ne dépassant
pas `chunk_size_chars`, avec un recouvrement de `overlap_chars` (dernières phrases du bloc
précédent répétées en tête du bloc suivant) pour ne pas couper le contexte à une frontière de
phrase.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str


def _split_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip())
    return sentences


def chunk_text(text: str, chunk_size_chars: int = 500, overlap_chars: int = 80) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current and current_len + sentence_len > chunk_size_chars:
            chunks.append(" ".join(current))
            # Recouvrement : on garde la fin du bloc précédent (par phrases entières) jusqu'à
            # atteindre ~overlap_chars, pour amorcer le bloc suivant.
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len >= overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap.copy()
            current_len = sum(len(s) + 1 for s in current)
        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(documents, chunk_size_chars: int = 500, overlap_chars: int = 80) -> list[Chunk]:
    """`documents` : liste de `rag.loader.RawDocument` déjà nettoyés (`text` = texte nettoyé)."""
    result: list[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc.text, chunk_size_chars, overlap_chars)
        for i, piece in enumerate(pieces):
            result.append(Chunk(chunk_id=f"{doc.doc_id}::chunk{i}", doc_id=doc.doc_id,
                                source=doc.source, text=piece))
    return result
