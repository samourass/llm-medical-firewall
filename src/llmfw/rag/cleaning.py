"""Nettoyage de texte avant découpage (chunking)."""
from __future__ import annotations

import re

_MD_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Nettoyage léger : enlève la ponctuation Markdown structurelle (titres, citations),
    normalise les espaces et sauts de ligne. Conserve le contenu textuel intact."""
    text = text.replace("\r\n", "\n")
    text = _MD_HEADER.sub("", text)
    text = _MD_BLOCKQUOTE.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()
