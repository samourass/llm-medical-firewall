"""Chargement des documents médicaux (`data/medical/*.md` ou `*.txt`)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawDocument:
    doc_id: str
    source: str
    text: str


def load_documents(docs_dir: Path) -> list[RawDocument]:
    """Charge tous les fichiers `.md`/`.txt` d'un dossier, triés par nom pour un ordre
    reproductible."""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Dossier de documents médicaux introuvable : {docs_dir}")
    paths = sorted([p for p in docs_dir.iterdir() if p.suffix.lower() in (".md", ".txt")])
    if not paths:
        raise FileNotFoundError(f"Aucun document .md/.txt trouvé dans {docs_dir}")
    return [RawDocument(doc_id=p.stem, source=p.name, text=p.read_text(encoding="utf-8")) for p in paths]
