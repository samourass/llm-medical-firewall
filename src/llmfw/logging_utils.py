"""Journalisation des événements de sécurité, sans jamais exposer de contenu sensible.

Ce que l'on enregistre :
    timestamp, request_id, classification d'entrée (label/confidence/decision/model/latence),
    décision de sécurité en sortie (action/model/latence).

Ce que l'on N'enregistre JAMAIS :
    - le texte brut de la requête utilisateur (seulement sa longueur et un hash SHA-256, utile
      pour corréler des événements sans pouvoir retrouver le contenu) ;
    - le texte brut de la réponse du LLM ou tout secret détecté par le firewall de sortie
      (seulement les catégories de "findings", jamais les valeurs matchées) ;
    - la `reason` de la classification d'entrée est conservée (elle décrit la décision, pas un
      secret), mais celle du firewall de sortie est volontairement omise si l'action est
      REDACT/BLOCK, car elle peut mentionner du contenu détecté.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from llmfw.config import Settings, get_settings


def _sha256_prefix(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


class SecurityLogger:
    def __init__(self, settings: Settings | None = None, log_path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.log_path = log_path or (self.settings.logs_dir / "security_events.jsonl")

    def log_event(self, request_id: str, input_result: dict, output_result: dict | None,
                 total_latency_ms: float, input_text: str) -> dict:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "input": {
                "text_sha256_prefix": _sha256_prefix(input_text or ""),
                "text_length": len(input_text or ""),
                "label": input_result.get("label"),
                "confidence": input_result.get("confidence"),
                "allowed": input_result.get("allowed"),
                "model": input_result.get("model"),
                "latency_ms": input_result.get("latency_ms"),
            },
            "output": None,
            "total_latency_ms": round(total_latency_ms, 4),
        }
        if output_result is not None:
            event["output"] = {
                "action": output_result.get("action"),
                "n_findings": len(output_result.get("findings", [])),
                "finding_categories": output_result.get("findings"),
                "model": output_result.get("model"),
                "latency_ms": output_result.get("latency_ms"),
            }

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event
