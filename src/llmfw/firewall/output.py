"""Firewall de SORTIE : inspecte la réponse générée par le LLM, avant qu'elle n'atteigne
l'utilisateur.

Contrairement au firewall d'entrée (qui CLASSE une requête en 6 catégories et décide
ALLOW/BLOCK), le firewall de sortie DÉTECTE des motifs de fuite précis (email, téléphone, PII,
clé API, jeton, identifiants, secrets synthétiques de test, fuite du prompt système) et peut soit
:
- **caviarder** (`REDACT`) les segments concernés quand ils peuvent être remplacés sans détruire
  le reste de la réponse (ex. un email, une clé API isolée) ;
- **bloquer** (`BLOCK`) la réponse entière quand la fuite ne peut pas être proprement caviardée
  (fuite du prompt système) ou quand trop de constatations distinctes apparaissent dans la même
  réponse (`OUTPUT_FIREWALL_BLOCK_THRESHOLD`, défaut 3) ;
- laisser passer (`ALLOW`) si rien n'est détecté.

Secrets synthétiques de test utilisés pour valider la détection (jamais de vrais identifiants) :
    TEST_API_KEY=TEST-123456789
    TEST_SECRET=MEDICAL-DEMO-SECRET
"""
from __future__ import annotations

import re
import time

from llmfw.config import Settings, get_settings

# Chaque motif est (regex compilée, nom de catégorie). L'ORDRE compte : en cas de
# chevauchement entre deux motifs (ex. une clé API contenant une longue suite de chiffres qui
# matcherait aussi le motif téléphone), le motif le plus HAUT dans cette liste gagne — les
# motifs les plus spécifiques (secrets/clés/identifiants) sont donc placés avant les motifs
# génériques (téléphone). Les catégories "system_prompt_leak" déclenchent toujours un BLOCK
# (voir `_UNREDACTABLE_CATEGORIES`) ; les autres sont caviardées.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bTEST_API_KEY\s*=\s*\S+"), "api_key"),
    (re.compile(r"\bTEST_SECRET\s*=\s*\S+"), "credential"),
    (re.compile(r"\bTEST-\d{6,}\b"), "synthetic_test_secret"),
    (re.compile(r"\bMEDICAL-DEMO-SECRET\b"), "synthetic_test_secret"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "api_key"),
    (re.compile(r"\bAKIA[0-9A-Z]{12,}\b"), "api_key"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "api_key"),
    (re.compile(r"\b(api[_\s-]?key|access[_\s-]?token|bearer\s+token|password|secret\s+key)\s*[:=]\s*\S+",
               re.IGNORECASE), "credential"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "pii_ssn"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "email"),
    (re.compile(r"(?<!\d)(\+?\d[\d\-. ]{7,}\d)(?!\d)"), "phone_number"),
    (re.compile(r"\b(you are (an?|the) (ai|assistant|language model)|as an ai language model|"
               r"my (system|developer) prompt (is|says)|i (was|am) instructed to)\b",
               re.IGNORECASE), "system_prompt_leak"),
]

_UNREDACTABLE_CATEGORIES = {"system_prompt_leak"}


def _non_overlapping_matches(text: str) -> list[dict]:
    """Trouve tous les matchs de `_PATTERNS`, puis résout les chevauchements en gardant, pour
    chaque zone de texte contestée, le match du motif le plus prioritaire (le plus haut dans
    `_PATTERNS`). Renvoie les matchs retenus, triés par position dans le texte."""
    candidates = []
    for priority, (pattern, category) in enumerate(_PATTERNS):
        for match in pattern.finditer(text):
            candidates.append({"priority": priority, "category": category,
                               "start": match.start(), "end": match.end(),
                               "text": match.group(0)})
    candidates.sort(key=lambda m: (m["priority"], m["start"]))

    selected: list[dict] = []
    covered: list[tuple[int, int]] = []
    for cand in candidates:
        if any(cand["start"] < end and cand["end"] > start for start, end in covered):
            continue  # zone déjà couverte par un motif plus prioritaire
        selected.append(cand)
        covered.append((cand["start"], cand["end"]))
    selected.sort(key=lambda m: m["start"])
    return selected


class OutputFirewall:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def inspect(self, response_text: str) -> dict:
        t0 = time.perf_counter()
        response_text = response_text or ""

        findings = _non_overlapping_matches(response_text)
        unredactable = [f for f in findings if f["category"] in _UNREDACTABLE_CATEGORIES]
        n_categories = len({f["category"] for f in findings})

        if unredactable:
            action = "BLOCK"
            sanitized = ("[response blocked by output firewall: potential system prompt leakage "
                        "detected]")
            reason = f"unredactable category detected: {sorted({f['category'] for f in unredactable})}"
        elif len(findings) >= self.settings.output_firewall_block_threshold:
            action = "BLOCK"
            sanitized = ("[response blocked by output firewall: too many distinct sensitive "
                        "findings in a single response]")
            reason = (f"{len(findings)} findings across {n_categories} categories >= "
                     f"threshold ({self.settings.output_firewall_block_threshold})")
        elif findings:
            action = "REDACT"
            # Remplacement de droite à gauche pour ne pas décaler les positions déjà calculées.
            parts = list(response_text)
            for f in sorted(findings, key=lambda m: m["start"], reverse=True):
                parts[f["start"]:f["end"]] = list("[REDACTED]")
            sanitized = "".join(parts)
            reason = f"redacted categories: {sorted({f['category'] for f in findings})}"
        else:
            action = "ALLOW"
            sanitized = response_text
            reason = "no sensitive pattern matched"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "action": action,
            "original": response_text,
            "sanitized": sanitized,
            "findings": [f["category"] for f in findings],
            "reason": reason,
            "model": "output_firewall_regex",
            "latency_ms": round(latency_ms, 4),
        }
