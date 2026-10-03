"""Pipeline complet du chatbot :

    USER -> INPUT FIREWALL -> RAG -> LLM -> OUTPUT FIREWALL -> USER

Composants injectés (modulaires, chacun testable indépendamment) :
    - `Firewall` (firewall d'entrée, `firewall/api.py`)
    - `RAGPipeline` (`rag/pipeline.py`)
    - `LLMProvider` (`llm/factory.py`)
    - `OutputFirewall` (`firewall/output.py`)
"""
from __future__ import annotations

import time
import uuid

from llmfw.config import Settings, get_settings
from llmfw.firewall.api import Firewall
from llmfw.firewall.output import OutputFirewall
from llmfw.llm.base import LLMProvider
from llmfw.llm.factory import get_llm
from llmfw.logging_utils import SecurityLogger
from llmfw.rag.pipeline import RAGPipeline


def _security_block_response(input_result: dict, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "stage_blocked": "input_firewall",
        "allowed": False,
        "response": (
            "Your request was blocked by the security firewall and was not sent to the "
            "medical assistant. If you believe this is a mistake, please rephrase your "
            "question."
        ),
        "input_firewall": input_result,
        "rag": None,
        "output_firewall": None,
    }


class ChatPipeline:
    def __init__(self, firewall: Firewall | None = None, rag: RAGPipeline | None = None,
                llm: LLMProvider | None = None, output_firewall: OutputFirewall | None = None,
                settings: Settings | None = None, logger: SecurityLogger | None = None) -> None:
        self.settings = settings or get_settings()
        self.firewall = firewall or Firewall(settings=self.settings)
        self.rag = rag or RAGPipeline(self.settings)
        self.llm = llm or get_llm(self.settings)
        self.output_firewall = output_firewall or OutputFirewall(self.settings)
        self.logger = logger or SecurityLogger(self.settings)

    def chat(self, user_input: str) -> dict:
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        input_result = self.firewall.inspect_input(user_input)
        if not input_result["allowed"]:
            self.logger.log_event(request_id=request_id, input_result=input_result,
                                  output_result=None,
                                  total_latency_ms=(time.perf_counter() - t0) * 1000.0,
                                  input_text=user_input)
            return _security_block_response(input_result, request_id)

        retrieved = self.rag.retrieve(user_input)
        context = RAGPipeline.build_context(retrieved)

        raw_response = self.llm.generate(user_input, context)

        output_result = self.output_firewall.inspect(raw_response)

        self.logger.log_event(request_id=request_id, input_result=input_result,
                              output_result=output_result,
                              total_latency_ms=(time.perf_counter() - t0) * 1000.0,
                              input_text=user_input)

        return {
            "request_id": request_id,
            "stage_blocked": None if output_result["action"] != "BLOCK" else "output_firewall",
            "allowed": output_result["action"] != "BLOCK",
            "response": output_result["sanitized"],
            "input_firewall": input_result,
            "rag": {"sources": [c.source for c in retrieved], "n_chunks_retrieved": len(retrieved)},
            "output_firewall": output_result,
        }


def chat(user_input: str, pipeline: ChatPipeline | None = None) -> dict:
    """Fonction de convenance : `chat("...")` avec un pipeline construit à partir de `.env`."""
    pipeline = pipeline or ChatPipeline()
    return pipeline.chat(user_input)
