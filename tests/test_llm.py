"""Tests unitaires de la couche LLM (Mock + factory)."""
import pytest

from llmfw.config import Settings
from llmfw.llm.factory import get_llm
from llmfw.llm.mock import MockLLM


def test_mock_llm_is_deterministic():
    llm = MockLLM()
    context = "[Source: flu.md] Common flu symptoms include fever and cough."
    r1 = llm.generate("What are flu symptoms?", context)
    r2 = llm.generate("What are flu symptoms?", context)
    assert r1 == r2


def test_mock_llm_handles_empty_context():
    llm = MockLLM()
    response = llm.generate("What is the meaning of life?", "")
    assert "don't have specific information" in response.lower()


def test_mock_llm_uses_context_content():
    llm = MockLLM()
    context = "[Source: flu.md] Rest and stay hydrated when you have the flu."
    response = llm.generate("What should I do for the flu?", context)
    assert "rest" in response.lower()


def test_factory_returns_mock_by_default():
    settings = Settings(llm_provider="mock")
    llm = get_llm(settings)
    assert llm.name == "mock"


def test_factory_rejects_unknown_provider():
    settings = Settings(llm_provider="not_a_real_provider")
    with pytest.raises(ValueError):
        get_llm(settings)
