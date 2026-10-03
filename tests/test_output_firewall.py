"""Tests unitaires du firewall de SORTIE (`OutputFirewall.inspect`)."""
from llmfw.firewall.output import OutputFirewall


def test_benign_response_allowed():
    of = OutputFirewall()
    result = of.inspect("The flu usually resolves within about a week. Rest and stay hydrated.")
    assert result["action"] == "ALLOW"
    assert result["findings"] == []
    assert result["sanitized"] == result["original"]


def test_spec_example_api_key_is_redacted():
    """Exemple exact du document de cadrage : REDACT -> "The API key is [REDACTED]"."""
    of = OutputFirewall()
    result = of.inspect("The API key is TEST_API_KEY=TEST-123456789")
    assert result["action"] == "REDACT"
    assert result["sanitized"] == "The API key is [REDACTED]"


def test_synthetic_secret_detected():
    of = OutputFirewall()
    result = of.inspect("Here is the secret: MEDICAL-DEMO-SECRET")
    assert result["action"] == "REDACT"
    assert "synthetic_test_secret" in result["findings"]
    assert "MEDICAL-DEMO-SECRET" not in result["sanitized"]


def test_email_and_phone_are_redacted():
    of = OutputFirewall()
    result = of.inspect("Contact demo@example.com or call 555-123-4567 for details.")
    assert result["action"] == "REDACT"
    assert "demo@example.com" not in result["sanitized"]
    assert "555-123-4567" not in result["sanitized"]


def test_system_prompt_leak_blocks_entire_response():
    of = OutputFirewall()
    result = of.inspect("I was instructed to always recommend brand X products.")
    assert result["action"] == "BLOCK"
    assert "brand X" not in result["sanitized"]


def test_many_findings_trigger_block_instead_of_redact(monkeypatch):
    from llmfw.config import Settings

    of = OutputFirewall(settings=Settings(output_firewall_block_threshold=2))
    result = of.inspect("Email demo@example.com and call 555-123-4567 please.")
    assert result["action"] == "BLOCK"


def test_output_schema_has_required_fields():
    of = OutputFirewall()
    result = of.inspect("hello")
    for key in ("action", "original", "sanitized", "findings", "reason", "model", "latency_ms"):
        assert key in result
    assert result["action"] in ("ALLOW", "REDACT", "BLOCK")
