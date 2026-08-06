"""Multimodal provider interface, real-LLM generator, schematic parser, fusion guards."""

from __future__ import annotations

import json
import os

import pytest

from agentic_raptor.specification.fusion import fuse_fields
from agentic_raptor.specification.parser import ExtractedField
from agentic_raptor.specification.schematic import ProviderSchematicParser
from agentic_raptor.topology_generation.generator import (
    LLMTopologyGenerator,
    build_five_transistor_ota,
)
from agentic_raptor.topology_generation.provider import (
    MockMultimodalModel,
    OpenAICompatibleModel,
    ProviderError,
    ProviderSettings,
    build_provider,
)
from agentic_raptor.utils.exceptions import GenerationError

_RUN_API = bool(os.environ.get("AGENTIC_RAPTOR_RUN_API_TESTS")) and bool(os.environ.get("OPENAI_API_KEY"))


def _valid_generator_payload() -> dict:
    graph = build_five_transistor_ota("llm-cand-0")
    payload = graph.to_dict()
    payload["reasoning_summary"] = "5T OTA reproduced from memory"
    payload["confidence"] = 0.8
    return {"candidates": [payload]}


# ---------------------------------------------------------------------------
# Credentials and provider errors
# ---------------------------------------------------------------------------
def test_missing_credential_detected(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    settings = ProviderSettings(api_key_env="MISSING_TEST_KEY")
    assert settings.missing_credential() == "MISSING_TEST_KEY"
    model = OpenAICompatibleModel(settings)
    with pytest.raises(ProviderError) as excinfo:
        model.generate_structured("prompt", [], {})
    assert excinfo.value.kind == "missing_credential"
    assert "MISSING_TEST_KEY" in excinfo.value.detail


def test_build_provider_unknown_name():
    with pytest.raises(ProviderError):
        build_provider(ProviderSettings(provider="nonexistent"))


# ---------------------------------------------------------------------------
# Real LLM generator over the mock provider
# ---------------------------------------------------------------------------
def test_llm_generator_produces_validated_graphs(spec):
    mock = MockMultimodalModel(responses=[_valid_generator_payload()])
    generator = LLMTopologyGenerator(mock, max_retries=0)
    graphs = generator.generate(spec, [], None, number_of_candidates=1)
    assert len(graphs) == 1
    assert graphs[0].metadata.extra["generator"] == "llm"
    assert mock.calls[0]["images"] == 0


def test_llm_generator_retries_after_malformed_then_succeeds(spec):
    mock = MockMultimodalModel(responses=[{"not_candidates": []}, _valid_generator_payload()])
    generator = LLMTopologyGenerator(mock, max_retries=1)
    graphs = generator.generate(spec, [], None, 1)
    assert len(graphs) == 1
    assert graphs[0].metadata.extra["generation_attempt"] == 2


def test_llm_generator_rejects_invalid_graphs_and_exhausts_retries(spec):
    bad_graph = {"graph_id": "bad", "nodes": [{"node_id": "r1", "device_type": "RESISTOR"}], "edges": []}
    mock = MockMultimodalModel(responses=[{"candidates": [bad_graph]}, {"candidates": [bad_graph]}])
    generator = LLMTopologyGenerator(mock, max_retries=1)
    with pytest.raises(GenerationError, match="failed after 2 attempts"):
        generator.generate(spec, [], None, 1)


def test_llm_generator_rejects_unsupported_device_type(spec):
    payload = _valid_generator_payload()
    payload["candidates"][0]["nodes"][0]["device_type"] = "TRIODE"
    mock = MockMultimodalModel(responses=[payload])
    generator = LLMTopologyGenerator(mock, max_retries=0)
    with pytest.raises(GenerationError):
        generator.generate(spec, [], None, 1)


# ---------------------------------------------------------------------------
# Schematic parser
# ---------------------------------------------------------------------------
def _schematic_response() -> dict:
    return {
        "circuit_class": "OTA",
        "devices": [{"label": "M1", "type": "NMOS"}, {"label": "M3", "type": "PMOS"}],
        "blocks": ["diff_pair", "current_mirror"],
        "ports": {"inputs": ["inp", "inn"], "outputs": ["out"], "supply": "VDD", "ground": "GND"},
        "component_values": [{"label": "CL", "value": "1pF"}],
        "supply_label": "VDD=1.8V",
        "connectivity_notes": ["M1/M2 sources joined at tail node"],
        "topology_hints": ["five-transistor OTA"],
        "confidence": 0.75,
        "unresolved": ["bulk connections not visible"],
    }


def test_schematic_parser_extracts_context_not_inventions(tmp_path):
    image = tmp_path / "s.png"
    image.write_bytes(b"\x89PNG\r\n")
    parser = ProviderSchematicParser(MockMultimodalModel(responses=[_schematic_response()]))
    result = parser.parse_rich(str(image))
    names = {f.name for f in result.fields}
    assert names == {"circuit_class", "supply_voltage"}  # only class + printed label
    supply = next(f for f in result.fields if f.name == "supply_voltage")
    assert supply.value == pytest.approx(1.8)
    assert supply.source == "image"
    assert result.unresolved == ["bulk connections not visible"]
    assert "five-transistor OTA" in result.hints["topology_hints"]
    assert "schematic_analysis" in parser.context_summary()


def test_image_cannot_override_explicit_specification():
    extracted = [
        ExtractedField("supply_voltage", 3.3, "structured"),
        ExtractedField("supply_voltage", 1.8, "image", confidence=0.9),
        ExtractedField("circuit_class", "ota", "image"),
        ExtractedField("technology", "180nm", "text"),
    ]
    spec, report = fuse_fields(extracted)
    assert spec is not None
    assert spec.supply_voltage == pytest.approx(3.3), "explicit value must win over image"
    assert spec.source_metadata["supply_voltage"] == "structured"
    assert spec.source_metadata["circuit_class"] == "image"  # image may fill gaps


# ---------------------------------------------------------------------------
# Real API (opt-in only: needs OPENAI_API_KEY AND AGENTIC_RAPTOR_RUN_API_TESTS=1)
# ---------------------------------------------------------------------------
@pytest.mark.requires_multimodal_api
@pytest.mark.skipif(not _RUN_API, reason="OPENAI_API_KEY and AGENTIC_RAPTOR_RUN_API_TESTS=1 required")
def test_real_provider_returns_json_object():
    model = OpenAICompatibleModel(ProviderSettings(retries=1, timeout_s=60.0))
    result = model.generate_structured(
        'Return exactly this JSON object: {"ok": true, "n": 3}', [], {}
    )
    assert isinstance(result, dict)
    assert result.get("ok") is True


def test_provider_never_logs_credentials(monkeypatch, caplog):
    monkeypatch.setenv("FAKE_KEY_ENV", "sk-secret-value-123")
    settings = ProviderSettings(api_key_env="FAKE_KEY_ENV", default_base_url="http://127.0.0.1:9", retries=0, timeout_s=0.2)
    model = OpenAICompatibleModel(settings)
    with pytest.raises(ProviderError), caplog.at_level("DEBUG"):
        model.generate_structured("x", [], {})
    assert "sk-secret-value-123" not in caplog.text
    assert "sk-secret-value-123" not in json.dumps([r.getMessage() for r in caplog.records])
