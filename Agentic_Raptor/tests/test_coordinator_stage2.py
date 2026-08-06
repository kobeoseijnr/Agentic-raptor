"""Coordinator Stage 2 behaviour: provider failures, simulator failures, no silent fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_raptor.coordinator.coordinator import AgenticCoordinator
from agentic_raptor.utils.config import AgenticConfig
from agentic_raptor.utils.exceptions import GenerationError

_SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "smoke_test.yaml"


def _config(tmp_path, **spice_overrides):
    config = AgenticConfig.from_yaml(_SMOKE_CONFIG)
    config.output_dir = str(tmp_path)
    config.logging.decision_log = str(tmp_path / "decisions.jsonl")
    config._base_dir = str(_SMOKE_CONFIG.parent)  # type: ignore[attr-defined]
    for key, value in spice_overrides.items():
        setattr(config.spice, key, value)
    return config


class _AlwaysFailingGenerator:
    def generate(self, *_args, **_kwargs):
        raise GenerationError("provider down: missing_credential: X not set")


def test_generator_failure_handled_and_logged(tmp_path):
    config = _config(tmp_path)
    coordinator = AgenticCoordinator(config, generator=_AlwaysFailingGenerator())
    result = coordinator.run_episode()
    reasons = [d for d in result.decisions if d["decision"] == "GENERATION_FAILED"]
    assert reasons, "generation failures must be logged as decisions with reasons"
    assert "provider down" in reasons[0]["reason"]
    assert result.final_state == "TERMINATE"
    assert result.best_candidate is None
    # No SPICE happened → reached_spice False propagated to credit report.
    assert result.update_report["credit"]["reached_spice"] is False


def test_unavailable_real_simulator_no_silent_fallback(tmp_path):
    config = _config(tmp_path, simulator="ngspice", ngspice_exe="Z:/missing/ngspice.exe", allow_mock_fallback=False)
    coordinator = AgenticCoordinator(config)
    assert coordinator.simulator_mode == "ngspice"
    result = coordinator.run_episode()
    failures = [d for d in result.decisions if d["decision"] == "SPICE_FAILURE"]
    assert failures and "simulator_unavailable" in failures[0]["reason"]
    assert result.update_report["credit"]["reached_spice"] is False
    assert result.success is False


def test_explicit_mock_fallback_when_allowed(tmp_path):
    config = _config(tmp_path, simulator="ngspice", ngspice_exe="Z:/missing/ngspice.exe", allow_mock_fallback=True)
    coordinator = AgenticCoordinator(config)
    assert coordinator.simulator_mode == "mock", "explicit fallback must switch to mock"
    result = coordinator.run_episode()
    assert result.final_state == "TERMINATE"
    assert result.update_report["credit"]["reached_spice"] is True


def test_mock_mode_unchanged_by_stage2(tmp_path):
    """Stage 1 mock smoke must still work through the Stage 2 coordinator."""
    coordinator = AgenticCoordinator(_config(tmp_path))
    result = coordinator.run_episode()
    assert result.final_state == "TERMINATE"
    changed = result.update_report["parameters_changed"]
    for component in ("policy_value_network", "sac_actor", "sac_critics", "dynamics_model"):
        assert changed.get(component) is True


@pytest.mark.requires_ngspice
def test_real_spice_episode_end_to_end(tmp_path):
    """Full episode against real ngspice (Smoke A superset): final reward is real."""
    from agentic_raptor.spice.ngspice_simulator import discover_ngspice

    if discover_ngspice() is None:
        pytest.skip("ngspice not installed")
    stage2 = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "stage2_real_spice.yaml"
    config = AgenticConfig.from_yaml(stage2)
    config.output_dir = str(tmp_path)
    config.logging.decision_log = str(tmp_path / "decisions.jsonl")
    config.spice.cache_path = None
    config._base_dir = str(stage2.parent)  # type: ignore[attr-defined]
    coordinator = AgenticCoordinator(config)
    result = coordinator.run_episode()
    assert result.final_state == "TERMINATE"
    assert result.update_report["credit"]["reached_spice"] is True
    assert result.metrics, "real SPICE metrics must be present"
    spice_events = [d for d in result.decisions if d["decision"] == "SPICE_RESULT"]
    assert spice_events and "backend=ngspice" in spice_events[0]["reason"]
    changed = result.update_report["parameters_changed"]
    assert changed.get("policy_value_network") is True
    assert changed.get("sac_actor") is True
