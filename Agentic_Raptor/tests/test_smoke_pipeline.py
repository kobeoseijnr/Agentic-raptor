"""Full mock end-to-end smoke pipeline with genuine-learning verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_raptor.coordinator.coordinator import AgenticCoordinator
from agentic_raptor.utils.config import AgenticConfig

_SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "smoke_test.yaml"


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    config = AgenticConfig.from_yaml(_SMOKE_CONFIG)
    out = tmp_path_factory.mktemp("smoke_outputs")
    config.output_dir = str(out)
    config.logging.decision_log = str(out / "decisions.jsonl")
    config._base_dir = str(_SMOKE_CONFIG.parent)  # type: ignore[attr-defined]
    coordinator = AgenticCoordinator(config)
    return coordinator.run_episode()


def test_pipeline_reaches_terminate(episode):
    assert episode.final_state == "TERMINATE"
    assert episode.error is None


def test_pipeline_produced_candidate_and_metrics(episode):
    assert episode.best_candidate is not None
    assert episode.best_candidate["sizing_state"], "sizing must have been applied"
    assert episode.metrics, "nominal SPICE metrics must be recorded"
    assert episode.final_reward is not None


def test_every_decision_logged_with_reason(episode):
    assert len(episode.decisions) >= 5
    for decision in episode.decisions:
        assert decision["decision"]
        assert decision["reason"], f"decision {decision['decision']} missing a reason"


def test_genuine_learning_updates_happened(episode):
    changed = episode.update_report["parameters_changed"]
    assert changed.get("policy_value_network") is True, "topology policy/value must train"
    assert changed.get("sac_actor") is True, "SAC actor must train"
    assert changed.get("sac_critics") is True, "SAC critics must train"
    assert changed.get("dynamics_model") is True, "dynamics model must train"


def test_cross_level_credit_propagated(episode):
    credit = episode.update_report["credit"]
    assert credit["steps_credited"] >= 1
    assert credit["final_reward"] == pytest.approx(episode.final_reward)


def test_memory_updated_and_summary_saved(episode):
    assert episode.memory_id is not None
    assert episode.summary_path and Path(episode.summary_path).is_file()


def test_budgets_respected(episode):
    snapshot = episode.budget_snapshot
    assert snapshot["spice_calls_remaining"] >= 0
    assert snapshot["topology_edits_remaining"] >= 0
    assert snapshot["sizing_steps_remaining"] >= 0
