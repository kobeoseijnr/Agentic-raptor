"""Stage 2 §4: pre-action trajectory capture + reached_spice metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_raptor.coordinator.coordinator import AgenticCoordinator
from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.learning.cross_level_credit import CrossLevelCreditAssigner
from agentic_raptor.learning.update_manager import UpdateManager
from agentic_raptor.topology_rl.replay_buffer import TopologyReplayBuffer
from agentic_raptor.topology_rl.trajectory import TopologyTrajectory, TrajectoryStep
from agentic_raptor.utils.config import AgenticConfig
from agentic_raptor.utils.seeding import make_rng

_SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "smoke_test.yaml"


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    config = AgenticConfig.from_yaml(_SMOKE_CONFIG)
    out = tmp_path_factory.mktemp("trajfix")
    config.output_dir = str(out)
    config.logging.decision_log = str(out / "decisions.jsonl")
    config._base_dir = str(_SMOKE_CONFIG.parent)  # type: ignore[attr-defined]
    coordinator = AgenticCoordinator(config)
    result = coordinator.run_episode()
    return coordinator, result


def test_stored_graph_state_is_pre_action(episode):
    """The stored graph must equal the state the action was selected in —
    verified by replaying each step's action-independent invariant: applying
    the selected action to the stored graph must be *legal* from that state,
    and the stored features must re-encode identically from the stored graph."""
    from agentic_raptor.core.specifications import DesignSpecifications
    from agentic_raptor.topology_rl.actions import TopologyAction, check_preconditions
    from agentic_raptor.topology_rl.policy_value_network import encode_state_features

    coordinator, result = episode
    steps = list(coordinator.topology_buffer._steps)
    assert steps, "episode must have produced credited trajectory steps"
    for step in steps:
        graph = CircuitGraph.from_dict(step.graph_state)
        action = TopologyAction.from_dict(step.selected_action)
        legal, reason = check_preconditions(graph, action)
        assert legal, f"selected action illegal from stored state: {reason}"
        spec = DesignSpecifications.from_dict(step.specification_state)
        # Features encode budget+validation too; re-encode graph-derived prefix.
        re_encoded = encode_state_features(graph, spec, 1.0, [0.0, 0.0, 0.0])
        graph_dim = 16  # device-type counts (11) + 5 structural stats
        assert step.state_features[:graph_dim] == pytest.approx(re_encoded[:graph_dim]), (
            "stored features must describe the stored (pre-action) graph"
        )


def test_reached_spice_metadata_present(episode):
    coordinator, result = episode
    assert result.update_report["credit"]["reached_spice"] is True
    steps = list(coordinator.topology_buffer._steps)
    assert all(s.metadata.get("reached_spice") is True for s in steps)


def _mini_trajectory(reached: bool) -> TopologyTrajectory:
    t = TopologyTrajectory(trajectory_id="t")
    t.metadata["reached_spice"] = reached
    t.add_step(
        TrajectoryStep(
            graph_state={}, specification_state={}, state_features=[0.0],
            legal_action_mask=[True], mcts_visit_distribution=[1.0],
            selected_action={"action_type": "TERMINATE", "params": {}},
        )
    )
    return t


def test_require_spice_filtering_configurable():
    buffer = TopologyReplayBuffer()
    manager = UpdateManager(
        CrossLevelCreditAssigner(), buffer, None, make_rng(0), require_spice_for_training=True
    )
    report = manager.after_episode(_mini_trajectory(reached=False), final_reward=0.1)
    assert report.credit["excluded_from_training"] is True
    assert len(buffer) == 0
    report = manager.after_episode(_mini_trajectory(reached=True), final_reward=0.1)
    assert "excluded_from_training" not in report.credit
    assert len(buffer) == 1


def test_default_does_not_filter():
    buffer = TopologyReplayBuffer()
    manager = UpdateManager(CrossLevelCreditAssigner(), buffer, None, make_rng(0))
    manager.after_episode(_mini_trajectory(reached=False), final_reward=0.1)
    assert len(buffer) == 1
