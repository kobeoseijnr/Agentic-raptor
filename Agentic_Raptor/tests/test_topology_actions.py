"""Typed topology actions: schemas, preconditions, application, environment."""

from __future__ import annotations

import pytest

from agentic_raptor.topology_rl.actions import (
    ActionType,
    TopologyAction,
    apply_action,
    check_preconditions,
    enumerate_candidate_actions,
)
from agentic_raptor.topology_rl.environment import TopologyEditEnv, TopologyEnvConfig
from agentic_raptor.utils.exceptions import ActionError


def test_schema_validation_rejects_missing_params():
    action = TopologyAction(ActionType.ADD_COMPENSATION_CAPACITOR, {"net_a": "n_out"})
    with pytest.raises(ActionError):
        action.validate_schema()


def test_schema_validation_rejects_extra_params():
    action = TopologyAction(ActionType.TERMINATE, {"bogus": 1})
    with pytest.raises(ActionError):
        action.validate_schema()


def test_add_compensation_capacitor(ota_graph):
    action = TopologyAction(
        ActionType.ADD_COMPENSATION_CAPACITOR, {"net_a": "n_out", "net_b": "n_gnd"}
    )
    legal, reason = check_preconditions(ota_graph, action)
    assert legal, reason
    new_graph, record = apply_action(ota_graph, action)
    assert len(new_graph.nodes) == len(ota_graph.nodes) + 1
    assert record.created_node_ids
    # Reversibility: undo returns the original structure.
    assert record.undo().is_structurally_equal(ota_graph)


def test_illegal_action_unknown_net(ota_graph):
    action = TopologyAction(
        ActionType.ADD_COMPENSATION_CAPACITOR, {"net_a": "n_out", "net_b": "no_such_net"}
    )
    legal, reason = check_preconditions(ota_graph, action)
    assert not legal
    assert "unknown net" in reason
    with pytest.raises(ActionError):
        apply_action(ota_graph, action)


def test_ports_cannot_be_removed(ota_graph):
    action = TopologyAction(ActionType.REMOVE_DEVICE, {"node_id": "vdd"})
    legal, reason = check_preconditions(ota_graph, action)
    assert not legal
    assert "ports cannot be removed" in reason


def test_remove_gain_stage_roundtrip(ota_graph):
    add = TopologyAction(ActionType.ADD_GAIN_STAGE, {"input_net": "n_inp", "output_net": "n_out"})
    with_stage, record = apply_action(ota_graph, add)
    assert len(record.created_node_ids) == 3
    stage_tag = with_stage.nodes[record.created_node_ids[0]].attributes["stage_tag"]
    remove = TopologyAction(ActionType.REMOVE_GAIN_STAGE, {"stage_tag": str(stage_tag)})
    without_stage, _ = apply_action(with_stage, remove)
    assert without_stage.is_structurally_equal(ota_graph)


def test_actions_never_touch_sizing(ota_graph):
    ota_graph.apply_sizing({"m1": {"width_m": 5e-6}})
    action = TopologyAction(ActionType.ADD_BIAS_BRANCH, {"from_net": "n_vdd", "to_net": "n_bias"})
    new_graph, _ = apply_action(ota_graph, action)
    assert new_graph.sizing_state()["m1"]["width_m"] == pytest.approx(5e-6)


def test_enumeration_bounded_and_terminate_first(ota_graph):
    actions = enumerate_candidate_actions(ota_graph, max_actions=10)
    assert actions[0].action_type == ActionType.TERMINATE
    assert len(actions) <= 10
    for action in actions:
        legal, reason = check_preconditions(ota_graph, action)
        assert legal, f"{action.key()}: {reason}"


def test_environment_toy_episode(ota_graph, spec):
    env = TopologyEditEnv(ota_graph, spec, config=TopologyEnvConfig(max_edits=3))
    obs, info = env.reset()
    assert obs["validation"].is_valid
    legal = env.legal_actions()
    assert len(legal) > 1
    # Take a structural edit (index 1), then terminate (index 0).
    obs, reward, terminated, truncated, info = env.step(1)
    assert not terminated
    assert "action" in info
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert env.is_terminal()
    assert reward > 0  # valid graph at termination
    assert "TopologyEditEnv" in env.render()


def test_environment_invalid_index_penalized(ota_graph, spec):
    config = TopologyEnvConfig(max_edits=3, invalid_action_penalty=-0.5)
    env = TopologyEditEnv(ota_graph, spec, config=config)
    env.reset()
    _obs, reward, terminated, _truncated, info = env.step(999)
    assert info["invalid_action"]
    assert reward == pytest.approx(-0.5)
    assert not terminated


def test_legal_action_mask_matches(ota_graph, spec):
    env = TopologyEditEnv(ota_graph, spec, config=TopologyEnvConfig(max_actions=16))
    env.reset()
    mask = env.legal_action_mask()
    assert len(mask) == 16
    assert sum(mask) == len(env.legal_actions())
