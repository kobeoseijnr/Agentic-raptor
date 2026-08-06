"""Genuine policy/value learning: real optimiser steps change parameters."""

from __future__ import annotations

import pytest

from agentic_raptor.learning.cross_level_credit import CrossLevelCreditAssigner
from agentic_raptor.topology_rl.policy_value_network import (
    STATE_FEATURE_DIM,
    NetworkEvaluator,
    PolicyValueConfig,
    build_policy_value_network,
    encode_state_features,
)
from agentic_raptor.topology_rl.replay_buffer import TopologyReplayBuffer
from agentic_raptor.topology_rl.trainer import PolicyValueTrainer, parameter_checksum
from agentic_raptor.topology_rl.trajectory import TopologyTrajectory, TrajectoryStep
from agentic_raptor.utils.seeding import make_rng


def _make_trajectory(ota_graph, spec, n_steps: int = 3, max_actions: int = 16) -> TopologyTrajectory:
    features = encode_state_features(ota_graph, spec, 1.0, [1.0, 0.0, 0.0])
    trajectory = TopologyTrajectory(trajectory_id="t-test")
    for i in range(n_steps):
        dist = [0.0] * max_actions
        dist[i % 4] = 0.7
        dist[(i + 1) % 4] = 0.3
        trajectory.add_step(
            TrajectoryStep(
                graph_state=ota_graph.to_dict(),
                specification_state=spec.to_dict(),
                state_features=features,
                legal_action_mask=[j < 6 for j in range(max_actions)],
                mcts_visit_distribution=dist,
                selected_action={"action_type": "TERMINATE", "params": {}},
            )
        )
    return trajectory


def test_state_feature_dim_matches_constant(ota_graph, spec):
    features = encode_state_features(ota_graph, spec, 1.0, [1.0, 0.0, 0.0])
    assert len(features) == STATE_FEATURE_DIM


def test_policy_and_value_update_change_parameters(ota_graph, spec):
    config = PolicyValueConfig(max_actions=16, hidden_dim=32, lr=1e-2)
    network = build_policy_value_network(config)
    trainer = PolicyValueTrainer(network, config)

    trajectory = _make_trajectory(ota_graph, spec)
    CrossLevelCreditAssigner(gamma=0.97).assign(trajectory, final_reward=0.8)
    buffer = TopologyReplayBuffer()
    assert buffer.add_trajectory(trajectory) == 3

    before = parameter_checksum(network)
    report = trainer.train_on_steps(buffer.sample(3, make_rng(0)))
    after = parameter_checksum(network)

    assert report.batch_size == 3
    assert report.policy_loss > 0.0
    assert report.value_loss >= 0.0
    assert before != after, "optimiser step must change network parameters"


def test_value_targets_come_from_final_post_sizing_reward(ota_graph, spec):
    trajectory = _make_trajectory(ota_graph, spec, n_steps=2)
    CrossLevelCreditAssigner(gamma=0.5).assign(trajectory, final_reward=1.0)
    # z_0 = 0.5^(1) * 1.0, z_1 = 0.5^0 * 1.0
    assert trajectory.steps[0].discounted_return == pytest.approx(0.5)
    assert trajectory.steps[1].discounted_return == pytest.approx(1.0)
    assert all(s.final_post_sizing_reward == pytest.approx(1.0) for s in trajectory.steps)


def test_network_evaluator_masks_priors(ota_graph, spec):
    config = PolicyValueConfig(max_actions=8, hidden_dim=16)
    network = build_policy_value_network(config)
    evaluator = NetworkEvaluator(network, config)
    from agentic_raptor.topology_rl.actions import enumerate_candidate_actions

    legal = enumerate_candidate_actions(ota_graph, max_actions=8)
    priors, value = evaluator.evaluate(ota_graph, spec, 1.0, [1.0, 0.0, 0.0], legal)
    assert len(priors) == len(legal)
    assert sum(priors) == pytest.approx(1.0, abs=1e-6)
    assert -1.0 <= value <= 1.0


def test_message_passing_encoder_forward(ota_graph, spec):
    config = PolicyValueConfig(max_actions=8, hidden_dim=16, encoder="message_passing", mp_node_dim=8)
    network = build_policy_value_network(config)
    evaluator = NetworkEvaluator(network, config)
    from agentic_raptor.topology_rl.actions import enumerate_candidate_actions

    legal = enumerate_candidate_actions(ota_graph, max_actions=8)
    priors, value = evaluator.evaluate(ota_graph, spec, 1.0, [1.0, 0.0, 0.0], legal)
    assert len(priors) == len(legal)
    assert -1.0 <= value <= 1.0


def test_replay_buffer_persistence(tmp_path, ota_graph, spec):
    trajectory = _make_trajectory(ota_graph, spec)
    CrossLevelCreditAssigner().assign(trajectory, 0.5)
    buffer = TopologyReplayBuffer()
    buffer.add_trajectory(trajectory)
    path = buffer.save(tmp_path / "steps.jsonl")
    restored = TopologyReplayBuffer.load(path)
    assert len(restored) == len(buffer)
