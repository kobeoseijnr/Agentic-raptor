"""Genuine SAC learning: sampling, updates, targets, dynamics, imagined rollouts."""

from __future__ import annotations

import pytest

from agentic_raptor.sizing.graph_conditioned_mb_sac import (
    GraphConditionedMBSAC,
    MBSACConfig,
    SizingStateContext,
    encode_sizing_state,
)
from agentic_raptor.sizing.parameter_space import SizingParameterSpace
from agentic_raptor.sizing.replay_buffer import SizingReplayBuffer, SizingTransition
from agentic_raptor.topology_rl.trainer import parameter_checksum
from agentic_raptor.utils.seeding import make_rng


@pytest.fixture
def sac(ota_graph):
    space = SizingParameterSpace.from_graph(ota_graph)
    return GraphConditionedMBSAC(space, MBSACConfig(hidden_dim=32, lr=1e-2, seed=3))


def _fill_replay(sac: GraphConditionedMBSAC, ota_graph, spec, n: int = 12) -> None:
    ctx = SizingStateContext(graph=ota_graph, spec=spec, sizing_vector=[0.0] * sac.action_dim)
    state = encode_sizing_state(ctx, sac.action_dim)
    for i in range(n):
        action = [((i * 7 + j) % 11 - 5) / 5.0 for j in range(sac.action_dim)]
        next_state = [s + 0.01 * (i % 3 - 1) for s in state]
        sac.add_transition(
            SizingTransition(
                state=state, action=action, reward=0.1 * (i % 5 - 2), next_state=next_state, done=(i == n - 1)
            )
        )
        state = next_state


def test_parameter_space_from_topology(ota_graph):
    space = SizingParameterSpace.from_graph(ota_graph)
    # 7 MOS × 3 params + 1 cap + 1 current source = 23 (cl + cc absent here; ib1 present)
    names = [s.name for s in space.specs]
    assert "m1.width_m" in names
    assert "ib1.current_a" in names
    assert "cl.capacitance_f" in names
    assert space.dim == len(names)
    # Round-trip normalize/denormalize.
    vec = [0.5] * space.dim
    sizing = space.denormalize(vec)
    back = space.normalize(sizing)
    assert back == pytest.approx(vec, abs=1e-9)


def test_stochastic_sampling_and_bounds(sac):
    state = [0.0] * sac.state_dim
    a1 = sac.select_action(state)
    a2 = sac.select_action(state)
    assert len(a1) == sac.action_dim
    assert all(-1.0 <= v <= 1.0 for v in a1)
    assert a1 != a2, "stochastic actor must sample different actions"
    d1 = sac.select_action(state, deterministic=True)
    d2 = sac.select_action(state, deterministic=True)
    assert d1 == d2, "deterministic mode must be repeatable"


def test_actor_critic_alpha_target_updates(sac, ota_graph, spec):
    _fill_replay(sac, ota_graph, spec)
    actor_before = parameter_checksum(sac.actor)
    critic_before = parameter_checksum(sac.q1) + parameter_checksum(sac.q2)
    target_before = parameter_checksum(sac.q1_target) + parameter_checksum(sac.q2_target)
    alpha_before = sac.alpha

    report = sac.update(batch_size=8)

    assert "critic_loss" in report and "actor_loss" in report
    assert parameter_checksum(sac.actor) != actor_before, "actor update must change parameters"
    assert parameter_checksum(sac.q1) + parameter_checksum(sac.q2) != critic_before, "critic update must change parameters"
    assert (
        parameter_checksum(sac.q1_target) + parameter_checksum(sac.q2_target) != target_before
    ), "Polyak update must move target networks"
    assert sac.alpha != alpha_before, "entropy temperature must adapt"


def test_dynamics_model_update_changes_parameters(sac, ota_graph, spec):
    _fill_replay(sac, ota_graph, spec)
    before = parameter_checksum(sac.dynamics)
    report = sac.update_dynamics(batch_size=8)
    assert "total_loss" in report
    assert parameter_checksum(sac.dynamics) != before


def test_imagined_rollout_generates_model_transitions(sac, ota_graph, spec):
    _fill_replay(sac, ota_graph, spec)
    sac.update_dynamics(batch_size=8)
    start = [0.0] * sac.state_dim
    count = sac.generate_imagined_transitions(start, horizon=3)
    assert 1 <= count <= 3
    assert sac.replay.counts()["model"] == count


def test_mixed_batch_sampling():
    buffer = SizingReplayBuffer()
    for _i in range(10):
        buffer.add(SizingTransition([0.0], [0.0], 0.0, [0.0], False, source="real"))
    for _i in range(10):
        buffer.add(SizingTransition([1.0], [0.0], 0.0, [1.0], False, source="model"))
    batch = buffer.sample_mixed(10, make_rng(0), real_fraction=0.8)
    sources = [t.source for t in batch]
    assert len(batch) == 10
    assert sources.count("real") == 8
    assert sources.count("model") == 2


def test_update_skips_gracefully_when_empty(sac):
    assert sac.update() == {"skipped": 1.0}
    assert sac.update_dynamics() == {"skipped": 1.0}
