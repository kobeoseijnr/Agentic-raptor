"""Stage 2 learning plumbing: cache keys, PVT aggregation, query policy, ensemble."""

from __future__ import annotations

import pytest

from agentic_raptor.core.candidate import CircuitCandidate
from agentic_raptor.core.types import GenerationSource
from agentic_raptor.sizing.graph_conditioned_mb_sac import GraphConditionedMBSAC, MBSACConfig
from agentic_raptor.sizing.parameter_space import SizingParameterSpace
from agentic_raptor.sizing.replay_buffer import SizingTransition
from agentic_raptor.sizing.spice_query_policy import RealSpiceQueryPolicy
from agentic_raptor.spice.cache import SimulationCache, make_cache_key
from agentic_raptor.spice.interface import SimulationResult
from agentic_raptor.spice.pvt import run_pvt, standard_corners
from agentic_raptor.spice.simulator_adapter import MockSpiceSimulator


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------
def test_cache_key_sensitive_to_every_component():
    base = dict(
        topology_hash="abc",
        sizing_state={"m1": {"width_m": 1e-6}},
        analyses=["op", "ac"],
        corner="typical",
        simulator_config={"version": "45.2", "model_library": "generic", "temperature_c": 27.0},
    )
    key = make_cache_key(**base)
    variants = [
        {**base, "topology_hash": "def"},
        {**base, "sizing_state": {"m1": {"width_m": 2e-6}}},
        {**base, "analyses": ["op"]},
        {**base, "corner": "ss"},
        {**base, "simulator_config": {**base["simulator_config"], "version": "44.0"}},
        {**base, "simulator_config": {**base["simulator_config"], "model_library": "tsmc180"}},
        {**base, "simulator_config": {**base["simulator_config"], "temperature_c": 85.0}},
    ]
    for variant in variants:
        assert make_cache_key(**variant) != key


def test_cache_disabled_is_passthrough(ota_graph, spec):
    cache = SimulationCache(enabled=False)
    candidate = CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)
    calls = []

    def run():
        calls.append(1)
        return SimulationResult(success=True, metrics={"gain_db": 1.0})

    for _ in range(2):
        _result, was_cached = cache.get_or_run(candidate, ["op"], "typical", {}, run)
        assert not was_cached
    assert len(calls) == 2


def test_failures_are_not_cached(ota_graph, spec):
    cache = SimulationCache()
    candidate = CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)
    outcomes = [
        SimulationResult(success=False, error_type="timeout"),
        SimulationResult(success=True, metrics={"gain_db": 2.0}),
    ]
    result1, _ = cache.get_or_run(candidate, ["op"], "typical", {}, lambda: outcomes.pop(0))
    assert not result1.success
    result2, was_cached = cache.get_or_run(candidate, ["op"], "typical", {}, lambda: outcomes.pop(0))
    assert result2.success and not was_cached  # failure was retried, not replayed


# ---------------------------------------------------------------------------
# PVT aggregation
# ---------------------------------------------------------------------------
def test_pvt_aggregation_fields(ota_graph, spec):
    candidate = CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)
    result = run_pvt(MockSpiceSimulator(seed=3), candidate, ["op", "ac"], standard_corners()[:3])
    assert 0.0 <= result.pass_rate <= 1.0
    assert result.pvt_score == result.pass_rate
    assert result.worst_corner in {c.name for c in standard_corners()[:3]}
    assert isinstance(result.mean_margin, float)
    assert set(result.failed_corners) <= {c.name for c in standard_corners()[:3]}
    # failed + passed == all corners
    assert len(result.failed_corners) == round((1 - result.pass_rate) * 3)


# ---------------------------------------------------------------------------
# Real-SPICE query policy
# ---------------------------------------------------------------------------
def test_query_policy_rules():
    policy = RealSpiceQueryPolicy(warmup_transitions=2, uncertainty_threshold=0.5, query_interval=3)
    d = policy.decide(0, 0, 0.0, spice_budget_remaining=5)
    assert d.use_real and "warm-up" in d.reason
    d = policy.decide(5, 0, 0.1, spice_budget_remaining=0)
    assert not d.use_real and "exhausted" in d.reason
    d = policy.decide(5, 0, 0.9, spice_budget_remaining=5)
    assert d.use_real and "uncertainty" in d.reason
    d = policy.decide(5, 3, 0.1, spice_budget_remaining=5)
    assert d.use_real and "re-anchor" in d.reason
    d = policy.decide(5, 1, 0.1, spice_budget_remaining=5)
    assert not d.use_real and "model prediction" in d.reason
    d = policy.decide(5, 0, 0.0, spice_budget_remaining=5, force_real=True)
    assert d.use_real and "verification" in d.reason


# ---------------------------------------------------------------------------
# Dynamics ensemble uncertainty
# ---------------------------------------------------------------------------
@pytest.fixture
def sac(ota_graph):
    space = SizingParameterSpace.from_graph(ota_graph)
    return GraphConditionedMBSAC(space, MBSACConfig(hidden_dim=32, lr=1e-2, seed=5, dynamics_ensemble_size=2))


def test_ensemble_uncertainty_nonnegative_and_single_disables(ota_graph, sac):
    state = [0.0] * sac.state_dim
    action = [0.0] * sac.action_dim
    assert sac.dynamics_uncertainty(state, action) >= 0.0
    space = SizingParameterSpace.from_graph(ota_graph)
    single = GraphConditionedMBSAC(space, MBSACConfig(hidden_dim=16, seed=5, dynamics_ensemble_size=1))
    assert single.dynamics_uncertainty(state, action) == 0.0


def test_ensemble_members_all_train(sac):
    from agentic_raptor.topology_rl.trainer import parameter_checksum

    state = [0.0] * sac.state_dim
    for i in range(10):
        sac.add_transition(
            SizingTransition(
                state=state,
                action=[0.1 * (i % 3 - 1)] * sac.action_dim,
                reward=0.05 * (i % 4 - 2),
                next_state=[s + 0.01 for s in state],
                done=False,
            )
        )
    checksums = [parameter_checksum(m) for m in sac.dynamics_models]
    report = sac.update_dynamics(batch_size=8)
    assert report.get("ensemble_size") == 2.0
    for model, before in zip(sac.dynamics_models, checksums, strict=True):
        assert parameter_checksum(model) != before, "every ensemble member must train"
