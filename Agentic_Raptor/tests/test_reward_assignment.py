"""Reward computation and cross-level credit assignment."""

from __future__ import annotations

import pytest

from agentic_raptor.core.budgets import BudgetState
from agentic_raptor.core.candidate import CircuitCandidate
from agentic_raptor.core.rewards import RewardComponents, feasibility_score
from agentic_raptor.core.types import GenerationSource
from agentic_raptor.learning.cross_level_credit import CrossLevelCreditAssigner
from agentic_raptor.learning.reward_assignment import (
    RewardWeights,
    assemble_components,
    compute_final_reward,
)
from agentic_raptor.spice.pvt import run_pvt, standard_corners
from agentic_raptor.spice.simulator_adapter import MockSpiceSimulator
from agentic_raptor.utils.exceptions import ConfigurationError


def _budgets() -> BudgetState:
    return BudgetState(
        max_topology_edits=4, max_sizing_steps=6, max_spice_calls=10, max_runtime_s=60.0
    )


def test_feasibility_score():
    assert feasibility_score({}) == 0.0
    assert feasibility_score({"a": 0.1, "b": -0.2}) == pytest.approx(0.5)
    assert feasibility_score({"a": 0.0}) == pytest.approx(1.0)


def test_weights_reject_unknown_keys():
    with pytest.raises(ConfigurationError):
        RewardWeights.from_dict({"bogus_weight": 1.0})


def test_final_reward_formula_matches_configuration():
    weights = RewardWeights(
        feasibility_weight=1.0,
        fom_weight=0.5,
        pvt_weight=0.3,
        spice_cost_weight=0.1,
        runtime_weight=0.05,
        invalidity_weight=0.5,
        validity_weight=0.0,
        sizing_progress_weight=0.0,
    )
    components = RewardComponents(
        spice_feasibility=1.0,
        normalized_fom=0.4,
        pvt_robustness=0.5,
        normalized_spice_calls=0.5,
        normalized_runtime=0.2,
        invalidity_penalty=1.0,
    )
    expected = 1.0 * 1.0 + 0.5 * 0.4 + 0.3 * 0.5 - 0.1 * 0.5 - 0.05 * 0.2 - 0.5 * 1.0
    assert compute_final_reward(components, weights) == pytest.approx(expected)


def test_assemble_components_with_mock_simulation(ota_graph, spec):
    candidate = CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)
    simulator = MockSpiceSimulator(seed=1)
    sim = simulator.simulate(candidate, ["op", "ac"], 5.0)
    budgets = _budgets()
    budgets.consume("spice_calls", 3)
    pvt = run_pvt(simulator, candidate, ["op"], standard_corners()[:2])
    components = assemble_components(
        validation=None,
        sizing_progress=0.2,
        sim=sim,
        pvt=pvt,
        budgets=budgets,
        invalid_action_count=1,
        load_capacitance_f=spec.load_capacitance_f,
    )
    assert 0.0 <= components.spice_feasibility <= 1.0
    assert 0.0 <= components.normalized_fom < 1.0
    assert components.normalized_spice_calls == pytest.approx(0.3)
    assert components.invalidity_penalty == pytest.approx(1.0)
    assert components.pvt_robustness == pytest.approx(pvt.pvt_score)


def test_credit_assigner_validates_arguments():
    with pytest.raises(ValueError):
        CrossLevelCreditAssigner(gamma=0.0)
    with pytest.raises(ValueError):
        CrossLevelCreditAssigner(shaping_blend=1.0)


def test_mock_simulator_deterministic(ota_graph, spec):
    candidate = CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)
    sim = MockSpiceSimulator(seed=5)
    a = sim.simulate(candidate, ["op", "ac"], 5.0)
    b = sim.simulate(candidate, ["op", "ac"], 5.0)
    assert a.metrics == b.metrics
    assert a.constraint_margins == b.constraint_margins


def test_mock_simulator_sensitive_to_sizing(ota_graph, spec):
    candidate = CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)
    sim = MockSpiceSimulator(seed=5)
    base = sim.simulate(candidate, ["op"], 5.0)
    candidate.sizing_state = {"ib1": {"current_a": 4e-4}}
    changed = sim.simulate(candidate, ["op"], 5.0)
    assert changed.metrics != base.metrics
