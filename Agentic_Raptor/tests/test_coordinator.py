"""Coordinator: state machine transitions and rule-based decisions."""

from __future__ import annotations

import pytest

from agentic_raptor.coordinator.budget_manager import BudgetManager
from agentic_raptor.coordinator.decision_policy import DecisionContext, RuleBasedDecisionPolicy
from agentic_raptor.coordinator.state_machine import (
    CoordinatorState,
    Decision,
    StateMachine,
)
from agentic_raptor.core.budgets import BudgetState
from agentic_raptor.utils.exceptions import CoordinatorError


def test_legal_transition_chain():
    sm = StateMachine()
    for state in (
        CoordinatorState.PARSE_SPECIFICATIONS,
        CoordinatorState.RETRIEVE,
        CoordinatorState.GENERATE,
        CoordinatorState.VALIDATE,
        CoordinatorState.EDIT_TOPOLOGY,
        CoordinatorState.SIZE,
        CoordinatorState.SIMULATE,
        CoordinatorState.DIAGNOSE,
        CoordinatorState.UPDATE_MEMORY,
        CoordinatorState.TERMINATE,
    ):
        sm.transition(state)
    assert sm.finished
    assert len(sm.history) == 10


def test_illegal_transition_rejected():
    sm = StateMachine()
    with pytest.raises(CoordinatorError):
        sm.transition(CoordinatorState.SIMULATE)


def test_terminal_states_have_no_exits():
    sm = StateMachine(state=CoordinatorState.TERMINATE)
    with pytest.raises(CoordinatorError):
        sm.transition(CoordinatorState.INITIALIZE)


def _ctx(**overrides) -> DecisionContext:
    base = DecisionContext(
        specifications_parsed=True,
        edit_budget_remaining=3,
        sizing_budget_remaining=3,
        spice_budget_remaining=3,
        generation_budget_remaining=2,
        retrieval_budget_remaining=2,
        topology_edits_done=1,  # past the mandatory initial refinement pass
    )
    for name, value in overrides.items():
        setattr(base, name, value)
    return base


def test_policy_mandatory_initial_refinement():
    """A fresh valid candidate gets one MCTS pass before sizing."""
    decision, reason = RuleBasedDecisionPolicy().decide(
        _ctx(has_candidate=True, candidate_valid=True, topology_edits_done=0)
    )
    assert decision == Decision.REPAIR_OR_EDIT_TOPOLOGY
    assert "topology RL precedes" in reason


def test_policy_stops_on_budget_exhaustion():
    decision, reason = RuleBasedDecisionPolicy().decide(_ctx(budgets_exhausted=True))
    assert decision == Decision.STOP_BUDGET_EXHAUSTED
    assert "budget" in reason


def test_policy_retrieves_then_generates():
    policy = RuleBasedDecisionPolicy()
    d1, _ = policy.decide(_ctx(has_candidate=False, retrieved_count=0))
    assert d1 == Decision.RETRIEVE_MORE
    d2, _ = policy.decide(_ctx(has_candidate=False, retrieved_count=3))
    assert d2 == Decision.GENERATE_NEW_TOPOLOGY


def test_policy_repairs_invalid_candidate():
    decision, _ = RuleBasedDecisionPolicy().decide(
        _ctx(has_candidate=True, candidate_valid=False, validation_error_count=2)
    )
    assert decision == Decision.REPAIR_OR_EDIT_TOPOLOGY


def test_policy_sizes_then_simulates():
    policy = RuleBasedDecisionPolicy()
    d1, _ = policy.decide(_ctx(has_candidate=True, candidate_valid=True, sized=False))
    assert d1 == Decision.CONTINUE_SIZING
    d2, _ = policy.decide(_ctx(has_candidate=True, candidate_valid=True, sized=True))
    assert d2 == Decision.RUN_SPICE


def test_policy_runs_pvt_then_accepts():
    policy = RuleBasedDecisionPolicy()
    d1, _ = policy.decide(
        _ctx(has_candidate=True, candidate_valid=True, sized=True, simulated=True, all_constraints_met=True)
    )
    assert d1 == Decision.RUN_PVT
    d2, _ = policy.decide(
        _ctx(
            has_candidate=True,
            candidate_valid=True,
            sized=True,
            simulated=True,
            all_constraints_met=True,
            pvt_done=True,
        )
    )
    assert d2 == Decision.ACCEPT_CANDIDATE


def test_policy_refines_near_feasible_sizing():
    decision, reason = RuleBasedDecisionPolicy().decide(
        _ctx(
            has_candidate=True,
            candidate_valid=True,
            sized=True,
            simulated=True,
            all_constraints_met=False,
            worst_margin=-0.1,
        )
    )
    assert decision == Decision.CONTINUE_SIZING
    assert "near-feasible" in reason


def test_policy_edits_on_structural_gap():
    decision, _ = RuleBasedDecisionPolicy().decide(
        _ctx(
            has_candidate=True,
            candidate_valid=True,
            sized=True,
            simulated=True,
            all_constraints_met=False,
            worst_margin=-0.8,
        )
    )
    assert decision == Decision.REPAIR_OR_EDIT_TOPOLOGY


def test_policy_stops_after_repeated_failures():
    decision, _ = RuleBasedDecisionPolicy().decide(_ctx(repeated_failures=3))
    assert decision == Decision.STOP_BUDGET_EXHAUSTED


def test_budget_manager_tracks_improvement_and_stagnation():
    manager = BudgetManager(
        BudgetState(max_topology_edits=2, max_sizing_steps=2, max_spice_calls=2, max_runtime_s=60.0),
        stagnation_patience=2,
    )
    assert manager.record_reward(0.1)
    assert not manager.record_reward(0.05)
    assert not manager.stagnated
    assert not manager.record_reward(0.05)
    assert manager.stagnated
    assert manager.record_reward(0.5)  # improvement resets stagnation
    assert not manager.stagnated
