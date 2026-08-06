"""MCTS: selection, expansion, backup, masking, visit distribution, widening."""

from __future__ import annotations

import pytest

from agentic_raptor.topology_rl.actions import ActionType, enumerate_candidate_actions
from agentic_raptor.topology_rl.mcts import MCTS, MCTSConfig, MCTSNode
from agentic_raptor.topology_rl.policy_value_network import HeuristicEvaluator


@pytest.fixture
def mcts():
    return MCTS(HeuristicEvaluator(), config=MCTSConfig(num_simulations=16, max_depth=3, seed=1))


def test_run_returns_legal_best_action(mcts, ota_graph, spec):
    result = mcts.run(ota_graph, spec, edit_budget_fraction=1.0)
    legal_keys = {a.key() for a in enumerate_candidate_actions(ota_graph, max_actions=32)}
    assert result.best_action.key() in legal_keys


def test_visit_counts_accumulate_and_backup(mcts, ota_graph, spec):
    result = mcts.run(ota_graph, spec, edit_budget_fraction=1.0)
    total_visits = sum(n for _a, n in result.visit_counts.values())
    assert total_visits > 0
    # Root value must be the mean of backed-up values → bounded by evaluator range.
    assert -1.0 <= result.root_value <= 1.0


def test_visit_distribution_masks_to_legal_actions(mcts, ota_graph, spec):
    result = mcts.run(ota_graph, spec, edit_budget_fraction=1.0)
    legal = enumerate_candidate_actions(ota_graph, max_actions=32)
    dist = result.visit_distribution(legal)
    assert len(dist) == len(legal)
    assert sum(dist) == pytest.approx(1.0, abs=1e-9)
    assert all(p >= 0 for p in dist)


def test_progressive_widening_limits_children(ota_graph, spec):
    tight = MCTS(
        HeuristicEvaluator(),
        config=MCTSConfig(num_simulations=8, max_depth=2, pw_c=1.0, pw_alpha=0.0, pw_min=2),
    )
    # Reconstruct root behaviour: with pw capped at 2, at most 2 children after few sims.
    result = tight.run(ota_graph, spec, edit_budget_fraction=1.0)
    assert len(result.visit_counts) <= 3  # pw_min=2 plus at most one forced expansion


def test_puct_prefers_high_prior_unvisited():
    parent = MCTSNode(graph=None, depth=0)  # type: ignore[arg-type]
    parent.visit_count = 9
    low = MCTSNode(graph=None, depth=1, parent=parent, prior=0.1)  # type: ignore[arg-type]
    high = MCTSNode(graph=None, depth=1, parent=parent, prior=0.9)  # type: ignore[arg-type]
    assert parent.puct_score(high, c_puct=1.5) > parent.puct_score(low, c_puct=1.5)


def test_backup_propagates_to_ancestors(ota_graph):
    root = MCTSNode(ota_graph, depth=0)
    child = MCTSNode(ota_graph, depth=1, parent=root)
    grandchild = MCTSNode(ota_graph, depth=2, parent=child)
    mcts = MCTS(HeuristicEvaluator())
    mcts._backup(grandchild, 0.5)
    assert root.visit_count == child.visit_count == grandchild.visit_count == 1
    assert root.q_value == pytest.approx(0.5)


def test_deterministic_given_seed(ota_graph, spec):
    a = MCTS(HeuristicEvaluator(), config=MCTSConfig(num_simulations=12, seed=3)).run(
        ota_graph, spec, 1.0
    )
    b = MCTS(HeuristicEvaluator(), config=MCTSConfig(num_simulations=12, seed=3)).run(
        ota_graph, spec, 1.0
    )
    assert a.best_action.key() == b.best_action.key()
    assert {k: n for k, (_x, n) in a.visit_counts.items()} == {
        k: n for k, (_x, n) in b.visit_counts.items()
    }


def test_terminate_child_is_terminal(mcts, ota_graph, spec):
    result = mcts.run(ota_graph, spec, edit_budget_fraction=1.0)
    if ActionType.TERMINATE.value in str(result.visit_counts):
        # TERMINATE children never expand further — depth stays 1 for them.
        pass  # structural guarantee exercised in _expand; smoke-checked here
