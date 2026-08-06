"""MCTS planning over topology edits, guided by a policy–value evaluator.

Implements: selection (PUCT), expansion with policy priors, value-network leaf
evaluation, backup, root visit-count distribution, legal-action masking
(children exist only for legal actions), progressive widening, max depth, and
a configurable simulation budget.

PUCT score:  Q(s,a) + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))

MCTS here is the *planning* component; learning lives in the policy–value
network trained by ``trainer.py``. MCTS alone is not reinforcement learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.topology_rl.actions import (
    ActionType,
    TopologyAction,
    apply_action,
    enumerate_candidate_actions,
)
from agentic_raptor.topology_rl.policy_value_network import PolicyValueEvaluator
from agentic_raptor.topology_validation.validator import TopologyValidator
from agentic_raptor.utils.exceptions import ActionError


@dataclass
class MCTSConfig:
    num_simulations: int = 24
    c_puct: float = 1.5
    max_depth: int = 4
    max_actions: int = 32
    # Progressive widening: a node may expand a new child only while
    # len(children) < max(pw_min, pw_c * N(s)^pw_alpha).
    pw_c: float = 2.0
    pw_alpha: float = 0.5
    pw_min: int = 3
    seed: int = 0


class MCTSNode:
    __slots__ = (
        "graph", "depth", "parent", "action", "prior", "visit_count",
        "value_sum", "children", "untried", "is_terminal_state", "primed",
    )

    def __init__(
        self,
        graph: CircuitGraph,
        depth: int,
        parent: MCTSNode | None = None,
        action: TopologyAction | None = None,
        prior: float = 1.0,
    ) -> None:
        self.graph = graph
        self.depth = depth
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[str, MCTSNode] = {}
        #: legal (action, prior) pairs not yet expanded, best prior first
        self.untried: list[tuple[TopologyAction, float]] = []
        self.is_terminal_state = False
        self.primed = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def puct_score(self, child: MCTSNode, c_puct: float) -> float:
        exploration = c_puct * child.prior * math.sqrt(max(self.visit_count, 1)) / (1 + child.visit_count)
        return child.q_value + exploration


@dataclass
class MCTSResult:
    best_action: TopologyAction
    root_value: float
    #: action key → (action, visit_count) at the root
    visit_counts: dict[str, tuple[TopologyAction, int]] = field(default_factory=dict)

    def visit_distribution(self, legal_actions: list[TopologyAction]) -> list[float]:
        """π_MCTS over the given legal-action list (0 for never-visited)."""
        total = sum(count for _a, count in self.visit_counts.values()) or 1
        return [self.visit_counts.get(a.key(), (a, 0))[1] / total for a in legal_actions]

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_action": self.best_action.to_dict(),
            "root_value": self.root_value,
            "visit_counts": {k: (a.to_dict(), n) for k, (a, n) in self.visit_counts.items()},
        }


class MCTS:
    """Planner. Evaluator injection keeps this testable and network-agnostic."""

    def __init__(
        self,
        evaluator: PolicyValueEvaluator,
        validator: TopologyValidator | None = None,
        config: MCTSConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.validator = validator or TopologyValidator()
        self.config = config or MCTSConfig()

    def run(
        self,
        root_graph: CircuitGraph,
        spec: DesignSpecifications,
        edit_budget_fraction: float,
    ) -> MCTSResult:
        cfg = self.config
        root = MCTSNode(root_graph.copy(), depth=0)
        root_value = self._evaluate(root, spec, edit_budget_fraction)

        for _sim in range(cfg.num_simulations):
            node = root
            # --- selection (PUCT) with progressive widening ---
            while not node.is_terminal_state and node.depth < cfg.max_depth:
                allowed = max(cfg.pw_min, int(cfg.pw_c * (max(node.visit_count, 1) ** cfg.pw_alpha)))
                if node.untried and len(node.children) < allowed:
                    node = self._expand(node)
                    break
                if not node.children:
                    break
                node = max(node.children.values(), key=lambda c: node.puct_score(c, cfg.c_puct))
            # --- leaf evaluation via the value network ---
            if node.is_terminal_state or node.depth >= cfg.max_depth:
                value = self._evaluate(node, spec, self._budget_at(node, edit_budget_fraction))
            else:
                value = self._evaluate(node, spec, self._budget_at(node, edit_budget_fraction))
            # --- backup ---
            self._backup(node, value)

        if not root.children:
            return MCTSResult(TopologyAction(ActionType.TERMINATE), root_value)
        best_child = max(root.children.values(), key=lambda c: c.visit_count)
        visit_counts = {
            key: (child.action, child.visit_count)
            for key, child in root.children.items()
            if child.action is not None
        }
        assert best_child.action is not None
        return MCTSResult(best_child.action, root.q_value, visit_counts)

    # -- internals ----------------------------------------------------------
    def _budget_at(self, node: MCTSNode, root_budget: float) -> float:
        depth_cost = node.depth / max(self.config.max_depth, 1)
        return max(0.0, root_budget * (1.0 - depth_cost))

    def _evaluate(self, node: MCTSNode, spec: DesignSpecifications, budget: float) -> float:
        """Evaluate a node with the policy–value evaluator; prime priors once."""
        validation = self.validator.validate(node.graph)
        legal = enumerate_candidate_actions(node.graph, max_actions=self.config.max_actions)
        priors, value = self.evaluator.evaluate(
            node.graph, spec, budget, validation.feature_vector(), legal
        )
        if not node.primed:
            pairs = sorted(zip(legal, priors, strict=True), key=lambda p: -p[1])
            node.untried = [(a, max(p, 1e-4)) for a, p in pairs]
            node.primed = True
        return value

    def _expand(self, node: MCTSNode) -> MCTSNode:
        action, prior = node.untried.pop(0)
        if action.action_type == ActionType.TERMINATE:
            child = MCTSNode(node.graph, node.depth + 1, parent=node, action=action, prior=prior)
            child.is_terminal_state = True
        else:
            try:
                new_graph, _record = apply_action(node.graph, action)
                child = MCTSNode(new_graph, node.depth + 1, parent=node, action=action, prior=prior)
            except ActionError:
                # Masked-out in practice; keep the tree consistent with a
                # terminal child that reuses the parent state.
                child = MCTSNode(node.graph, node.depth + 1, parent=node, action=action, prior=prior)
                child.is_terminal_state = True
        node.children[action.key()] = child
        return child

    def _backup(self, node: MCTSNode | None, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
