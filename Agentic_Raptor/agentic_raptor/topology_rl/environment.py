"""Gymnasium-compatible topology-edit environment.

API matches the Gymnasium convention — ``reset() -> (obs, info)`` and
``step(a) -> (obs, reward, terminated, truncated, info)`` — plus the
domain-required ``legal_actions()``, ``is_terminal()``, ``render()``.
The action argument is an index into the current ``legal_actions()`` list
(index 0 is always TERMINATE). Illegal indices yield the configured
invalid-action penalty and leave the topology unchanged.

The environment only edits discrete topology structure; continuous device
sizes are never touched here (that is the sizing level).
"""

from __future__ import annotations

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
from agentic_raptor.topology_validation.validator import TopologyValidator, ValidationResult
from agentic_raptor.utils.exceptions import ActionError


@dataclass
class TopologyEnvConfig:
    max_edits: int = 8
    max_actions: int = 32
    invalid_action_penalty: float = -0.5
    step_penalty: float = -0.02
    validity_reward: float = 0.25
    warning_penalty: float = -0.02


@dataclass
class TopologyEnvState:
    graph: CircuitGraph
    validation: ValidationResult
    edits_used: int = 0
    terminated: bool = False
    edit_records: list[dict[str, Any]] = field(default_factory=list)


class TopologyEditEnv:
    """Topology refinement as a sequential decision process."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        initial_graph: CircuitGraph,
        specifications: DesignSpecifications,
        validator: TopologyValidator | None = None,
        config: TopologyEnvConfig | None = None,
    ) -> None:
        self._initial_graph = initial_graph.copy()
        self.spec = specifications
        self.validator = validator or TopologyValidator()
        self.config = config or TopologyEnvConfig()
        self.state: TopologyEnvState | None = None

    # -- gymnasium API ------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        graph = self._initial_graph.copy()
        validation = self.validator.validate(graph)
        self.state = TopologyEnvState(graph=graph, validation=validation)
        return self._observation(), {"validation": validation.to_dict()}

    def step(self, action_index: int) -> tuple[dict, float, bool, bool, dict]:
        state = self._require_state()
        if state.terminated:
            raise ActionError("episode already terminated; call reset()")
        legal = self.legal_actions()
        info: dict[str, Any] = {}

        if not 0 <= action_index < len(legal):
            reward = self.config.invalid_action_penalty
            info["invalid_action"] = True
            info["reason"] = f"action index {action_index} outside legal range 0..{len(legal) - 1}"
            state.edits_used += 1
            truncated = state.edits_used >= self.config.max_edits
            state.terminated = truncated
            return self._observation(), reward, False, truncated, info

        action = legal[action_index]
        info["action"] = action.to_dict()

        if action.action_type == ActionType.TERMINATE:
            state.terminated = True
            reward = self.config.validity_reward if state.validation.is_valid else -self.config.validity_reward
            return self._observation(), reward, True, False, info

        was_valid = state.validation.is_valid
        try:
            new_graph, record = apply_action(state.graph, action)
        except ActionError as exc:
            reward = self.config.invalid_action_penalty
            info["invalid_action"] = True
            info["reason"] = str(exc)
            state.edits_used += 1
            truncated = state.edits_used >= self.config.max_edits
            state.terminated = truncated
            return self._observation(), reward, False, truncated, info

        validation = self.validator.validate(new_graph)
        state.graph = new_graph
        state.validation = validation
        state.edits_used += 1
        state.edit_records.append(record.to_dict())

        reward = self.config.step_penalty
        reward += self.config.warning_penalty * len(validation.warnings)
        if validation.is_valid and not was_valid:
            reward += self.config.validity_reward
        elif not validation.is_valid and was_valid:
            reward -= self.config.validity_reward

        truncated = state.edits_used >= self.config.max_edits
        state.terminated = truncated
        info["validation"] = validation.to_dict()
        return self._observation(), reward, False, truncated, info

    # -- domain API ---------------------------------------------------------
    def legal_actions(self) -> list[TopologyAction]:
        state = self._require_state()
        return enumerate_candidate_actions(state.graph, max_actions=self.config.max_actions)

    def legal_action_mask(self) -> list[bool]:
        """Fixed-width mask over max_actions slots."""
        n = len(self.legal_actions())
        return [i < n for i in range(self.config.max_actions)]

    def is_terminal(self) -> bool:
        state = self._require_state()
        return state.terminated

    def render(self) -> str:
        state = self._require_state()
        text = (
            f"TopologyEditEnv(graph={state.graph.graph_id}, nodes={len(state.graph.nodes)}, "
            f"edges={len(state.graph.edges)}, valid={state.validation.is_valid}, "
            f"edits={state.edits_used}/{self.config.max_edits})"
        )
        return text

    def edit_budget_fraction(self) -> float:
        state = self._require_state()
        return max(0.0, 1.0 - state.edits_used / self.config.max_edits)

    # -- internals ----------------------------------------------------------
    def _require_state(self) -> TopologyEnvState:
        if self.state is None:
            raise ActionError("environment not reset; call reset() first")
        return self.state

    def _observation(self) -> dict[str, Any]:
        from agentic_raptor.topology_rl.policy_value_network import encode_state_features

        state = self._require_state()
        return {
            "features": encode_state_features(
                state.graph,
                self.spec,
                self.edit_budget_fraction(),
                state.validation.feature_vector(),
            ),
            "legal_action_mask": self.legal_action_mask(),
            "graph": state.graph,
            "validation": state.validation,
        }
