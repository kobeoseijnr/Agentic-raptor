"""Topology trajectories: the data MCTS produces and the trainer consumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryStep:
    """One topology decision point."""

    graph_state: dict[str, Any]              # serialized CircuitGraph
    specification_state: dict[str, Any]      # serialized DesignSpecifications
    state_features: list[float]              # encoded network input
    legal_action_mask: list[bool]
    mcts_visit_distribution: list[float]     # π_MCTS over legal-action slots
    selected_action: dict[str, Any]          # serialized TopologyAction
    final_post_sizing_reward: float | None = None
    discounted_return: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryStep:
        return cls(
            graph_state=dict(data["graph_state"]),
            specification_state=dict(data["specification_state"]),
            state_features=[float(v) for v in data["state_features"]],
            legal_action_mask=[bool(v) for v in data["legal_action_mask"]],
            mcts_visit_distribution=[float(v) for v in data["mcts_visit_distribution"]],
            selected_action=dict(data["selected_action"]),
            final_post_sizing_reward=data.get("final_post_sizing_reward"),
            discounted_return=data.get("discounted_return"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class TopologyTrajectory:
    """Ordered decision sequence for one candidate topology."""

    trajectory_id: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_post_sizing_reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "steps": [s.to_dict() for s in self.steps],
            "final_post_sizing_reward": self.final_post_sizing_reward,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopologyTrajectory:
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            steps=[TrajectoryStep.from_dict(s) for s in (data.get("steps") or [])],
            final_post_sizing_reward=data.get("final_post_sizing_reward"),
            metadata=dict(data.get("metadata") or {}),
        )
