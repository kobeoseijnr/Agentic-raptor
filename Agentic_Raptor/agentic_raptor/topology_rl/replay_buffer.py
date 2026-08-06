"""Replay storage for topology trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

from agentic_raptor.topology_rl.trajectory import TopologyTrajectory, TrajectoryStep


class TopologyReplayBuffer:
    """FIFO buffer of credit-assigned trajectory steps."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._steps: list[TrajectoryStep] = []

    def add_trajectory(self, trajectory: TopologyTrajectory) -> int:
        """Add all steps that already carry a return; returns number added."""
        added = 0
        for step in trajectory.steps:
            if step.discounted_return is None:
                continue
            self._steps.append(step)
            added += 1
        overflow = len(self._steps) - self.capacity
        if overflow > 0:
            self._steps = self._steps[overflow:]
        return added

    def sample(self, batch_size: int, rng: Random) -> list[TrajectoryStep]:
        if not self._steps:
            return []
        k = min(batch_size, len(self._steps))
        return rng.sample(self._steps, k)

    def __len__(self) -> int:
        return len(self._steps)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for step in self._steps:
                f.write(json.dumps(step.to_dict(), ensure_ascii=False) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path, capacity: int = 10_000) -> TopologyReplayBuffer:
        buffer = cls(capacity=capacity)
        p = Path(path)
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        buffer._steps.append(TrajectoryStep.from_dict(json.loads(line)))
        return buffer
