"""Replay buffer for sizing transitions with real/model source labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Literal

TransitionSource = Literal["real", "model"]


@dataclass
class SizingTransition:
    state: list[float]
    action: list[float]
    reward: float
    next_state: list[float]
    done: bool
    source: TransitionSource = "real"
    metadata: dict[str, Any] = field(default_factory=dict)


class SizingReplayBuffer:
    """FIFO buffer keeping real and model transitions separately addressable."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._real: list[SizingTransition] = []
        self._model: list[SizingTransition] = []

    def add(self, transition: SizingTransition) -> None:
        store = self._real if transition.source == "real" else self._model
        store.append(transition)
        overflow = len(store) - self.capacity
        if overflow > 0:
            del store[:overflow]

    def counts(self) -> dict[str, int]:
        return {"real": len(self._real), "model": len(self._model)}

    def __len__(self) -> int:
        return len(self._real) + len(self._model)

    def sample_mixed(
        self, batch_size: int, rng: Random, real_fraction: float = 0.8
    ) -> list[SizingTransition]:
        """Sample a real/model mixture; falls back to whatever is available."""
        if not self._real and not self._model:
            return []
        n_real = round(batch_size * real_fraction)
        n_model = batch_size - n_real
        batch: list[SizingTransition] = []
        if self._real:
            k = min(n_real + max(0, n_model - len(self._model)), len(self._real))
            batch += [self._real[i] for i in sorted(rng.sample(range(len(self._real)), min(k, len(self._real))))]
        if self._model:
            k = min(n_model + max(0, n_real - len(self._real)), len(self._model))
            batch += [self._model[i] for i in sorted(rng.sample(range(len(self._model)), min(k, len(self._model))))]
        rng.shuffle(batch)
        return batch[:batch_size]

    def as_batch(self, transitions: list[SizingTransition]) -> dict[str, Any]:
        return {
            "state": [t.state for t in transitions],
            "action": [t.action for t in transitions],
            "reward": [t.reward for t in transitions],
            "next_state": [t.next_state for t in transitions],
            "done": [t.done for t in transitions],
            "source": [t.source for t in transitions],
        }
