"""Budget + progress tracking for coordinator decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_raptor.core.budgets import BudgetState


@dataclass
class BudgetManager:
    budgets: BudgetState
    stagnation_patience: int = 3
    max_repeated_failures: int = 3

    best_reward: float | None = None
    repeated_failure_count: int = 0
    _no_improvement_count: int = 0
    _rewards: list[float] = field(default_factory=list)

    def record_reward(self, reward: float) -> bool:
        """Track episode reward; returns True when it improved the best."""
        self._rewards.append(reward)
        if self.best_reward is None or reward > self.best_reward:
            self.best_reward = reward
            self._no_improvement_count = 0
            return True
        self._no_improvement_count += 1
        return False

    def record_failure(self) -> None:
        self.repeated_failure_count += 1

    def reset_failures(self) -> None:
        self.repeated_failure_count = 0

    @property
    def stagnated(self) -> bool:
        return self._no_improvement_count >= self.stagnation_patience

    @property
    def failing_repeatedly(self) -> bool:
        return self.repeated_failure_count >= self.max_repeated_failures

    def snapshot(self) -> dict[str, float | int | bool | None]:
        return {
            **self.budgets.snapshot(),
            "best_reward": self.best_reward,
            "repeated_failures": self.repeated_failure_count,
            "stagnated": self.stagnated,
        }
