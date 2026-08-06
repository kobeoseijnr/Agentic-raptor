"""Episode budgets shared between coordinator, environment, and sizing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agentic_raptor.utils.exceptions import BudgetExhaustedError


@dataclass
class BudgetState:
    """Mutable per-episode budget counters."""

    max_topology_edits: int
    max_sizing_steps: int
    max_spice_calls: int
    max_runtime_s: float
    max_generations: int = 5
    max_retrievals: int = 10

    used_topology_edits: int = 0
    used_sizing_steps: int = 0
    used_spice_calls: int = 0
    used_generations: int = 0
    used_retrievals: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    # -- consumption --------------------------------------------------------
    def consume(self, kind: str, amount: int = 1) -> None:
        used_attr, max_attr = f"used_{kind}", f"max_{kind}"
        used, cap = getattr(self, used_attr), getattr(self, max_attr)
        if used + amount > cap:
            raise BudgetExhaustedError(f"budget {kind!r} exhausted ({used}/{cap})")
        setattr(self, used_attr, used + amount)

    # -- queries ------------------------------------------------------------
    def remaining(self, kind: str) -> int:
        return int(getattr(self, f"max_{kind}") - getattr(self, f"used_{kind}"))

    def elapsed_s(self) -> float:
        return time.monotonic() - self._start_time

    def runtime_remaining_s(self) -> float:
        return max(0.0, self.max_runtime_s - self.elapsed_s())

    def runtime_exhausted(self) -> bool:
        return self.elapsed_s() >= self.max_runtime_s

    def any_exhausted(self) -> bool:
        return (
            self.runtime_exhausted()
            or self.remaining("topology_edits") <= 0
            or self.remaining("sizing_steps") <= 0
            or self.remaining("spice_calls") <= 0
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "topology_edits_remaining": self.remaining("topology_edits"),
            "sizing_steps_remaining": self.remaining("sizing_steps"),
            "spice_calls_remaining": self.remaining("spice_calls"),
            "generations_remaining": self.remaining("generations"),
            "retrievals_remaining": self.remaining("retrievals"),
            "runtime_remaining_s": self.runtime_remaining_s(),
        }

    #: Normalized remaining-budget features for network inputs.
    def feature_vector(self) -> list[float]:
        def frac(kind: str) -> float:
            cap = float(getattr(self, f"max_{kind}"))
            return self.remaining(kind) / cap if cap > 0 else 0.0

        runtime_frac = self.runtime_remaining_s() / self.max_runtime_s if self.max_runtime_s > 0 else 0.0
        return [
            frac("topology_edits"),
            frac("sizing_steps"),
            frac("spice_calls"),
            frac("generations"),
            runtime_frac,
        ]
