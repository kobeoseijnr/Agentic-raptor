"""Real-SPICE query policy for the budgeted sizing loop.

Simple, documented, deterministic rules — evaluated in order:

1. WARM-UP    — the first ``warmup_transitions`` per topology are always real
                (includes the very first transition of a new topology).
2. NO BUDGET  — with no SPICE budget left, real is impossible → model.
3. UNCERTAIN  — dynamics-ensemble disagreement above ``uncertainty_threshold``
                → real (the model cannot be trusted here).
4. PERIODIC   — every ``query_interval``-th step is real to re-anchor the
                model (drift check).
5. TERMINAL   — final/best-candidate verification is ALWAYS real (enforced by
                the coordinator, not this policy; ``force_real`` covers it).
6. otherwise  — model prediction.

Imagined/model transitions may train SAC but never produce the accepted
candidate or the final topology return (coordinator responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryDecision:
    use_real: bool
    reason: str


@dataclass
class RealSpiceQueryPolicy:
    warmup_transitions: int = 2
    uncertainty_threshold: float = 0.5
    query_interval: int = 3

    def decide(
        self,
        transitions_done: int,
        model_steps_since_real: int,
        uncertainty: float,
        spice_budget_remaining: int,
        force_real: bool = False,
    ) -> QueryDecision:
        if force_real:
            if spice_budget_remaining <= 0:
                return QueryDecision(False, "verification requested but SPICE budget exhausted")
            return QueryDecision(True, "terminal/best-candidate verification requires real SPICE")
        if transitions_done < self.warmup_transitions:
            if spice_budget_remaining <= 0:
                return QueryDecision(False, "warm-up requires real SPICE but budget is exhausted")
            return QueryDecision(True, f"warm-up transition {transitions_done + 1}/{self.warmup_transitions}")
        if spice_budget_remaining <= 0:
            return QueryDecision(False, "SPICE budget exhausted; model prediction only")
        if uncertainty > self.uncertainty_threshold:
            return QueryDecision(
                True, f"dynamics uncertainty {uncertainty:.3f} > threshold {self.uncertainty_threshold}"
            )
        if self.query_interval > 0 and model_steps_since_real >= self.query_interval:
            return QueryDecision(True, f"periodic re-anchor after {model_steps_since_real} model steps")
        return QueryDecision(False, f"model prediction (uncertainty {uncertainty:.3f} within threshold)")
