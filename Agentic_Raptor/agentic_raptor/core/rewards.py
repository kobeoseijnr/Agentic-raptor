"""Reward component definitions and helper computations.

The *weights* are configuration (YAML), never hard-coded here; see
``agentic_raptor.learning.reward_assignment`` for the weighted combination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardComponents:
    """Unweighted reward signals produced by one candidate evaluation."""

    topology_validity: float = 0.0        # 1.0 valid, 0.0 invalid (minus warning shading)
    sizing_progress: float = 0.0          # improvement of constraint margins during sizing
    spice_feasibility: float = 0.0        # fraction of constraints met at final SPICE
    normalized_fom: float = 0.0           # figure of merit mapped to [0, 1]
    pvt_robustness: float = 0.0           # worst-corner feasibility in [0, 1]
    normalized_spice_calls: float = 0.0   # spice calls used / budget, in [0, 1]
    normalized_runtime: float = 0.0       # runtime used / budget, in [0, 1]
    invalidity_penalty: float = 0.0       # accumulated invalid-action penalty (>= 0)
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)


def feasibility_score(constraint_margins: dict[str, float]) -> float:
    """Fraction of constraints with non-negative margin. Empty dict → 0.0.

    Margins follow the convention: ``margin >= 0`` means the constraint is met,
    normalized so that -1.0 means "missed by 100% of the target".
    """
    if not constraint_margins:
        return 0.0
    met = sum(1 for m in constraint_margins.values() if m >= 0.0)
    return met / len(constraint_margins)


def soft_feasibility(constraint_margins: dict[str, float]) -> float:
    """Smooth [0, 1] feasibility: mean of sigmoid-squashed margins (dense signal)."""
    if not constraint_margins:
        return 0.0
    return sum(1.0 / (1.0 + math.exp(-4.0 * m)) for m in constraint_margins.values()) / len(constraint_margins)


def normalize_fom(fom: float, scale: float = 1.0) -> float:
    """Map an unbounded figure of merit to [0, 1) via a smooth squash."""
    if scale <= 0:
        raise ValueError("fom scale must be positive")
    return max(0.0, 1.0 - math.exp(-max(fom, 0.0) / scale))
