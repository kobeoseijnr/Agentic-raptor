"""PVT corner evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.core.candidate import CircuitCandidate
from agentic_raptor.spice.interface import SimulationResult


@dataclass(frozen=True)
class PvtCorner:
    name: str
    process: str           # "typical" | "ss" | "ff" | "sf" | "fs"
    voltage_scale: float   # multiplier on nominal supply
    temperature_c: float


def standard_corners() -> list[PvtCorner]:
    return [
        PvtCorner("tt_nom_27", "typical", 1.00, 27.0),
        PvtCorner("ss_low_85", "ss", 0.90, 85.0),
        PvtCorner("ff_high_m40", "ff", 1.10, -40.0),
        PvtCorner("sf_nom_85", "sf", 1.00, 85.0),
        PvtCorner("fs_low_m40", "fs", 0.90, -40.0),
    ]


@dataclass
class PvtResult:
    corner_results: dict[str, SimulationResult] = field(default_factory=dict)
    worst_case_margins: dict[str, float] = field(default_factory=dict)
    pvt_score: float = 0.0  # == pass_rate; kept for Stage 1 compatibility
    # --- Stage 2 aggregation ---
    pass_rate: float = 0.0
    mean_margin: float = 0.0          # mean of per-corner worst margins
    worst_corner: str | None = None   # corner with the lowest worst margin
    failed_corners: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corner_results": {k: v.to_dict() for k, v in self.corner_results.items()},
            "worst_case_margins": dict(self.worst_case_margins),
            "pvt_score": self.pvt_score,
            "pass_rate": self.pass_rate,
            "mean_margin": self.mean_margin,
            "worst_corner": self.worst_corner,
            "failed_corners": list(self.failed_corners),
        }


def run_pvt(
    simulator: Any,
    candidate: CircuitCandidate,
    analyses: list[str],
    corners: list[PvtCorner] | None = None,
    timeout_s: float = 10.0,
) -> PvtResult:
    """Simulate every corner; aggregate worst-case margins and a robustness score.

    ``simulator`` must accept corner/voltage_scale/temperature_c keyword
    arguments (both MockSpiceSimulator and future adapters do).
    """
    corners = corners or standard_corners()
    result = PvtResult()
    feasible = 0
    corner_worst: dict[str, float] = {}
    for corner in corners:
        sim = simulator.simulate(
            candidate,
            analyses,
            timeout_s=timeout_s,
            corner=corner.process,
            voltage_scale=corner.voltage_scale,
            temperature_c=corner.temperature_c,
        )
        result.corner_results[corner.name] = sim
        passed = sim.success and sim.constraint_margins and all(
            m >= 0 for m in sim.constraint_margins.values()
        )
        if passed:
            feasible += 1
        else:
            result.failed_corners.append(corner.name)
        corner_worst[corner.name] = (
            min(sim.constraint_margins.values()) if sim.constraint_margins else -1.0
        )
        for key, value in sim.constraint_margins.items():
            if key not in result.worst_case_margins or value < result.worst_case_margins[key]:
                result.worst_case_margins[key] = value
    if corners:
        result.pass_rate = feasible / len(corners)
        result.pvt_score = result.pass_rate
        result.mean_margin = sum(corner_worst.values()) / len(corner_worst)
        result.worst_corner = min(corner_worst, key=corner_worst.get)  # type: ignore[arg-type]
    return result
