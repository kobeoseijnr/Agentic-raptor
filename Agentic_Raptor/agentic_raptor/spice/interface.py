"""Simulator-neutral SPICE interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic_raptor.core.candidate import CircuitCandidate


class AnalysisType:
    """Supported analysis identifiers (strings so YAML stays simple)."""

    OPERATING_POINT = "op"
    AC = "ac"
    TRANSIENT = "tran"
    NOISE = "noise"
    PVT = "pvt"
    MONTE_CARLO = "monte_carlo"

    ALL: tuple[str, ...] = (OPERATING_POINT, AC, TRANSIENT, NOISE, PVT, MONTE_CARLO)


@dataclass
class SimulationResult:
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    constraint_margins: dict[str, float] = field(default_factory=dict)
    raw_output_path: str | None = None
    runtime_s: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    corner: str = "typical"
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationResult:
        return cls(
            success=bool(data["success"]),
            metrics={k: float(v) for k, v in (data.get("metrics") or {}).items()},
            constraint_margins={k: float(v) for k, v in (data.get("constraint_margins") or {}).items()},
            raw_output_path=data.get("raw_output_path"),
            runtime_s=float(data.get("runtime_s", 0.0)),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            corner=str(data.get("corner", "typical")),
            seed=int(data.get("seed", 0)),
        )


class SpiceSimulator(Protocol):
    """Anything that can evaluate a candidate. Implementations: mock, legacy ngspice."""

    def simulate(
        self,
        candidate: CircuitCandidate,
        analyses: list[str],
        timeout_s: float,
    ) -> SimulationResult: ...
