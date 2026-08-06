"""Sizing parameter space generated dynamically from the topology graph.

Every sizable device contributes typed continuous parameters with bounds;
actions live in normalized [-1, 1] space and map back to physical values
(log-scaled where ranges span decades). Topology and sizing stay separated:
this module only reads device types from the graph and writes values into
``CircuitNode.sizing_parameters`` via an explicit apply step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.types import DeviceType

#: device type → (param name, low, high, log_scale)
_PARAM_TEMPLATES: dict[DeviceType, tuple[tuple[str, float, float, bool], ...]] = {
    DeviceType.NMOS: (
        ("width_m", 0.2e-6, 100e-6, True),
        ("length_m", 0.05e-6, 5e-6, True),
        ("multiplier", 1.0, 16.0, False),
    ),
    DeviceType.PMOS: (
        ("width_m", 0.2e-6, 200e-6, True),
        ("length_m", 0.05e-6, 5e-6, True),
        ("multiplier", 1.0, 16.0, False),
    ),
    DeviceType.RESISTOR: (("resistance_ohm", 100.0, 1e6, True),),
    DeviceType.CAPACITOR: (("capacitance_f", 10e-15, 10e-12, True),),
    DeviceType.CURRENT_SOURCE: (("current_a", 1e-6, 500e-6, True),),
    DeviceType.VOLTAGE_SOURCE: (("voltage_v", 0.0, 5.0, False),),
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str          # "<node_id>.<param>"
    node_id: str
    param: str
    low: float
    high: float
    log_scale: bool

    def denormalize(self, u: float) -> float:
        """[-1, 1] → physical value."""
        t = (max(-1.0, min(1.0, u)) + 1.0) / 2.0
        if self.log_scale:
            lo, hi = math.log10(self.low), math.log10(self.high)
            return 10.0 ** (lo + t * (hi - lo))
        return self.low + t * (self.high - self.low)

    def normalize(self, value: float) -> float:
        """physical value → [-1, 1] (clamped)."""
        v = max(self.low, min(self.high, value))
        if self.log_scale:
            lo, hi = math.log10(self.low), math.log10(self.high)
            t = (math.log10(v) - lo) / (hi - lo)
        else:
            t = (v - self.low) / (self.high - self.low)
        return 2.0 * t - 1.0


class SizingParameterSpace:
    """Ordered, deterministic parameter space for one topology."""

    def __init__(self, specs: list[ParameterSpec]) -> None:
        self.specs = specs

    @classmethod
    def from_graph(cls, graph: CircuitGraph) -> SizingParameterSpace:
        specs: list[ParameterSpec] = []
        for node in sorted(graph.nodes.values(), key=lambda n: n.node_id):
            for param, low, high, log_scale in _PARAM_TEMPLATES.get(node.device_type, ()):
                specs.append(
                    ParameterSpec(
                        name=f"{node.node_id}.{param}",
                        node_id=node.node_id,
                        param=param,
                        low=low,
                        high=high,
                        log_scale=log_scale,
                    )
                )
        return cls(specs)

    @property
    def dim(self) -> int:
        return len(self.specs)

    def default_vector(self) -> list[float]:
        """Normalized mid-range starting point (deterministic)."""
        return [0.0] * self.dim

    def denormalize(self, u: list[float]) -> dict[str, dict[str, float]]:
        """Normalized vector → {node_id: {param: physical value}}."""
        out: dict[str, dict[str, float]] = {}
        for spec, value in zip(self.specs, u, strict=True):
            out.setdefault(spec.node_id, {})[spec.param] = spec.denormalize(value)
        return out

    def normalize(self, sizing: dict[str, dict[str, float]]) -> list[float]:
        """{node_id: {param: value}} → normalized vector (missing → 0.0)."""
        out: list[float] = []
        for spec in self.specs:
            value = sizing.get(spec.node_id, {}).get(spec.param)
            out.append(0.0 if value is None else spec.normalize(value))
        return out

    def apply_to_graph(self, graph: CircuitGraph, u: list[float]) -> None:
        graph.apply_sizing(self.denormalize(u))

    def clip(self, u: list[float]) -> list[float]:
        return [max(-1.0, min(1.0, v)) for v in u]
