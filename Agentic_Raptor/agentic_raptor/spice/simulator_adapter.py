"""Simulator implementations: deterministic mock + legacy ngspice adapter stub."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

from agentic_raptor.core.candidate import CircuitCandidate
from agentic_raptor.core.types import DeviceType
from agentic_raptor.spice.interface import SimulationResult
from agentic_raptor.spice.result_parser import compute_constraint_margins
from agentic_raptor.utils.exceptions import SimulationError


def _hash_unit_floats(*parts: str, n: int) -> list[float]:
    """n deterministic floats in [0, 1) derived from the sha256 of the parts."""
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
    out: list[float] = []
    data = digest
    while len(out) < n:
        for i in range(0, len(data) - 7, 8):
            (value,) = struct.unpack_from(">Q", data, i)
            out.append((value >> 11) / float(1 << 53))
            if len(out) == n:
                break
        data = hashlib.sha256(data).digest()
    return out


@dataclass
class MockSpiceSimulator:
    """Deterministic physics-flavoured mock.

    Metrics are smooth functions of the sizing state (so SAC gets learnable
    structure) plus a topology-hash-dependent offset (so different topologies
    genuinely differ), with corner-dependent derating. No randomness at all —
    identical inputs give identical results.
    """

    seed: int = 0

    def simulate(
        self,
        candidate: CircuitCandidate,
        analyses: list[str],
        timeout_s: float = 10.0,
        corner: str = "typical",
        voltage_scale: float = 1.0,
        temperature_c: float | None = None,
    ) -> SimulationResult:
        graph = candidate.topology
        spec = candidate.specifications
        topo_hash = graph.structural_hash()

        sizing = candidate.sizing_state or graph.sizing_state()
        sizing_items = sorted((nid, p, v) for nid, params in sizing.items() for p, v in params.items())
        sizing_repr = ";".join(f"{nid}.{p}={v:.6e}" for nid, p, v in sizing_items)
        h = _hash_unit_floats(topo_hash, sizing_repr, ",".join(sorted(analyses)), corner, str(self.seed), n=6)

        # Structure-derived drivers.
        n_mos = len(graph.nodes_of_type(DeviceType.NMOS)) + len(graph.nodes_of_type(DeviceType.PMOS))
        n_caps = len(graph.nodes_of_type(DeviceType.CAPACITOR))
        n_stages = max(1, sum(1 for n in graph.nodes.values() if n.block_role == "gain_stage") // 3 + 1)
        comp_caps = sum(1 for n in graph.nodes.values() if n.block_role == "compensation")

        # Sizing-derived drivers (smooth, bounded).
        widths = [v for nid, p, v in sizing_items if p == "width_m"]
        currents = [v for nid, p, v in sizing_items if p == "current_a"]
        mean_width = sum(widths) / len(widths) if widths else 5e-6
        total_current = sum(currents) if currents else 50e-6
        width_factor = math.tanh(mean_width / 20e-6)          # 0..1
        current_factor = math.tanh(total_current / 200e-6)    # 0..1

        vdd = spec.supply_voltage * voltage_scale
        temp = temperature_c if temperature_c is not None else spec.temperature_c
        temp_derate = 1.0 - 0.0015 * max(0.0, temp - 27.0)
        corner_derate = {"typical": 1.0, "ss": 0.85, "ff": 1.08, "sf": 0.95, "fs": 0.95}.get(corner, 0.9)

        gain_db = (
            (34.0 + 26.0 * n_stages) * temp_derate * corner_derate
            + 10.0 * width_factor
            + 6.0 * (h[0] - 0.5)
        )
        load_c = spec.load_capacitance_f or 1e-12
        gbw_hz = (
            (total_current / max(load_c, 1e-14)) * 0.02 * corner_derate * temp_derate
            * (1.0 + 0.3 * (h[1] - 0.5))
            / (1.0 + 0.5 * comp_caps)
        )
        phase_margin_deg = min(
            89.0,
            max(5.0, 38.0 + 22.0 * comp_caps + 8.0 * (h[2] - 0.5) - 9.0 * (n_stages - 1) + 10.0 * (1.0 - current_factor)),
        )
        power_w = vdd * total_current * (1.0 + 0.05 * n_mos) * (1.0 + 0.1 * (h[3] - 0.5))
        slew = total_current / max(load_c, 1e-14) * (0.8 + 0.4 * h[4])
        swing = max(0.1, vdd - 0.4 - 0.15 * n_stages + 0.1 * (h[5] - 0.5))
        area = (sum(w * 1e6 for w in widths) * 2.0 + 50.0 * n_caps + 20.0) * (1.0 + 0.2 * n_mos / 10.0)

        metrics = {
            "gain_db": round(gain_db, 3),
            "gbw_hz": round(gbw_hz, 1),
            "phase_margin_deg": round(phase_margin_deg, 2),
            "power_w": power_w,
            "slew_rate_v_per_s": slew,
            "output_swing_v": round(swing, 3),
            "area_um2": round(area, 1),
        }
        margins = compute_constraint_margins(metrics, spec)
        runtime = 0.001 * (1 + len(analyses))
        return SimulationResult(
            success=True,
            metrics=metrics,
            constraint_margins=margins,
            raw_output_path=None,
            runtime_s=runtime,
            corner=corner,
            seed=self.seed,
        )


class LegacyNgspiceSimulator:
    """Adapter target for the legacy ngspice pipeline.

    The legacy execution sites are ``mb_sac/sizing_environment.py`` and
    ``controller.module_adapters.run_spice_validation`` (takes an exported
    netlist path + ngspice exe + timeout). Wiring requires netlist export via
    ``graph/export_graph_to_netlist.py`` from the legacy graph form, which is
    stage-2 work (see docs/IMPLEMENTATION_PLAN.md); this class documents the
    seam and fails loudly rather than pretending to simulate.
    """

    def __init__(self, ngspice_exe: str) -> None:
        self.ngspice_exe = ngspice_exe

    def simulate(
        self,
        candidate: CircuitCandidate,
        analyses: list[str],
        timeout_s: float = 60.0,
        corner: str = "typical",
        voltage_scale: float = 1.0,
        temperature_c: float | None = None,
    ) -> SimulationResult:
        raise SimulationError(
            "LegacyNgspiceSimulator is a stage-2 integration: requires typed-graph → "
            "legacy-netlist export (graph/export_graph_to_netlist.py) before invoking "
            "controller.module_adapters.run_spice_validation. Use MockSpiceSimulator "
            "for the smoke pipeline."
        )
