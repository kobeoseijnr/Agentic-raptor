"""Testbench generation for OTA/amplifier benchmarks.

Operating conditions come from the unified specification (supply, temperature,
load capacitance, common-mode input). Anything not specified uses an explicit
benchmark default and is marked with a ``; default:`` comment in the netlist —
nothing is silently invented.

Wiring (two-input amplifier, op/ac):
    * non-inverting input driven with DC=vcm, AC=1;
    * inverting input tied to the output through a huge inductor (DC feedback
      stabilises the operating point) and to AC ground through a huge
      capacitor — the standard open-loop gain measurement configuration.
Inverting-input selection: ``additional_constraints["inverting_input_port"]``
when provided, else the port whose id suggests an inverting role ("inn",
"minus", "neg"), else the last input in sorted order (assumption is recorded
in the netlist header).

Transient (slew) uses a separate unity-gain wiring, which is why ``tran``
runs as its own netlist.

Noise and Monte Carlo: interfaces exist (`build_noise_testbench`,
`build_monte_carlo_testbench`) but deliberately raise until they are actually
implemented and parsed — no unearned support claims.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.spice.netlist_builder import BuiltCircuit
from agentic_raptor.utils.exceptions import SimulationError

_HUGE_L = "1e9"   # DC-short / AC-open feedback element
_HUGE_C = "1e9"   # DC-open / AC-short element


@dataclass(frozen=True)
class OperatingConditions:
    vdd: float
    temperature_c: float
    vcm: float
    vcm_is_default: bool
    load_capacitance_f: float | None
    ac_start_hz: float = 1.0
    ac_stop_hz: float = 100e9

    @classmethod
    def from_spec(
        cls,
        spec: DesignSpecifications,
        voltage_scale: float = 1.0,
        temperature_c: float | None = None,
    ) -> OperatingConditions:
        vdd = spec.supply_voltage * voltage_scale
        vcm = spec.common_mode_input_v
        vcm_is_default = vcm is None
        if vcm is None:
            vcm = vdd / 2.0  # benchmark default, marked in the netlist
        stop = 100e9
        if spec.target_gbw_hz:
            stop = max(1e9, spec.target_gbw_hz * 1e3)
        return cls(
            vdd=vdd,
            temperature_c=temperature_c if temperature_c is not None else spec.temperature_c,
            vcm=vcm,
            vcm_is_default=vcm_is_default,
            load_capacitance_f=spec.load_capacitance_f,
            ac_stop_hz=stop,
        )


def _select_inputs(built: BuiltCircuit, spec: DesignSpecifications, graph: CircuitGraph) -> tuple[str, str | None, str]:
    """(driven_node, feedback_node_or_None, assumption_note)."""
    inputs = built.input_nodes
    if not inputs:
        raise SimulationError("testbench requires at least one connected INPUT_PORT")
    if len(inputs) == 1:
        return inputs[0], None, "single input: open-loop drive (no DC feedback possible)"
    override = spec.additional_constraints.get("inverting_input_port")
    port_ids = [n.node_id for n in sorted(graph.nodes.values(), key=lambda n: n.node_id)
                if n.device_type.value == "INPUT_PORT"]
    inverting_index = None
    if isinstance(override, str) and override in port_ids:
        inverting_index = port_ids.index(override)
        note = f"inverting input from additional_constraints: {override}"
    else:
        for i, pid in enumerate(port_ids):
            if any(tag in pid.lower() for tag in ("inn", "minus", "neg", "inv")):
                inverting_index = i
                note = f"inverting input inferred from port id {pid!r}"
                break
        else:
            inverting_index = len(port_ids) - 1
            note = f"inverting input ASSUMED to be last sorted port {port_ids[-1]!r}"
    driven = inputs[(inverting_index + 1) % len(inputs)]
    feedback = inputs[inverting_index]
    return driven, feedback, note


def _internal_load_present(graph: CircuitGraph) -> bool:
    return any(n.block_role == "load" for n in graph.nodes.values())


def _common_sections(built: BuiltCircuit, oc: OperatingConditions) -> list[str]:
    vcm_marker = "  ; default: vdd/2 (common_mode_input_v unspecified)" if oc.vcm_is_default else ""
    lines = [
        "",
        "* --- testbench sources ---",
        f"VVDD {built.supply_nodes[0]} 0 DC {oc.vdd:.6g}",
        f"VCM cm_node 0 DC {oc.vcm:.6g}{vcm_marker}",
        f".temp {oc.temperature_c:.6g}",
        ".options nomod",
        ".option reltol=1e-3 gmin=1e-12",
    ]
    for extra in built.supply_nodes[1:]:
        lines.append(f"V{extra} {extra} 0 DC {oc.vdd:.6g}")
    return lines


def build_op_ac_testbench(
    built: BuiltCircuit,
    graph: CircuitGraph,
    spec: DesignSpecifications,
    oc: OperatingConditions,
) -> str:
    """Full netlist for operating-point + open-loop AC with .meas extraction."""
    out = built.output_nodes[0]
    driven, feedback, note = _select_inputs(built, spec, graph)
    lines = [built.circuit_text(), *_common_sections(built, oc), f"* input wiring: {note}"]
    lines.append(f"VIN {driven} cm_node DC 0 AC 1")
    if feedback is not None:
        lines.append(f"LFB {out} {feedback} {_HUGE_L}")
        lines.append(f"CFB {feedback} cm_node {_HUGE_C}")
    if oc.load_capacitance_f and not _internal_load_present(graph):
        lines.append(f"CLOAD {out} 0 {oc.load_capacitance_f:.6g}")
    elif oc.load_capacitance_f:
        lines.append("* internal load-role capacitor present; external CLOAD omitted")
    lines += [
        "",
        ".control",
        "set units=degrees",
        "op",
        f"print v({out}) vvdd#branch",
        f"ac dec 25 {built_num(oc.ac_start_hz)} {built_num(oc.ac_stop_hz)}",
        f"meas ac dc_gain_db FIND vdb({out}) AT={built_num(max(10.0, oc.ac_start_hz))}",
        f"meas ac ugf_hz WHEN vdb({out})=0 CROSS=1",
        f"meas ac phase_at_ugf FIND vp({out}) WHEN vdb({out})=0 CROSS=1",
        "quit 0",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def build_tran_testbench(
    built: BuiltCircuit,
    graph: CircuitGraph,
    spec: DesignSpecifications,
    oc: OperatingConditions,
) -> str:
    """Unity-gain step testbench for slew-rate extraction."""
    out = built.output_nodes[0]
    driven, feedback, note = _select_inputs(built, spec, graph)
    if feedback is None:
        raise SimulationError("transient slew testbench requires a two-input amplifier (unity-gain wiring)")
    step = min(0.2, oc.vdd * 0.15)
    lo, hi = oc.vcm - step, oc.vcm + step
    m_lo, m_hi = oc.vcm - 0.6 * step, oc.vcm + 0.6 * step
    lines = [built.circuit_text(), *_common_sections(built, oc), f"* input wiring: {note} (unity-gain)"]
    lines.append(f"VIN {driven} 0 PULSE({lo:.6g} {hi:.6g} 1u 1n 1n 20u 40u)")
    lines.append(f"VFB {feedback} {out} DC 0")  # unity feedback via 0 V source
    if oc.load_capacitance_f and not _internal_load_present(graph):
        lines.append(f"CLOAD {out} 0 {oc.load_capacitance_f:.6g}")
    lines += [
        "",
        ".control",
        "tran 5n 25u",
        f"meas tran t_rise_lo WHEN v({out})={m_lo:.6g} RISE=1",
        f"meas tran t_rise_hi WHEN v({out})={m_hi:.6g} RISE=1",
        "quit 0",
        ".endc",
        ".end",
        "",
    ]
    return "\n".join(lines)


def build_noise_testbench(*_args: object, **_kwargs: object) -> str:
    raise SimulationError(
        "noise analysis interface exists but is not yet implemented/parsed (Stage 2 scope excludes it)"
    )


def build_monte_carlo_testbench(*_args: object, **_kwargs: object) -> str:
    raise SimulationError(
        "Monte Carlo interface exists but is not yet implemented/parsed (Stage 2 scope excludes it)"
    )


def built_num(value: float) -> str:
    """Plain SPICE-friendly number formatting (no python exponent quirks)."""
    return f"{value:.6g}"
