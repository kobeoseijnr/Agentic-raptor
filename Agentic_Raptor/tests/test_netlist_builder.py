"""Graph → netlist conversion: cards, ordering, sanitisation, testbenches, legacy adapter."""

from __future__ import annotations

import pytest

from agentic_raptor.core.circuit_graph import CircuitEdge, CircuitGraph, CircuitNode
from agentic_raptor.core.types import DeviceType, TerminalType
from agentic_raptor.spice.device_mapping import NetNameMap, sanitize_node_name
from agentic_raptor.spice.model_library import load_model_library
from agentic_raptor.spice.netlist_builder import build_circuit
from agentic_raptor.spice.testbench_builder import (
    OperatingConditions,
    build_monte_carlo_testbench,
    build_noise_testbench,
    build_op_ac_testbench,
    build_tran_testbench,
)
from agentic_raptor.utils.exceptions import SimulationError

LIBRARY = load_model_library(None, "generic_1u_level1")


def _built(ota_graph, sizing=None):
    return build_circuit(ota_graph, sizing or {}, LIBRARY, candidate_id="test-cand")


def test_node_sanitisation_and_ground_zero(ota_graph):
    net_map = NetNameMap.from_graph(ota_graph)
    assert net_map.node("n_gnd") == "0", "ground net must map to node 0"
    assert sanitize_node_name("weird net:name!") == "weird_net_name_"


def test_mos_card_terminal_order_and_sizing(ota_graph):
    sizing = {"m1": {"width_m": 5e-6, "length_m": 0.25e-6, "multiplier": 2.0}}
    built = _built(ota_graph, sizing)
    m1 = next(line for line in built.circuit_lines if line.startswith("M_m1 "))
    tokens = m1.split()
    # D G S B order: drain=n_mirror, gate=n_inp, source=n_tail, bulk=gnd(0)
    assert tokens[1:5] == ["n_mirror", "n_inp", "n_tail", "0"]
    assert tokens[5] == "nmos_generic"
    assert "W=5e-06" in m1 and "L=2.5e-07" in m1 and "M=2" in m1
    assert "; defaults" not in m1, "explicit sizing must not be marked as default"


def test_default_sizing_marked(ota_graph):
    built = _built(ota_graph)
    m1 = next(line for line in built.circuit_lines if line.startswith("M_m1 "))
    assert "; defaults" in m1, "parameter-space defaults must be visibly marked"


def test_passives_and_sources_emitted(ota_graph):
    built = _built(ota_graph, {"ib1": {"current_a": 3e-5}, "cl": {"capacitance_f": 2e-12}})
    ib = next(line for line in built.circuit_lines if line.startswith("I_ib1 "))
    cl = next(line for line in built.circuit_lines if line.startswith("C_cl "))
    assert "DC 3e-05" in ib
    assert "2e-12" in cl
    assert "n_vdd" in ib and "n_bias" in ib


def test_header_contains_candidate_and_hash(ota_graph):
    built = _built(ota_graph)
    text = built.circuit_text()
    assert "candidate=test-cand" in text
    assert ota_graph.structural_hash() in text
    assert ".model nmos_generic" in text


def test_partially_connected_device_rejected(ota_graph):
    broken = ota_graph.copy()
    broken.add_node(CircuitNode("m_half", DeviceType.NMOS))
    broken.add_edge(CircuitEdge("e_h", "m_half", TerminalType.GATE, "n_out"))
    with pytest.raises(SimulationError, match="unconnected"):
        _built(broken)


def test_subcircuit_requires_metadata(ota_graph):
    g = ota_graph.copy()
    g.add_node(CircuitNode("blk", DeviceType.SUBCIRCUIT_BLOCK))
    g.add_edge(CircuitEdge("e_b", "blk", TerminalType.BLOCK_PIN, "n_out"))
    with pytest.raises(SimulationError, match="subckt_name"):
        _built(g)


def test_missing_rails_rejected():
    graph = CircuitGraph(
        "no-rails",
        nodes=[
            CircuitNode("inp", DeviceType.INPUT_PORT),
            CircuitNode("out", DeviceType.OUTPUT_PORT),
            CircuitNode("r1", DeviceType.RESISTOR),
        ],
        edges=[
            CircuitEdge("e1", "inp", TerminalType.PORT, "a"),
            CircuitEdge("e2", "out", TerminalType.PORT, "b"),
            CircuitEdge("e3", "r1", TerminalType.PLUS, "a"),
            CircuitEdge("e4", "r1", TerminalType.MINUS, "b"),
        ],
    )
    with pytest.raises(SimulationError, match="SUPPLY_PORT"):
        build_circuit(graph, {}, LIBRARY)


def test_op_ac_testbench_contents(ota_graph, spec):
    built = _built(ota_graph)
    oc = OperatingConditions.from_spec(spec)
    netlist = build_op_ac_testbench(built, ota_graph, spec, oc)
    assert "VVDD n_vdd 0 DC 1.8" in netlist
    assert "LFB n_out n_inn 1e9" in netlist          # feedback to inferred inverting input
    assert "VIN n_inp cm_node DC 0 AC 1" in netlist  # drive on the other input
    assert "meas ac dc_gain_db" in netlist and "meas ac ugf_hz" in netlist
    assert "internal load-role capacitor present" in netlist  # mock OTA carries cl
    assert ".temp 27" in netlist
    assert netlist.rstrip().endswith(".end")


def test_vcm_default_marked(ota_graph):
    from agentic_raptor.core.specifications import DesignSpecifications

    spec = DesignSpecifications(circuit_class="ota", technology="g", supply_voltage=2.0)
    oc = OperatingConditions.from_spec(spec)
    built = _built(ota_graph)
    netlist = build_op_ac_testbench(built, ota_graph, spec, oc)
    assert "default: vdd/2" in netlist
    assert "VCM cm_node 0 DC 1" in netlist


def test_tran_testbench_unity_gain(ota_graph, spec):
    built = _built(ota_graph)
    oc = OperatingConditions.from_spec(spec)
    netlist = build_tran_testbench(built, ota_graph, spec, oc)
    assert "VFB n_inn n_out DC 0" in netlist
    assert "PULSE(" in netlist and "meas tran t_rise_lo" in netlist


def test_noise_and_monte_carlo_interfaces_refuse():
    with pytest.raises(SimulationError, match="noise"):
        build_noise_testbench()
    with pytest.raises(SimulationError, match="Monte Carlo"):
        build_monte_carlo_testbench()


@pytest.mark.integration
def test_legacy_export_adapter_comparison(ota_graph):
    """Legacy exporter cross-check on supported devices.

    Verified behaviour of the legacy exporter (its own docstring says it is
    intentionally conservative): without preserved raw device lines it
    reconstructs only passive elements. The adapter therefore serves as a
    passive-device cross-check + legacy interop path; the Stage 2 native
    builder (which emits every device) is the simulation path.
    """
    from agentic_raptor.adapters import legacy_netlist

    if not legacy_netlist.is_available():
        pytest.skip("legacy exporter not importable")
    sizing = {"m1": {"width_m": 5e-6}}
    result = legacy_netlist.export_via_legacy(ota_graph, sizing)
    assert result.ok, result.errors
    legacy_text = (result.netlist_text or "").lower()
    assert "cl" in legacy_text, "legacy export must contain the load capacitor"
    assert "n_out" in legacy_text and "n_gnd" in legacy_text
    # Native builder emits the full device set — the authoritative path.
    native = _built(ota_graph, sizing)
    native_devices = {line.split()[0].split("_", 1)[1] for line in native.circuit_lines}
    assert {"m1", "m2", "m3", "m4", "m5", "m6", "ib1", "cl"} <= native_devices


def test_legacy_translation_reports_unsupported(ota_graph):
    from agentic_raptor.adapters.legacy_netlist import to_legacy_graph_dict

    g = ota_graph.copy()
    g.add_node(CircuitNode("blk", DeviceType.SUBCIRCUIT_BLOCK))
    g.add_edge(CircuitEdge("e_b", "blk", TerminalType.BLOCK_PIN, "n_out"))
    _legacy, unsupported = to_legacy_graph_dict(g, {})
    assert unsupported == ["blk"]
