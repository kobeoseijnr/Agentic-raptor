"""Deterministic validator on valid and invalid toy circuits."""

from __future__ import annotations

from agentic_raptor.core.circuit_graph import CircuitEdge, CircuitGraph, CircuitNode
from agentic_raptor.core.types import DeviceType, TerminalType
from agentic_raptor.topology_validation.validator import TopologyValidator


def _codes(result) -> set[str]:
    return {i.code for i in result.issues}


def test_valid_ota_passes(ota_graph):
    result = TopologyValidator().validate(ota_graph)
    assert result.is_valid
    assert result.graph_hash
    assert not result.errors


def test_empty_graph_rejected():
    result = TopologyValidator().validate(CircuitGraph("empty"))
    assert not result.is_valid
    assert "EMPTY_GRAPH" in _codes(result)


def test_missing_ports_detected():
    graph = CircuitGraph(
        "no-ports",
        nodes=[CircuitNode("m1", DeviceType.NMOS)],
        edges=[
            CircuitEdge("e1", "m1", TerminalType.DRAIN, "n1"),
            CircuitEdge("e2", "m1", TerminalType.GATE, "n1"),
            CircuitEdge("e3", "m1", TerminalType.SOURCE, "n2"),
            CircuitEdge("e4", "m1", TerminalType.BULK, "n2"),
        ],
    )
    result = TopologyValidator().validate(graph)
    assert not result.is_valid
    codes = _codes(result)
    assert {"MISSING_INPUT_PORT", "MISSING_OUTPUT_PORT", "MISSING_SUPPLY_PORT", "MISSING_GROUND_PORT"} <= codes


def test_floating_device_detected(ota_graph):
    floating = ota_graph.copy()
    floating.add_node(CircuitNode("m_orphan", DeviceType.NMOS))
    result = TopologyValidator().validate(floating)
    assert not result.is_valid
    assert "FLOATING_DEVICE" in _codes(result)


def test_partially_connected_device_warns(ota_graph):
    partial = ota_graph.copy()
    partial.add_node(CircuitNode("m_half", DeviceType.NMOS))
    partial.add_edge(CircuitEdge("e_h1", "m_half", TerminalType.GATE, "n_out"))
    result = TopologyValidator().validate(partial)
    assert "PARTIALLY_CONNECTED_DEVICE" in _codes(result)
    assert any(i.severity == "warning" and i.node_ids == ["m_half"] for i in result.issues)


def test_supply_ground_short_detected(ota_graph):
    shorted = ota_graph.to_dict()
    # Rewire the ground port onto the supply net.
    for edge in shorted["edges"]:
        if edge["edge_id"] == "e_gnd":
            edge["net"] = "n_vdd"
    result = TopologyValidator().validate(CircuitGraph.from_dict(shorted))
    assert not result.is_valid
    assert "SUPPLY_GROUND_SHORT" in _codes(result)


def test_output_not_driven_detected():
    graph = CircuitGraph(
        "undriven",
        nodes=[
            CircuitNode("vdd", DeviceType.SUPPLY_PORT),
            CircuitNode("gnd", DeviceType.GROUND_PORT),
            CircuitNode("inp", DeviceType.INPUT_PORT),
            CircuitNode("out", DeviceType.OUTPUT_PORT),
            CircuitNode("r1", DeviceType.RESISTOR),
            CircuitNode("m1", DeviceType.NMOS),
        ],
        edges=[
            CircuitEdge("e1", "vdd", TerminalType.PORT, "n_vdd"),
            CircuitEdge("e2", "gnd", TerminalType.PORT, "n_gnd"),
            CircuitEdge("e3", "inp", TerminalType.PORT, "n_in"),
            CircuitEdge("e4", "out", TerminalType.PORT, "n_out"),
            # Output only touches a resistor to a dead-end net; MOS is elsewhere.
            CircuitEdge("e5", "r1", TerminalType.PLUS, "n_out"),
            CircuitEdge("e6", "r1", TerminalType.MINUS, "n_dead"),
            CircuitEdge("e7", "m1", TerminalType.GATE, "n_in"),
            CircuitEdge("e8", "m1", TerminalType.DRAIN, "n_vdd"),
            CircuitEdge("e9", "m1", TerminalType.SOURCE, "n_gnd"),
            CircuitEdge("e10", "m1", TerminalType.BULK, "n_gnd"),
        ],
    )
    result = TopologyValidator().validate(graph)
    assert not result.is_valid
    assert "OUTPUT_NOT_DRIVEN" in _codes(result)


def test_max_graph_size_enforced(ota_graph):
    result = TopologyValidator(max_nodes=3, max_edges=5).validate(ota_graph)
    assert not result.is_valid
    assert {"TOO_MANY_NODES", "TOO_MANY_EDGES"} <= _codes(result)


def test_dc_floating_net_detected(ota_graph):
    """A bias net driven by a current source into a gate only (no diode) is
    a guaranteed SPICE singular matrix and must be a validation error."""
    broken = ota_graph.copy()
    broken.remove_node("m6")  # remove the diode-connected mirror device
    result = TopologyValidator().validate(broken)
    assert not result.is_valid
    floating = [i for i in result.issues if i.code == "DC_FLOATING_NET"]
    assert floating and "n_bias" in floating[0].message


def test_hash_stable_for_same_structure(ota_graph):
    r1 = TopologyValidator().validate(ota_graph)
    r2 = TopologyValidator().validate(ota_graph.copy())
    assert r1.graph_hash == r2.graph_hash
