"""Circuit graph: serialization, hashing, comparison, sizing separation."""

from __future__ import annotations

import pytest

from agentic_raptor.core.circuit_graph import CircuitEdge, CircuitGraph, CircuitNode
from agentic_raptor.core.types import DeviceType, TerminalType
from agentic_raptor.utils.exceptions import GraphError


def test_json_roundtrip(ota_graph):
    restored = CircuitGraph.from_json(ota_graph.to_json())
    assert restored.is_structurally_equal(ota_graph)
    assert restored.graph_id == ota_graph.graph_id
    assert set(restored.nodes) == set(ota_graph.nodes)


def test_stable_hash_across_renames(ota_graph):
    baseline = ota_graph.structural_hash()
    renamed = ota_graph.to_dict()
    # Rename every node and net consistently; structure unchanged.
    for node in renamed["nodes"]:
        node["node_id"] = "x_" + node["node_id"]
    for edge in renamed["edges"]:
        edge["node_id"] = "x_" + edge["node_id"]
        edge["net"] = "netx_" + edge["net"]
    assert CircuitGraph.from_dict(renamed).structural_hash() == baseline


def test_hash_changes_on_structural_change(ota_graph):
    baseline = ota_graph.structural_hash()
    modified = ota_graph.copy()
    cap = CircuitNode("cc_extra", DeviceType.CAPACITOR)
    modified.add_node(cap)
    modified.add_edge(CircuitEdge("e_x1", "cc_extra", TerminalType.PLUS, "n_out"))
    modified.add_edge(CircuitEdge("e_x2", "cc_extra", TerminalType.MINUS, "n_gnd"))
    assert modified.structural_hash() != baseline
    assert not modified.is_structurally_equal(ota_graph)


def test_sizing_separated_from_hash(ota_graph):
    baseline = ota_graph.structural_hash()
    ota_graph.apply_sizing({"m1": {"width_m": 1e-5, "length_m": 2e-7}})
    assert ota_graph.structural_hash() == baseline
    assert ota_graph.sizing_state()["m1"]["width_m"] == pytest.approx(1e-5)


def test_duplicate_node_id_rejected(ota_graph):
    with pytest.raises(GraphError):
        ota_graph.add_node(CircuitNode("m1", DeviceType.NMOS))


def test_duplicate_node_in_serialized_form_rejected(ota_graph):
    data = ota_graph.to_dict()
    data["nodes"].append(dict(data["nodes"][0]))
    with pytest.raises(GraphError):
        CircuitGraph.from_dict(data)


def test_duplicate_terminal_connection_rejected(ota_graph):
    with pytest.raises(GraphError):
        ota_graph.add_edge(CircuitEdge("e_dup", "m1", TerminalType.GATE, "n_gnd"))


def test_invalid_terminal_rejected(ota_graph):
    with pytest.raises(GraphError):
        ota_graph.add_edge(CircuitEdge("e_bad", "cl", TerminalType.GATE, "n_out"))


def test_unsupported_device_type_rejected():
    with pytest.raises(ValueError):
        CircuitNode("bad", "TRIODE")  # type: ignore[arg-type]


def test_networkx_conversion(ota_graph):
    g = ota_graph.to_networkx()
    device_nodes = [n for n, d in g.nodes(data=True) if d["kind"] == "device"]
    net_nodes = [n for n, d in g.nodes(data=True) if d["kind"] == "net"]
    assert len(device_nodes) == len(ota_graph.nodes)
    assert len(net_nodes) == len(ota_graph.nets())
