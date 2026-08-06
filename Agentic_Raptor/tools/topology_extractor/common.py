"""Shared canonicalisation for the topology corpus.

Every source converts into the SAME representation: `agentic_raptor.core.
circuit_graph.CircuitGraph` (device-level for SPICE sources; block-level via
SUBCIRCUIT_BLOCK nodes with `block_role` for DAG sources). Canonical hash =
the existing WL `structural_hash()` (rename-invariant, order-normalised).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from agentic_raptor.core.circuit_graph import CircuitEdge, CircuitGraph, CircuitNode  # noqa: E402
from agentic_raptor.core.types import DeviceType, TerminalType  # noqa: E402

__all__ = ["CircuitEdge", "CircuitGraph", "CircuitNode", "DeviceType", "TerminalType",
           "ExtractedTopology", "block_graph", "canonical_hash", "isomorphic"]


@dataclass
class ExtractedTopology:
    name: str
    repository: str            # analoggym | opamp_generator | cktgnn
    graph: CircuitGraph
    netlist_path: str | None = None
    schematic_path: str | None = None
    mapping_status: str = "exact"      # exact | approximate_dag_decode | skipped
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def graph_hash(self) -> str:
        return self.graph.structural_hash()


def canonical_hash(graph: CircuitGraph) -> str:
    return graph.structural_hash()


def isomorphic(a: CircuitGraph, b: CircuitGraph) -> bool:
    import networkx as nx
    from networkx.algorithms.isomorphism import categorical_node_match

    return nx.is_isomorphic(a.to_networkx(), b.to_networkx(),
                            node_match=categorical_node_match("label", ""))


def block_graph(name: str, nodes: list[tuple[str, str]], edges: list[tuple[str, str]],
                family_hint: str = "") -> CircuitGraph:
    """Build a block-level canonical graph.

    nodes: (node_id, block_role); edges: (src_id, dst_id) → net per edge.
    IN/OUT/VDD/GND ports are added automatically when referenced.
    """
    g = CircuitGraph(name)
    port_types = {"IN": DeviceType.INPUT_PORT, "OUT": DeviceType.OUTPUT_PORT,
                  "VDD": DeviceType.SUPPLY_PORT, "GND": DeviceType.GROUND_PORT}
    for nid, role in nodes:
        if nid in port_types:
            continue
        g.add_node(CircuitNode(nid, DeviceType.SUBCIRCUIT_BLOCK, block_role=role,
                               attributes={"subckt_name": role}))
    made_ports: set[str] = set()
    for k, (src, dst) in enumerate(edges):
        net = f"net_{src}_{dst}_{k}"
        for endpoint in (src, dst):
            if endpoint in port_types and endpoint not in made_ports:
                g.add_node(CircuitNode(endpoint, port_types[endpoint]))
                made_ports.add(endpoint)
            term = TerminalType.PORT if endpoint in port_types else TerminalType.BLOCK_PIN
            g.add_edge(CircuitEdge(f"e{k}_{endpoint}", endpoint, term, net))
    g.metadata.circuit_family = family_hint
    return g
