"""Technology-independent typed circuit graph.

Model
-----
* :class:`CircuitNode` — a device or port with typed terminals, free-form
  attributes, and **separate** continuous ``sizing_parameters``.
* :class:`CircuitEdge` — one terminal-to-net attachment ``(node_id, terminal) → net``.
  A net is the set of terminals attached to the same net name; this matches
  SPICE semantics and converts directly to the legacy RAPTOR bipartite
  device–net representation (``graph/graph_schema.py``).
* :class:`CircuitGraph` — container with JSON (de)serialization, NetworkX
  conversion, stable structural hashing (sizing excluded), and comparison.

Future PyTorch Geometric conversion: `to_networkx()` output feeds
`torch_geometric.utils.from_networkx` once PyG is installed (see
docs/DECISIONS.md D3); no PyG import happens here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.core.types import (
    EXPECTED_TERMINALS,
    DeviceType,
    TerminalType,
)
from agentic_raptor.utils.exceptions import GraphError


@dataclass
class TopologyMetadata:
    """Provenance and description of a topology (not part of the structural hash)."""

    name: str = ""
    description: str = ""
    circuit_family: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitNode:
    """A device, port, or subcircuit block."""

    node_id: str
    device_type: DeviceType
    terminals: tuple[TerminalType, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    sizing_parameters: dict[str, float] = field(default_factory=dict)
    block_role: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.device_type, str):
            self.device_type = DeviceType(self.device_type)
        if not self.terminals:
            self.terminals = EXPECTED_TERMINALS[self.device_type]
        else:
            self.terminals = tuple(TerminalType(t) for t in self.terminals)


@dataclass
class CircuitEdge:
    """Attachment of one node terminal to an electrical net."""

    edge_id: str
    node_id: str
    terminal: TerminalType
    net: str

    def __post_init__(self) -> None:
        if isinstance(self.terminal, str):
            self.terminal = TerminalType(self.terminal)


class CircuitGraph:
    """Mutable typed circuit graph. Topology only — sizing lives on nodes separately."""

    def __init__(
        self,
        graph_id: str,
        nodes: list[CircuitNode] | None = None,
        edges: list[CircuitEdge] | None = None,
        metadata: TopologyMetadata | None = None,
    ) -> None:
        self.graph_id = graph_id
        self.metadata = metadata or TopologyMetadata()
        self._nodes: dict[str, CircuitNode] = {}
        self._edges: dict[str, CircuitEdge] = {}
        for n in nodes or []:
            self.add_node(n)
        for e in edges or []:
            self.add_edge(e)

    # -- accessors ----------------------------------------------------------
    @property
    def nodes(self) -> dict[str, CircuitNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> dict[str, CircuitEdge]:
        return dict(self._edges)

    def node(self, node_id: str) -> CircuitNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise GraphError(f"unknown node_id {node_id!r}") from exc

    def nodes_of_type(self, device_type: DeviceType) -> list[CircuitNode]:
        return [n for n in self._nodes.values() if n.device_type == device_type]

    def nets(self) -> dict[str, list[tuple[str, TerminalType]]]:
        """Net name → attached (node_id, terminal) pairs, deterministically ordered."""
        out: dict[str, list[tuple[str, TerminalType]]] = {}
        for e in sorted(self._edges.values(), key=lambda e: e.edge_id):
            out.setdefault(e.net, []).append((e.node_id, e.terminal))
        return out

    def net_of(self, node_id: str, terminal: TerminalType) -> str | None:
        for e in self._edges.values():
            if e.node_id == node_id and e.terminal == terminal:
                return e.net
        return None

    # -- mutation -----------------------------------------------------------
    def add_node(self, node: CircuitNode) -> None:
        if node.node_id in self._nodes:
            raise GraphError(f"duplicate node_id {node.node_id!r}")
        self._nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> tuple[CircuitNode, list[CircuitEdge]]:
        """Remove a node and its attachments; returns removed pieces for undo."""
        node = self.node(node_id)
        removed_edges = [e for e in self._edges.values() if e.node_id == node_id]
        for e in removed_edges:
            del self._edges[e.edge_id]
        del self._nodes[node_id]
        return node, removed_edges

    def add_edge(self, edge: CircuitEdge) -> None:
        if edge.edge_id in self._edges:
            raise GraphError(f"duplicate edge_id {edge.edge_id!r}")
        if edge.node_id not in self._nodes:
            raise GraphError(f"edge {edge.edge_id!r} references unknown node {edge.node_id!r}")
        node = self._nodes[edge.node_id]
        if edge.terminal not in node.terminals:
            raise GraphError(
                f"terminal {edge.terminal.value!r} invalid for {node.device_type.value} node {edge.node_id!r}"
            )
        # BLOCK_PIN and PORT are fan-out capable (block-level topology corpora);
        # device terminals (D/G/S/B, P/N) stay strictly single-attachment.
        if edge.terminal not in (TerminalType.BLOCK_PIN, TerminalType.PORT):
            for existing in self._edges.values():
                if existing.node_id == edge.node_id and existing.terminal == edge.terminal:
                    raise GraphError(
                        f"terminal ({edge.node_id!r}, {edge.terminal.value!r}) already attached to net "
                        f"{existing.net!r}"
                    )
        self._edges[edge.edge_id] = edge

    def remove_edge(self, edge_id: str) -> CircuitEdge:
        if edge_id not in self._edges:
            raise GraphError(f"unknown edge_id {edge_id!r}")
        return self._edges.pop(edge_id)

    def next_id(self, prefix: str) -> str:
        """Deterministic fresh identifier with the given prefix."""
        existing = set(self._nodes) | set(self._edges)
        i = 1
        while f"{prefix}{i}" in existing:
            i += 1
        return f"{prefix}{i}"

    def copy(self) -> CircuitGraph:
        return CircuitGraph.from_dict(self.to_dict())

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return {
            "graph_id": self.graph_id,
            "metadata": to_jsonable(self.metadata),
            "nodes": [to_jsonable(n) for n in sorted(self._nodes.values(), key=lambda n: n.node_id)],
            "edges": [to_jsonable(e) for e in sorted(self._edges.values(), key=lambda e: e.edge_id)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CircuitGraph:
        raw_nodes = data.get("nodes") or []
        seen: set[str] = set()
        for rn in raw_nodes:
            nid = str(rn.get("node_id"))
            if nid in seen:
                raise GraphError(f"duplicate node_id {nid!r} in serialized graph")
            seen.add(nid)
        meta_raw = dict(data.get("metadata") or {})
        metadata = TopologyMetadata(
            name=str(meta_raw.get("name", "")),
            description=str(meta_raw.get("description", "")),
            circuit_family=str(meta_raw.get("circuit_family", "")),
            source=str(meta_raw.get("source", "")),
            tags=list(meta_raw.get("tags") or []),
            extra=dict(meta_raw.get("extra") or {}),
        )
        nodes = [
            CircuitNode(
                node_id=str(rn["node_id"]),
                device_type=DeviceType(rn["device_type"]),
                terminals=tuple(TerminalType(t) for t in (rn.get("terminals") or ())),
                attributes=dict(rn.get("attributes") or {}),
                sizing_parameters={k: float(v) for k, v in (rn.get("sizing_parameters") or {}).items()},
                block_role=rn.get("block_role"),
            )
            for rn in raw_nodes
        ]
        edges = [
            CircuitEdge(
                edge_id=str(re["edge_id"]),
                node_id=str(re["node_id"]),
                terminal=TerminalType(re["terminal"]),
                net=str(re["net"]),
            )
            for re in (data.get("edges") or [])
        ]
        return cls(graph_id=str(data.get("graph_id", "graph")), nodes=nodes, edges=edges, metadata=metadata)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> CircuitGraph:
        return cls.from_dict(json.loads(text))

    # -- structural views ---------------------------------------------------
    def to_networkx(self) -> Any:
        """Bipartite device–net NetworkX graph (matches legacy ``bipartite_device_net``).

        Device nodes are labelled by device type, net nodes by ``"net"``, edges by
        terminal role. Sizing parameters are intentionally excluded so the view is
        purely structural.
        """
        import networkx as nx

        g = nx.Graph()
        for n in self._nodes.values():
            # block_role participates in the label so semantic attributes
            # (e.g. signed gm '+gm+' vs '-gm+') affect the WL hash.
            label = n.device_type.value + (f":{n.block_role}" if n.block_role else "")
            g.add_node(f"dev::{n.node_id}", label=label, kind="device")
        for net_name in self.nets():
            g.add_node(f"net::{net_name}", label="net", kind="net")
        # Merge parallel attachments (e.g. diode-connected MOS: G and D on the
        # same net) into one edge with a deterministic combined label —
        # nx.Graph would otherwise keep only the last-inserted terminal label.
        combined: dict[tuple[str, str], list[str]] = {}
        for e in self._edges.values():
            combined.setdefault((f"dev::{e.node_id}", f"net::{e.net}"), []).append(e.terminal.value)
        for (dev, net), terminals in combined.items():
            g.add_edge(dev, net, label="+".join(sorted(terminals)))
        return g

    def structural_hash(self) -> str:
        """Stable topology hash. Sizing values and metadata are excluded.

        Uses the Weisfeiler–Lehman graph hash over the bipartite device–net view,
        so renaming nodes/nets without changing structure yields the same hash.
        """
        import networkx as nx

        g = self.to_networkx()
        # Fold edge (terminal) labels into node labels so WL sees terminal roles.
        for u, v, data in g.edges(data=True):
            for endpoint in (u, v):
                g.nodes[endpoint]["label"] += f"|{data['label']}"
        for _node, data in g.nodes(data=True):
            data["label"] = "|".join(sorted(data["label"].split("|")))
        return nx.weisfeiler_lehman_graph_hash(g, node_attr="label", iterations=3, digest_size=16)

    def is_structurally_equal(self, other: CircuitGraph) -> bool:
        """Topology comparison via structural hash plus cheap invariants."""
        return (
            len(self._nodes) == len(other._nodes)
            and len(self._edges) == len(other._edges)
            and self.structural_hash() == other.structural_hash()
        )

    # -- sizing separation --------------------------------------------------
    def sizing_state(self) -> dict[str, dict[str, float]]:
        """node_id → sizing parameters (continuous values only)."""
        return {n.node_id: dict(n.sizing_parameters) for n in self._nodes.values() if n.sizing_parameters}

    def apply_sizing(self, sizing: dict[str, dict[str, float]]) -> None:
        for node_id, params in sizing.items():
            self.node(node_id).sizing_parameters.update({k: float(v) for k, v in params.items()})

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CircuitGraph(id={self.graph_id!r}, nodes={len(self._nodes)}, edges={len(self._edges)})"
