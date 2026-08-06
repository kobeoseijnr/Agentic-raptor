"""Read-only adapter around the legacy ``graph.export_graph_to_netlist``.

The legacy exporter takes a dict-based bipartite device–net graph (legacy
``graph/graph_schema.py`` form) and returns SPICE-like netlist text. Its own
docstring warns that exported netlists are NOT guaranteed electrically valid,
so this adapter is used for cross-checks and legacy interop — the Stage 2
native builder (``spice/netlist_builder``) remains the simulation path.

Never modifies legacy code or data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.adapters.legacy_raptor import ensure_repo_root_on_path, legacy_available
from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.types import PORT_DEVICE_TYPES, DeviceType, TerminalType


def is_available() -> bool:
    return legacy_available("graph.export_graph_to_netlist")


#: typed device type → legacy device_type string
_LEGACY_DEVICE_TYPES: dict[DeviceType, str] = {
    DeviceType.NMOS: "nmos",
    DeviceType.PMOS: "pmos",
    DeviceType.RESISTOR: "resistor",
    DeviceType.CAPACITOR: "capacitor",
    DeviceType.CURRENT_SOURCE: "isource",
    DeviceType.VOLTAGE_SOURCE: "vsource",
}

#: nets attached to these port types get legacy net_type labels
_LEGACY_NET_TYPES: dict[DeviceType, str] = {
    DeviceType.SUPPLY_PORT: "supply",
    DeviceType.GROUND_PORT: "ground",
    DeviceType.INPUT_PORT: "input",
    DeviceType.OUTPUT_PORT: "output",
}


@dataclass
class LegacyExportResult:
    ok: bool
    netlist_text: str | None = None
    errors: list[str] = field(default_factory=list)
    unsupported_nodes: list[str] = field(default_factory=list)


def to_legacy_graph_dict(graph: CircuitGraph, sizing_state: dict[str, dict[str, float]]) -> tuple[dict[str, Any], list[str]]:
    """Typed CircuitGraph → legacy bipartite device–net dict.

    Returns (legacy_dict, unsupported_node_ids). Ports become net-type labels
    rather than devices (matching the legacy convention); sizing parameters are
    attached as node features.
    """
    net_types: dict[str, str] = {}
    for port_type, label in _LEGACY_NET_TYPES.items():
        for port in graph.nodes_of_type(port_type):
            net = graph.net_of(port.node_id, TerminalType.PORT)
            if net:
                net_types.setdefault(net, label)

    nodes: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for node in sorted(graph.nodes.values(), key=lambda n: n.node_id):
        if node.device_type in PORT_DEVICE_TYPES:
            continue
        legacy_type = _LEGACY_DEVICE_TYPES.get(node.device_type)
        if legacy_type is None:
            unsupported.append(node.node_id)
            continue
        sizing = {**node.sizing_parameters, **sizing_state.get(node.node_id, {})}
        nodes.append(
            {
                "node_id": node.node_id,
                "node_type": "device",
                "name": node.node_id,
                "device_type": legacy_type,
                "features": {"block_role": node.block_role, **sizing},
            }
        )
    for net in sorted(graph.nets()):
        nodes.append(
            {
                "node_id": f"net::{net}",
                "node_type": "net",
                "name": net,
                "net_type": net_types.get(net, "internal"),
            }
        )
    edges = [
        {
            "edge_id": e.edge_id,
            "src": e.node_id,
            "dst": f"net::{e.net}",
            "terminal_role": e.terminal.value,
        }
        for e in sorted(graph.edges.values(), key=lambda e: e.edge_id)
        if graph.nodes[e.node_id].device_type not in PORT_DEVICE_TYPES
    ]
    return (
        {
            "graph_id": graph.graph_id,
            "graph_type": "bipartite_device_net",
            "topology_id": graph.graph_id,
            "nodes": nodes,
            "edges": edges,
        },
        unsupported,
    )


def export_via_legacy(graph: CircuitGraph, sizing_state: dict[str, dict[str, float]]) -> LegacyExportResult:
    """Call the legacy exporter on a translated graph; structured errors on failure."""
    if not is_available():
        return LegacyExportResult(ok=False, errors=["legacy exporter graph.export_graph_to_netlist not importable"])
    legacy_dict, unsupported = to_legacy_graph_dict(graph, sizing_state)
    if unsupported:
        return LegacyExportResult(
            ok=False,
            errors=[f"unsupported device types for legacy export on nodes: {unsupported}"],
            unsupported_nodes=unsupported,
        )
    ensure_repo_root_on_path()
    try:
        from graph.export_graph_to_netlist import export_graph_to_netlist  # noqa: PLC0415

        text = export_graph_to_netlist(legacy_dict)
    except Exception as exc:  # legacy code raises plain exceptions
        return LegacyExportResult(ok=False, errors=[f"legacy exporter raised: {exc!r}"])
    if not isinstance(text, str) or not text.strip():
        return LegacyExportResult(ok=False, errors=["legacy exporter returned empty output"])
    return LegacyExportResult(ok=True, netlist_text=text)
