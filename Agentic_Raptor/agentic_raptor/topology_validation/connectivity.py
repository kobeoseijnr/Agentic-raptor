"""Net-level connectivity analysis used by the validator."""

from __future__ import annotations

from collections import deque

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.types import ACTIVE_DEVICE_TYPES, DeviceType
from agentic_raptor.topology_validation.rules import ValidationIssue


def _adjacency(graph: CircuitGraph) -> dict[str, set[str]]:
    """node_id → neighbouring node_ids (through shared nets)."""
    adj: dict[str, set[str]] = {nid: set() for nid in graph.nodes}
    for attachments in graph.nets().values():
        ids = [node_id for node_id, _ in attachments]
        for a in ids:
            for b in ids:
                if a != b:
                    adj[a].add(b)
    return adj


def reachable_from(graph: CircuitGraph, start_ids: set[str]) -> set[str]:
    adj = _adjacency(graph)
    seen: set[str] = set()
    queue: deque[str] = deque(sorted(start_ids))
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        for nxt in sorted(adj.get(current, ())):
            if nxt not in seen:
                queue.append(nxt)
    return seen


def rule_output_on_active_path(graph: CircuitGraph) -> list[ValidationIssue]:
    """The output port must reach at least one active device, and that region
    must also touch the supply/ground rails (a crude 'driven output' check)."""
    outputs = graph.nodes_of_type(DeviceType.OUTPUT_PORT)
    if not outputs:
        return []  # MISSING_OUTPUT_PORT already reported by rule_required_ports
    issues: list[ValidationIssue] = []
    reach = reachable_from(graph, {n.node_id for n in outputs})
    active_reached = [
        nid for nid in reach if graph.nodes[nid].device_type in ACTIVE_DEVICE_TYPES
    ]
    if not active_reached:
        issues.append(
            ValidationIssue(
                "OUTPUT_NOT_DRIVEN",
                "output port does not reach any active device",
                "error",
                node_ids=[n.node_id for n in outputs],
            )
        )
        return issues
    rail_ids = {
        n.node_id
        for t in (DeviceType.SUPPLY_PORT, DeviceType.GROUND_PORT)
        for n in graph.nodes_of_type(t)
    }
    if rail_ids and not (reach & rail_ids):
        issues.append(
            ValidationIssue(
                "OUTPUT_PATH_UNPOWERED",
                "output-connected region does not reach supply/ground rails",
                "warning",
                node_ids=[n.node_id for n in outputs],
            )
        )
    return issues


def rule_disconnected_islands(graph: CircuitGraph) -> list[ValidationIssue]:
    """Warn when devices are unreachable from every port (isolated islands)."""
    port_ids = {
        n.node_id
        for n in graph.nodes.values()
        if n.device_type
        in (DeviceType.INPUT_PORT, DeviceType.OUTPUT_PORT, DeviceType.SUPPLY_PORT, DeviceType.GROUND_PORT)
    }
    if not port_ids:
        return []
    reach = reachable_from(graph, port_ids)
    isolated = sorted(set(graph.nodes) - reach)
    if not isolated:
        return []
    return [
        ValidationIssue(
            "ISOLATED_SUBGRAPH",
            f"{len(isolated)} node(s) unreachable from any port: {isolated}",
            "warning",
            node_ids=isolated,
        )
    ]
