"""Individual validation rules. Each rule returns a list of ValidationIssue.

Structured results live in ``validator.py``; rules are pure functions over
:class:`~agentic_raptor.core.circuit_graph.CircuitGraph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.types import (
    EXPECTED_TERMINALS,
    PORT_DEVICE_TYPES,
    DeviceType,
    TerminalType,
)


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: Literal["error", "warning"]
    node_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)


def rule_non_empty(graph: CircuitGraph) -> list[ValidationIssue]:
    if not graph.nodes:
        return [ValidationIssue("EMPTY_GRAPH", "graph has no nodes", "error")]
    return []


def rule_required_ports(graph: CircuitGraph) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for dev_type, code in (
        (DeviceType.INPUT_PORT, "MISSING_INPUT_PORT"),
        (DeviceType.OUTPUT_PORT, "MISSING_OUTPUT_PORT"),
        (DeviceType.SUPPLY_PORT, "MISSING_SUPPLY_PORT"),
        (DeviceType.GROUND_PORT, "MISSING_GROUND_PORT"),
    ):
        if not graph.nodes_of_type(dev_type):
            issues.append(ValidationIssue(code, f"no {dev_type.value} node present", "error"))
    return issues


def rule_valid_terminals(graph: CircuitGraph) -> list[ValidationIssue]:
    """Device terminals must match the expected terminal set for the type.

    Construction-time checks in CircuitGraph already reject most violations;
    this re-checks defensively for graphs built through from_dict with edits.
    """
    issues: list[ValidationIssue] = []
    for node in graph.nodes.values():
        expected = set(EXPECTED_TERMINALS[node.device_type])
        actual = set(node.terminals)
        if actual != expected:
            issues.append(
                ValidationIssue(
                    "INVALID_TERMINAL_SET",
                    f"node {node.node_id!r} ({node.device_type.value}) has terminals "
                    f"{sorted(t.value for t in actual)} but expects {sorted(t.value for t in expected)}",
                    "error",
                    node_ids=[node.node_id],
                )
            )
    for edge in graph.edges.values():
        node = graph.nodes.get(edge.node_id)
        if node is not None and edge.terminal not in node.terminals:
            issues.append(
                ValidationIssue(
                    "IMPOSSIBLE_TERMINAL_CONNECTION",
                    f"edge {edge.edge_id!r} attaches nonexistent terminal {edge.terminal.value!r} "
                    f"of node {edge.node_id!r}",
                    "error",
                    node_ids=[edge.node_id],
                    edge_ids=[edge.edge_id],
                )
            )
    return issues


def rule_no_floating_devices(graph: CircuitGraph) -> list[ValidationIssue]:
    """A non-port device with zero attached terminals is an error; a partially
    attached device (some terminals unconnected) is a warning."""
    issues: list[ValidationIssue] = []
    attach_counts: dict[str, int] = {nid: 0 for nid in graph.nodes}
    for edge in graph.edges.values():
        attach_counts[edge.node_id] = attach_counts.get(edge.node_id, 0) + 1
    for node in graph.nodes.values():
        if node.device_type in PORT_DEVICE_TYPES:
            if attach_counts.get(node.node_id, 0) == 0:
                issues.append(
                    ValidationIssue(
                        "FLOATING_PORT",
                        f"port {node.node_id!r} is not attached to any net",
                        "error",
                        node_ids=[node.node_id],
                    )
                )
            continue
        count = attach_counts.get(node.node_id, 0)
        if count == 0:
            issues.append(
                ValidationIssue(
                    "FLOATING_DEVICE",
                    f"device {node.node_id!r} ({node.device_type.value}) has no connections",
                    "error",
                    node_ids=[node.node_id],
                )
            )
        elif count < len(node.terminals):
            issues.append(
                ValidationIssue(
                    "PARTIALLY_CONNECTED_DEVICE",
                    f"device {node.node_id!r} has {count}/{len(node.terminals)} terminals connected",
                    "warning",
                    node_ids=[node.node_id],
                )
            )
    return issues


def rule_supply_ground_short(graph: CircuitGraph) -> list[ValidationIssue]:
    """Detect nets that attach both a supply port and a ground port directly."""
    issues: list[ValidationIssue] = []
    supply_ids = {n.node_id for n in graph.nodes_of_type(DeviceType.SUPPLY_PORT)}
    ground_ids = {n.node_id for n in graph.nodes_of_type(DeviceType.GROUND_PORT)}
    for net_name, attachments in graph.nets().items():
        attached_ids = {node_id for node_id, _ in attachments}
        if attached_ids & supply_ids and attached_ids & ground_ids:
            issues.append(
                ValidationIssue(
                    "SUPPLY_GROUND_SHORT",
                    f"net {net_name!r} directly connects supply and ground",
                    "error",
                    node_ids=sorted((attached_ids & supply_ids) | (attached_ids & ground_ids)),
                )
            )
    return issues


#: (device type, terminal) pairs that provide a low-impedance DC path into a net.
_LOW_IMPEDANCE: frozenset[tuple[DeviceType, TerminalType]] = frozenset(
    {
        (DeviceType.NMOS, TerminalType.DRAIN),
        (DeviceType.NMOS, TerminalType.SOURCE),
        (DeviceType.PMOS, TerminalType.DRAIN),
        (DeviceType.PMOS, TerminalType.SOURCE),
        (DeviceType.RESISTOR, TerminalType.PLUS),
        (DeviceType.RESISTOR, TerminalType.MINUS),
        (DeviceType.VOLTAGE_SOURCE, TerminalType.PLUS),
        (DeviceType.VOLTAGE_SOURCE, TerminalType.MINUS),
        (DeviceType.INPUT_PORT, TerminalType.PORT),
        (DeviceType.OUTPUT_PORT, TerminalType.PORT),
        (DeviceType.SUPPLY_PORT, TerminalType.PORT),
        (DeviceType.GROUND_PORT, TerminalType.PORT),
        (DeviceType.SUBCIRCUIT_BLOCK, TerminalType.BLOCK_PIN),
    }
)


def rule_dc_floating_nets(graph: CircuitGraph) -> list[ValidationIssue]:
    """Nets reachable only through gates/capacitors/current sources have no DC
    path — a guaranteed singular matrix in SPICE (e.g. a bias net driven by a
    current source into a MOS gate with no diode device)."""
    issues: list[ValidationIssue] = []
    for net_name, attachments in graph.nets().items():
        has_low_impedance = any(
            (graph.nodes[node_id].device_type, terminal) in _LOW_IMPEDANCE
            for node_id, terminal in attachments
        )
        if attachments and not has_low_impedance:
            issues.append(
                ValidationIssue(
                    "DC_FLOATING_NET",
                    f"net {net_name!r} has no DC path: only gates/capacitors/current sources "
                    "attach to it (add a diode-connected device, resistor, or channel terminal)",
                    "error",
                    node_ids=sorted({node_id for node_id, _t in attachments}),
                )
            )
    return issues


def rule_max_graph_size(graph: CircuitGraph, max_nodes: int, max_edges: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if len(graph.nodes) > max_nodes:
        issues.append(
            ValidationIssue("TOO_MANY_NODES", f"{len(graph.nodes)} nodes > limit {max_nodes}", "error")
        )
    if len(graph.edges) > max_edges:
        issues.append(
            ValidationIssue("TOO_MANY_EDGES", f"{len(graph.edges)} edges > limit {max_edges}", "error")
        )
    return issues
