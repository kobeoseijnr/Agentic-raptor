"""Typed topology-edit actions.

Every action:
1. has a typed schema (:data:`ACTION_SCHEMAS` — required parameter names/types);
2. checks preconditions (:func:`check_preconditions`);
3. preserves a reversible edit record (:class:`EditRecord` stores the prior
   serialized graph — simple and always correct at scaffold graph sizes);
4. triggers validation (done by the environment after each apply);
5. yields an invalid-action penalty when illegal (environment responsibility);
6. never changes continuous transistor sizes — sizing is the MB-SAC level.

Macro-actions (bias branch, gain stage, compensation) tag created nodes with
``block_role`` so the matching REMOVE_* actions can find them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentic_raptor.core.circuit_graph import CircuitEdge, CircuitGraph, CircuitNode
from agentic_raptor.core.types import DeviceType, TerminalType
from agentic_raptor.utils.exceptions import ActionError


class ActionType(str, Enum):
    ADD_DEVICE = "ADD_DEVICE"
    REMOVE_DEVICE = "REMOVE_DEVICE"
    CONNECT_TERMINALS = "CONNECT_TERMINALS"
    DISCONNECT_TERMINALS = "DISCONNECT_TERMINALS"
    REPLACE_DEVICE_TYPE = "REPLACE_DEVICE_TYPE"
    ADD_BIAS_BRANCH = "ADD_BIAS_BRANCH"
    REMOVE_BIAS_BRANCH = "REMOVE_BIAS_BRANCH"
    ADD_COMPENSATION_CAPACITOR = "ADD_COMPENSATION_CAPACITOR"
    REMOVE_COMPENSATION_COMPONENT = "REMOVE_COMPENSATION_COMPONENT"
    ADD_GAIN_STAGE = "ADD_GAIN_STAGE"
    REMOVE_GAIN_STAGE = "REMOVE_GAIN_STAGE"
    CHANGE_BLOCK_CONNECTION = "CHANGE_BLOCK_CONNECTION"
    TERMINATE = "TERMINATE"


#: Required parameters per action type (typed schema).
ACTION_SCHEMAS: dict[ActionType, dict[str, type]] = {
    ActionType.ADD_DEVICE: {"device_type": str, "connections": dict},
    ActionType.REMOVE_DEVICE: {"node_id": str},
    ActionType.CONNECT_TERMINALS: {"node_id": str, "terminal": str, "net": str},
    ActionType.DISCONNECT_TERMINALS: {"node_id": str, "terminal": str},
    ActionType.REPLACE_DEVICE_TYPE: {"node_id": str, "new_device_type": str},
    ActionType.ADD_BIAS_BRANCH: {"from_net": str, "to_net": str},
    ActionType.REMOVE_BIAS_BRANCH: {"node_id": str},
    ActionType.ADD_COMPENSATION_CAPACITOR: {"net_a": str, "net_b": str},
    ActionType.REMOVE_COMPENSATION_COMPONENT: {"node_id": str},
    ActionType.ADD_GAIN_STAGE: {"input_net": str, "output_net": str},
    ActionType.REMOVE_GAIN_STAGE: {"stage_tag": str},
    ActionType.CHANGE_BLOCK_CONNECTION: {"node_id": str, "terminal": str, "new_net": str},
    ActionType.TERMINATE: {},
}

#: Type-compatible replacements for REPLACE_DEVICE_TYPE (same terminal sets).
_REPLACEABLE: dict[DeviceType, frozenset[DeviceType]] = {
    DeviceType.NMOS: frozenset({DeviceType.PMOS}),
    DeviceType.PMOS: frozenset({DeviceType.NMOS}),
    DeviceType.RESISTOR: frozenset({DeviceType.CAPACITOR, DeviceType.CURRENT_SOURCE}),
    DeviceType.CAPACITOR: frozenset({DeviceType.RESISTOR}),
    DeviceType.CURRENT_SOURCE: frozenset({DeviceType.RESISTOR, DeviceType.VOLTAGE_SOURCE}),
    DeviceType.VOLTAGE_SOURCE: frozenset({DeviceType.CURRENT_SOURCE}),
}


@dataclass(frozen=True)
class TopologyAction:
    """One concrete, parameterized topology edit."""

    action_type: ActionType
    params: dict[str, Any] = field(default_factory=dict)

    def validate_schema(self) -> None:
        schema = ACTION_SCHEMAS[self.action_type]
        for name, expected in schema.items():
            if name not in self.params:
                raise ActionError(f"{self.action_type.value}: missing parameter {name!r}")
            if not isinstance(self.params[name], expected):
                raise ActionError(
                    f"{self.action_type.value}: parameter {name!r} must be {expected.__name__}"
                )
        extra = set(self.params) - set(schema)
        if extra:
            raise ActionError(f"{self.action_type.value}: unexpected parameters {sorted(extra)}")

    def key(self) -> str:
        """Stable identity string (used for MCTS child keys and dedup)."""
        parts = [self.action_type.value] + [f"{k}={self.params[k]}" for k in sorted(self.params)]
        return "|".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"action_type": self.action_type.value, "params": copy.deepcopy(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopologyAction:
        return cls(ActionType(data["action_type"]), dict(data.get("params") or {}))


@dataclass
class EditRecord:
    """Reversible record of one applied edit."""

    action: TopologyAction
    prior_graph: dict[str, Any]
    created_node_ids: list[str] = field(default_factory=list)
    removed_node_ids: list[str] = field(default_factory=list)

    def undo(self) -> CircuitGraph:
        return CircuitGraph.from_dict(self.prior_graph)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "created_node_ids": list(self.created_node_ids),
            "removed_node_ids": list(self.removed_node_ids),
        }


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
def check_preconditions(graph: CircuitGraph, action: TopologyAction) -> tuple[bool, str]:
    """Return (legal, reason). Never mutates the graph."""
    try:
        action.validate_schema()
    except ActionError as exc:
        return False, str(exc)

    p = action.params
    nets = graph.nets()
    at = action.action_type

    if at == ActionType.TERMINATE:
        return True, "always legal"

    if at == ActionType.ADD_DEVICE:
        try:
            dev = DeviceType(p["device_type"])
        except ValueError:
            return False, f"unsupported device type {p['device_type']!r}"
        if dev in (DeviceType.INPUT_PORT, DeviceType.OUTPUT_PORT, DeviceType.SUPPLY_PORT, DeviceType.GROUND_PORT):
            return False, "ports cannot be added by edit actions"
        for terminal, net in p["connections"].items():
            try:
                TerminalType(terminal)
            except ValueError:
                return False, f"unknown terminal {terminal!r}"
            if net not in nets:
                return False, f"connection targets unknown net {net!r}"
        return True, "ok"

    if at in (ActionType.REMOVE_DEVICE, ActionType.REMOVE_BIAS_BRANCH, ActionType.REMOVE_COMPENSATION_COMPONENT):
        node = graph.nodes.get(p["node_id"])
        if node is None:
            return False, f"unknown node {p['node_id']!r}"
        if node.device_type in (
            DeviceType.INPUT_PORT,
            DeviceType.OUTPUT_PORT,
            DeviceType.SUPPLY_PORT,
            DeviceType.GROUND_PORT,
        ):
            return False, "ports cannot be removed"
        if at == ActionType.REMOVE_BIAS_BRANCH and node.block_role != "bias_branch":
            return False, f"node {p['node_id']!r} is not a bias branch"
        if at == ActionType.REMOVE_COMPENSATION_COMPONENT and node.block_role != "compensation":
            return False, f"node {p['node_id']!r} is not a compensation component"
        return True, "ok"

    if at == ActionType.CONNECT_TERMINALS:
        node = graph.nodes.get(p["node_id"])
        if node is None:
            return False, f"unknown node {p['node_id']!r}"
        try:
            terminal = TerminalType(p["terminal"])
        except ValueError:
            return False, f"unknown terminal {p['terminal']!r}"
        if terminal not in node.terminals:
            return False, f"terminal {p['terminal']!r} invalid for {node.device_type.value}"
        if graph.net_of(node.node_id, terminal) is not None:
            return False, "terminal already connected (disconnect first)"
        return True, "ok"

    if at == ActionType.DISCONNECT_TERMINALS:
        node = graph.nodes.get(p["node_id"])
        if node is None:
            return False, f"unknown node {p['node_id']!r}"
        try:
            terminal = TerminalType(p["terminal"])
        except ValueError:
            return False, f"unknown terminal {p['terminal']!r}"
        if graph.net_of(node.node_id, terminal) is None:
            return False, "terminal is not connected"
        return True, "ok"

    if at == ActionType.REPLACE_DEVICE_TYPE:
        node = graph.nodes.get(p["node_id"])
        if node is None:
            return False, f"unknown node {p['node_id']!r}"
        try:
            new_type = DeviceType(p["new_device_type"])
        except ValueError:
            return False, f"unsupported device type {p['new_device_type']!r}"
        allowed = _REPLACEABLE.get(node.device_type, frozenset())
        if new_type not in allowed:
            return False, f"cannot replace {node.device_type.value} with {new_type.value}"
        return True, "ok"

    if at == ActionType.ADD_BIAS_BRANCH:
        for key in ("from_net", "to_net"):
            if p[key] not in nets:
                return False, f"unknown net {p[key]!r}"
        if p["from_net"] == p["to_net"]:
            return False, "bias branch endpoints must differ"
        return True, "ok"

    if at == ActionType.ADD_COMPENSATION_CAPACITOR:
        for key in ("net_a", "net_b"):
            if p[key] not in nets:
                return False, f"unknown net {p[key]!r}"
        if p["net_a"] == p["net_b"]:
            return False, "capacitor endpoints must differ"
        return True, "ok"

    if at == ActionType.ADD_GAIN_STAGE:
        for key in ("input_net", "output_net"):
            if p[key] not in nets:
                return False, f"unknown net {p[key]!r}"
        if p["input_net"] == p["output_net"]:
            return False, "gain stage input and output must differ"
        return True, "ok"

    if at == ActionType.REMOVE_GAIN_STAGE:
        tagged = [
            n.node_id
            for n in graph.nodes.values()
            if n.block_role == "gain_stage" and n.attributes.get("stage_tag") == p["stage_tag"]
        ]
        if not tagged:
            return False, f"no gain stage with tag {p['stage_tag']!r}"
        return True, "ok"

    if at == ActionType.CHANGE_BLOCK_CONNECTION:
        node = graph.nodes.get(p["node_id"])
        if node is None:
            return False, f"unknown node {p['node_id']!r}"
        try:
            terminal = TerminalType(p["terminal"])
        except ValueError:
            return False, f"unknown terminal {p['terminal']!r}"
        if graph.net_of(node.node_id, terminal) is None:
            return False, "terminal is not connected; use CONNECT_TERMINALS"
        if p["new_net"] not in nets:
            return False, f"unknown net {p['new_net']!r}"
        return True, "ok"

    return False, f"unhandled action type {at.value}"  # pragma: no cover


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def apply_action(graph: CircuitGraph, action: TopologyAction) -> tuple[CircuitGraph, EditRecord]:
    """Apply an action to a *copy* of the graph. Raises ActionError when illegal."""
    legal, reason = check_preconditions(graph, action)
    if not legal:
        raise ActionError(f"illegal action {action.key()}: {reason}")

    record = EditRecord(action=action, prior_graph=graph.to_dict())
    g = graph.copy()
    p = action.params
    at = action.action_type

    if at == ActionType.TERMINATE:
        return g, record

    if at == ActionType.ADD_DEVICE:
        node_id = g.next_id("dev")
        node = CircuitNode(node_id=node_id, device_type=DeviceType(p["device_type"]))
        g.add_node(node)
        for terminal, net in sorted(p["connections"].items()):
            g.add_edge(CircuitEdge(g.next_id("e"), node_id, TerminalType(terminal), str(net)))
        record.created_node_ids.append(node_id)

    elif at in (ActionType.REMOVE_DEVICE, ActionType.REMOVE_BIAS_BRANCH, ActionType.REMOVE_COMPENSATION_COMPONENT):
        g.remove_node(p["node_id"])
        record.removed_node_ids.append(p["node_id"])

    elif at == ActionType.CONNECT_TERMINALS:
        g.add_edge(CircuitEdge(g.next_id("e"), p["node_id"], TerminalType(p["terminal"]), p["net"]))

    elif at == ActionType.DISCONNECT_TERMINALS:
        terminal = TerminalType(p["terminal"])
        for edge in list(g.edges.values()):
            if edge.node_id == p["node_id"] and edge.terminal == terminal:
                g.remove_edge(edge.edge_id)

    elif at == ActionType.REPLACE_DEVICE_TYPE:
        node = g.node(p["node_id"])
        node.device_type = DeviceType(p["new_device_type"])
        node.sizing_parameters.clear()  # sizing is invalidated by a type change

    elif at == ActionType.ADD_BIAS_BRANCH:
        node_id = g.next_id("bias")
        g.add_node(
            CircuitNode(node_id=node_id, device_type=DeviceType.CURRENT_SOURCE, block_role="bias_branch")
        )
        g.add_edge(CircuitEdge(g.next_id("e"), node_id, TerminalType.PLUS, p["from_net"]))
        g.add_edge(CircuitEdge(g.next_id("e"), node_id, TerminalType.MINUS, p["to_net"]))
        record.created_node_ids.append(node_id)

    elif at == ActionType.ADD_COMPENSATION_CAPACITOR:
        node_id = g.next_id("cc")
        g.add_node(
            CircuitNode(node_id=node_id, device_type=DeviceType.CAPACITOR, block_role="compensation")
        )
        g.add_edge(CircuitEdge(g.next_id("e"), node_id, TerminalType.PLUS, p["net_a"]))
        g.add_edge(CircuitEdge(g.next_id("e"), node_id, TerminalType.MINUS, p["net_b"]))
        record.created_node_ids.append(node_id)

    elif at == ActionType.ADD_GAIN_STAGE:
        # Minimal common-source stage: NMOS driver + current-source load.
        tag = g.next_id("stage")
        internal_net = f"n_{tag}"
        mos_id = g.next_id("m")
        g.add_node(
            CircuitNode(
                node_id=mos_id,
                device_type=DeviceType.NMOS,
                block_role="gain_stage",
                attributes={"stage_tag": tag},
            )
        )
        load_id = g.next_id("il")
        g.add_node(
            CircuitNode(
                node_id=load_id,
                device_type=DeviceType.CURRENT_SOURCE,
                block_role="gain_stage",
                attributes={"stage_tag": tag},
            )
        )
        ground_net = _rail_net(g, DeviceType.GROUND_PORT) or p["input_net"]
        supply_net = _rail_net(g, DeviceType.SUPPLY_PORT) or p["output_net"]
        g.add_edge(CircuitEdge(g.next_id("e"), mos_id, TerminalType.GATE, p["input_net"]))
        g.add_edge(CircuitEdge(g.next_id("e"), mos_id, TerminalType.DRAIN, internal_net))
        g.add_edge(CircuitEdge(g.next_id("e"), mos_id, TerminalType.SOURCE, ground_net))
        g.add_edge(CircuitEdge(g.next_id("e"), mos_id, TerminalType.BULK, ground_net))
        g.add_edge(CircuitEdge(g.next_id("e"), load_id, TerminalType.PLUS, supply_net))
        g.add_edge(CircuitEdge(g.next_id("e"), load_id, TerminalType.MINUS, internal_net))
        # Couple the stage output onto the requested output net with a capacitor
        # (keeps the edit purely additive; direct net merging is a later stage).
        couple_id = g.next_id("ccpl")
        g.add_node(
            CircuitNode(
                node_id=couple_id,
                device_type=DeviceType.CAPACITOR,
                block_role="gain_stage",
                attributes={"stage_tag": tag},
            )
        )
        g.add_edge(CircuitEdge(g.next_id("e"), couple_id, TerminalType.PLUS, internal_net))
        g.add_edge(CircuitEdge(g.next_id("e"), couple_id, TerminalType.MINUS, p["output_net"]))
        record.created_node_ids += [mos_id, load_id, couple_id]

    elif at == ActionType.REMOVE_GAIN_STAGE:
        for node in list(g.nodes.values()):
            if node.block_role == "gain_stage" and node.attributes.get("stage_tag") == p["stage_tag"]:
                g.remove_node(node.node_id)
                record.removed_node_ids.append(node.node_id)

    elif at == ActionType.CHANGE_BLOCK_CONNECTION:
        terminal = TerminalType(p["terminal"])
        for edge in list(g.edges.values()):
            if edge.node_id == p["node_id"] and edge.terminal == terminal:
                g.remove_edge(edge.edge_id)
        g.add_edge(CircuitEdge(g.next_id("e"), p["node_id"], terminal, p["new_net"]))

    else:  # pragma: no cover
        raise ActionError(f"unhandled action type {at.value}")

    return g, record


def _rail_net(graph: CircuitGraph, port_type: DeviceType) -> str | None:
    # Sorted for determinism: the same net is chosen for a graph and its copy.
    for node in sorted(graph.nodes_of_type(port_type), key=lambda n: n.node_id):
        net = graph.net_of(node.node_id, TerminalType.PORT)
        if net:
            return net
    return None


# ---------------------------------------------------------------------------
# Bounded deterministic action enumeration
# ---------------------------------------------------------------------------
def enumerate_candidate_actions(graph: CircuitGraph, max_actions: int = 32) -> list[TopologyAction]:
    """Deterministic bounded set of *schema-legal* candidate actions.

    Enumerates from templates over the graph's current nets and role-tagged
    nodes, then filters by preconditions. TERMINATE is always first so index 0
    is stable across states (useful for masking).
    """
    actions: list[TopologyAction] = [TopologyAction(ActionType.TERMINATE)]
    nets = sorted(graph.nets())
    out_net = _port_net(graph, DeviceType.OUTPUT_PORT)
    gnd_net = _rail_net(graph, DeviceType.GROUND_PORT)
    vdd_net = _rail_net(graph, DeviceType.SUPPLY_PORT)
    in_net = _port_net(graph, DeviceType.INPUT_PORT)
    internal_nets = [n for n in nets if n not in {out_net, gnd_net, vdd_net, in_net}]

    def _add(a: TopologyAction) -> None:
        legal, _ = check_preconditions(graph, a)
        if legal and len(actions) < max_actions:
            actions.append(a)

    if out_net and gnd_net:
        _add(TopologyAction(ActionType.ADD_COMPENSATION_CAPACITOR, {"net_a": out_net, "net_b": gnd_net}))
    for net in internal_nets[:4]:
        if out_net:
            _add(TopologyAction(ActionType.ADD_COMPENSATION_CAPACITOR, {"net_a": net, "net_b": out_net}))
        if gnd_net:
            _add(TopologyAction(ActionType.ADD_COMPENSATION_CAPACITOR, {"net_a": net, "net_b": gnd_net}))
    if vdd_net:
        for net in internal_nets[:4]:
            _add(TopologyAction(ActionType.ADD_BIAS_BRANCH, {"from_net": vdd_net, "to_net": net}))
        if gnd_net:
            _add(TopologyAction(ActionType.ADD_BIAS_BRANCH, {"from_net": vdd_net, "to_net": gnd_net}))
    if in_net and out_net:
        _add(TopologyAction(ActionType.ADD_GAIN_STAGE, {"input_net": in_net, "output_net": out_net}))
    for node in sorted(graph.nodes.values(), key=lambda n: n.node_id):
        if node.block_role == "compensation":
            _add(TopologyAction(ActionType.REMOVE_COMPENSATION_COMPONENT, {"node_id": node.node_id}))
        elif node.block_role == "bias_branch":
            _add(TopologyAction(ActionType.REMOVE_BIAS_BRANCH, {"node_id": node.node_id}))
        elif node.block_role == "gain_stage" and node.device_type == DeviceType.NMOS:
            _add(TopologyAction(ActionType.REMOVE_GAIN_STAGE, {"stage_tag": str(node.attributes.get("stage_tag"))}))
    return actions


def _port_net(graph: CircuitGraph, port_type: DeviceType) -> str | None:
    return _rail_net(graph, port_type)
