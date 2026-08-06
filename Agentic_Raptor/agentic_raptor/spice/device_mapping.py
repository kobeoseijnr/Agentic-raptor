"""Device-level mapping from typed CircuitGraph nodes to SPICE card fragments.

Responsibilities: stable device names, SPICE terminal ordering, node-name
sanitisation, model names, and sizing-parameter emission. No I/O here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_raptor.core.circuit_graph import CircuitGraph, CircuitNode
from agentic_raptor.core.types import DeviceType, TerminalType
from agentic_raptor.utils.exceptions import SimulationError

#: SPICE element prefix per device type (stable naming: <prefix>_<node_id>).
_ELEMENT_PREFIX: dict[DeviceType, str] = {
    DeviceType.NMOS: "M",
    DeviceType.PMOS: "M",
    DeviceType.RESISTOR: "R",
    DeviceType.CAPACITOR: "C",
    DeviceType.CURRENT_SOURCE: "I",
    DeviceType.VOLTAGE_SOURCE: "V",
    DeviceType.SUBCIRCUIT_BLOCK: "X",
}

#: SPICE terminal emission order per device type.
SPICE_TERMINAL_ORDER: dict[DeviceType, tuple[TerminalType, ...]] = {
    DeviceType.NMOS: (TerminalType.DRAIN, TerminalType.GATE, TerminalType.SOURCE, TerminalType.BULK),
    DeviceType.PMOS: (TerminalType.DRAIN, TerminalType.GATE, TerminalType.SOURCE, TerminalType.BULK),
    DeviceType.RESISTOR: (TerminalType.PLUS, TerminalType.MINUS),
    DeviceType.CAPACITOR: (TerminalType.PLUS, TerminalType.MINUS),
    DeviceType.CURRENT_SOURCE: (TerminalType.PLUS, TerminalType.MINUS),
    DeviceType.VOLTAGE_SOURCE: (TerminalType.PLUS, TerminalType.MINUS),
}

#: Default sizing values used when a parameter is missing. These are the
#: documented mid-range defaults of the sizing parameter space (log-mid),
#: emitted with a "default" marker comment — never silently invented.
DEFAULT_SIZING: dict[DeviceType, dict[str, float]] = {
    DeviceType.NMOS: {"width_m": 4.47e-6, "length_m": 0.5e-6, "multiplier": 1.0},
    DeviceType.PMOS: {"width_m": 6.32e-6, "length_m": 0.5e-6, "multiplier": 1.0},
    DeviceType.RESISTOR: {"resistance_ohm": 10_000.0},
    DeviceType.CAPACITOR: {"capacitance_f": 3.16e-13},
    DeviceType.CURRENT_SOURCE: {"current_a": 2.24e-5},
    DeviceType.VOLTAGE_SOURCE: {"voltage_v": 0.0},
}


def sanitize_node_name(net: str) -> str:
    """SPICE-safe node name: alphanumerics and underscore only."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", net.strip())
    return cleaned or "_"


@dataclass
class NetNameMap:
    """Deterministic net → SPICE node mapping. Ground nets map to node 0."""

    mapping: dict[str, str]

    @classmethod
    def from_graph(cls, graph: CircuitGraph) -> NetNameMap:
        ground_nets: set[str] = set()
        for node in graph.nodes_of_type(DeviceType.GROUND_PORT):
            net = graph.net_of(node.node_id, TerminalType.PORT)
            if net:
                ground_nets.add(net)
        mapping: dict[str, str] = {}
        used: set[str] = {"0"}
        for net in sorted(graph.nets()):
            if net in ground_nets:
                mapping[net] = "0"
                continue
            name = sanitize_node_name(net)
            base, i = name, 2
            while name in used:
                name = f"{base}_{i}"
                i += 1
            used.add(name)
            mapping[net] = name
        return cls(mapping)

    def node(self, net: str) -> str:
        try:
            return self.mapping[net]
        except KeyError as exc:
            raise SimulationError(f"net {net!r} has no SPICE node mapping") from exc


def _sizing(node: CircuitNode, sizing_state: dict[str, dict[str, float]]) -> tuple[dict[str, float], bool]:
    """(effective sizing for node, used_defaults). Explicit values win."""
    explicit = {**node.sizing_parameters, **sizing_state.get(node.node_id, {})}
    defaults = DEFAULT_SIZING.get(node.device_type, {})
    merged = {**defaults, **explicit}
    used_defaults = any(k not in explicit for k in defaults)
    return merged, used_defaults


def device_card(
    node: CircuitNode,
    graph: CircuitGraph,
    net_map: NetNameMap,
    sizing_state: dict[str, dict[str, float]],
    nmos_model: str,
    pmos_model: str,
) -> str:
    """One SPICE card for one device node. Raises SimulationError on gaps."""
    dev = node.device_type
    prefix = _ELEMENT_PREFIX.get(dev)
    if prefix is None:
        raise SimulationError(f"device type {dev.value} has no SPICE element mapping")
    name = f"{prefix}_{sanitize_node_name(node.node_id)}"

    order = SPICE_TERMINAL_ORDER.get(dev)
    if order is None and dev is not DeviceType.SUBCIRCUIT_BLOCK:
        raise SimulationError(f"no terminal order for {dev.value}")

    def terminal_node(terminal: TerminalType) -> str:
        net = graph.net_of(node.node_id, terminal)
        if net is None:
            raise SimulationError(
                f"device {node.node_id!r} terminal {terminal.value!r} is unconnected; "
                "netlist emission requires fully connected devices"
            )
        return net_map.node(net)

    sizing, used_defaults = _sizing(node, sizing_state)
    marker = "  ; defaults" if used_defaults else ""

    if dev in (DeviceType.NMOS, DeviceType.PMOS):
        nodes = " ".join(terminal_node(t) for t in order)  # type: ignore[union-attr]
        model = nmos_model if dev is DeviceType.NMOS else pmos_model
        return (
            f"{name} {nodes} {model} "
            f"W={sizing['width_m']:.6g} L={sizing['length_m']:.6g} M={int(round(sizing['multiplier']))}{marker}"
        )
    if dev is DeviceType.RESISTOR:
        nodes = " ".join(terminal_node(t) for t in order)  # type: ignore[union-attr]
        return f"{name} {nodes} {sizing['resistance_ohm']:.6g}{marker}"
    if dev is DeviceType.CAPACITOR:
        nodes = " ".join(terminal_node(t) for t in order)  # type: ignore[union-attr]
        return f"{name} {nodes} {sizing['capacitance_f']:.6g}{marker}"
    if dev is DeviceType.CURRENT_SOURCE:
        nodes = " ".join(terminal_node(t) for t in order)  # type: ignore[union-attr]
        return f"{name} {nodes} DC {sizing['current_a']:.6g}{marker}"
    if dev is DeviceType.VOLTAGE_SOURCE:
        nodes = " ".join(terminal_node(t) for t in order)  # type: ignore[union-attr]
        return f"{name} {nodes} DC {sizing['voltage_v']:.6g}{marker}"
    if dev is DeviceType.SUBCIRCUIT_BLOCK:
        subckt = node.attributes.get("subckt_name")
        if not subckt:
            raise SimulationError(
                f"SUBCIRCUIT_BLOCK {node.node_id!r} needs attributes['subckt_name']"
            )
        pins = node.attributes.get("pin_nets")
        if not isinstance(pins, list) or not pins:
            raise SimulationError(
                f"SUBCIRCUIT_BLOCK {node.node_id!r} needs attributes['pin_nets'] (ordered net list)"
            )
        nodes = " ".join(net_map.node(str(p)) for p in pins)
        return f"{name} {nodes} {subckt}{marker}"
    raise SimulationError(f"unhandled device type {dev.value}")  # pragma: no cover
