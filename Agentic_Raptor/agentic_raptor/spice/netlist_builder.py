"""Typed CircuitGraph + sizing → simulator-ready SPICE netlist core.

Emits the *circuit* section (devices + models + header comments). The
testbench (sources, load, analyses, measurements) is added by
``testbench_builder`` so the same circuit can be reused across analyses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.types import DeviceType, TerminalType
from agentic_raptor.spice.device_mapping import NetNameMap, device_card
from agentic_raptor.spice.model_library import ModelLibrary
from agentic_raptor.utils.exceptions import SimulationError

#: Device types emitted as circuit elements (ports are testbench boundaries).
_CIRCUIT_DEVICES = (
    DeviceType.NMOS,
    DeviceType.PMOS,
    DeviceType.RESISTOR,
    DeviceType.CAPACITOR,
    DeviceType.CURRENT_SOURCE,
    DeviceType.VOLTAGE_SOURCE,
    DeviceType.SUBCIRCUIT_BLOCK,
)


@dataclass
class BuiltCircuit:
    """Circuit section plus the boundary information the testbench needs."""

    circuit_lines: list[str]
    net_map: NetNameMap
    supply_nodes: list[str]
    ground_node: str
    input_nodes: list[str]           # SPICE node names of INPUT_PORT nets, sorted by port id
    output_nodes: list[str]
    model_library: ModelLibrary
    header: list[str] = field(default_factory=list)

    def circuit_text(self) -> str:
        return "\n".join(self.header + [self.model_library.netlist_section(), ""] + self.circuit_lines)


def build_circuit(
    graph: CircuitGraph,
    sizing_state: dict[str, dict[str, float]],
    model_library: ModelLibrary,
    candidate_id: str = "unknown",
) -> BuiltCircuit:
    """Convert a typed graph into SPICE circuit cards.

    Raises :class:`SimulationError` for structures that cannot be emitted
    (unconnected terminals, unmapped devices, missing rails).
    """
    net_map = NetNameMap.from_graph(graph)

    def port_nodes(port_type: DeviceType) -> list[str]:
        nodes: list[str] = []
        for port in sorted(graph.nodes_of_type(port_type), key=lambda n: n.node_id):
            net = graph.net_of(port.node_id, TerminalType.PORT)
            if net is not None:
                nodes.append(net_map.node(net))
        return nodes

    supply_nodes = port_nodes(DeviceType.SUPPLY_PORT)
    ground_nets = port_nodes(DeviceType.GROUND_PORT)
    input_nodes = port_nodes(DeviceType.INPUT_PORT)
    output_nodes = port_nodes(DeviceType.OUTPUT_PORT)
    if not supply_nodes:
        raise SimulationError("no connected SUPPLY_PORT; cannot build a powered netlist")
    if not ground_nets:
        raise SimulationError("no connected GROUND_PORT; cannot reference node 0")
    if not output_nodes:
        raise SimulationError("no connected OUTPUT_PORT; nothing to measure")

    header = [
        f"* Agentic RAPTOR netlist  candidate={candidate_id}",
        f"* topology_id={graph.graph_id}  topology_hash={graph.structural_hash()}",
        f"* model_library={model_library.label}",
    ]

    circuit_lines: list[str] = []
    for node in sorted(graph.nodes.values(), key=lambda n: n.node_id):
        if node.device_type in _CIRCUIT_DEVICES:
            circuit_lines.append(
                device_card(
                    node,
                    graph,
                    net_map,
                    sizing_state,
                    nmos_model=model_library.nmos_model,
                    pmos_model=model_library.pmos_model,
                )
            )

    if not circuit_lines:
        raise SimulationError("graph contains no emittable circuit devices")

    return BuiltCircuit(
        circuit_lines=circuit_lines,
        net_map=net_map,
        supply_nodes=supply_nodes,
        ground_node="0",
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        model_library=model_library,
        header=header,
    )
