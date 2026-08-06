"""Functional block decomposition over canonical CircuitGraphs.

Device-level graphs (AnalogGym): structural detectors for differential pair,
current mirror, cascode, tail current source, compensation network, load, bias
network, gain/output stage (name-role hints from the netlist reinforce the
structural rules, never replace them).
Block-level graphs (OPAMP-Generator/CktGNN): node roles ARE the blocks.
"""

from __future__ import annotations

from collections import defaultdict

from tools.topology_extractor.common import CircuitGraph, DeviceType, TerminalType


def detect_blocks(graph: CircuitGraph) -> dict[str, list[list[str]]]:
    blocks: dict[str, list[list[str]]] = defaultdict(list)
    mos = [n for n in graph.nodes.values() if n.device_type in (DeviceType.NMOS, DeviceType.PMOS)]
    if not mos:  # block-level graph
        for n in graph.nodes.values():
            if n.block_role:
                blocks[n.block_role].append([n.node_id])
        return dict(blocks)

    net_of = graph.net_of
    src = {m.node_id: net_of(m.node_id, TerminalType.SOURCE) for m in mos}
    gate = {m.node_id: net_of(m.node_id, TerminalType.GATE) for m in mos}
    drain = {m.node_id: net_of(m.node_id, TerminalType.DRAIN) for m in mos}
    rails = set()
    for pt in (DeviceType.SUPPLY_PORT, DeviceType.GROUND_PORT):
        for p in graph.nodes_of_type(pt):
            rails.add(net_of(p.node_id, TerminalType.PORT))
    inputs = {net_of(p.node_id, TerminalType.PORT) for p in graph.nodes_of_type(DeviceType.INPUT_PORT)}
    out_nets = {net_of(p.node_id, TerminalType.PORT) for p in graph.nodes_of_type(DeviceType.OUTPUT_PORT)}

    by_source: dict[str, list] = defaultdict(list)
    for m in mos:
        if src[m.node_id] not in rails:
            by_source[src[m.node_id]].append(m)
    for net, group in by_source.items():
        pairs = [m for m in group if gate[m.node_id] not in rails]
        if len(pairs) >= 2 and any(gate[m.node_id] in inputs for m in pairs):
            blocks["differential_pair"].append(sorted(m.node_id for m in pairs[:2]))
            tails = [m for m in mos if drain[m.node_id] == net]
            if tails:
                blocks["tail_current_source"].append([tails[0].node_id])

    by_gate: dict[str, list] = defaultdict(list)
    for m in mos:
        by_gate[gate[m.node_id]].append(m)
    for net, group in by_gate.items():
        diode = [m for m in group if drain[m.node_id] == net]
        if diode and len(group) >= 2:
            blocks["current_mirror"].append(sorted(m.node_id for m in group))
    for m in mos:  # cascode: source stacked on same-type drain (both off-rail)
        if src[m.node_id] in rails:
            continue
        below = [x for x in mos if x.device_type == m.device_type and drain[x.node_id] == src[m.node_id]]
        if below:
            blocks["cascode"].append(sorted([m.node_id, below[0].node_id]))
    for c in graph.nodes_of_type(DeviceType.CAPACITOR):
        nets = {net_of(c.node_id, TerminalType.PLUS), net_of(c.node_id, TerminalType.MINUS)}
        if nets & out_nets and nets & rails:
            blocks["load"].append([c.node_id])
        elif not nets & rails:
            blocks["compensation_network"].append([c.node_id])
    for m in mos:  # role hints from netlist parameter names
        role = (m.block_role or "").lower()
        if role.startswith("bias"):
            blocks["bias_network"].append([m.node_id])
        elif role.startswith("gm"):
            blocks["gain_stage"].append([m.node_id])
        elif role.startswith("load"):
            blocks["load"].append([m.node_id])
    for m in mos:
        if drain[m.node_id] in out_nets:
            blocks["output_stage"].append([m.node_id])
    return dict(blocks)


def block_signature(blocks: dict[str, list[list[str]]]) -> tuple:
    """Cluster signature: sorted (block, count-bucket) pairs."""
    def bucket(n: int) -> int:
        return 1 if n <= 1 else 2 if n <= 3 else 3
    return tuple(sorted((k, bucket(len(v))) for k, v in blocks.items()))
