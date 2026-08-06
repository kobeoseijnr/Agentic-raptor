"""AnalogGym amplifier extractor: device-level SPICE .subckt → CircuitGraph."""

from __future__ import annotations

import re
from pathlib import Path

from tools.topology_extractor.common import (
    CircuitEdge, CircuitGraph, CircuitNode, DeviceType, ExtractedTopology, TerminalType,
)

_MOS_TERMS = (TerminalType.DRAIN, TerminalType.GATE, TerminalType.SOURCE, TerminalType.BULK)
_PORT_MAP = {"vinp": DeviceType.INPUT_PORT, "vinn": DeviceType.INPUT_PORT,
             "vout": DeviceType.OUTPUT_PORT, "vdda": DeviceType.SUPPLY_PORT,
             "gnda": DeviceType.GROUND_PORT}


def parse_netlist(path: Path) -> CircuitGraph:
    g = CircuitGraph(path.stem)
    edge_n = 0

    def attach(nid: str, term: TerminalType, net: str) -> None:
        nonlocal edge_n
        edge_n += 1
        g.add_edge(CircuitEdge(f"e{edge_n}", nid, term, net.lower()))

    ports_added = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        low = line.lower()
        if low.startswith(".subckt") and not ports_added:
            for p in low.split()[2:]:
                if p in _PORT_MAP:
                    g.add_node(CircuitNode(f"port_{p}", _PORT_MAP[p]))
                    attach(f"port_{p}", TerminalType.PORT, p)
            ports_added = True
        elif low.startswith("xm") or low.startswith("m"):
            t = low.split()
            if len(t) < 6:
                continue
            name, nets, model = t[0], t[1:5], t[5]
            dev = DeviceType.PMOS if "pfet" in model or "pmos" in model else DeviceType.NMOS
            role = None
            m = re.search(r"_(gm\w*|bias\w*|load\w*)_", low)
            if m:
                role = m.group(1)
            g.add_node(CircuitNode(name, dev, block_role=role, attributes={"model": model}))
            for term, net in zip(_MOS_TERMS, nets, strict=True):
                attach(name, term, net)
        elif low and low[0] in "rc" and not low.startswith(".") and len(low.split()) >= 3:
            t = low.split()
            dev = DeviceType.RESISTOR if low[0] == "r" else DeviceType.CAPACITOR
            g.add_node(CircuitNode(t[0], dev))
            attach(t[0], TerminalType.PLUS, t[1])
            attach(t[0], TerminalType.MINUS, t[2])
    g.metadata.circuit_family = "multi_stage_opamp"
    g.metadata.source = "analoggym"
    return g


def extract(analoggym_root: Path) -> list[ExtractedTopology]:
    amp = analoggym_root / "AnalogGym" / "Amplifier"
    out: list[ExtractedTopology] = []
    for netlist in sorted((amp / "spice_netlist").iterdir()):
        if not netlist.is_file():
            continue
        try:
            graph = parse_netlist(netlist)
        except Exception as exc:  # malformed → recorded, not fabricated
            out.append(ExtractedTopology(netlist.stem, "analoggym", CircuitGraph(netlist.stem),
                                         mapping_status="skipped", metadata={"error": str(exc)}))
            continue
        png = amp / "schematic" / f"{netlist.stem}.png"
        out.append(ExtractedTopology(
            name=netlist.stem, repository="analoggym", graph=graph,
            netlist_path=str(netlist), schematic_path=str(png) if png.is_file() else None,
            metadata={"technology": "sky130", "license": "AnalogGym repo license",
                      "paper": netlist.stem.split("_")[0]},
        ))
    return out
