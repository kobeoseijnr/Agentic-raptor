"""Neutral result schema for the external-baseline evaluation (STAGE 5).

Adapters convert baseline-native outputs into ExternalTopologyResult.
Rules (from the evaluation brief):
  * adapters may invoke, parse, convert and count -- never alter a baseline's
    algorithm;
  * a failed topology is recorded as failed; it is never rewritten into a
    valid one unless the baseline's own algorithm includes repair;
  * every raw output is preserved at raw_output_path.
Baselines are invoked via SUBPROCESS only -- this package never imports
baseline code (enforced by tests/test_external_baseline_leakage.py).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ExternalTopologyResult:
    baseline: str
    spec_id: str
    seed: int
    raw_output_path: str = ""
    netlist_path: str = ""
    graph_path: str = ""
    valid_syntax: bool | None = None
    valid_graph: bool | None = None
    simulatable: bool | None = None
    topology_hash: str = ""
    num_devices: int | None = None
    generation_runtime_s: float | None = None
    llm_calls: int | None = None
    llm_tokens: int | None = None
    spice_calls: int | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def netlist_topology_hash(netlist_text: str) -> str:
    """Canonical-ish hash for dedup/diversity: device lines only, whitespace
    normalized, parameter values stripped (topology = structure, not sizing)."""
    lines = []
    for ln in netlist_text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("*", "//", ".title", ".TITLE")):
            continue
        if s[0].upper() in "MRCLVIQDXE" or s.lower().startswith((".subckt", ".ends")):
            toks = [t for t in s.split() if "=" not in t]
            lines.append(" ".join(toks).upper())
    lines.sort()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def count_devices(netlist_text: str) -> int:
    n = 0
    for ln in netlist_text.splitlines():
        s = ln.strip()
        if s and s[0].upper() in "MRCLQDX" and not s.startswith("."):
            n += 1
    return n


def _net_tokens(toks, kind, n_nets):
    """Normalize net tokens: strip PANDA-style pin annotations Vx(I|O) -> VX,
    and cut X-lines at the '/' cell-name delimiter."""
    if "/" in toks:
        toks = toks[:toks.index("/")]
    nets = [t.split("(")[0].upper() for t in toks[1:1 + n_nets]]
    return [n for n in nets if n]


def common_netlist_graph_check(netlist_text: str) -> tuple[bool, str]:
    """Neutral STRUCTURAL check applied uniformly to every external netlist
    (pilot finding #3). Deliberately simulator- and PDK-agnostic:
      * at least one active device (M/Q);
      * no dangling internal net (a non-port net touched by exactly one
        terminal);
      * every device line has plausibly enough terminals.
    Never repairs anything -- verdict + reason only."""
    devices = []
    net_use: dict[str, int] = {}
    ports = {"0", "GND", "VSS", "VDD", "VDDA", "GNDA", "VIN", "VINP", "VINN",
             "VOUT", "VOUTP", "VOUTN", "VBIAS", "VB", "CLK", "VCM", "VDD!", "GND!"}
    # declared subcircuit ports are ports by definition
    for ln in netlist_text.splitlines():
        t = ln.strip()
        if t.lower().startswith(".subckt"):
            ports |= {x.split("(")[0].upper() for x in t.split()[2:]}
    for ln in netlist_text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("*", ".", "//", "+")):
            continue
        kind = s[0].upper()
        toks = [t for t in s.split() if "=" not in t]
        need = {"M": 5, "Q": 4, "R": 3, "C": 3, "L": 3, "V": 3, "I": 3,
                "D": 3, "X": 3, "E": 5}.get(kind)
        if need is None:
            continue
        if len(toks) < need:
            return False, f"device line too short: {s[:60]}"
        n_nets = {"M": 4, "Q": 3, "R": 2, "C": 2, "L": 2, "V": 2, "I": 2,
                  "D": 2, "E": 4}.get(kind, max(2, len(toks) - 2))
        nets = _net_tokens(toks, kind, n_nets)
        devices.append((kind, nets))
        for n in nets:
            net_use[n] = net_use.get(n, 0) + 1
    if not any(k in ("M", "Q", "X") for k, _ in devices):
        # X-instances count as active: subcircuit-library netlists (e.g.
        # PANDA's cell library) express all devices as X lines
        return False, "no active device"
    dangling = [n for n, c in net_use.items()
                if c == 1 and n not in ports and not n.endswith("!")]
    if dangling:
        return False, f"dangling internal nets: {dangling[:4]}"
    return True, "ok"


def topology_iso_key(netlist_text: str) -> str:
    """Isomorphism-approximating canonical key (pilot finding #5): the
    multiset of (device kind, sorted degree profile of its nets), which is
    invariant to net renaming and line order. Two circuits with the same key
    are structurally indistinguishable at this granularity; a full VF2 check
    can refine ties in the paper phase (documented approximation)."""
    net_use: dict[str, int] = {}
    devs: list[tuple[str, list[str]]] = []
    for ln in netlist_text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("*", ".", "//", "+")):
            continue
        kind = s[0].upper()
        if kind not in "MQRCLVIDXE":
            continue
        toks = [t for t in s.split() if "=" not in t]
        n_nets = {"M": 4, "Q": 3, "R": 2, "C": 2, "L": 2, "V": 2, "I": 2,
                  "D": 2, "E": 4}.get(kind, max(2, len(toks) - 2))
        nets = _net_tokens(toks, kind, n_nets)
        devs.append((kind, nets))
        for n in nets:
            net_use[n] = net_use.get(n, 0) + 1
    sig = sorted(f"{k}:{'-'.join(str(net_use[n]) for n in sorted(nets, key=lambda x: net_use[x]))}"
                 for k, nets in devs)
    return hashlib.sha256("|".join(sig).encode()).hexdigest()[:16]


def write_results(rows: list[ExternalTopologyResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")
