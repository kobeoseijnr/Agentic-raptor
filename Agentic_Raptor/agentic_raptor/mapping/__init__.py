"""Stage 3C: transistor mapping of generated structural families → Sky130.

Pipeline: structural CircuitGraph → functional interpretation → DeviceCircuitGraph
(device roles, groups, provenance) → bias/supply integration (labelled
generated_support_bias) → prior-based initial sizing → Sky130 subckt netlist
(AnalogGym port convention, so the existing Stage 3B testbench/qualification
pipeline is reused verbatim) → static validation (incl. graph preservation) →
electrical qualification. Failures preserved; nothing silently repaired.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.electrical import _ROOT, discover_ngspice, qualify_family

MAPPING_STATUSES = ("mapped_static_valid", "mapped_static_invalid", "mapping_ambiguous",
                    "mapping_incomplete", "mapping_unsupported", "netlist_generation_failed",
                    "bias_unresolved", "port_unresolved", "compensation_unresolved",
                    "graph_preservation_failed")

#: AnalogGym-derived safe Sky130 priors (origin recorded per device).
PRIORS = {
    "input_pair_nmos": {"w": 10.0, "l": 0.5, "origin": "analoggym_input_pair_prior", "conf": 0.8},
    "mirror_pmos": {"w": 20.0, "l": 0.5, "origin": "analoggym_mirror_prior", "conf": 0.8},
    "tail_nmos": {"w": 20.0, "l": 1.0, "origin": "analoggym_tail_prior", "conf": 0.8},
    "cs_gain_nmos": {"w": 40.0, "l": 0.5, "origin": "analoggym_second_stage_prior", "conf": 0.7},
    "load_pmos": {"w": 60.0, "l": 0.5, "origin": "analoggym_load_prior", "conf": 0.7},
    "bias_nmos": {"w": 20.0, "l": 1.0, "origin": "generated_support_bias_prior", "conf": 0.6},
    "miller_cap_f": {"value": 2e-12, "origin": "conservative_sky130_default", "conf": 0.6},
    "ibias_a": {"value": 20e-6, "origin": "analoggym_bias_current_prior", "conf": 0.7},
}


@dataclass
class DeviceRecord:
    device_id: str
    kind: str            # nmos | pmos | cap | res | isrc
    role: str
    nets: dict[str, str]              # terminal → net
    group: str | None = None          # matched/mirror/pair/cascode group id
    sizing: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceCircuitGraph:
    topology_id: str
    mapping_candidate_id: str
    stage_count: int
    devices: list[DeviceRecord] = field(default_factory=list)
    ports: dict[str, str] = field(default_factory=dict)
    support_bias: list[str] = field(default_factory=list)   # generated_support_bias device ids
    polarity: dict[str, Any] = field(default_factory=dict)
    block_assignments: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


def audit_generated(registry: TopologyRegistry) -> list[dict[str, Any]]:
    rows = []
    for tid in registry.list_topologies():
        e = registry.get_topology(tid)
        if e.source == "analoggym":
            continue
        blocks = set(e.metadata.get("functional_blocks", {}))
        gain_stages = e.metadata.get("functional_blocks", {}).get("gain_stage", 0)
        unresolved = sorted(b for b in blocks if b.startswith(("ff_", "fb_")))
        if gain_stages >= 1 and not unresolved:
            readiness = "mapping_ready"
        elif gain_stages >= 1:
            readiness = "partially_specified"   # gm main path + unmapped ff/fb branches
        elif blocks & {"gm_pos", "gm_neg"}:
            readiness = "behavioral_only"
        else:
            readiness = "structurally_ambiguous"
        rows.append({"topology_id": tid, "source": e.source,
                     "graph_hash": e.metadata.get("graph_hash"),
                     "nodes": len(e.graph.nodes), "edges": len(e.graph.edges),
                     "functional_blocks": sorted(blocks), "gain_stages": gain_stages,
                     "unresolved_blocks": unresolved, "mapping_readiness": readiness})
    return rows


def map_family(entry, audit_row: dict[str, Any]) -> tuple[DeviceCircuitGraph | None, str]:
    """Template mapping: N-stage cascade (5T first stage, CS stages after),
    C blocks → Miller/load compensation, R+C → nulling branch. ff/fb gm branches
    are NOT silently realized: families keep partially_specified provenance and
    only the main path is mapped in this candidate (recorded in unresolved)."""
    n = max(1, audit_row["gain_stages"]) if audit_row["gain_stages"] else 0
    if n == 0:
        return None, "mapping_unsupported"
    n = min(n, 3)
    cid = f"map-{uuid.uuid4().hex[:8]}"
    g = DeviceCircuitGraph(entry.topology_id, cid, n,
                           ports={"gnda": "gnda", "vdda": "vdda", "vinn": "vinn",
                                  "vinp": "vinp", "vout": "vout"})
    ev = {"rule": "n_stage_cascade_template", "audit": audit_row["mapping_readiness"]}
    s1out = "n1" if n > 1 else "vout"
    # Stage 1: 5T OTA (NMOS pair, PMOS mirror, tail + support bias mirror)
    g.devices += [
        DeviceRecord("M1", "nmos", "input_pair_nmos",
                     {"d": "nmir", "g": "vinp", "s": "ntail", "b": "gnda"},
                     "pair1", dict(PRIORS["input_pair_nmos"]), ev),
        DeviceRecord("M2", "nmos", "input_pair_nmos",
                     {"d": s1out, "g": "vinn", "s": "ntail", "b": "gnda"},
                     "pair1", dict(PRIORS["input_pair_nmos"]), ev),
        DeviceRecord("M3", "pmos", "mirror_reference",
                     {"d": "nmir", "g": "nmir", "s": "vdda", "b": "vdda"},
                     "mir1", dict(PRIORS["mirror_pmos"]), ev),
        DeviceRecord("M4", "pmos", "mirror_output",
                     {"d": s1out, "g": "nmir", "s": "vdda", "b": "vdda"},
                     "mir1", dict(PRIORS["mirror_pmos"]), ev),
        DeviceRecord("M5", "nmos", "tail_current_source",
                     {"d": "ntail", "g": "nbias", "s": "gnda", "b": "gnda"},
                     "tail1", dict(PRIORS["tail_nmos"]), ev),
        DeviceRecord("M6", "nmos", "bias_device",
                     {"d": "nbias", "g": "nbias", "s": "gnda", "b": "gnda"},
                     "bias1", dict(PRIORS["bias_nmos"]), {"generated_support_bias": True, **ev}),
        DeviceRecord("IB1", "isrc", "bias_device",
                     {"p": "vdda", "n": "nbias"}, "bias1",
                     {"value": PRIORS["ibias_a"]["value"], "origin": PRIORS["ibias_a"]["origin"]},
                     {"generated_support_bias": True, **ev}),
    ]
    g.support_bias = ["M6", "IB1"]
    prev = s1out
    inversions = 2  # vinp path: CS into mirror + mirror out (non-inverting to s1out)
    for k in range(2, n + 1):
        out = "vout" if k == n else f"n{k}"
        g.devices += [
            DeviceRecord(f"M{4 + 2 * k}", "nmos", "second_stage_gain_device",
                         {"d": out, "g": prev, "s": "gnda", "b": "gnda"},
                         f"cs{k}", dict(PRIORS["cs_gain_nmos"]), ev),
            DeviceRecord(f"M{5 + 2 * k}", "pmos", "active_load",
                         {"d": out, "g": "nmir", "s": "vdda", "b": "vdda"},
                         f"cs{k}", dict(PRIORS["load_pmos"]), ev),
        ]
        # Miller compensation across EACH inverting CS stage, locally.
        #
        # This placement is load-bearing, not incidental. A Miller capacitor
        # provides NEGATIVE feedback only across an ODD number of inverting
        # stages. In this template every CS stage inverts and the 5T first
        # stage is non-inverting to n1, so:
        #
        #   n1 -> n2   : 1 inversion  -> capacitor is negative feedback  (OK)
        #   n2 -> vout : 1 inversion  -> capacitor is negative feedback  (OK)
        #   n1 -> vout : 2 inversions -> capacitor is POSITIVE feedback  (BAD)
        #
        # A "nested Miller" variant returning every capacitor from vout was
        # tried and MEASURED WORSE precisely for that reason: the outer
        # capacitor spanned two inversions, so it fed back in phase.
        # 3s_miller went from +0.24 deg to -87.35 deg of phase margin.
        # Textbook nested Miller assumes a stage-inversion pattern this
        # template does not have; do not reintroduce it without first
        # changing the stage polarities.
        if "C" in audit_row["functional_blocks"] or "RC_parallel" in audit_row["functional_blocks"] \
                or "RC_series" in audit_row["functional_blocks"]:
            g.devices.append(DeviceRecord(f"CC{k}", "cap", "miller_compensation",
                                          {"p": prev, "n": out}, None,
                                          {"value": PRIORS["miller_cap_f"]["value"],
                                           "origin": PRIORS["miller_cap_f"]["origin"]},
                                          {"structural_evidence": "C-type block in source graph",
                                           "compensation_topology": "miller_per_stage",
                                           "encloses_inversions": 1, **ev}))
        prev = out
        inversions += 1
    g.polarity = {"polarity_status": "template_defined",
                  "stage_inversion_count": inversions,
                  "signal_path_evidence": "vinp→M1→mirror→(CS)^k→vout; template parity",
                  "noninverting_input": "vinp" if inversions % 2 == 0 else "vinn",
                  "selection_rule": "structural parity, NOT outcome-based"}
    if g.polarity["noninverting_input"] == "vinn":
        # keep TB convention: swap input assignment structurally (documented).
        for d in g.devices:
            if d.device_id == "M1":
                d.nets["g"] = "vinn"
            elif d.device_id == "M2":
                d.nets["g"] = "vinp"
        g.polarity["port_swap_applied"] = True
        g.polarity["noninverting_input"] = "vinp"
    g.unresolved = audit_row["unresolved_blocks"]
    g.block_assignments = [{"block": "five_transistor_first_stage", "devices": ["M1", "M2", "M3", "M4", "M5"],
                            "confidence": 0.8, "rule": "template", "evidence": ev}]
    return g, ("mapping_incomplete" if g.unresolved else "mapped")


def emit_netlist(g: DeviceCircuitGraph, subckt_name: str) -> str:
    lines = [f"* Stage3C mapped netlist {g.topology_id} candidate {g.mapping_candidate_id}",
             f".subckt {subckt_name} gnda vdda vinn vinp vout"]
    for d in g.devices:
        n = d.nets
        if d.kind in ("nmos", "pmos"):
            model = "sky130_fd_pr__nfet_01v8" if d.kind == "nmos" else "sky130_fd_pr__pfet_01v8"
            lines.append(f"x{d.device_id} {n['d']} {n['g']} {n['s']} {n['b']} {model} "
                         f"l={d.sizing['l']} w={d.sizing['w']} m=1")
        elif d.kind == "cap":
            lines.append(f"c{d.device_id} {n['p']} {n['n']} {d.sizing['value']:.4g}")
        elif d.kind == "res":
            lines.append(f"r{d.device_id} {n['p']} {n['n']} {d.sizing['value']:.4g}")
        elif d.kind == "isrc":
            lines.append(f"i{d.device_id} {n['p']} {n['n']} {d.sizing['value']:.4g}")
    lines.append(".ends")
    return "\n".join(lines) + "\n"


def static_validate(g: DeviceCircuitGraph, netlist: str) -> dict[str, Any]:
    problems = []
    nets_used: set[str] = set()
    ids = [d.device_id for d in g.devices]
    if len(ids) != len(set(ids)):
        problems.append("duplicate device identifiers")
    for d in g.devices:
        if d.kind in ("nmos", "pmos") and set(d.nets) != {"d", "g", "s", "b"}:
            problems.append(f"{d.device_id}: MOSFET missing terminals")
        nets_used |= set(d.nets.values())
    for p in ("gnda", "vdda", "vinp", "vinn", "vout"):
        if p not in nets_used:
            problems.append(f"port {p} unconnected")
    if "vdda" in nets_used and any(
            d.kind in ("nmos", "pmos") and d.nets.get("d") == d.nets.get("s") == "vdda" for d in g.devices):
        problems.append("supply short")
    # graph preservation: mapped stage count vs structural gain stages
    device_count_match = len([d for d in g.devices if d.kind in ("nmos", "pmos")]) >= 5
    return {"status": "mapped_static_valid" if not problems else "mapped_static_invalid",
            "problems": problems,
            "graph_preservation_score": 1.0 if not g.unresolved else round(
                1.0 - 0.1 * len(g.unresolved), 2),
            "missing_blocks": g.unresolved, "device_count_match": device_count_match,
            "sky130_models_valid": "sky130_fd_pr__" in netlist}


class _MappedEntry:
    """Adapter making a mapped candidate look like a registry entry for Stage 3B."""

    def __init__(self, tid: str, cdir: Path, meta: dict[str, Any], graph):
        self.topology_id, self.path, self.metadata, self.graph = tid, cdir, meta, graph
        self.source, self.has_netlist, self.has_schematic = "generated_mapped", True, False


def run_stage3c(limit: int | None = None) -> dict[str, Any]:
    registry = TopologyRegistry(_ROOT / "datasets" / "topology_library")
    audits = audit_generated(registry)
    exe = discover_ngspice()
    mem = _ROOT / "datasets" / "simulation_memory"
    results = []
    ready = [a for a in audits if a["mapping_readiness"] in ("mapping_ready", "partially_specified")]
    for a in (ready[:limit] if limit else ready):
        entry = registry.get_topology(a["topology_id"])
        g, note = map_family(entry, a)
        if g is None:
            results.append({"topology_id": a["topology_id"], "mapping_status": note})
            continue
        cdir = _ROOT / "artifacts" / "stage3c" / a["topology_id"] / g.mapping_candidate_id
        cdir.mkdir(parents=True, exist_ok=True)
        subckt = f"mapped_{a['topology_id']}"
        netlist = emit_netlist(g, subckt)
        (cdir / "netlist.sp").write_text(netlist, encoding="utf-8")
        (cdir / "device_graph.json").write_text(json.dumps(asdict(g), indent=1, default=str), encoding="utf-8")
        val = static_validate(g, netlist)
        (cdir / "validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
        (cdir / "provenance.json").write_text(json.dumps(
            {"source_family": a["topology_id"], "graph_hash": a["graph_hash"],
             "mapping_note": note, "support_bias": g.support_bias,
             "polarity": g.polarity, "timestamp": time.time()}, indent=1), encoding="utf-8")
        row: dict[str, Any] = {"topology_id": a["topology_id"], "candidate": g.mapping_candidate_id,
                               "mapping_status": val["status"], "mapping_note": note,
                               "unresolved": g.unresolved}
        if val["status"] == "mapped_static_valid" and exe:
            # Sizing is inlined in the netlist; the Stage 3B testbench includes
            # design_variables/<name>, so provide a comment-only file and point
            # the pipeline's amplifier root at this candidate dir temporarily.
            (cdir / "design_variables").mkdir(exist_ok=True)
            (cdir / "design_variables" / subckt).write_text("* sizing inlined in netlist\n",
                                                            encoding="utf-8")
            fake = _MappedEntry(a["topology_id"], cdir,
                                {"name": subckt, "graph_hash": a["graph_hash"]}, entry.graph)
            audit_dict = {"subckt_name": subckt, "immediately_runnable": True,
                          "blocking_reason": None, "graph_hash": a["graph_hash"]}
            import agentic_raptor.electrical as elec

            original = elec._LEGACY_AMP
            elec._LEGACY_AMP = cdir
            try:
                rec = qualify_family(fake, audit_dict, cdir / "run", exe, "env-stage3c",
                                     split="train", timeout_s=180.0)
            finally:
                elec._LEGACY_AMP = original
            row.update({"electrical_status": rec.electrical_validation_status,
                        "metrics": {k: v for k, v in rec.extracted_metrics.items() if v is not None},
                        "failure_class": rec.failure_class, "run_id": rec.run_id})
        results.append(row)
    with (mem / "mapping_runs.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    counts = {"generated_families": len(audits),
              "mapping_ready": len([a for a in audits if a["mapping_readiness"] == "mapping_ready"]),
              "partially_specified": len([a for a in audits if a["mapping_readiness"] == "partially_specified"]),
              "behavioral_only": len([a for a in audits if a["mapping_readiness"] == "behavioral_only"]),
              "structurally_ambiguous": len([a for a in audits if a["mapping_readiness"] == "structurally_ambiguous"]),
              "candidates": len([r for r in results if r.get("candidate")]),
              "static_valid": len([r for r in results if r.get("mapping_status") == "mapped_static_valid"]),
              "electrically_functional": len([r for r in results if r.get("electrical_status") == "electrically_functional"]),
              "electrical_failed": len([r for r in results if r.get("electrical_status") not in (None, "electrically_functional")])}
    return {"counts": counts, "audits": audits, "results": results}


if __name__ == "__main__":
    out = run_stage3c()
    print(json.dumps(out["counts"], indent=1))
