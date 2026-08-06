"""Stage 3C.2b: operational registry v3, RAG rebuild, realisation, qualification."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Any

from agentic_raptor.corpus import TopologyRegistry, build_rag_index
from agentic_raptor.electrical import discover_ngspice, qualify_family
from agentic_raptor.mapping import _MappedEntry, emit_netlist, map_family, static_validate

_ROOT = Path(__file__).resolve().parents[2]
V3 = _ROOT / "artifacts" / "topology_registry_operational_v3"


def _copy_family(src: Path, dst: Path, extra_meta: dict[str, Any]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    meta = json.loads((dst / "metadata.json").read_text(encoding="utf-8"))
    meta.update(extra_meta)
    (dst / "metadata.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    if not (dst / "blocks.json").is_file():
        graph = json.loads((dst / "graph.json").read_text(encoding="utf-8"))
        blocks: dict[str, list] = defaultdict(list)
        for n in graph.get("nodes", []):
            if n.get("block_role"):
                blocks[n["block_role"]].append([n["node_id"]])
        (dst / "blocks.json").write_text(json.dumps(blocks, indent=0), encoding="utf-8")


def assemble() -> dict[str, Any]:
    if V3.exists():
        shutil.rmtree(V3)
    V3.mkdir(parents=True)
    v1 = TopologyRegistry(_ROOT / "datasets" / "topology_library")
    counts = {"analoggym": 0, "opamp_generator": 0, "cktgnn_v2": 0}
    for tid in v1.filter_by_source("analoggym"):
        _copy_family(v1.get_topology(tid).path, V3 / tid,
                     {"registry_version": "operational_v3", "active": True,
                      "source_type": "analoggym"})
        counts["analoggym"] += 1
    for d in sorted((_ROOT / "artifacts" / "topology_registry_v1_hash_v2").glob("topology_og2_*")):
        _copy_family(d, V3 / d.name, {"registry_version": "operational_v3", "active": True,
                                      "source_type": "opamp_generator"})
        counts["opamp_generator"] += 1
    for d in sorted((_ROOT / "artifacts" / "topology_registry_v2").glob("topology_v2_*")):
        _copy_family(d, V3 / d.name, {"registry_version": "operational_v3", "active": True,
                                      "source_type": "cktgnn_v2"})
        counts["cktgnn_v2"] += 1
    return counts


def build_splits_v3(registry: TopologyRegistry, seed: int = 42) -> dict[str, str]:
    legacy = json.loads((_ROOT / "datasets/topology_splits/family_split.json").read_text())
    legacy_of = {t: s for s in ("train", "validation", "test") for t in legacy.get(s, [])}
    og2 = sorted(t for t in registry.list_topologies() if t.startswith("topology_og2"))
    v2 = sorted(t for t in registry.list_topologies() if t.startswith("topology_v2"))
    ag = registry.filter_by_source("analoggym")
    split: dict[str, str] = {t: legacy_of.get(t, "train") for t in ag}  # preserve AG
    og_group_split = legacy_of.get("topology_0057", "train")
    for t in og2:  # inherited split-group constraint: all descendants together
        split[t] = og_group_split
    rng = Random(seed)
    shuffled = v2[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    for i, t in enumerate(shuffled):
        split[t] = "test" if i < round(n * 0.15) else "validation" if i < round(n * 0.3) else "train"
    return split


def run_all(sim_limit: int | None = None) -> dict[str, Any]:
    started = time.time()
    counts = assemble()
    registry = TopologyRegistry(V3)
    assert registry.get_family_count() == 174, registry.get_family_count()
    assert "topology_0057" not in registry.list_topologies()
    split = build_splits_v3(registry)
    (V3 / "split_v3.json").write_text(json.dumps(
        {"seed": 42, "algorithm": "preserve_AG+og2_group+seeded_v2",
         "assignments": split}, indent=0), encoding="utf-8")
    rag = build_rag_index(registry, _ROOT / "datasets" / "topology_rag")

    exe = discover_ngspice()
    results: list[dict[str, Any]] = []
    generated = [t for t in registry.list_topologies() if not t.startswith("topology_0")]
    for tid in (generated[:sim_limit] if sim_limit else generated):
        e = registry.get_topology(tid)
        meta = e.metadata
        interp = e.path / "interpretation.json"
        if tid.startswith("topology_v2") and interp.is_file():
            fsg = json.loads(interp.read_text())
            stages = len([s for s in fsg["stages"] if s.get("path_class") == "main_forward_path"])
            fb = [s for s in fsg["stages"] if s.get("path_class") == "feedback_path"]
            blocks = set(meta.get("functional_blocks", {}))
            if fb:
                results.append({"topology_id": tid, "status": "withheld_unsupported_template",
                                "reason": "feedback transconductor branch has no verified template"})
                continue
        else:  # og2
            blocks = set(meta.get("members", [""])) and set()
            g = e.graph
            stages = sum(1 for n in g.nodes.values() if n.block_role == "gain_stage")
            blocks = {n.block_role for n in g.nodes.values() if n.block_role}
        if stages == 0:
            results.append({"topology_id": tid, "status": "withheld_at_mapping_audit",
                            "reason": "no main-path gain stage"})
            continue
        audit_row = {"topology_id": tid, "gain_stages": stages,
                     "functional_blocks": sorted(blocks) + (["C"] if "C" in str(blocks) else []),
                     "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
                     "graph_hash": meta.get("graph_hash")}
        g_dev, note = map_family(e, audit_row)
        if g_dev is None:
            results.append({"topology_id": tid, "status": "withheld_unsupported_template"})
            continue
        cdir = _ROOT / "artifacts" / "stage3c2b" / meta.get("source_type", "gen") / tid / g_dev.mapping_candidate_id
        cdir.mkdir(parents=True, exist_ok=True)
        subckt = f"m3b_{tid}"
        net = emit_netlist(g_dev, subckt)
        (cdir / "netlist.sp").write_text(net, encoding="utf-8")
        val = static_validate(g_dev, net)
        (cdir / "static_validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
        row = {"topology_id": tid, "realisation_id": g_dev.mapping_candidate_id,
               "split": split[tid], "static": val["status"]}
        if val["status"] != "mapped_static_valid":
            row["status"] = "failed_static_validation"
            results.append(row)
            continue
        row["status"] = "ready_for_simulation"
        if exe:
            (cdir / "design_variables").mkdir(exist_ok=True)
            (cdir / "design_variables" / subckt).write_text("* inlined\n", encoding="utf-8")
            import agentic_raptor.electrical as elec
            fake = _MappedEntry(tid, cdir, {"name": subckt, "graph_hash": meta.get("graph_hash")}, e.graph)
            orig = elec._LEGACY_AMP
            elec._LEGACY_AMP = cdir
            try:
                rec = qualify_family(fake, {"subckt_name": subckt, "immediately_runnable": True,
                                            "blocking_reason": None},
                                     cdir / "run", exe, "env-3c2b", split[tid], 120.0)
            finally:
                elec._LEGACY_AMP = orig
            pm = rec.extracted_metrics.get("phase_margin_deg")
            row.update({"electrical": rec.electrical_validation_status,
                        "failure_class": rec.failure_class,
                        "metrics": {k: v for k, v in rec.extracted_metrics.items() if v is not None},
                        "stability": ("verified_stable" if pm is not None and pm > 0
                                      else "verified_unstable" if pm is not None
                                      else "phase_margin_unavailable")})
        results.append(row)

    # tiers (strict: A requires functional AND verified_stable)
    tiers = defaultdict(int)
    for tid in registry.filter_by_source("analoggym"):
        ev = registry.get_topology(tid).path / "electrical_validation.json"
        st = json.loads(ev.read_text()).get("electrical_validation_status") if ev.is_file() else None
        summ = {}
        sfile = _ROOT / "datasets/simulation_memory/topology_summaries.jsonl"
        for line in sfile.read_text().splitlines():
            r = json.loads(line)
            if r["topology_id"] == tid:
                summ = r
        stab = summ.get("stability_status")
        if st == "electrically_functional" and stab == "verified_stable":
            tiers["A1"] += 1
        elif st == "electrically_functional":
            tiers["D2"] += 1  # functional but verified_unstable
        else:
            tiers["B2"] += 1
    for r in results:
        if r.get("electrical") == "electrically_functional" and r.get("stability") == "verified_stable":
            tiers["A2"] += 1
        elif r.get("electrical") == "electrically_functional":
            tiers["D2"] += 1
        elif r.get("status") == "ready_for_simulation" and r.get("electrical"):
            tiers["B1"] += 1
        elif r.get("status", "").startswith("withheld"):
            tiers["C1"] += 1
        else:
            tiers["D1"] += 1

    mem = _ROOT / "datasets" / "simulation_memory" / "stage3c2b_runs.jsonl"
    with mem.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    summary = {
        "active_families": registry.get_family_count(), "source_counts": counts,
        "split_counts": {s: sum(1 for v in split.values() if v == s) for s in ("train", "validation", "test")},
        "rag": rag,
        "generated_processed": len(results),
        "withheld": sum(1 for r in results if str(r.get("status", "")).startswith("withheld")),
        "static_failed": sum(1 for r in results if r.get("status") == "failed_static_validation"),
        "simulated": sum(1 for r in results if "electrical" in r),
        "electrically_functional": sum(1 for r in results if r.get("electrical") == "electrically_functional"),
        "verified_stable": sum(1 for r in results if r.get("stability") == "verified_stable"),
        "verified_unstable": sum(1 for r in results if r.get("stability") == "verified_unstable"),
        "tiers": dict(tiers), "wall_clock_s": round(time.time() - started, 1),
    }
    (V3 / "SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    manifest = {f.name: hashlib.sha256((f).read_bytes()).hexdigest()
                for f in [V3 / "SUMMARY.json", V3 / "split_v3.json"]}
    (V3 / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=1, default=str))
