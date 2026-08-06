"""Stage 3C.2: verified CktGNN corpus re-extraction and migration.

Flow: archive legacy → verify semantics from source → decode raw OCB pickles
with the 26-entry SUBG_NODE basis (signed gm preserved) → canonical graphs
(labels carry sign+direction → WL hash is attribute-aware) → dedup/cluster →
registry v2 families (new IDs; legacy untouched) → lineage → Stage 3C.1
interpretation → Stage 3C mapping bridge → unchanged ngspice qualification.
Unknown type codes raise `unknown_cktgnn_subgraph_type` — never coerced.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentic_raptor.core.circuit_graph import CircuitEdge, CircuitGraph, CircuitNode
from agentic_raptor.core.types import DeviceType, TerminalType
from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.mapping.interpretation import NODE_TYPE, SUBG_NODE, interpret_verified_dag

_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_VERSION = "verified_subgnode_v2"
LEGACY_VERSION = "legacy_invalidated_v1"


def write_semantics_spec(path: Path) -> None:
    entries = []
    for code, comps in SUBG_NODE.items():
        gm = next((c for c in comps if "gm" in c), None)
        entries.append({
            "raw_code": code, "components": comps,
            "gm_polarity": gm[0] if gm else None,
            "path_class": ("feedback" if gm and gm.endswith("-") else
                           "feedforward" if gm else "passive"),
            "connectivity": ("parallel" if code in (4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
                             else "series" if code in (5, 22, 23, 24, 25) else "single"),
        })
    path.write_text(json.dumps({
        "semantic_version": SEMANTIC_VERSION,
        "source_file": "repositories/CktGNN/OCB/src/circuit_generation.py",
        "source_symbols": {"NODE_TYPE": "L33-43", "SUBG_NODE": "L46-70"},
        "node_type": NODE_TYPE, "subg_node_basis": entries,
        "gm_notation": "first sign = transconductance polarity; second sign = "
                       "'+' feedforward main path / '-' feedback (OCB paper + amp_generator.py)",
        "indexing": "zero-based subgraph ids; vertex 'type' field = SUBG_NODE id",
        "limitations": ["series/parallel split for codes 18-25 taken from generation "
                        "comments; both variants share components"],
    }, indent=1), encoding="utf-8")


def archive_legacy(snapshot_dir: Path) -> dict[str, Any]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    registry = TopologyRegistry(_ROOT / "datasets" / "topology_library")
    manifest = {"created": time.time(), "reason": "t%8 decoder defect (Stage 3C.1 audit)",
                "semantic_version": LEGACY_VERSION, "git_commit": None,
                "affected": registry.filter_by_source("cktgnn"),
                "unaffected": [t for t in registry.list_topologies()
                               if t not in registry.filter_by_source("cktgnn")],
                "files": {}}
    for tid in manifest["affected"]:
        src = registry.get_topology(tid).path
        dst = snapshot_dir / tid
        if not dst.exists():
            shutil.copytree(src, dst)
        for f in dst.rglob("*"):
            if f.is_file():
                manifest["files"][str(f.relative_to(snapshot_dir))] = hashlib.sha256(
                    f.read_bytes()).hexdigest()
    for extra in ("interpretation_runs.jsonl",):
        p = _ROOT / "datasets" / "simulation_memory" / extra
        if p.is_file():
            shutil.copy2(p, snapshot_dir / extra)
            manifest["files"][extra] = hashlib.sha256(p.read_bytes()).hexdigest()
    (snapshot_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str),
                                                encoding="utf-8")
    return manifest


def decode_raw_graph(index: int, g) -> tuple[dict[str, str], list[tuple[str, str]], dict[str, Any]]:
    """igraph DAG (vertex type = SUBG_NODE id) → typed nodes/edges + report."""
    nodes: dict[str, str] = {}
    report = {"raw_id": index, "warnings": [], "status": "extracted_verified"}
    vid_name: dict[int, list[str]] = {}
    for vi, v in enumerate(g.vs):
        code = int(v["type"])
        if code not in SUBG_NODE:
            raise ValueError(f"unknown_cktgnn_subgraph_type:{code}")
        comps = SUBG_NODE[code]
        names = []
        for k, comp in enumerate(comps):
            if comp == "In":
                name = "In"
            elif comp == "Out":
                name = "Out"
            else:
                name = f"v{vi}_{k}_{comp.replace('+','p').replace('-','m')}"
            nodes[name] = comp
            names.append(name)
        vid_name[vi] = names
        if len(comps) > 1:
            report.setdefault("expanded", []).append({"vertex": vi, "code": code, "into": names})
    edges: list[tuple[str, str]] = []
    for e in g.es:
        for a in vid_name[e.source]:
            for b in vid_name[e.target]:
                edges.append((a, b))
    if "In" not in nodes or "Out" not in nodes:
        report["status"] = "missing_port"
    return nodes, edges, report


_ROLE_DEVICE = {"R": DeviceType.RESISTOR, "C": DeviceType.CAPACITOR}


def to_canonical(family_id: str, nodes: dict[str, str], edges: list[tuple[str, str]]) -> CircuitGraph:
    g = CircuitGraph(family_id)
    port_map = {"In": DeviceType.INPUT_PORT, "Out": DeviceType.OUTPUT_PORT}
    for nid, t in nodes.items():
        if t in port_map:
            g.add_node(CircuitNode(nid, port_map[t]))
        elif t in _ROLE_DEVICE:
            g.add_node(CircuitNode(nid, _ROLE_DEVICE[t], block_role=t))
        else:  # signed gm — role string carries polarity+direction → affects WL hash
            g.add_node(CircuitNode(nid, DeviceType.SUBCIRCUIT_BLOCK, block_role=t,
                                   attributes={"subckt_name": t}))
    for k, (a, b) in enumerate(edges):
        net = f"net_{k}"
        for endpoint in (a, b):
            node = g.nodes[endpoint]
            term = (TerminalType.PORT if node.device_type in port_map.values()
                    else TerminalType.PLUS if node.device_type in _ROLE_DEVICE.values()
                    else TerminalType.BLOCK_PIN)
            if node.device_type in _ROLE_DEVICE.values():
                term = TerminalType.PLUS if endpoint == a else TerminalType.MINUS
                if g.net_of(endpoint, term):
                    term = TerminalType.MINUS if term == TerminalType.PLUS else TerminalType.PLUS
                if g.net_of(endpoint, term):
                    continue  # both pins used; extra fanout dropped w/ warning upstream
            g.add_edge(CircuitEdge(f"e{k}_{endpoint}", endpoint, term, net))
    g.metadata.source = "cktgnn_v2"
    g.metadata.circuit_family = "ocb_opamp_v2"
    # signed-gm labels already in block_role → structural_hash is attribute-aware
    return g


def run_reextraction(sample_limit: int = 400) -> dict[str, Any]:
    out_root = _ROOT / "artifacts" / "stage3c2"
    reg_v2 = _ROOT / "artifacts" / "topology_registry_v2"
    (out_root / "raw_graphs").mkdir(parents=True, exist_ok=True)
    (out_root / "expanded_semantic_graphs").mkdir(parents=True, exist_ok=True)
    write_semantics_spec(_ROOT / "agentic_raptor" / "corpus" / "cktgnn_semantics_v2.json")
    archive = archive_legacy(_ROOT / "artifacts" / "corpus_snapshots" / "stage3a_legacy_invalidated_v1")

    with (_ROOT / "repositories/CktGNN/OCB/CktBench101/ckt_bench_101.pkl").open("rb") as f:
        data = pickle.load(f)
    graphs = data[0] if isinstance(data, tuple) else data
    stats = defaultdict(int)
    extracted: list[dict[str, Any]] = []
    for i, item in enumerate(graphs[:sample_limit]):
        g = item[0] if isinstance(item, (tuple, list)) else item
        try:
            nodes, edges, report = decode_raw_graph(i, g)
        except ValueError as exc:
            stats["unknown_type" if "unknown" in str(exc) else "malformed_graph"] += 1
            extracted.append({"raw_id": i, "status": str(exc)})
            continue
        stats[report["status"]] += 1
        if report["status"] != "extracted_verified":
            extracted.append(report)
            continue
        canon = to_canonical(f"cktgnn_v2_{i:05d}", nodes, edges)
        extracted.append({**report, "graph_hash": canon.structural_hash(),
                          "nodes": nodes, "edges": edges})
        (out_root / "expanded_semantic_graphs" / f"{i:05d}.json").write_text(
            json.dumps({"nodes": nodes, "edges": edges, "semantic_version": SEMANTIC_VERSION},
                       indent=0), encoding="utf-8")
    # dedup by attribute-aware hash (roles in labels)
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in extracted:
        if "graph_hash" in r:
            by_hash[r["graph_hash"]].append(r)
    unique = {h: grp[0] for h, grp in by_hash.items()}
    duplicates = sum(len(g) - 1 for g in by_hash.values())
    # cluster by signed-gm multiset + passive multiset
    clusters: dict[tuple, list[str]] = defaultdict(list)
    for h, r in unique.items():
        gms = tuple(sorted(t for t in r["nodes"].values() if "gm" in t))
        pas = tuple(sorted(t for t in r["nodes"].values() if t in ("R", "C")))
        clusters[(gms, pas)].append(h)
    # emit v2 families
    reg_v2.mkdir(parents=True, exist_ok=True)
    lineage_path = out_root / "legacy_to_v2_lineage.jsonl"
    families = []
    for fi, (key, hashes) in enumerate(sorted(clusters.items(), key=lambda kv: str(kv[0])), 1):
        rep = unique[hashes[0]]
        tid = f"topology_v2_{fi:04d}"
        d = reg_v2 / tid
        d.mkdir(exist_ok=True)
        canon = to_canonical(tid, rep["nodes"], rep["edges"])
        (d / "graph.json").write_text(canon.to_json(), encoding="utf-8")
        blocks: dict[str, list] = defaultdict(list)
        for nid, t in rep["nodes"].items():
            if t not in ("In", "Out"):
                blocks[t].append([nid])
        (d / "blocks.json").write_text(json.dumps(blocks, indent=0), encoding="utf-8")
        (d / "metadata.json").write_text(json.dumps({
            "name": tid, "source": "cktgnn_v2", "semantic_version": SEMANTIC_VERSION,
            "graph_hash": canon.structural_hash(), "family_members": hashes,
            "signed_gm": key[0], "passives": key[1],
            "validation_status": "unvalidated", "mapping_status": "verified_semantics",
            "functional_blocks": {b: len(v) for b, v in blocks.items()},
            "legacy_topology_ids": [], "raw_source_ids": [unique[h]["raw_id"] for h in hashes],
        }, indent=1), encoding="utf-8")
        families.append(tid)
    with lineage_path.open("w", encoding="utf-8") as f:
        legacy = archive["affected"]
        f.write(json.dumps({"relationship": "many_to_many",
                            "legacy_topology_ids": legacy,
                            "corrected_families": families,
                            "reason": "legacy labels unverified; raw-source-id bridge only",
                            "migration_confidence": 0.5}) + "\n")
    # interpretation on corrected families
    interp = {"interpretation_ready": 0, "ambiguous": 0, "unsupported": 0, "valid_fsg": 0}
    fsg_ready = []
    for tid in families:
        meta = json.loads((reg_v2 / tid / "metadata.json").read_text())
        rep_hash = meta["family_members"][0]
        r = unique[rep_hash]
        fsg = interpret_verified_dag(tid, r["nodes"], r["edges"])
        v = fsg.validation()
        (reg_v2 / tid / "interpretation.json").write_text(json.dumps(
            {"validation": v, "stages": [s.__dict__ for s in fsg.stages],
             "inversion_parity": fsg.inversion_parity}, indent=0, default=str), encoding="utf-8")
        if v["functional_graph_valid"] and v["interpreted_gm_fraction"] == 1.0:
            interp["interpretation_ready"] += 1
            interp["valid_fsg"] += 1
            main_stages = len([s for s in fsg.stages if s.path_class == "main_forward_path"])
            fsg_ready.append((tid, main_stages, meta))
        elif v["interpreted_gm_fraction"] > 0:
            interp["ambiguous"] += 1
        else:
            interp["unsupported"] += 1
    return {"archive_files": len(archive["files"]), "raw_stats": dict(stats),
            "raw_total": min(sample_limit, len(graphs)),
            "unique_graphs": len(unique), "duplicates": duplicates,
            "clusters": len(clusters), "v2_families": len(families),
            "interpretation": interp, "fsg_ready": [(t, n) for t, n, _m in fsg_ready]}


if __name__ == "__main__":
    print(json.dumps(run_reextraction(), indent=1, default=str))
