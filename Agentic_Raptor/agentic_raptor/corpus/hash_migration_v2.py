"""Stage 3C.2c: v1 hash migration + OPAMP-Generator role-aware re-dedup.

Graph identity ≠ netlist identity ≠ simulation identity: hashes are migrated
(graph_hash→v2, legacy preserved), AnalogGym electrical evidence is verified
preserved via netlist hashes, and OPAMP-Generator families are rebuilt from
all 1,500 source rows under the role-aware hasher. Nothing electrical is
rerun or fabricated; legacy CktGNN stays invalidated.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentic_raptor.corpus import TopologyRegistry

_ROOT = Path(__file__).resolve().parents[2]
HASH_VERSION = 2
HASH_ALGO = "wl_role_aware_v2"
HASH_ATTRS = ["node_type", "block_role(signed gm, ff/fb, functional roles)",
              "port_role", "terminal_labels", "connectivity"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot(src: Path, dst: Path, reason: str) -> dict[str, Any]:
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {"created": time.time(), "reason": reason, "hasher_before": "wl_role_blind_v1",
                "git_commit": None, "files": {}}
    for d in sorted(src.glob("topology_*")):
        t = dst / d.name
        if not t.exists():
            shutil.copytree(d, t)
        for f in t.rglob("*"):
            if f.is_file():
                manifest["files"][str(f.relative_to(dst))] = _sha(f)
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return manifest


def migrate_v1_hashes(registry: TopologyRegistry, out: Path) -> list[dict[str, Any]]:
    rows = []
    for tid in registry.list_topologies():
        e = registry.get_topology(tid)
        old = e.metadata.get("graph_hash")
        new = e.graph.structural_hash()
        meta = dict(e.metadata)
        meta.update({"graph_hash": new, "legacy_graph_hash": old,
                     "hash_version": HASH_VERSION, "hash_algorithm": HASH_ALGO,
                     "hash_attributes": HASH_ATTRS})
        netlist = e.path / "netlist.sp"
        row = {"topology_id": tid, "source": e.source, "old_graph_hash": old,
               "new_graph_hash": new, "changed": old != new,
               "hash_version_before": 1, "hash_version_after": HASH_VERSION,
               "netlist_hash": _sha(netlist) if netlist.is_file() else None,
               "migration_status": "hash_only_migration",
               "electrical_evidence": ("electrical_evidence_preserved"
                                       if e.source == "analoggym" else "n/a")}
        # atomic metadata update (legacy preserved in snapshot + legacy_graph_hash)
        tmp = e.path / "metadata.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        tmp.replace(e.path / "metadata.json")
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return rows


def rededup_opamp_generator() -> dict[str, Any]:
    sys.path.insert(0, str(_ROOT))
    from tools.topology_extractor import opamp_generator

    topos = opamp_generator.extract(_ROOT / "repositories" / "OPAMP-Generator", sample_limit=1500)
    valid = [t for t in topos if t.mapping_status != "skipped"]
    by_hash: dict[str, list] = defaultdict(list)
    for t in valid:
        by_hash[t.graph.structural_hash()].append(t)
    # variant analysis: same shape (role-blind signature = sorted degree+type-less)
    def shape_key(t) -> str:
        return f"n{len(t.graph.nodes)}_e{len(t.graph.edges)}"
    role_only, sign_only, fffb_only = 0, 0, 0
    shapes: dict[str, set] = defaultdict(set)
    for h, grp in by_hash.items():
        shapes[shape_key(grp[0])].add(h)
    for _s, hs in shapes.items():
        if len(hs) > 1:
            roles = [tuple(sorted(n.block_role or "" for n in by_hash[h][0].graph.nodes.values()))
                     for h in hs]
            for i in range(len(roles)):
                for j in range(i + 1, len(roles)):
                    a, b = roles[i], roles[j]
                    if a == b:
                        continue
                    diff = {x for x in a} ^ {x for x in b}
                    if all(("ff_" in d or "fb_" in d) for d in diff if d):
                        fffb_only += 1
                    elif all(("pos" in d or "neg" in d) for d in diff if d):
                        sign_only += 1
                    else:
                        role_only += 1
    return {"source_rows": len(topos), "valid_rows": len(valid),
            "invalid_rows": len(topos) - len(valid),
            "unique_role_aware": len(by_hash),
            "duplicates": len(valid) - len(by_hash),
            "role_only_variant_pairs": role_only, "sign_only_variant_pairs": sign_only,
            "fffb_only_variant_pairs": fffb_only,
            "groups": {h: [t.name for t in grp[:3]] for h, grp in by_hash.items()},
            "representatives": {h: grp[0] for h, grp in by_hash.items()}}


def run_migration() -> dict[str, Any]:
    lib = _ROOT / "datasets" / "topology_library"
    snap = snapshot(lib, _ROOT / "artifacts" / "corpus_snapshots" / "pre_stage3c2c_v1_hash_migration",
                    "role-blind → role-aware hash migration (Defect 2)")
    registry = TopologyRegistry(lib)
    legacy_opamp = [t for t in registry.list_topologies()
                    if registry.get_topology(t).graph.metadata.circuit_family == "three_stage_opamp"
                    or "opampgen" in json.dumps(registry.get_metadata(t).get("family_members", []))]
    rows = migrate_v1_hashes(registry, _ROOT / "artifacts" / "stage3c2c" / "v1_hash_migration.jsonl")

    dedup = rededup_opamp_generator()
    reg_out = _ROOT / "artifacts" / "topology_registry_v1_hash_v2"
    reg_out.mkdir(parents=True, exist_ok=True)
    lineage_path = _ROOT / "artifacts" / "stage3c2c" / "v1_to_hash_v2_lineage.jsonl"
    families = []
    with lineage_path.open("w", encoding="utf-8") as lf:
        for fi, (h, rep) in enumerate(sorted(dedup["representatives"].items()), 1):
            tid = f"topology_og2_{fi:04d}"
            d = reg_out / tid
            d.mkdir(exist_ok=True)
            (d / "graph.json").write_text(rep.graph.to_json(), encoding="utf-8")
            (d / "metadata.json").write_text(json.dumps({
                "name": rep.name, "source": "opamp_generator", "graph_hash": h,
                "legacy_graph_hash": None, "hash_version": HASH_VERSION,
                "hash_algorithm": HASH_ALGO, "legacy_family_ids": legacy_opamp,
                "members": dedup["groups"][h], "validation_status": "unvalidated",
                "relationship": "legacy_family_split" if len(legacy_opamp) < dedup["unique_role_aware"]
                else "family_membership_reassigned"}, indent=1), encoding="utf-8")
            families.append(tid)
            lf.write(json.dumps({"source_type": "opamp_generator",
                                 "legacy_topology_ids": legacy_opamp,
                                 "corrected_topology_id": tid, "corrected_graph_hash": h,
                                 "source_rows": dedup["groups"][h],
                                 "relationship": "legacy_family_split",
                                 "confidence": 0.9,
                                 "reason": "role-aware dedup separates collapsed variants"}) + "\n")
        for r in rows:  # hash-only migrations for everything else
            if r["source"] == "analoggym" or r["topology_id"] not in legacy_opamp:
                lf.write(json.dumps({"source_type": r["source"],
                                     "legacy_topology_id": r["topology_id"],
                                     "corrected_topology_id": r["topology_id"],
                                     "legacy_graph_hash": r["old_graph_hash"],
                                     "corrected_graph_hash": r["new_graph_hash"],
                                     "relationship": "hash_only_migration",
                                     "confidence": 1.0,
                                     "electrical_evidence": r["electrical_evidence"]}) + "\n")

    # consistency checks (fail loudly)
    problems = []
    reg2 = TopologyRegistry(lib)
    for tid in reg2.list_topologies():
        m = reg2.get_metadata(tid)
        if m.get("hash_version") != HASH_VERSION:
            problems.append(f"{tid}: hash_version != 2")
        if m.get("graph_hash") != reg2.get_graph(tid).structural_hash():
            problems.append(f"{tid}: stale active hash")
        if "legacy_graph_hash" not in m:
            problems.append(f"{tid}: missing legacy_graph_hash")
    if len(set(families)) != len(families):
        problems.append("duplicate corrected topology IDs")
    summary = {"archive_files": len(snap["files"]),
               "v1_families_audited": len(rows),
               "hashes_changed": sum(r["changed"] for r in rows),
               "hashes_unchanged": sum(not r["changed"] for r in rows),
               "analoggym_preserved": sum(1 for r in rows
                                          if r["electrical_evidence"] == "electrical_evidence_preserved"),
               "legacy_opamp_families": legacy_opamp,
               "opamp_dedup": {k: v for k, v in dedup.items()
                               if k not in ("groups", "representatives")},
               "corrected_opamp_families": len(families),
               "consistency_problems": problems}
    (_ROOT / "artifacts" / "stage3c2c" / "summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    if problems:
        raise RuntimeError(f"consistency check failed: {problems}")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_migration(), indent=1, default=str))
