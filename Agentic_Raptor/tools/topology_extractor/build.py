"""Corpus builder: extract → dedup → cluster → emit datasets/topology_library."""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from tools.topology_extractor import analoggym, cktgnn, opamp_generator  # noqa: E402
from tools.topology_extractor.blocks import block_signature, detect_blocks  # noqa: E402
from tools.topology_extractor.common import ExtractedTopology, isomorphic  # noqa: E402


def build(root: Path | None = None, out_dir: Path | None = None) -> dict:
    root = root or _PKG_ROOT
    out_dir = out_dir or root / "datasets" / "topology_library"
    sources = {
        "analoggym": analoggym.extract(root.parent / "RAPTOR_Legacy" / "AnalogGym"),
        "opamp_generator": opamp_generator.extract(root / "repositories" / "OPAMP-Generator"),
        "cktgnn": cktgnn.extract(root / "repositories" / "CktGNN"),
    }
    all_topos = [t for lst in sources.values() for t in lst]
    usable = [t for t in all_topos if t.mapping_status != "skipped"]
    skipped = [t for t in all_topos if t.mapping_status == "skipped"]

    # --- dedup: canonical WL hash, isomorphism check on collisions ---
    by_hash: dict[str, list[ExtractedTopology]] = defaultdict(list)
    for t in usable:
        by_hash[t.graph_hash].append(t)
    unique: list[ExtractedTopology] = []
    duplicates = 0
    for _h, group in sorted(by_hash.items()):
        keep = [group[0]]
        for cand in group[1:]:
            if any(isomorphic(cand.graph, k.graph) for k in keep):
                duplicates += 1
            else:  # rare WL collision, structurally distinct
                keep.append(cand)
        unique.extend(keep)

    # --- cluster into canonical families ---
    clusters: dict[tuple, list[ExtractedTopology]] = defaultdict(list)
    for t in unique:
        blocks = detect_blocks(t.graph)
        t.metadata["functional_blocks"] = {k: len(v) for k, v in blocks.items()}
        t.metadata["_blocks_detail"] = blocks
        key = (
            (t.repository, t.name) if t.repository == "analoggym"
            else ("generated", t.graph.metadata.circuit_family, tuple(sorted(blocks)))
        )
        _ = block_signature  # fine-grained signature retained in metadata only
        clusters[key].append(t)
    # AnalogGym literature amplifiers are canonical families by construction;
    # generated sources cluster by (family, block signature).
    families: list[list[ExtractedTopology]] = []
    for key, members in sorted(clusters.items(), key=lambda kv: str(kv[0])):
        if key[0] == "analoggym":
            families.extend([[m] for m in members])
        else:
            families.append(sorted(members, key=lambda m: len(m.graph.nodes)))

    # --- emit library ---
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts = {"missing_schematics": 0, "missing_netlists": 0}
    for i, fam in enumerate(families, 1):
        rep = fam[0]
        d = out_dir / f"topology_{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "graph.json").write_text(rep.graph.to_json(), encoding="utf-8")
        (d / "blocks.json").write_text(
            json.dumps(rep.metadata.pop("_blocks_detail", {}), indent=1), encoding="utf-8")
        meta = {
            "name": rep.name, "source": rep.repository, "repository": rep.repository,
            "paper": rep.metadata.get("paper"), "year": None,
            "graph_hash": rep.graph_hash, "graph_size": len(rep.graph.nodes),
            "num_stages": rep.metadata.get("functional_blocks", {}).get("gain_stage", 0) or None,
            "functional_blocks": rep.metadata.get("functional_blocks", {}),
            "technology": rep.metadata.get("technology"),
            "license": rep.metadata.get("license", "see source repository"),
            "mapping_status": rep.mapping_status, "validation_status": "unvalidated",
            "family_members": [m.name for m in fam], "family_size": len(fam),
        }
        (d / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        if rep.netlist_path and Path(rep.netlist_path).is_file():
            shutil.copy2(rep.netlist_path, d / "netlist.sp")
        else:
            counts["missing_netlists"] += 1
        if rep.schematic_path and Path(rep.schematic_path).is_file():
            shutil.copy2(rep.schematic_path, d / "schematic.png")
        else:
            counts["missing_schematics"] += 1

    report = {
        "per_source_extracted": {k: len(v) for k, v in sources.items()},
        "usable": len(usable), "skipped_unknown_mappings": len(skipped),
        "skipped_reasons": [t.metadata.get("error", "?")[:100] for t in skipped][:5],
        "duplicates_removed": duplicates, "unique_graphs": len(unique),
        "clusters": len(clusters), "canonical_families": len(families),
        **counts,
    }
    lines = ["# Topology Library Report", ""]
    lines += [f"- AnalogGym topologies extracted: {len(sources['analoggym'])}"]
    lines += [f"- OPAMP-Generator topologies extracted: {len(sources['opamp_generator'])}"]
    lines += [f"- CktGNN topologies extracted: {len(sources['cktgnn'])}"]
    lines += [f"- Duplicates removed (isomorphism-confirmed): {duplicates}"]
    lines += [f"- Unique graphs after dedup: {len(unique)}"]
    lines += [f"- Graph clusters: {len(clusters)}"]
    lines += [f"- Canonical topology families emitted: {len(families)}"]
    lines += [f"- Missing schematics: {counts['missing_schematics']} (generated sources have none)"]
    lines += [f"- Missing netlists: {counts['missing_netlists']} (block-level DAG sources)"]
    lines += [f"- Unknown mappings / skipped: {len(skipped)} — {report['skipped_reasons']}"]
    lines += ["", "Mapping caveats: OPAMP-Generator rows use approximate_dag_decode "
              "(stage-level faithful; branch semantics approximated); CktGNN requires "
              "python-igraph for OCB pickles. No sizing/SPICE performed by design."]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
