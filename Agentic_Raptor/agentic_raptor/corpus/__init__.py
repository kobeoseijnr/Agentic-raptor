"""Topology-corpus registry, hierarchical RAG index, splits, and selection.

Replaces the single-template Stage 3A restriction with dynamic discovery of
`datasets/topology_library/topology_*/`. Distinctions preserved everywhere:
structural vs electrical validation; literature (analoggym) vs generated
provenance. No electrical performance is ever fabricated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.utils.exceptions import AgenticRaptorError

REQUIRED_FILES = ("graph.json", "metadata.json", "blocks.json")


@dataclass
class TopologyEntry:
    topology_id: str
    path: Path
    metadata: dict[str, Any]
    blocks: dict[str, Any]
    graph: CircuitGraph
    has_netlist: bool
    has_schematic: bool

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def stage_count(self) -> int | None:
        return self.metadata.get("num_stages")

    @property
    def structural_validation_status(self) -> str:
        return "structurally_loaded"  # schema-validated at load; validator run separately

    @property
    def electrical_validation_status(self) -> str:
        return str(self.metadata.get("validation_status", "unvalidated"))

    def retrieval_document(self) -> dict[str, Any]:
        blocks = sorted(self.metadata.get("functional_blocks", {}))
        summary = (
            f"{self.source} amplifier family '{self.metadata.get('name')}', "
            f"{len(self.graph.nodes)} nodes, stages={self.stage_count}, "
            f"blocks: {', '.join(blocks) or 'unknown'}; "
            f"mapping={self.metadata.get('mapping_status')}, "
            f"electrical={self.electrical_validation_status}"
        )
        return {
            "topology_id": self.topology_id,
            "source": self.source,
            "paper": self.metadata.get("paper"),
            "stage_count": self.stage_count,
            "functional_blocks": blocks,
            "graph_hash": self.metadata.get("graph_hash"),
            "mapping_status": self.metadata.get("mapping_status"),
            "validation_status": self.electrical_validation_status,
            "structural_validation_status": self.structural_validation_status,
            "has_netlist": self.has_netlist,
            "has_schematic": self.has_schematic,
            "structural_summary": summary,
            "retrieval_text": summary,  # metadata/structure only — no invented performance
        }


class TopologyRegistry:
    """Dynamic discovery; never hard-codes the family count."""

    def __init__(self, library_root: str | Path = "datasets/topology_library") -> None:
        self.root = Path(library_root)
        self.entries: dict[str, TopologyEntry] = {}
        self.excluded: list[dict[str, str]] = []
        for d in sorted(self.root.glob("topology_*")):
            if not d.is_dir():
                continue
            missing = [f for f in REQUIRED_FILES if not (d / f).is_file()]
            if missing:
                self.excluded.append({"topology_id": d.name, "reason": f"missing {missing}"})
                continue
            try:
                graph = CircuitGraph.from_json((d / "graph.json").read_text(encoding="utf-8"))
                meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
                blocks = json.loads((d / "blocks.json").read_text(encoding="utf-8"))
            except (AgenticRaptorError, ValueError, KeyError) as exc:
                self.excluded.append({"topology_id": d.name, "reason": f"schema: {exc}"})
                continue
            self.entries[d.name] = TopologyEntry(
                d.name, d, meta, blocks, graph,
                has_netlist=(d / "netlist.sp").is_file(),
                has_schematic=(d / "schematic.png").is_file(),
            )

    # -- required interface -------------------------------------------------
    def list_topologies(self) -> list[str]:
        return sorted(self.entries)

    def get_topology(self, tid: str) -> TopologyEntry:
        return self.entries[tid]

    def get_graph(self, tid: str) -> CircuitGraph:
        return self.entries[tid].graph

    def get_metadata(self, tid: str) -> dict[str, Any]:
        return self.entries[tid].metadata

    def get_blocks(self, tid: str) -> dict[str, Any]:
        return self.entries[tid].blocks

    def get_netlist(self, tid: str) -> str | None:
        p = self.entries[tid].path / "netlist.sp"
        return p.read_text(encoding="utf-8") if p.is_file() else None

    def get_schematic(self, tid: str) -> Path | None:
        p = self.entries[tid].path / "schematic.png"
        return p if p.is_file() else None

    def filter_by_source(self, source: str) -> list[str]:
        return [t for t, e in sorted(self.entries.items()) if e.source == source]

    def filter_by_stage_count(self, n: int) -> list[str]:
        return [t for t, e in sorted(self.entries.items()) if e.stage_count == n]

    def filter_by_blocks(self, required: list[str]) -> list[str]:
        return [t for t, e in sorted(self.entries.items())
                if set(required) <= set(e.metadata.get("functional_blocks", {}))]

    def filter_by_validation_status(self, status: str) -> list[str]:
        return [t for t, e in sorted(self.entries.items())
                if e.electrical_validation_status == status]

    def find_by_graph_hash(self, h: str) -> list[str]:
        return [t for t, e in sorted(self.entries.items()) if e.metadata.get("graph_hash") == h]

    def get_family_count(self) -> int:
        return len(self.entries)

    def duplicate_hashes(self) -> dict[str, list[str]]:
        seen: dict[str, list[str]] = {}
        for t, e in self.entries.items():
            seen.setdefault(str(e.metadata.get("graph_hash")), []).append(t)
        return {h: sorted(ts) for h, ts in seen.items() if len(ts) > 1}


# -- hierarchical RAG index ---------------------------------------------------
def build_rag_index(registry: TopologyRegistry, out_root: str | Path = "datasets/topology_rag") -> dict[str, int]:
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    topo_records = [registry.entries[t].retrieval_document() for t in registry.list_topologies()]
    block_records, graph_records = [], []
    for t in registry.list_topologies():
        e = registry.entries[t]
        for block, members in e.blocks.items():
            block_records.append({"topology_id": t, "block": block,
                                  "instances": len(members), "source": e.source,
                                  "validation_status": e.electrical_validation_status})
        g = e.graph
        graph_records.append({"topology_id": t, "graph_hash": e.metadata.get("graph_hash"),
                              "nodes": len(g.nodes), "edges": len(g.edges),
                              "nets": len(g.nets()),
                              "block_composition": sorted(e.metadata.get("functional_blocks", {})),
                              "stage_count": e.stage_count})
    def _dump(name: str, rows: list[dict]) -> None:
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _dump("topology_records.jsonl", topo_records)
    _dump("block_records.jsonl", block_records)
    _dump("graph_records.jsonl", graph_records)
    counts = {"topology_records": len(topo_records), "block_records": len(block_records),
              "graph_records": len(graph_records), "simulation_memory_records": 0}
    (out / "index_metadata.json").write_text(json.dumps(
        {"levels": ["topology_family", "functional_block", "structural_graph",
                    "simulation_memory (interface only — empty until SPICE-labelled data exists)"],
         **counts}, indent=1), encoding="utf-8")
    (out / "build_report.md").write_text(
        "# Topology RAG index\n" + "\n".join(f"- {k}: {v}" for k, v in counts.items()) +
        "\n- Level 4 simulation memory: interface reserved, intentionally empty.\n", encoding="utf-8")
    return counts


# -- selection ---------------------------------------------------------------
def select_topology_candidates(
    registry: TopologyRegistry, spec: Any, k: int = 5,
    required_blocks: list[str] | None = None, stage_count: int | None = None,
) -> list[dict[str, Any]]:
    """Top-K family candidates with score decomposition + provenance tiers."""
    scored = []
    for t in registry.list_topologies():
        e = registry.entries[t]
        parts = {"base": 0.1}
        if e.source == "analoggym":
            parts["literature"] = 0.4
        if e.has_netlist:
            parts["transistor_mapped"] = 0.3
        if stage_count is not None and e.stage_count == stage_count:
            parts["stage_match"] = 0.4
        if required_blocks:
            hit = len(set(required_blocks) & set(e.metadata.get("functional_blocks", {})))
            parts["block_match"] = 0.2 * hit
        gain = getattr(spec, "target_gain_db", None)
        if gain and gain >= 60 and (e.stage_count or 1) >= 2:
            parts["gain_intent"] = 0.2
        # Tier wiring: A = independently electrically qualified (Level-4 record);
        # B = device-level netlist, simulation incomplete; C = mapping required
        # (still retrievable for exploration); D = invalid/incomplete.
        ev = (e.path / "electrical_validation.json")
        level4_status = None
        if ev.is_file():
            try:
                level4_status = json.loads(ev.read_text(encoding="utf-8")).get("electrical_validation_status")
            except ValueError:
                level4_status = None
        if level4_status == "electrically_functional":
            tier = "tier_A_electrically_qualified"
            parts["tier_A"] = 0.6
        elif e.has_netlist:
            tier = "tier_B_netlist_unqualified"
            parts["tier_B"] = 0.2
        elif e.graph.nodes:
            tier = "tier_C_mapping_required"
        else:
            tier = "tier_D_invalid"
            parts["tier_D"] = -1.0
        scored.append({"topology_id": t, "score": round(sum(parts.values()), 3),
                       "score_decomposition": parts, "candidate_tier": tier,
                       "graph_hash": e.metadata.get("graph_hash"), "source": e.source,
                       "blocks": sorted(e.metadata.get("functional_blocks", {})),
                       "retrieval_document": e.retrieval_document()})
    scored.sort(key=lambda r: (-r["score"], r["topology_id"]))
    return scored[:k]


# -- splits -------------------------------------------------------------------
def build_splits(registry: TopologyRegistry, out_root: str | Path = "datasets/topology_splits",
                 seed: int = 42) -> dict[str, Any]:
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    fams = registry.list_topologies()
    rng = Random(seed)
    shuffled = fams[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test, n_val = round(n * 0.15), round(n * 0.15)
    family_split = {"seed": seed,
                    "test": sorted(shuffled[:n_test]),
                    "validation": sorted(shuffled[n_test:n_test + n_val]),
                    "train": sorted(shuffled[n_test + n_val:])}
    lit = registry.filter_by_source("analoggym")
    holdout = sorted(Random(seed).sample(lit, max(1, len(lit) // 4))) if lit else []
    bench = {"seed": seed, "heldout_analoggym_families": holdout,
             "development": sorted(set(fams) - set(holdout)),
             "excluded_from": ["sft", "complete_topology_rag", "selection_examples", "training_trajectories"],
             "block_level_retrieval": "allowed only if protocol explicitly permits"}
    (out / "family_split.json").write_text(json.dumps(family_split, indent=1), encoding="utf-8")
    (out / "benchmark_holdout_split.json").write_text(json.dumps(bench, indent=1), encoding="utf-8")
    overlap = set(family_split["train"]) & set(family_split["test"])
    (out / "leakage_report.md").write_text(
        f"# Split leakage report\n- families: {n}\n- train/test overlap: {len(overlap)} (must be 0)\n"
        f"- benchmark holdout: {len(holdout)} AnalogGym families excluded from SFT, "
        "complete-topology RAG, selection examples, and training trajectories.\n", encoding="utf-8")
    return {"families": n, "train": len(family_split["train"]), "validation": len(family_split["validation"]),
            "test": len(family_split["test"]), "benchmark_holdout": len(holdout), "overlap": len(overlap)}
