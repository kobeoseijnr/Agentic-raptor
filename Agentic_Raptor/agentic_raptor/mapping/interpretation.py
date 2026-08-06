"""Stage 3C.1: gm-block interpretation with VERIFIED CktGNN semantics.

Verified from repositories/CktGNN/OCB/src/circuit_generation.py L33-70:
NODE_TYPE {R:0, C:1, '+gm+':2, '-gm+':3, '+gm-':4, '-gm-':5, sudo_in:6,
sudo_out:7, In:8, Out:9}; SUBG_NODE basis 0-25 (In/Out, R, C, R+C par/ser,
lone ±gm±, C∥gm, R∥gm, C-R-gm par/ser combos). gm sign convention (OCB paper +
amp_generator.py): FIRST sign = transconductance polarity, SECOND sign =
path direction (+ feedforward main-path, − feedback).

CONSEQUENCE (honest): the Stage 3A extractor labelled corpus CktGNN nodes with
an unverified modulo heuristic → stored roles are unreliable for types ≥4.
Therefore ALL 37 behavioral families are classified
`withheld_at_semantic_audit` pending re-extraction with this verified table.
The interpreter below is implemented and unit-tested on synthetic graphs using
the verified semantics, so re-extraction → interpretation is one step.
No interpretation is ever selected using simulation outcomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: VERIFIED tables (evidence: circuit_generation.py L33-70)
NODE_TYPE = {"R": 0, "C": 1, "+gm+": 2, "-gm+": 3, "+gm-": 4, "-gm-": 5,
             "sudo_in": 6, "sudo_out": 7, "In": 8, "Out": 9}
SUBG_NODE = {0: ["In"], 1: ["Out"], 2: ["R"], 3: ["C"], 4: ["R", "C"], 5: ["R", "C"],
             6: ["+gm+"], 7: ["-gm+"], 8: ["+gm-"], 9: ["-gm-"],
             10: ["C", "+gm+"], 11: ["C", "-gm+"], 12: ["C", "+gm-"], 13: ["C", "-gm-"],
             14: ["R", "+gm+"], 15: ["R", "-gm+"], 16: ["R", "+gm-"], 17: ["R", "-gm-"],
             18: ["C", "R", "+gm+"], 19: ["C", "R", "-gm+"], 20: ["C", "R", "+gm-"],
             21: ["C", "R", "-gm-"], 22: ["C", "R", "+gm+"], 23: ["C", "R", "-gm+"],
             24: ["C", "R", "+gm-"], 25: ["C", "R", "-gm-"]}

GM_CATEGORIES = ("differential_input_transconductor", "single_ended_transconductor",
                 "common_source_equivalent", "output_transconductor",
                 "feedforward_transconductor", "feedback_transconductor",
                 "composite_transconductor", "unresolved_transconductor")


@dataclass
class FunctionalStage:
    stage_id: str
    function: str                       # GM_CATEGORIES ∪ {compensation_branch, load_branch}
    polarity: str                       # "+" | "-" | "unresolved"
    path_class: str                     # main_forward_path | feedforward_path | feedback_path | load_path
    source_nodes: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    rule: str = ""
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    unresolved_alternatives: list[str] = field(default_factory=list)


@dataclass
class FunctionalStageGraph:
    family_id: str
    candidate_id: str
    stages: list[FunctionalStage] = field(default_factory=list)
    stage_order: list[str] = field(default_factory=list)
    inversion_parity: int | None = None
    unresolved: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def validation(self) -> dict[str, Any]:
        main = [s for s in self.stages if s.path_class == "main_forward_path"]
        gm = [s for s in self.stages if "transconductor" in s.function or "common_source" in s.function]
        resolved = [s for s in gm if s.function != "unresolved_transconductor"]
        return {
            "functional_graph_valid": bool(main) and not self.unresolved,
            "main_path_coverage": len(main) / max(1, len(gm)) if gm else 0.0,
            "interpreted_gm_fraction": len(resolved) / max(1, len(gm)) if gm else 0.0,
            "unresolved_gm_fraction": 1 - (len(resolved) / max(1, len(gm))) if gm else 1.0,
            "polarity_coverage": sum(1 for s in gm if s.polarity != "unresolved") / max(1, len(gm)),
        }


# ---- rule library (IDs R1-R5; outcome-independent by construction) ----------
def interpret_verified_dag(family_id: str, nodes: dict[str, str],
                           edges: list[tuple[str, str]]) -> FunctionalStageGraph:
    """Interpret a DAG whose node values are VERIFIED CktGNN type strings.

    nodes: node_id → type string ("In","Out","R","C","±gm±"); edges directed.
    """
    g = FunctionalStageGraph(family_id, "interp_0001",
                             provenance={"semantics": "verified circuit_generation.py L33-70"})
    succ: dict[str, list[str]] = {}
    pred: dict[str, list[str]] = {}
    for a, b in edges:
        succ.setdefault(a, []).append(b)
        pred.setdefault(b, []).append(a)
    in_nodes = [n for n, t in nodes.items() if t == "In"]
    out_nodes = [n for n, t in nodes.items() if t == "Out"]
    parity = 0
    order: list[str] = []
    for nid, t in nodes.items():
        if "gm" not in t:
            if t == "C":
                g.stages.append(FunctionalStage(f"comp_{nid}", "compensation_branch", "n/a",
                                                "load_path", [nid], pred.get(nid, []),
                                                succ.get(nid, []),
                                                rule="R5_capacitor_branch", confidence=0.9))
            continue
        sign, direction = t[0], t[-1]           # verified convention
        from_in = any(p in in_nodes for p in pred.get(nid, []))
        to_out = any(s in out_nodes for s in succ.get(nid, []))
        if direction == "-":
            fn, path, rule, conf = "feedback_transconductor", "feedback_path", "R4_second_sign_minus", 0.95
        elif from_in and not to_out:
            fn, path, rule, conf = "differential_input_transconductor", "main_forward_path", "R1_input_connected_gm", 0.85
        elif to_out:
            fn, path, rule, conf = "output_transconductor", "main_forward_path", "R3_gm_feeds_output", 0.85
        elif from_in and to_out:
            fn, path, rule, conf = "composite_transconductor", "main_forward_path", "R1+R3", 0.6
        else:
            fn, path, rule, conf = "common_source_equivalent", "main_forward_path", "R2_internal_gm", 0.75
        # feedforward: parallel main-path gm skipping stages
        if direction == "+" and from_in and to_out and len(nodes) > 3:
            fn, path, rule = "feedforward_transconductor", "feedforward_path", "R6_parallel_short_path"
        stage = FunctionalStage(f"gm_{nid}", fn, sign, path, [nid],
                                pred.get(nid, []), succ.get(nid, []), rule, conf,
                                evidence={"type": t, "from_in": from_in, "to_out": to_out})
        g.stages.append(stage)
        if path == "main_forward_path":
            order.append(stage.stage_id)
            if sign == "-":
                parity += 1
    g.stage_order = order
    g.inversion_parity = parity
    g.unresolved = [s.stage_id for s in g.stages if s.function == "unresolved_transconductor"]
    return g


def audit_behavioral_families(registry) -> list[dict[str, Any]]:
    """Classify all behavioral-only corpus families against verified semantics."""
    rows = []
    for tid in registry.list_topologies():
        e = registry.get_topology(tid)
        if e.source != "cktgnn":
            continue
        rows.append({
            "topology_id": tid, "graph_hash": e.metadata.get("graph_hash"),
            "interpretation_status": "withheld_at_semantic_audit",
            "reason": ("corpus node labels derived from an UNVERIFIED modulo type map "
                       "(tools/topology_extractor/cktgnn.py _role); verified basis "
                       "(circuit_generation.py L33-70) differs for types >=4 — "
                       "re-extraction with SUBG_NODE/NODE_TYPE required before interpretation"),
            "semantics_status": "semantics_verified_for_source__corpus_labels_unverified",
        })
    return rows


def write_withheld_memory(rows: list[dict[str, Any]], mem_root: Path) -> int:
    with (mem_root / "interpretation_runs.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)
