"""OPAMP-Generator extractor: three-stage op-amp DAG rows → block-level graphs.

Row format (topo_opt/dataset_withoutY_1w.txt): nested lists, one list per DAG
node; element 0 = node type (netlist_generator.get_node_info: 1=R, 2=C,
3/4=RC par/ser, 5/6=feedforward ±gm, 15/16=feedback ±gm, 7-14/17-24=R|C combos
with ff/fb gm); remaining elements describe links to earlier nodes (nonzero →
edge). The main amplification path IN→gm1→gm2→gm3→OUT is implicit in the
three-stage template. mapping_status="approximate_dag_decode" — stage-level
structure is faithful; exact branch-node semantics are approximated.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tools.topology_extractor.common import ExtractedTopology, block_graph

_TYPE_ROLE = {1: "R", 2: "C", 3: "RC_parallel", 4: "RC_series",
              5: "ff_gm_pos", 6: "ff_gm_neg", 15: "fb_gm_pos", 16: "fb_gm_neg"}
for t in (7, 8, 11, 12):
    _TYPE_ROLE[t] = "ff_gm_pos_RC"
for t in (9, 10, 13, 14):
    _TYPE_ROLE[t] = "ff_gm_neg_RC"
for t in (17, 18, 19, 20):
    _TYPE_ROLE[t] = "fb_gm_pos_RC"
for t in (21, 22, 23, 24):
    _TYPE_ROLE[t] = "fb_gm_neg_RC"


def row_to_graph(name: str, row: list[list[int]]) -> "ExtractedTopology":
    nodes = [("gm1", "gain_stage"), ("gm2", "gain_stage"), ("gm3", "gain_stage")]
    edges = [("IN", "gm1"), ("gm1", "gm2"), ("gm2", "gm3"), ("gm3", "OUT"),
             ("VDD", "gm1"), ("GND", "gm1")]
    anchors = ["IN", "gm1", "gm2", "gm3", "OUT"]
    for j, spec in enumerate(row):
        if not spec:
            continue
        t = spec[0]
        role = _TYPE_ROLE.get(t)
        if role is None:
            continue
        nid = f"b{j}_{role}"
        nodes.append((nid, role))
        srcs = [k for k, v in enumerate(spec[1:]) if v] or [max(0, j - 1)]
        src = anchors[min(srcs[0], len(anchors) - 1)]
        dst = anchors[min(j + 1, len(anchors) - 1)]
        if role.startswith("fb"):
            src, dst = dst, src
        if src == dst:
            dst = "OUT"
        edges.append((src, nid))
        edges.append((nid, dst))
    graph = block_graph(name, nodes, edges, family_hint="three_stage_opamp")
    graph.metadata.source = "opamp_generator"
    return ExtractedTopology(name=name, repository="opamp_generator", graph=graph,
                             mapping_status="approximate_dag_decode",
                             metadata={"paper": "Lu et al. OPAMP-Generator", "encoding": row})


def extract(repo_root: Path, sample_limit: int = 1500) -> list[ExtractedTopology]:
    dataset = repo_root / "topo_opt" / "dataset_withoutY_1w.txt"
    out: list[ExtractedTopology] = []
    for i, line in enumerate(dataset.read_text(encoding="utf-8").splitlines()):
        if i >= sample_limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            row = ast.literal_eval(line)
            out.append(row_to_graph(f"opampgen_{i:05d}", row))
        except (ValueError, SyntaxError, Exception):
            continue
    return out
