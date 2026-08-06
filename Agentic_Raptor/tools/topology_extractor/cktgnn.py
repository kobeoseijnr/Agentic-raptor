"""CktGNN extractor: OCB CktBench101 igraph pickles â†’ block-level graphs.

Requires python-igraph to unpickle. If unavailable or the pickle cannot be
decoded, topologies are reported as skipped (unknown mappings) â€” never
fabricated. Node types follow the OCB amplifier subgraph basis (gmÂ± / R / C
combinations on 2-3 stage op-amps); type ints are mapped to coarse roles.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from tools.topology_extractor.common import ExtractedTopology, block_graph


def _role(t: int) -> str:
    """VERIFIED decoding (Stage 3C.2): SUBG_NODE basis; unknown codes raise."""
    from agentic_raptor.mapping.interpretation import SUBG_NODE
    if t not in SUBG_NODE:
        raise ValueError(f"unknown_cktgnn_subgraph_type:{t}")
    comps = SUBG_NODE[t]
    if comps == ["In"]: return "port"
    if comps == ["Out"]: return "port"
    return "+".join(comps)

def extract(repo_root: Path, sample_limit: int = 400) -> list[ExtractedTopology]:
    pkl = repo_root / "OCB" / "CktBench101" / "ckt_bench_101.pkl"
    out: list[ExtractedTopology] = []
    try:
        import igraph  # noqa: F401
        with pkl.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        return [ExtractedTopology("cktbench101", "cktgnn",
                                  block_graph("cktbench101_skipped", [], []),
                                  mapping_status="skipped",
                                  metadata={"error": f"{type(exc).__name__}: {exc}",
                                            "note": "install python-igraph to decode OCB pickles"})]
    graphs = data[0] if isinstance(data, tuple) else data
    for i, item in enumerate(graphs[:sample_limit]):
        g = item[0] if isinstance(item, (tuple, list)) else item
        try:
            vs_types = [int(v["type"]) for v in g.vs]
            nodes, edges = [], []
            ids = {}
            for vi, t in enumerate(vs_types):
                if t == 0:
                    ids[vi] = "IN"
                elif t == 1 or vi == len(vs_types) - 1 and _role(t) == "port":
                    ids[vi] = "OUT"
                else:
                    ids[vi] = f"n{vi}_{_role(t)}"
                    nodes.append((ids[vi], _role(t)))
            for e in g.es:
                edges.append((ids[e.source], ids[e.target]))
            if not any(dst == "OUT" for _s, dst in edges):
                edges.append((nodes[-1][0] if nodes else "IN", "OUT"))
            graph = block_graph(f"cktgnn_{i:05d}", nodes, edges, family_hint="ocb_opamp")
            graph.metadata.source = "cktgnn"
            out.append(ExtractedTopology(f"cktgnn_{i:05d}", "cktgnn", graph,
                                         metadata={"paper": "Dong et al. CktGNN (OCB)"}))
        except Exception as exc:
            out.append(ExtractedTopology(f"cktgnn_{i:05d}", "cktgnn",
                                         block_graph(f"cktgnn_{i:05d}_skipped", [], []),
                                         mapping_status="skipped", metadata={"error": str(exc)}))
    return out

