"""Topology corpus: registry, RAG index, selection, splits, leakage, MCTS compat."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_raptor.corpus import (
    TopologyRegistry,
    build_rag_index,
    build_splits,
    select_topology_candidates,
)

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registry() -> TopologyRegistry:
    return TopologyRegistry(_ROOT / "datasets" / "topology_library")


def test_dynamic_discovery_no_hardcoded_count(registry):
    n = registry.get_family_count()
    assert n >= 30 and n == len(registry.list_topologies())
    assert not registry.excluded, f"unexpected exclusions: {registry.excluded}"


def test_required_and_optional_files(registry):
    tid = registry.list_topologies()[0]
    e = registry.get_topology(tid)
    assert e.graph.nodes and e.metadata and isinstance(e.blocks, dict)
    with_netlist = [t for t in registry.list_topologies() if registry.get_netlist(t)]
    without = [t for t in registry.list_topologies() if registry.get_netlist(t) is None]
    assert with_netlist and without, "corpus mixes netlist/non-netlist entries; loader must handle both"


def test_filters_and_hash_lookup(registry):
    ag = registry.filter_by_source("analoggym")
    assert len(ag) >= 10
    tid = ag[0]
    h = registry.get_metadata(tid)["graph_hash"]
    assert tid in registry.find_by_graph_hash(h)
    assert registry.filter_by_validation_status("unvalidated"), "corpus is honest: unvalidated exists"
    assert registry.duplicate_hashes() == {}, "library must not contain duplicate canonical hashes"


def test_rag_index_no_fabricated_performance(registry, tmp_path):
    counts = build_rag_index(registry, tmp_path)
    assert counts["topology_records"] == registry.get_family_count()
    assert counts["block_records"] > 0 and counts["graph_records"] == counts["topology_records"]
    assert counts["simulation_memory_records"] == 0  # level 4 empty by design
    text = (tmp_path / "topology_records.jsonl").read_text()
    for banned in ("gain_db", "phase_margin", "bandwidth_hz", "noise", "power_w"):
        assert banned not in text, "no invented electrical performance in retrieval docs"


def test_selection_topk_with_tiers(registry, spec):
    cands = select_topology_candidates(registry, spec, k=5, stage_count=3)
    assert len(cands) == 5
    assert all(c["score_decomposition"] for c in cands)
    tiers = {c["candidate_tier"] for c in cands}
    assert tiers <= {"tier_A_electrically_qualified", "tier_B_netlist_unqualified",
                     "tier_C_mapping_required", "tier_D_invalid"}
    again = select_topology_candidates(registry, spec, k=5, stage_count=3)
    assert [c["topology_id"] for c in cands] == [c["topology_id"] for c in again]  # deterministic


def test_splits_reproducible_and_leak_free(registry, tmp_path):
    a = build_splits(registry, tmp_path, seed=42)
    b = build_splits(registry, tmp_path, seed=42)
    assert a == b and a["overlap"] == 0
    assert a["train"] + a["validation"] + a["test"] == a["families"]
    assert a["benchmark_holdout"] >= 1


def test_mcts_operates_on_corpus_graph(registry, spec):
    from agentic_raptor.topology_rl.mcts import MCTS, MCTSConfig
    from agentic_raptor.topology_rl.policy_value_network import HeuristicEvaluator

    tid = registry.filter_by_source("analoggym")[0]
    graph = registry.get_graph(tid)
    result = MCTS(HeuristicEvaluator(), config=MCTSConfig(num_simulations=4, max_depth=2)).run(
        graph, spec, 1.0
    )
    assert result.best_action is not None  # canonical CircuitGraph, one MCTS for all sources
