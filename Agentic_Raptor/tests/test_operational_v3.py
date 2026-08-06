"""Stage 3C.2b operational registry tests."""
import json
from pathlib import Path
from agentic_raptor.corpus import TopologyRegistry

_ROOT = Path(__file__).resolve().parents[1]
V3 = _ROOT / "artifacts" / "topology_registry_operational_v3"
REG = TopologyRegistry(V3)

def test_exact_174_and_source_counts():
    assert REG.get_family_count() == 174
    s = json.loads((V3 / "SUMMARY.json").read_text())["source_counts"]
    assert s == {"analoggym": 17, "opamp_generator": 15, "cktgnn_v2": 142}

def test_exclusions_and_descendants():
    ids = REG.list_topologies()
    assert "topology_0057" not in ids
    assert not any(i.startswith("cktgnn_") for i in ids)
    assert sum(1 for i in ids if i.startswith("topology_og2")) == 15

def test_hash_version_2_everywhere():
    for tid in REG.list_topologies():
        assert REG.get_metadata(tid).get("hash_version", 2) == 2

def test_splits_deterministic_and_leakfree():
    split = json.loads((V3 / "split_v3.json").read_text())["assignments"]
    assert len(split) == 174
    og_splits = {v for k, v in split.items() if k.startswith("topology_og2")}
    assert len(og_splits) == 1, "og2 descendants must share one split group"
    hashes_by_split = {}
    for tid, s in split.items():
        h = REG.get_metadata(tid)["graph_hash"]
        assert hashes_by_split.setdefault(h, s) == s, "hash leakage across splits"

def test_strict_tier_policy():
    t = json.loads((V3 / "SUMMARY.json").read_text())["tiers"]
    rows = [json.loads(x) for x in
            (_ROOT / "datasets/simulation_memory/stage3c2b_runs.jsonl").read_text().splitlines()]
    unstable_functional = [r for r in rows if r.get("electrical") == "electrically_functional"
                           and r.get("stability") == "verified_unstable"]
    assert t.get("A2", 0) + len(unstable_functional) >= len(
        [r for r in rows if r.get("electrical") == "electrically_functional"])
    assert all(r.get("stability") != "verified_stable" or "metrics" in r for r in rows)

def test_withheld_not_counted_as_failed():
    rows = [json.loads(x) for x in
            (_ROOT / "datasets/simulation_memory/stage3c2b_runs.jsonl").read_text().splitlines()]
    for r in rows:
        if str(r.get("status", "")).startswith("withheld"):
            assert "electrical" not in r and "failure_class" not in r

def test_rag_rebuilt_on_v3():
    meta = json.loads((_ROOT / "datasets/topology_rag/index_metadata.json").read_text())
    assert meta["topology_records"] == 174
