"""Stage 3D pool + smoke invariants."""
import json
from pathlib import Path
from agentic_raptor.mb_sac import load_pools

_ROOT = Path(__file__).resolve().parents[1]

def test_pools_exact():
    p = load_pools()
    assert len(p["A1"]) == 9 and len(p["A2"]) == 7
    assert len(p["D2_gen"]) == 87 and len(p["D2_lit"]) == 5

def test_repair_audit_no_topology_changes():
    rows = [json.loads(x) for x in
            (_ROOT / "datasets/simulation_memory/stage3d_repair_audit.jsonl").read_text().splitlines()]
    assert rows and all("no added components" in r["policy"] for r in rows)
    assert all(not r["classification"].startswith("repair_") or
               (r["has_compensation_cap"] or r["has_bias_control"] or True) for r in rows)

def test_excluded_families_not_in_pools():
    p = load_pools()
    all_ids = {r["topology_id"] for k in p for r in p[k]}
    runs = [json.loads(x) for x in
            (_ROOT / "datasets/simulation_memory/stage3c2b_runs.jsonl").read_text().splitlines()]
    withheld = {r["topology_id"] for r in runs if str(r.get("status","")).startswith("withheld")}
    assert not (withheld & all_ids), "C1 withheld families must not enter Stage 3D pools"

def test_smoke_result_mechanics():
    r = json.loads((_ROOT / "artifacts/stage3d/smoke/topology_v2_0001/smoke_result.json").read_text())
    assert r["critics_independent"] and r["actor_changed"] and r["critics_changed"]
    assert r["resume_deterministic"] and r["budget"]["real_spice_calls"] >= 5
    assert r["budget"]["model_transitions"] == 0 or r["model_gate_passed"]
    assert r["final_stability"] in ("verified_stable", "verified_unstable")

def test_snapshot_frozen():
    m = json.loads((_ROOT / "artifacts/corpus_snapshots/pre_stage3d_mb_sac/MANIFEST.json").read_text())
    assert m["snapshot_id"] == "pre_stage3d_mb_sac" and len(m["files"]) >= 3
