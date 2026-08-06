"""Stage 3A: identifiers, splits, validators, and the four smoke tests (mock-mode pipeline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_raptor.stage3a import (
    assign_splits,
    canonical_spec_hash,
    check_leakage,
    content_id,
    duplicate_report,
    validate_records,
)

SPEC = {"circuit_class": "ota", "technology": "g", "supply_voltage": 1.8, "source_metadata": {"x": "text"}}


def test_spec_hash_deterministic_and_ignores_provenance():
    assert canonical_spec_hash(SPEC) == canonical_spec_hash({**SPEC, "source_metadata": {}})
    assert canonical_spec_hash(SPEC) != canonical_spec_hash({**SPEC, "supply_voltage": 3.3})
    assert content_id("graph", {"a": 1}) == content_id("graph", {"a": 1})


def test_split_algorithm_fixture():  # correction 3: fixture-tested splitter
    groups = [f"g{i}" for i in range(10)]
    split = assign_splits(groups, seed=1)
    assert set(split.values()) == {"train", "validation", "test"}
    assert assign_splits(groups, seed=1) == split  # deterministic
    tiny = assign_splits(["a", "b"], seed=1)       # empty debug splits permitted
    assert set(tiny.values()) <= {"train", "validation", "test"}


def test_validator_quarantines_not_drops():
    rows = [
        {"x": 1, "execution_mode": "REAL", "split": "train", "group_key": "g"},
        {"x": None, "execution_mode": "REAL", "split": "train", "group_key": "g"},
        {"x": 1, "execution_mode": "MOCK", "split": "train", "group_key": "g"},
        {"x": 1, "execution_mode": "REAL", "real_or_imagined": "MODEL", "split": "t", "group_key": "g"},
        {"x": 1, "execution_mode": "REAL", "reached_spice": True, "split": "t", "group_key": "g"},
    ]
    ok, quar = validate_records(rows, ["x"], real_only=True)
    assert len(ok) == 1 and len(quar) == 4
    assert all(q["reasons"] for q in quar)


def test_leakage_and_duplicates():
    clean = [{"split": "train", "group_key": "a"}, {"split": "test", "group_key": "b"}]
    leaky = clean + [{"split": "test", "group_key": "a"}]
    assert check_leakage({"d": clean})["prohibited_leakage"] == 0
    assert check_leakage({"d": leaky})["prohibited_leakage"] == 1
    dup = duplicate_report([{"k": "x"}, {"k": "x"}, {"k": "y"}], "k")
    assert dup["duplicate_fraction"] == pytest.approx(1 / 3)


@pytest.fixture(scope="module")
def debug_run(tmp_path_factory):
    """Smoke A–D backbone in MOCK mode (real mode exercised via CLI, reported separately)."""
    from agentic_raptor.stage3a.generate import run_debug_generation

    root = tmp_path_factory.mktemp("s3a") / "stage3a"
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "experiments" / "smoke_test.yaml")
    out = run_debug_generation(cfg, str(root), seeds=[1, 2, 3, 4, 5], mode="MOCK")
    return root, out["report"]


def test_smoke_A_linkage(debug_run):
    root, report = debug_run
    assert report["runs"] == 5 and report["reached_spice"] >= 1
    import json
    unified = [json.loads(x) for x in (root / "processed/unified/unified_design_runs.jsonl").read_text().splitlines()]
    for r in unified:
        assert r["specification_id"].startswith("spec-") and r["run_id"] and r["episode_id"]


def test_smoke_B_sft_roundtrip(debug_run):
    import json
    root, _ = debug_run
    rows = []
    for s in ("train", "validation", "test"):
        p = root / f"processed/sft/sft_{s}.jsonl"
        rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert rows, "at least one SFT example"
    r = rows[0]
    assert r["target_circuit_graph"]["nodes"] and r["spice_evidence"]
    assert r["multimodal_assets"][0]["modality"] == "text"  # correction 6 provenance


def test_smoke_C_preferences_gated(debug_run):
    import json
    root, _ = debug_run
    rows = []
    for s in ("train", "validation", "test"):
        rows += [json.loads(x) for x in (root / f"processed/search_ranker/search_ranker_{s}.jsonl").read_text().splitlines() if x.strip()]
    for r in rows:
        assert r["preferred_candidate"] in ("candidate_a", "candidate_b")
        assert r["preference_basis"] == "real_spice_evidence"
        assert "chosen_response" not in r  # not an LLM-DPO format


def test_smoke_D_rl_datasets(debug_run):
    import json
    root, report = debug_run
    topo = []
    for s in ("train", "validation", "test"):
        topo += [json.loads(x) for x in (root / f"processed/topology_rl/topology_rl_{s}.jsonl").read_text().splitlines() if x.strip()]
    assert topo and all(t["pre_action_graph"] for t in topo)
    sac = []
    for s in ("train", "validation", "test"):
        sac += [json.loads(x) for x in (root / f"processed/mb_sac/mb_sac_{s}.jsonl").read_text().splitlines() if x.strip()]
    assert all(t["real_or_imagined"] == "REAL" for t in sac)
    assert report["leakage"] == 0
    assert report["verdict"] in ("DEBUG_DATASET_VALID", "DEBUG_DATASET_INVALID")
