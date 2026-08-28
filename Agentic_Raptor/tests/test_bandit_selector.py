"""BANDIT_TOP2_V1 (2026-08-14): hash-pinned linear top-2 selector.
Real-candidate fixture pattern from test_alphazero_stage5d.py (real graphs
from corpus_diverse.json, no LLM calls, zero SPICE)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from agentic_raptor.topology_rl import bandit_selector as bs

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"

SPEC = {"spec_id": "bandit_test", "gain_target_db": 60.0,
       "phase_margin_target_deg": 55.0, "load_capacitance_pf": 100.0,
       "ugbw_target_hz": 1e5}


def _real_candidates(n=3):
    if not CORPUS.is_file():
        pytest.skip("corpus_diverse.json not present in this checkout")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    seen = {}
    for r in corpus["records"]:
        h = r["canonical_graph_hash"]
        if h not in seen:
            seen[h] = r
        if len(seen) >= n:
            break
    return [{"llm_proposal_id": f"p{i:02d}", "canonical_graph_hash": h,
            "canonical_family": r["topology_signature"],
            "obj": json.loads(r["response"]), "source": "llm"}
            for i, (h, r) in enumerate(seen.items())]


# ---------------------------------------------------------------------------
# weights artifact: hash pin + feature-drift guard
# ---------------------------------------------------------------------------
def test_load_verifies_sha_pin_and_feature_names():
    art = bs.load_bandit_v1()
    assert len(art["weights"]) == 24 == len(art["mu"]) == len(art["sd"])
    assert art["feasibility_gate_2026_08_14"]["verdict"] == "PASS"
    assert "TRAIN only" in art["training_data"]["domain"]


def test_tampered_weights_hard_fail(tmp_path, monkeypatch):
    fake = tmp_path / "BANDIT_TOP2_V1.json"
    fake.write_text(bs.BANDIT_V1_PATH.read_text(encoding="utf-8") + " ",
                    encoding="utf-8")
    monkeypatch.setattr(bs, "BANDIT_V1_PATH", fake)
    with pytest.raises(bs.BanditSelectorError, match="hash mismatch"):
        bs.load_bandit_v1()


def test_missing_weights_hard_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "BANDIT_V1_PATH", tmp_path / "nope.json")
    with pytest.raises(bs.BanditSelectorError, match="not found"):
        bs.load_bandit_v1()


# ---------------------------------------------------------------------------
# selection contract
# ---------------------------------------------------------------------------
def test_selects_two_distinct_deterministically():
    cands = _real_candidates(3)
    a = bs.bandit_select_two(cands, SPEC, "ctx")
    b = bs.bandit_select_two(cands, SPEC, "ctx")
    assert [c["canonical_graph_hash"] for c in a["selected"]] == \
           [c["canonical_graph_hash"] for c in b["selected"]]
    assert a["selected"][0]["canonical_graph_hash"] != \
           a["selected"][1]["canonical_graph_hash"]
    assert a["search"] == "bandit_top2"
    assert a["weights_sha256"] == bs.PROMOTED_BANDIT_SHA256  # V2 since 2026-08-16
    # ranked entries carry the keys the harvest/trace paths read
    for c in a["ranked"]:
        for k in ("bandit_score", "rank", "visit_count", "selected_top2",
                  "llm_proposal_id", "canonical_graph_hash", "obj"):
            assert k in c
    # scores strictly ordered (rank 0 has the max score)
    scores = [c["bandit_score"] for c in a["ranked"]]
    assert scores == sorted(scores, reverse=True)


def test_fewer_than_two_candidates_hard_fail():
    with pytest.raises(bs.BanditSelectorError, match=">= 2"):
        bs.bandit_select_two(_real_candidates(3)[:1], SPEC, "ctx")


def test_option_c_az_pool_dedupes_by_canonical_hash():
    cands = _real_candidates(3)
    # an AZ pick that IS an original seed must not double-enter the pool
    az_pick = dict(cands[0])
    az_pick.update({"device_graph": None, "edit_depth": 0,
                   "is_edited_descendant": False})
    out = bs.bandit_select_two(cands, SPEC, "ctx", az_selected=[az_pick])
    assert out["pool_size"] == 3
    assert out["az_contributed"] == 0
    assert out["search"] == "bandit_top2_az_hybrid"
    dup = next(c for c in out["ranked"]
               if c["canonical_graph_hash"] == cands[0]["canonical_graph_hash"])
    assert dup.get("also_alphazero_selected") is True


# ---------------------------------------------------------------------------
# pipeline wiring: bandit IS the live selector (2026-08-15 promotion);
# AlphaZero + hybrid stay opt-in
# ---------------------------------------------------------------------------
def test_run_pipeline_dispatch_and_promoted_default():
    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    assert 'search == "bandit_top2"' in src
    assert 'search == "bandit_top2_az"' in src
    # 2026-08-15 SELECTOR PROMOTION (user decision): bandit_top2 is the
    # live default; AlphaZero ("one_root") is opt-in research/ablation
    assert inspect.signature(v2.run_pipeline).parameters["search"].default \
        == "bandit_top2"
    # hybrid mode runs AlphaZero as generator then hands top-2 decision
    # to the bandit
    assert "az_selected=_az[\"selected\"]" in src


def test_feature_impl_is_the_shared_one():
    src = inspect.getsource(bs.bandit_select_two)
    assert "physical_features_core" in src   # single shared feature impl
    art = bs.load_bandit_v1()
    from agentic_raptor.topology_rl.linear_value import FEATURE_NAMES
    assert tuple(art["feature_names"]) == tuple(FEATURE_NAMES)
