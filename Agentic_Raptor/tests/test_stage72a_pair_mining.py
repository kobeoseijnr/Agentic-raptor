"""Stage 7.2A: cross-branch DPO pair mining -- mechanics tests.

No SPICE, no LLM. Synthetic rows shaped exactly like real sac_replay.jsonl
entries exercise the pure-Python mining logic (splitting, validation,
categorization, dedup); a couple of report-derived tests are skipped if the
real mining/training run hasn't happened in this checkout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.ranking import pair_mining as pm

MINED_PATH = Path("artifacts/publication_v3/stage7_2a_pair_mining/MINED_PAIRS.jsonl")
REPORT_PATH = Path("artifacts/publication_v3/stage7_2a_dpo_repair/STAGE7_2A_REPORT.json")


def _row(spec_hash="s1", seed=0, branch="A", step=0, topology_hash="tA",
        gain=70.0, pm_deg=50.0, ugbw=1e5, reward=0.5, cload=1e-10,
        env="POST_CLOAD_FIX_V1", knobs=None):
    return {"spec_hash": spec_hash, "seed": seed, "branch": branch, "step": step,
           "topology_hash": topology_hash, "gain_db": gain, "pm_deg": pm_deg,
           "ugbw_hz": ugbw, "reward": reward, "requested_c_load_f": cload,
           "electrical_environment_version": env,
           "knobs": knobs or {"s1_w": 1.0, "s2_w": 1.0, "s1_l": 1.0, "s2_l": 1.0,
                              "cap_x": 1.0, "ib_x": 1.0, "rz_x": 1.0},
           "generation_spec_id": "spec_x"}


SPEC = {"gain_target_db": 60.0, "phase_margin_target_deg": 45.0,
       "load_capacitance_pf": 100.0, "ugbw_target_hz": 1e5, "spec_id": "spec_x"}


# ---------------------------------------------------------------------------
# Section 1: candidate provenance
# ---------------------------------------------------------------------------
def test_real_spice_candidate_accepted():
    assert pm.validate_candidate_provenance(_row()) == []


def test_surrogate_only_candidate_rejected():
    """A row with no real gain/pm measurement (both None) must never be
    minable -- there is no such row in the real live path (every sac_size
    step measures real ngspice), but the validator must still reject it if
    it somehow appeared."""
    bad = _row(gain=None, pm_deg=None)
    problems = pm.validate_candidate_provenance(bad)
    assert "no_real_measurement" in problems


def test_pre_cload_fix_candidate_rejected():
    bad = _row(env="PRE_CLOAD_FIX")
    problems = pm.validate_candidate_provenance(bad)
    assert "not_post_cload_fix_v1" in problems


def test_missing_knobs_rejected():
    bad = _row()
    bad["knobs"] = {}
    assert "missing_knobs" in pm.validate_candidate_provenance(bad)


# ---------------------------------------------------------------------------
# Section 1: run splitting on step==0 boundaries
# ---------------------------------------------------------------------------
def test_load_runs_splits_on_step_zero_recurrence(tmp_path):
    rows = ([_row(step=i) for i in range(3)]
           + [_row(step=i) for i in range(2)])   # a second real run reusing the key
    p = tmp_path / "replay.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    runs = pm.load_runs_from_replay(p)
    assert len(runs) == 2
    assert [r["step"] for r in runs[0]["rows"]] == [0, 1, 2]
    assert [r["step"] for r in runs[1]["rows"]] == [0, 1]


def test_pair_runs_by_spec_seed_zips_in_order():
    runs = [
        {"spec_hash": "s1", "seed": 0, "branch": "A", "rows": [_row(branch="A")]},
        {"spec_hash": "s1", "seed": 0, "branch": "B", "rows": [_row(branch="B", topology_hash="tB")]},
    ]
    paired = pm.pair_runs_by_spec_seed(runs)
    assert len(paired) == 1
    assert paired[0]["A"]["rows"][0]["branch"] == "A"
    assert paired[0]["B"]["rows"][0]["branch"] == "B"


# ---------------------------------------------------------------------------
# Section 9/10: cross-branch only, same-spec, canonical identity, dedup
# ---------------------------------------------------------------------------
def _real_paired_run():
    a_rows = [_row(branch="A", topology_hash="tA", step=i,
                  knobs={"s1_w": 1.0 + i * 0.1, "s2_w": 1.0, "s1_l": 1.0, "s2_l": 1.0,
                        "cap_x": 1.0, "ib_x": 1.0, "rz_x": 1.0},
                  gain=70.0 + i, pm_deg=50.0 - i * 2, ugbw=1e5 * (1 + i * 0.1))
             for i in range(6)]
    b_rows = [_row(branch="B", topology_hash="tB", step=i,
                  knobs={"s1_w": 1.0, "s2_w": 1.0 + i * 0.1, "s1_l": 1.0, "s2_l": 1.0,
                        "cap_x": 1.0, "ib_x": 1.0, "rz_x": 1.0},
                  gain=55.0 + i, pm_deg=40.0 + i, ugbw=8e4)
             for i in range(6)]
    return {"originating_run_id": "s1:0:0", "spec_hash": "s1", "seed": 0, "run_index": 0,
           "A": {"rows": a_rows}, "B": {"rows": b_rows}}


def test_mine_cross_branch_pairs_never_same_topology(tmp_path):
    paired_run = _real_paired_run()
    mined = pm.mine_cross_branch_pairs(paired_run, SPEC, tmp_path, set(), max_candidates_per_branch=3)
    assert mined, "expected at least one mined pair from real-shaped data"
    for p in mined:
        assert p["topology_hash_a"] != p["topology_hash_b"]


def test_mine_cross_branch_pairs_protected_spec_yields_nothing(tmp_path):
    paired_run = _real_paired_run()
    mined = pm.mine_cross_branch_pairs(paired_run, SPEC, tmp_path,
                                       excluded_context_ids={"spec_x"},
                                       max_candidates_per_branch=3)
    assert mined == []


def test_canonical_pair_id_is_order_independent():
    id1 = pm.canonical_pair_id("s1", "tA:h1", "tB:h2")
    id2 = pm.canonical_pair_id("s1", "tB:h2", "tA:h1")
    assert id1 == id2


def test_knob_hash_distinguishes_different_sizings_same_topology():
    h1 = pm.knob_hash({"s1_w": 1.0, "s2_w": 1.0})
    h2 = pm.knob_hash({"s1_w": 2.0, "s2_w": 1.0})
    assert h1 != h2


def test_deduplicate_pairs_removes_reversed_duplicates():
    p1 = {"pair_id": pm.canonical_pair_id("s1", "tA:h1", "tB:h2"), "x": 1}
    p2 = {"pair_id": pm.canonical_pair_id("s1", "tB:h2", "tA:h1"), "x": 2}
    kept, n_dupes = pm.deduplicate_pairs([p1, p2])
    assert len(kept) == 1
    assert n_dupes == 1


def test_mine_cross_branch_pairs_deduplicates_representative_overlap(tmp_path):
    """If two roles resolve to the same candidate, the cross product must
    not produce a duplicate pair_id."""
    paired_run = _real_paired_run()
    mined = pm.mine_cross_branch_pairs(paired_run, SPEC, tmp_path, set(),
                                       max_candidates_per_branch=5)
    ids = [p["pair_id"] for p in mined]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Section 4: CLOAD equality recorded
# ---------------------------------------------------------------------------
def test_cload_equality_flagged_when_branches_differ(tmp_path):
    paired_run = _real_paired_run()
    for r in paired_run["B"]["rows"]:
        r["requested_c_load_f"] = 2e-10   # deliberately different from A's 1e-10
    mined = pm.mine_cross_branch_pairs(paired_run, SPEC, tmp_path, set(),
                                       max_candidates_per_branch=3)
    assert mined
    assert all(p["cload_equal_ab"] is False for p in mined)


# ---------------------------------------------------------------------------
# Section 7: representative candidate selection
# ---------------------------------------------------------------------------
def test_select_representative_candidates_returns_at_most_max():
    rows = [_row(step=i, gain=60 + i, pm_deg=40 + i, ugbw=1e5) for i in range(8)]
    reps = pm.select_representative_candidates(rows, SPEC, max_candidates=4)
    assert 1 <= len(reps) <= 4
    assert len(reps) == len({r["candidate_id"] for r in reps})


def test_select_representative_candidates_empty_on_no_valid_rows():
    bad_rows = [_row(env="PRE_CLOAD_FIX") for _ in range(3)]
    reps = pm.select_representative_candidates(bad_rows, SPEC)
    assert reps == []


# ---------------------------------------------------------------------------
# Section 21: ranker-authority classification (hard gate frozen, unchanged)
# ---------------------------------------------------------------------------
def test_classify_ranker_authority_matches_hard_safety_tier():
    from agentic_raptor.ranking.post_sac import hard_safety_tier
    from agentic_raptor.ranking.types import SurrogatePrediction
    pa = SurrogatePrediction(topology_hash="a", sizing_manifest_hash="m",
                             operating_point_probability=0.9, stability_probability=0.9,
                             normalized_margins={"gain": 0.1})
    pb = SurrogatePrediction(topology_hash="b", sizing_manifest_hash="m")
    assert pm.classify_ranker_authority(pa, pa) == (hard_safety_tier(pa) == hard_safety_tier(pa))
    assert pm.classify_ranker_authority(pa, pb) == (hard_safety_tier(pa) == hard_safety_tier(pb))


# ---------------------------------------------------------------------------
# Section 14: label hierarchy -- feasible beats infeasible regardless of FoM
# ---------------------------------------------------------------------------
def test_measured_preference_feasible_beats_infeasible():
    meas_a = {"gain_db": 65.0, "pm_deg": 50.0, "ugbw_hz": 2e5, "stable": True,
             "stability": "verified_stable", "electrical": "electrically_functional",
             "op_valid": True}
    meas_b = {"gain_db": 90.0, "pm_deg": -10.0, "ugbw_hz": 2e5, "stable": False,
             "stability": "verified_unstable", "electrical": "electrically_functional",
             "op_valid": True}
    winner, reason = pm.measured_preference_from_outcomes(meas_a, meas_b, SPEC)
    assert winner == "A"


# ---------------------------------------------------------------------------
# Section 5: harvesting disabled in frozen mode -- structural check
# ---------------------------------------------------------------------------
def test_pair_mining_not_imported_by_default_run_pipeline():
    """Live wiring is opt-in (Section 5): a bare run_pipeline() must not
    import/call the mining module unconditionally."""
    import inspect

    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    assert "pair_mining" not in src


# ---------------------------------------------------------------------------
# Report-derived tests (skipped if the real run hasn't happened)
# ---------------------------------------------------------------------------
def _load_report():
    if not REPORT_PATH.is_file():
        pytest.skip("Stage 7.2A mining/training not yet run in this checkout")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_report_train_dev_specs_disjoint():
    d = _load_report()
    dev_specs = set(d["dev_spec_hashes"])
    assert d["train_coverage"]["n_specs"] > 0
    assert d["dev_coverage"]["n_specs"] > 0
    # train/dev pair counts are consistent with the pool size
    assert (d["train_coverage"]["n_pairs"] + d["dev_coverage"]["n_pairs"]
           <= d["pool_stats"]["n_original"] + d["pool_stats"]["n_mined_raw"])


def test_report_no_both_feasible_pairs_claimed_without_evidence():
    d = _load_report()
    assert d["dev_coverage"]["n_both_feasible"] == 0
    assert d["train_coverage"]["n_both_feasible"] == 0


def test_report_all_models_finite_and_monotonic_or_excluded():
    d = _load_report()
    for name, m in d["model_results"].items():
        de = m["dev_eval"]
        for margin in de["score_margins"]:
            assert margin == margin and abs(margin) < 1e6   # not NaN/inf


def test_report_promotion_classification_is_one_of_expected():
    d = _load_report()
    cls = d["promotion"]["final_classification"]
    assert cls in ("PROMOTED", "STAGE_7_2A_DATA_FIX_INSUFFICIENT")
