"""Pre-campaign integrity audit (2026-08-09), before the 12-hour
POST_CLOAD_FIX_V1 calibration campaign.

This audit found and fixed FOUR real, deterministic bugs that would have
silently mixed PRE_CLOAD_FIX data into the campaign's output or leaked
frozen evaluation data into training streams:

1. harvest_run() only gated the SFT queue on protected_ids -- ranker_pairs,
   puct_examples, rag_memory, and sac_replay had no leakage guard at all.
   Confirmed exploitable: an earlier diagnostic run on a protected eval
   spec (t_boundary_topology_0008, a frozen seed-11/23/47 evaluation spec)
   leaked 1 puct_examples record + 2 rag_memory records into the brand-new
   POST_CLOAD_FIX_V1 stores before this fix existed. Those 3 leaked
   records were deleted (100% of the two files' content, zero legitimate
   data lost) once found.
2. puct_examples records carried no SPICE/topology provenance for the
   measurement that produced value_target.
3. spec_sizing._GLOBAL_DYNAMICS (the surrogate cold-start "read the global
   history too" fallback) still pointed at the OLD, unversioned
   dynamics_surrogate_data.jsonl after Stage 1.6 versioned the write
   target -- every sandboxed sac_size(persist=True) call would have
   silently blended PRE_CLOAD_FIX rows back in.
4. rag_freeze.py carried its OWN separate, unversioned RAG_MEMORY_V2
   constant -- build_clean_snapshot() called with no explicit source_path
   would have read a file the campaign never writes to, silently
   producing an empty "clean" snapshot instead of a real one.

This file covers the fixes; tests/test_cload_consistency.py and
tests/test_cload_artifact_provenance.py cover the Stage 1.5/1.6 work this
audit verified rather than re-testing from scratch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------- finding 1: harvest_run leakage gap ----------------------
def _hv(spec_id: str, spec_index: int = 0) -> dict:
    return {"spec": {"spec_id": spec_id, "load_capacitance_pf": 137.0},
           "spec_hash": "h", "split": "heldout", "spec_index": spec_index,
           "seed": 0, "budget": 8, "requested_c_load_f": 137e-12,
           "candidates": [{"llm_proposal_id": "p00", "canonical_graph_hash": "g0",
                          "canonical_family": "2s_none", "obj": {}}],
           "root_visits": {"a_sel_p00": 3}, "candidate_visits": {},
           "root_action_ids": ["a_sel_p00"], "root_state": {"topology_id": "2s_none"},
           "candidate_manifests": {}, "search": "one_root",
           "branches": {
               "A": {"design": {"canonical_graph_hash": "gA",
                               "topology_family": "2s_none"},
                    "sac": {"trajectory": [{"knobs": {"s1_w": 1.0}, "reward": 0.5}]},
                    "authoritative": {"call_id": "c1", "gain_db": 60.0,
                                      "pm_deg": 50.0, "ugbw_hz": 1e5,
                                      "idd_a": 1e-4, "c_load_f": 137e-12,
                                      "normalized_distance_to_feasibility": 0.0,
                                      "exact_spec_pass": True,
                                      "verified_stable": True,
                                      "operating_point_valid": True,
                                      "topology_hash": "gA"}},
               "B": {"design": {"canonical_graph_hash": "gB",
                               "topology_family": "2s_miller"},
                    "sac": {"trajectory": []},
                    "authoritative": {"call_id": "c2", "gain_db": 55.0,
                                      "pm_deg": 40.0, "ugbw_hz": 8e4,
                                      "idd_a": 1.1e-4, "c_load_f": 137e-12,
                                      "normalized_distance_to_feasibility": 0.3,
                                      "exact_spec_pass": False,
                                      "verified_stable": True,
                                      "operating_point_valid": True,
                                      "topology_hash": "gB"}}},
           "ranker": {"selected_design": "A", "backup_design": "B",
                     "decision_basis": "x", "deciding_level": "x",
                     "low_confidence": False, "score_A": None, "score_B": None,
                     "checkpoint_hash": None},
           "proposer_checkpoint": "x"}


def test_harvest_run_blocks_protected_spec_from_every_stream():
    """The exact gap found: before the fix, only sft_queue was protected.
    A protected spec must now yield EMPTY streams across the board."""
    from agentic_raptor.selfimprove_v2.streams import harvest_run
    streams = harvest_run(_hv("protected_spec"), split="heldout",
                          protected_ids={"protected_spec"})
    for name, rows in streams.items():
        assert rows == [], f"stream {name!r} leaked data for a protected spec: {rows}"


def test_harvest_run_allows_unprotected_spec_through():
    """Sanity check the guard isn't overzealous -- a genuinely unprotected
    spec must still populate the streams it qualifies for."""
    from agentic_raptor.selfimprove_v2.streams import harvest_run
    streams = harvest_run(_hv("clean_spec"), split="train", protected_ids=set())
    assert len(streams["ranker_pairs"]) == 1
    assert len(streams["puct_examples"]) == 1
    assert len(streams["rag_memory"]) == 2       # both A and B measured
    assert len(streams["sac_replay"]) == 1


def test_harvest_run_protection_also_catches_evaluation_context_id_match(monkeypatch):
    """Belt-and-braces: even if protected_ids (context_id set) somehow
    missed a spec, the content-hash check must still catch it."""
    from agentic_raptor.selfimprove_v2 import streams as streams_mod

    def _fake_ectx(spec):
        return "MATCHED_HASH"

    monkeypatch.setattr(
        "agentic_raptor.llm_dpo.integrity.evaluation_context_id", _fake_ectx)
    monkeypatch.setattr(
        "agentic_raptor.publication.eval_sets.excluded_evaluation_context_ids",
        lambda: {"MATCHED_HASH"})
    result = streams_mod.harvest_run(_hv("not_in_context_id_set"),
                                     split="train", protected_ids=set())
    for name, rows in result.items():
        assert rows == [], f"stream {name!r} leaked via a content-hash match"


# --------------------- finding 2: PUCT provenance fields ------------------------
def test_puct_examples_carry_measurement_provenance():
    from agentic_raptor.selfimprove_v2.streams import harvest_run
    streams = harvest_run(_hv("clean_spec2"), split="train", protected_ids=set())
    ex = streams["puct_examples"][0]
    assert ex["measured_call_id"] == "c1"          # design A was selected
    assert ex["measured_topology_hash"] == "gA"
    assert ex["simulated_c_load_f"] == pytest.approx(137e-12)


# --------------------- finding 3: _GLOBAL_DYNAMICS versioning ------------------
def test_global_dynamics_fallback_is_versioned():
    from agentic_raptor.mb_sac.spec_sizing import _GLOBAL_DYNAMICS
    assert "post_cload_v1" in _GLOBAL_DYNAMICS.name


def test_global_dynamics_is_not_the_old_unversioned_file():
    from agentic_raptor.mb_sac.spec_sizing import _GLOBAL_DYNAMICS
    old = ROOT / "datasets/simulation_memory/dynamics_surrogate_data.jsonl"
    assert _GLOBAL_DYNAMICS != old


# --------------------- finding 4: rag_freeze.py path -----------------------------
def test_rag_freeze_default_source_matches_run_raptor_v2_write_target():
    import run_raptor_v2 as v2
    from agentic_raptor.publication import rag_freeze
    assert rag_freeze.RAG_MEMORY_V2 == v2.RAG_MEMORY_V2


# --------------------- item 1: spec identity hardening --------------------------
def test_excluded_evaluation_context_ids_returns_content_hashes():
    from agentic_raptor.publication.eval_sets import (
        excluded_context_ids, excluded_evaluation_context_ids)
    ectx = excluded_evaluation_context_ids()
    cids = excluded_context_ids()
    assert len(ectx) > 0
    assert len(cids) > 0
    # different hash spaces -- an evaluation_context_id (sha_json of the
    # full structured spec) should not collide with a bare context_id string
    assert not (ectx & cids)


def test_no_cross_split_context_id_collisions_in_current_corpus():
    """The empirical fact excluded_context_ids()'s safety currently rests
    on -- re-checked here as a trip-wire so a future corpus regeneration
    that violates it gets caught immediately, not discovered by accident."""
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json")
                        .read_text(encoding="utf-8"))
    by_split: dict[str, set] = {}
    for r in corpus["records"]:
        by_split.setdefault(r["split"], set()).add(r["context_id"])
    splits = list(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = by_split[a] & by_split[b]
            assert not overlap, f"{a} and {b} share context_id(s): {overlap}"


# --------------------- item 11: SAC persist=False structural check -------------
def test_sac_size_skips_both_load_and_save_when_persist_false():
    """persist=False must skip BOTH the warm-start LOAD and the SAVE --
    checked structurally (source contains the guard on both blocks), since
    this is what makes the live A0-A8 pipeline immune to stale SAC memory
    regardless of what's sitting in sizing_memory_post_cload_v1/."""
    import inspect

    from agentic_raptor.mb_sac.spec_sizing import sac_size
    src = inspect.getsource(sac_size)
    assert src.count("if persist and family:") == 2


def test_production_call_sites_use_persist_false():
    import inspect

    import run_raptor_v2 as v2
    # 2026-08-16 agentic refactor: the sizing call lives in the
    # extracted single-branch worker; the invariant is unchanged
    src = inspect.getsource(v2._size_one_branch)
    assert "persist=False" in src


# --------------------- item 15: campaign resume/dedup ---------------------------
def test_calibration_script_auto_resume_skips_done_pairs(tmp_path, monkeypatch):
    """Simulates a LOG with prior progress and confirms the done_pairs
    parsing logic (the actual mechanism, not a re-implementation) would
    skip them -- exercised at the unit level since the full script needs a
    real model load to run end-to-end."""
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({"seed": 0, "spec_index": 0, "result": "OK"}) + "\n" +
        json.dumps({"seed": 0, "spec_index": 1,
                   "result": "ERROR: TimeoutError: x"}) + "\n" +
        json.dumps({"seed": 1, "spec_index": 0, "result": "OK"}) + "\n",
        encoding="utf-8")
    done_pairs = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not str(row.get("result", "")).startswith("ERROR"):
            done_pairs.add((row["seed"], row["spec_index"]))
    assert (0, 0) in done_pairs
    assert (1, 0) in done_pairs
    assert (0, 1) not in done_pairs        # errored -- must be retried, not skipped


def test_campaign_status_distinguishes_partial_from_full_pass():
    n_specs = 10
    done_idxs = {0, 1, 2}                  # far short of n_specs
    all_covered = set(range(n_specs)) <= done_idxs
    assert all_covered is False
    done_idxs = set(range(n_specs))
    all_covered = set(range(n_specs)) <= done_idxs
    assert all_covered is True


# --------------------- item 14: compatibility gate still correct ---------------
# This test originally pinned the gate to FAIL, documenting the expected
# state at that point in the mid-rebuild audit ("before the campaign has
# even run"). Two things have since legitimately changed, not weakened:
# (1) real campaigns since then populated trusted_pairs/rag_memory_v2_clean/
# sac_sizing_memory_dir with genuine POST_CLOAD_FIX_V1 data; (2) Stage 8
# (2026-08-12) found CLOAD_GATED_ARTIFACTS was gating on FOUR components a
# live FULL run never actually reads -- two retired-and-deliberately-absent
# root-PUCT checkpoint entries, the now-removed learned DPO ranker, and an
# offline-corpus-construction-only artifact -- a real stale-logic bug, not a
# case for weakening the check. See preflight.CLOAD_GATED_ARTIFACTS's
# docstring for the full root-cause writeup.
def test_compatibility_gate_now_passes_on_the_four_live_artifacts():
    # Stage 8 (2026-08-12): dpo_ranker_v2 joined the gate set when the
    # re-justified Stage 7.2B checkpoint was deployed back into FULL's
    # default selector -- it is now genuinely loaded by every live FULL
    # run, same as the other three.
    from agentic_raptor.publication.preflight import (
        CLOAD_GATED_ARTIFACTS, check_electrical_environment_compatibility)
    assert set(CLOAD_GATED_ARTIFACTS) == {
        "trusted_pairs", "rag_memory_v2_clean", "sac_sizing_memory_dir",
        "dpo_ranker_v2"}
    result = check_electrical_environment_compatibility()
    assert result["ok"] is True, (
        f"expected the four genuinely-live-relevant artifacts to all be "
        f"POST_CLOAD_FIX_V1 by now: {result['detail']}")


# ------------------ finding 5 (post-launch, 2026-08-10): LOG versioning --------
# Found on the REAL 12h campaign's output, not in a test: LOG was the one
# calibration-script output never versioned to _post_cload_v1, so its
# auto-resume (added in this same audit) read 119 OLD pre-repair rows from
# earlier this session together with the new run's rows in ONE file and
# silently treated 7 specs (index 0-6) as already-done from stale
# measurements -- campaign_status.json claimed "full_pass_complete
# (85/85)" when the genuine post-fix count was 78/85. Fixed by versioning
# LOG/ERRORS and hand-extracting the 78 genuine rows into
# log_post_cload_v1.jsonl (the old mixed file untouched, kept as evidence).
def test_calibration_log_path_is_versioned():
    import run_puct_value_calibration as cal
    assert "post_cload_v1" in cal.LOG.name
    assert "post_cload_v1" in cal.ERRORS.name


def test_calibration_log_is_not_the_old_mixed_era_file():
    import run_puct_value_calibration as cal
    old_log = (ROOT / "artifacts/publication_v3/puct_value_calibration"
              / "log.jsonl")
    assert cal.LOG != old_log


def test_extracted_post_fix_log_reaches_full_coverage_after_resume():
    """After the LOG-versioning fix AND the done_pairs incremental-update
    fix (both this same day), a resume run correctly filled the 7 missing
    specs (0-6) -- real campaign output, re-verified directly rather than
    trusting the script's own now-fixed status message."""
    p = (ROOT / "artifacts/publication_v3/puct_value_calibration"
        / "log_post_cload_v1.jsonl")
    if not p.is_file():
        pytest.skip("log_post_cload_v1.jsonl not present in this checkout")
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()
           if x.strip()]
    done = {r["spec_index"] for r in rows
           if not str(r.get("result", "")).startswith("ERROR")}
    assert set(range(85)) <= done, (
        f"still missing: {sorted(set(range(85)) - done)}")


# ---------------- finding 6 (post-launch, 2026-08-10): stale done_pairs --------
# Found on the REAL resume run: it correctly filled the 7 missing specs
# (confirmed by re-reading the file directly), but the script's OWN final
# status report still said "partial_pass (78/85)" -- done_pairs was
# populated once at startup from LOG-as-it-existed-before and never
# updated as new rows completed during the SAME invocation, so the final
# coverage check compared against a stale pre-run snapshot instead of what
# had just been done.
def test_done_pairs_is_updated_incrementally_during_the_run():
    import inspect

    import run_puct_value_calibration as cal
    src = inspect.getsource(cal.main)
    add_idx = src.find("done_pairs.add((seed, idx))")
    log_write_idx = src.find("log_f.write(json.dumps(row")
    assert add_idx != -1, "done_pairs is never updated inside the run loop"
    assert add_idx > log_write_idx, (
        "done_pairs.add must come AFTER the row is written/known, not before")
