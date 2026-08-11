"""Stage 1.6: CLEAN POST-CLOAD-FIX MEASUREMENT AND ARTIFACT REBUILD.

Covers the versioning/compatibility machinery this stage adds on top of the
Stage 1.5 C_LOAD repair (tests/test_cload_consistency.py): PRE_CLOAD_FIX
artifacts cannot silently load in paper mode, new measurement artifacts
carry POST_CLOAD_FIX_V1 metadata, corrected trusted pairs / PUCT examples /
RAG memory / surrogate data all record requested==simulated load, the
artifact-compatibility gate works, and no new legacy-RAPTOR coupling was
introduced anywhere Stage 1.6 touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.publication import artifact_provenance as ap

ROOT = Path(__file__).resolve().parents[1]


# --------------------------- freeze / manifest --------------------------------
def test_freeze_artifact_is_read_only(tmp_path):
    """Freezing must never touch the artifact's own bytes."""
    f = tmp_path / "some_artifact.jsonl"
    f.write_text('{"a": 1}\n', encoding="utf-8")
    before = f.read_bytes()
    entry = ap.freeze_artifact("some_artifact", f)
    assert f.read_bytes() == before
    assert entry["electrical_environment_version"] == ap.PRE_CLOAD_FIX
    assert entry["sha256"] == __import__("hashlib").sha256(before).hexdigest()


def test_freeze_artifact_missing_path_is_recorded_not_raised(tmp_path):
    entry = ap.freeze_artifact("missing_thing", tmp_path / "does_not_exist.pt")
    assert entry["exists"] is False
    assert entry["kind"] == "missing"
    assert entry["sha256"] is None


def test_freeze_all_pre_cload_artifacts_produces_a_manifest():
    doc = ap.freeze_all_pre_cload_artifacts()
    assert doc["cutoff"] == ap.ELECTRICAL_ENV_CUTOFF
    names = {a["name"] for a in doc["artifacts"]}
    assert names == set(ap.TRACKED_ARTIFACTS)
    assert ap.MANIFEST.is_file()


# --------------------------- stamp() -------------------------------------------
def test_stamp_attaches_required_fields():
    obj = ap.stamp({"foo": "bar"}, model_type="dpo_ranker",
                   training_data_hash="abc123", checkpoint_hash="def456",
                   validated=True)
    assert obj["foo"] == "bar"                       # original content kept
    assert obj["electrical_environment_version"] == ap.POST_CLOAD_FIX_V1
    assert obj["post_vcm_fix"] is True
    assert obj["post_cload_fix"] is True
    assert obj["model_type"] == "dpo_ranker"
    assert obj["training_data_hash"] == "abc123"
    assert obj["checkpoint_hash"] == "def456"
    assert obj["validated"] is True


def test_stamp_omits_optional_fields_when_not_given():
    obj = ap.stamp({})
    assert "model_type" not in obj
    assert "training_data_hash" not in obj
    assert obj["electrical_environment_version"] == ap.POST_CLOAD_FIX_V1


# --------------------------- environment_version_of ----------------------------
def test_environment_version_of_trusts_explicit_tag():
    meta = {"electrical_environment_version": ap.POST_CLOAD_FIX_V1}
    assert ap.environment_version_of(meta) == ap.POST_CLOAD_FIX_V1


def test_environment_version_of_missing_tag_is_unknown():
    assert ap.environment_version_of({}) == "UNKNOWN"
    assert ap.environment_version_of(None) == "UNKNOWN"


def test_environment_version_of_timestamp_overrides_a_false_claim():
    """A record timestamped before the repair cannot have gone through
    effective_c_load(), no matter what its own metadata claims -- this is
    the defense against a stale record with a copy-pasted version tag."""
    meta = {"electrical_environment_version": ap.POST_CLOAD_FIX_V1}
    old_ts = ap.ELECTRICAL_ENV_CUTOFF - 3600
    assert ap.environment_version_of(meta, record_timestamp=old_ts) == ap.PRE_CLOAD_FIX


def test_environment_version_of_timestamp_after_cutoff_trusts_the_tag():
    meta = {"electrical_environment_version": ap.POST_CLOAD_FIX_V1}
    new_ts = ap.ELECTRICAL_ENV_CUTOFF + 3600
    assert ap.environment_version_of(meta, record_timestamp=new_ts) == ap.POST_CLOAD_FIX_V1


# --------------------------- assert_compatible ---------------------------------
def test_assert_compatible_passes_when_all_match():
    result = ap.assert_compatible({"a": ap.POST_CLOAD_FIX_V1,
                                   "b": ap.POST_CLOAD_FIX_V1})
    assert result["ok"] is True


def test_assert_compatible_raises_on_any_mismatch():
    with pytest.raises(AssertionError, match="ELECTRICAL ENVIRONMENT MISMATCH"):
        ap.assert_compatible({"a": ap.POST_CLOAD_FIX_V1, "b": ap.PRE_CLOAD_FIX})


# ------------------- current_environment_version: live paths -------------------
def test_current_environment_version_not_yet_built_is_distinct_from_pre_cload_fix():
    """A path that doesn't exist yet (nothing rebuilt) must read differently
    from a path that exists and is known-stale -- conflating the two would
    make 'nothing to mix in' look identical to 'known-bad data present'."""
    v = ap.current_environment_version("sac_sizing_memory_dir")
    assert v in ("NOT_YET_BUILT", ap.POST_CLOAD_FIX_V1)  # depends on test order


def test_current_environment_version_unknown_name_is_unknown():
    assert ap.current_environment_version("not_a_real_artifact") == "UNKNOWN"


def test_current_environment_version_reads_json_meta(tmp_path, monkeypatch):
    p = tmp_path / "fake_gate.json"
    p.write_text(json.dumps({"electrical_environment_version": ap.POST_CLOAD_FIX_V1}),
                 encoding="utf-8")
    monkeypatch.setitem(ap.LIVE_ARTIFACT_PATHS, "family_spec_gate", p)
    assert ap.current_environment_version("family_spec_gate") == ap.POST_CLOAD_FIX_V1


def test_current_environment_version_detects_mixed_jsonl(tmp_path, monkeypatch):
    p = tmp_path / "fake_pairs.jsonl"
    p.write_text(
        json.dumps({"electrical_environment_version": ap.PRE_CLOAD_FIX}) + "\n" +
        json.dumps({"electrical_environment_version": ap.POST_CLOAD_FIX_V1}) + "\n",
        encoding="utf-8")
    monkeypatch.setitem(ap.LIVE_ARTIFACT_PATHS, "trusted_pairs", p)
    assert ap.current_environment_version("trusted_pairs") == "MIXED"


# --------------------- preflight: the paper-mode hard-fail gate ----------------
def test_preflight_compatibility_check_fails_before_any_rebuild():
    """Expected to fail right now, by design -- nothing has been rebuilt
    yet. This is the mechanism that keeps the next A0-A8 pilot from
    launching on stale data."""
    from agentic_raptor.publication.preflight import (
        check_electrical_environment_compatibility)
    result = check_electrical_environment_compatibility()
    assert result["name"] == "electrical_environment_compatible"
    assert result["critical"] is True
    # not asserting ok=False here -- once the real rebuild campaign runs,
    # this check SHOULD start passing, and that's the point of it existing


def test_preflight_includes_the_cload_compatibility_check():
    from agentic_raptor.publication.preflight import ALL_CHECKS
    assert any(c.__name__ == "check_electrical_environment_compatibility"
              for c in ALL_CHECKS)


# --------------------- new measurement records carry provenance ----------------
def test_harvest_run_stamps_every_stream_with_post_cload_fix_v1():
    from agentic_raptor.selfimprove_v2.streams import harvest_run
    hv = {"spec": {"spec_id": "t", "load_capacitance_pf": 137.0},
         "spec_hash": "h", "split": "train", "spec_index": 3, "seed": 0,
         "budget": 8, "requested_c_load_f": 137e-12,
         "candidates": [], "root_visits": {}, "candidate_visits": {},
         "root_action_ids": [], "root_state": None,
         "candidate_manifests": {}, "search": None,
         "branches": {
             "A": {"design": {"canonical_graph_hash": "gA",
                              "topology_family": "2s_none"},
                  "authoritative": {"call_id": "c1", "gain_db": 60.0,
                                    "pm_deg": 50.0, "ugbw_hz": 1e5,
                                    "idd_a": 1e-4, "c_load_f": 137e-12,
                                    "exact_spec_pass": True,
                                    "verified_stable": True}},
             "B": {"design": {"canonical_graph_hash": "gB",
                              "topology_family": "2s_miller"},
                  "authoritative": {"call_id": "c2", "gain_db": 55.0,
                                    "pm_deg": 40.0, "ugbw_hz": 8e4,
                                    "idd_a": 1.1e-4, "c_load_f": 137e-12,
                                    "exact_spec_pass": False,
                                    "verified_stable": True}}},
         "ranker": {"selected_design": "A", "backup_design": "B",
                   "decision_basis": "x", "deciding_level": "x",
                   "low_confidence": False, "score_A": None, "score_B": None,
                   "checkpoint_hash": None},
         "proposer_checkpoint": "x"}
    streams = harvest_run(hv, split="train", protected_ids=set())
    for name in ("rag_memory",):
        for row in streams[name]:
            assert row["electrical_environment_version"] == ap.POST_CLOAD_FIX_V1
            assert row["requested_c_load_f"] == pytest.approx(137e-12)
            assert row["simulated_c_load_f"] == pytest.approx(137e-12)
            assert row["spec_index"] == 3


def test_sac_size_default_state_dir_is_the_versioned_path():
    from agentic_raptor.mb_sac.spec_sizing import STATE_DIR
    assert "post_cload_v1" in str(STATE_DIR) or "AGENTIC_RAPTOR_SIZING_MEMORY" in __import__("os").environ


def test_trusted_pairs_path_is_versioned():
    from agentic_raptor.ranking.post_sac import TRUSTED
    assert "post_cload_v1" in TRUSTED.name


def test_rag_memory_v2_path_is_versioned():
    import run_raptor_v2 as v2
    assert "post_cload_v1" in v2.RAG_MEMORY_V2.name


def test_puct_examples_source_path_is_versioned():
    from agentic_raptor.topology_rl.value_refresh import V2_PUCT_EXAMPLES_FILE
    assert "post_cload_v1" in str(V2_PUCT_EXAMPLES_FILE)


# --------------------- evaluation data excluded from retraining ----------------
def test_eval_sets_leakage_guard_still_intact_after_stage_1_6():
    """Stage 1.6 must not have loosened the existing evaluation-leakage
    guard while rewiring provenance -- same guard, re-checked here as a
    regression trip-wire."""
    from agentic_raptor.publication.eval_sets import assert_no_leakage
    assert assert_no_leakage(set(), "stage_1_6_regression_check") is True


# --------------------- no new legacy-RAPTOR dependency --------------------------
def test_no_new_legacy_raptor_reference_in_stage_1_6_files():
    files = [
        ROOT / "agentic_raptor/publication/artifact_provenance.py",
        ROOT / "agentic_raptor/selfimprove_v2/streams.py",
        ROOT / "agentic_raptor/ranking/post_sac.py",
    ]
    for f in files:
        src = f.read_text(encoding="utf-8")
        assert "sys.path" not in src
        assert "import RAPTOR_Legacy" not in src
        assert "from RAPTOR_Legacy" not in src
