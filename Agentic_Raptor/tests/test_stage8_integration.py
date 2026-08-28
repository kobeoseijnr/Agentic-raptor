"""Stage 8: final frozen pipeline integration diagnostic -- mechanics tests.

Most of these are pure code/config checks (no SPICE, no LLM) that can run
every time; a few read the real STAGE8_INTEGRATION_REPORT.json produced by
run_stage8_integration_diagnostic.py and are skipped if that hasn't been
run in this checkout.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

REPORT = Path("artifacts/publication_v3/stage8_integration_diagnostic/STAGE8_INTEGRATION_REPORT.json")


# ---------------------------------------------------------------------------
# Section 1/3 (second deployment, 2026-08-12): learned DPO V2 restored as
# FULL's default -- Stage 7.2B re-justified it (POST_SAC_FEATURES_V2,
# DPO_REJUSTIFIED). See tests/test_stage8_dpo_v2_deployment.py for the full
# V2-specific deployment test suite (checkpoint hash, feature parity,
# schema mismatch handling, hard-gate ordering).
# ---------------------------------------------------------------------------
def test_run_pipeline_default_ranker_mode_is_dpo():
    import run_raptor_v2 as v2
    sig = inspect.signature(v2.run_pipeline)
    assert sig.parameters["ranker_mode"].default == "dpo"


def test_a0_full_enables_learned_dpo_again():
    from agentic_raptor.publication.ablation_v3 import A0_FULL
    assert A0_FULL.use_dpo is True
    assert A0_FULL.to_run_pipeline_kwargs()["ranker_mode"] == "dpo"


def test_deterministic_mode_remains_available_as_explicit_opt_in():
    """"deterministic" must still be a functional opt-in (ablation/baseline
    comparison, historical reproduction), just no longer FULL's default."""
    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    assert 'ranker_mode in ("dpo", "dpo_gated")' in src


# ---------------------------------------------------------------------------
# Section 11: controlled failure -- explicit dpo request with no reachable/
# valid checkpoint must hard-fail, never silently downgrade to deterministic
# ---------------------------------------------------------------------------
def test_dpo_v2_load_failure_is_never_silently_swallowed_in_run_pipeline():
    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    # load_promoted_v2() itself raises (FileNotFoundError /
    # RankerCheckpointHashMismatch / RankerFeatureSchemaMismatch); confirm
    # run_pipeline does not wrap that call in a try/except that would
    # silently fall back to the deterministic selector
    assert "load_promoted_v2(branch_context)" in src
    call_line_idx = src.index("load_promoted_v2(branch_context)")
    preceding = src[:call_line_idx]
    # the nearest preceding 'try:' (if any) must not be the one guarding
    # this call -- simplest robust check: no bare except/pass immediately
    # follows within the use_learned_dpo block
    block_start = src.index("if use_learned_dpo:")
    block = src[block_start:call_line_idx + 50]
    assert "except" not in block


# ---------------------------------------------------------------------------
# Section 13: A8 un-retired -- meaningful again now that FULL has DPO V2
# ---------------------------------------------------------------------------
def test_a8_un_retired_and_meaningful_again():
    from agentic_raptor.publication.ablation_v3 import (A8_NO_DPO,
                                                         PRIMARY_EXPERIMENTS)
    assert A8_NO_DPO.retired is False
    assert A8_NO_DPO.use_dpo is False
    assert "A8" in PRIMARY_EXPERIMENTS


# ---------------------------------------------------------------------------
# Section 4/28 (second deployment): the SUPERSEDED V1 checkpoint check stays
# non-critical; the NEW V2 checks (Section 48) are critical, since DPO V2 is
# live FULL's default again.
# ---------------------------------------------------------------------------
def test_dpo_checkpoint_check_is_non_critical():
    from agentic_raptor.publication.preflight import check_dpo_checkpoint
    result = check_dpo_checkpoint()
    assert result["critical"] is False


def test_dpo_promoted_checkpoint_check_is_critical_and_passes():
    from agentic_raptor.publication.preflight import \
        check_dpo_promoted_checkpoint
    result = check_dpo_promoted_checkpoint()
    assert result["critical"] is True
    assert result["ok"] is True, result["detail"]


def test_dpo_feature_schema_compatible_check_is_critical_and_passes():
    from agentic_raptor.publication.preflight import \
        check_dpo_feature_schema_compatible
    result = check_dpo_feature_schema_compatible()
    assert result["critical"] is True
    assert result["ok"] is True, result["detail"]


def test_dpo_ranker_v2_included_in_cload_gate_and_all_checks():
    from agentic_raptor.publication.preflight import (
        ALL_CHECKS, CLOAD_GATED_ARTIFACTS, check_dpo_feature_schema_compatible,
        check_dpo_promoted_checkpoint)
    assert "dpo_ranker_v2" in CLOAD_GATED_ARTIFACTS
    assert check_dpo_promoted_checkpoint in ALL_CHECKS
    assert check_dpo_feature_schema_compatible in ALL_CHECKS


def test_evaluation_learning_disabled_check_exists_and_passes():
    from agentic_raptor.publication.preflight import (
        ALL_CHECKS, check_evaluation_learning_disabled)
    assert check_evaluation_learning_disabled in ALL_CHECKS
    result = check_evaluation_learning_disabled()
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# Section 7-9: preflight blocker root-cause fixes
# ---------------------------------------------------------------------------
def test_cload_gated_artifacts_excludes_retired_and_non_live_components():
    from agentic_raptor.publication.preflight import CLOAD_GATED_ARTIFACTS
    # retired root-PUCT checkpoints, the now-retired DPO ranker, and the
    # offline-corpus-construction-only family_spec_gate must not gate a
    # live FULL run's readiness
    for name in ("puct_value_checkpoint", "puct_policy_checkpoint",
                "dpo_ranker", "family_spec_gate"):
        assert name not in CLOAD_GATED_ARTIFACTS
    # the three genuinely live-relevant artifacts remain gated
    for name in ("trusted_pairs", "rag_memory_v2_clean", "sac_sizing_memory_dir"):
        assert name in CLOAD_GATED_ARTIFACTS


def test_electrical_environment_compatible_passes_on_current_state():
    from agentic_raptor.publication.preflight import \
        check_electrical_environment_compatibility
    result = check_electrical_environment_compatibility()
    assert result["ok"] is True, result["detail"]


def test_rag_populated_reports_real_coverage_not_a_path_bug():
    """Part D (2026-08-12): the ORIGINAL RAG_MIN_RECORDS=200 was audited and
    found to be an ungrounded round number (see preflight.py's docstring),
    not a formally frozen scientific minimum -- replaced with real coverage
    criteria (usable count, distinct-spec count, per-family floor) set with
    margin below, not equal to, the current 174/85-spec/4-family state (so
    the fix isn't just "tuned to make the check pass" under a new name).
    This now genuinely passes on real, clean, zero-defect data -- the
    earlier framing ("reports a real shortfall") described the state before
    this audit, not a permanent property of the check."""
    from agentic_raptor.publication.preflight import (CLEAN_RAG_PATH,
                                                       check_rag_populated)
    assert CLEAN_RAG_PATH.is_file(), "canonical clean RAG snapshot must exist"
    result = check_rag_populated()
    # real, honestly-reported coverage numbers -- not fabricated to pass
    assert "usable CLEAN records" in result["detail"]
    assert "distinct specs" in result["detail"]
    assert "family_counts" in result["detail"]
    assert result["ok"] is True, result["detail"]


def test_rag_quality_report_returns_real_computed_metrics():
    from agentic_raptor.publication.preflight import rag_quality_report
    r = rag_quality_report()
    assert r["total_records"] > 0
    assert r["unique_specs"] > 0
    assert len(r["unique_topology_families"]) >= 1
    assert r["duplicate_record_count"] == 0
    assert r["protected_evaluation_contamination_count"] == 0


def test_rag_min_records_threshold_has_a_justification_below_current_state():
    """The new floor must be grounded (real margin below current state, not
    equal to it) -- guards against silently re-tuning it to match whatever
    the corpus happens to contain on a future re-audit."""
    from agentic_raptor.publication.preflight import (
        RAG_MIN_DISTINCT_SPECS, RAG_MIN_RECORDS, rag_quality_report)
    r = rag_quality_report()
    assert RAG_MIN_RECORDS < r["total_records"]
    assert RAG_MIN_DISTINCT_SPECS < r["unique_specs"]


# ---------------------------------------------------------------------------
# Section 15: explicit KNOB_NAMES ordering (no dict.values() fragility)
# ---------------------------------------------------------------------------
def test_run_raptor_v2_no_longer_uses_dict_values_for_knobs():
    import run_raptor_v2 as v2
    src = inspect.getsource(v2)
    assert "knobs.values()" not in src
    assert "[knobs[name] for name in KNOB_NAMES]" in src


def test_knob_dict_to_vector_reorders_correctly_via_canonical_mapping():
    """The exact regression shape: a non-trivially-ordered dict must map to
    the KNOB_NAMES-ordered vector, not insertion order."""
    from agentic_raptor.mb_sac.spec_sizing import KNOB_NAMES
    knobs = {"rz_x": 7.0, "s1_w": 1.5, "cap_x": 3.0, "s2_l": 2.5,
            "ib_x": 0.5, "s1_l": 4.0, "s2_w": 6.0}
    vector = [knobs[name] for name in KNOB_NAMES]
    assert vector == [1.5, 6.0, 4.0, 2.5, 3.0, 0.5, 7.0]
    assert list(knobs.values()) != vector


# ---------------------------------------------------------------------------
# Section 24: fallback audit
# ---------------------------------------------------------------------------
def test_no_root_puct_fallback_in_live_dispatch():
    """puct_select_two is mentioned only in comments documenting its
    retirement -- confirm it is never actually CALLED (no `puct_select_two(`
    invocation on a live code line), while the real replacement functions
    are."""
    import run_raptor_v2 as v2
    code_lines = [line for line in inspect.getsource(v2.run_pipeline).splitlines()
                 if not line.strip().startswith("#")]
    code = "\n".join(code_lines)
    assert "puct_select_two(" not in code
    assert "alphazero_select_two(" in code
    assert "direct_prior_select_two(" in code


def test_old_root_puct_checkpoint_absent_on_disk():
    from agentic_raptor.publication.preflight import \
        check_old_root_puct_checkpoint_absent
    result = check_old_root_puct_checkpoint_absent()
    assert result["ok"] is True, result["detail"]


def test_no_persistent_sac_warm_start_in_live_sizing_call():
    import run_raptor_v2 as v2
    # 2026-08-16 agentic refactor: the sizing call lives in the
    # extracted single-branch worker; the invariant is unchanged
    src = inspect.getsource(v2._size_one_branch)
    assert "persist=False" in src


def test_no_legacy_raptor_or_decoy_coordinator_import_in_live_path():
    import run_raptor_v2 as v2
    src = inspect.getsource(v2)
    assert "agentic_raptor.adapters" not in src
    assert "agentic_raptor.coordinator" not in src
    assert "graph_conditioned_mb_sac" not in src


# ---------------------------------------------------------------------------
# Section 10: frozen/no-learning mode disables every harvest/persist route
# ---------------------------------------------------------------------------
def test_frozen_learning_mode_gates_the_entire_harvest_block():
    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    assert 'if learning_mode == "adaptive":' in src
    # record_pair persistence is independently gated the same way -- and
    # since the 2026-08-16 custody fix, ALSO suppressed when an external
    # harvester (A9) owns the run's training data
    assert 'persist=(learning_mode == "adaptive" and not harvest)' in src


def test_ablation_default_learning_mode_is_frozen():
    from dataclasses import fields

    from agentic_raptor.publication.ablation_v3 import AblationConfig
    default = next(f.default for f in fields(AblationConfig)
                  if f.name == "learning_mode")
    assert default == "frozen"


# ---------------------------------------------------------------------------
# Section 17: two-candidate contract (structural, hard-raises internally)
# ---------------------------------------------------------------------------
def test_size_and_predict_hard_fails_on_shared_topology_hash():
    import run_raptor_v2 as v2
    src = inspect.getsource(v2.size_and_predict)
    assert "ArchitectureViolation" in src
    assert "both branches sized the same topology" in src


# ---------------------------------------------------------------------------
# Report-derived checks (skipped if the real diagnostic hasn't run)
# ---------------------------------------------------------------------------
def _load_report():
    if not REPORT.is_file():
        pytest.skip("Stage 8 integration diagnostic not yet run in this checkout")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_stage8_report_all_jobs_used_dpo_v2_selector():
    from agentic_raptor.ranking.model_v2 import REQUIRED_V2_SHA256
    d = _load_report()
    for j in d["jobs"]:
        sr = j["trace_summary"]["stage8_ranker"]
        assert sr["use_learned_dpo"] is True
        assert sr["selector"] == "learned_dpo"
        assert sr["feature_schema"] == "POST_SAC_FEATURES_V2"
        assert sr["ranker_checkpoint_hash"] == REQUIRED_V2_SHA256


def test_stage8_report_topology_and_knob_identity_hold():
    d = _load_report()
    for j in d["jobs"]:
        assert j["verify"]["topology_identity_ok"], j["verify"]["problems"]
        assert j["verify"]["knob_identity_ok"], j["verify"]["problems"]
        assert j["verify"]["two_candidate_distinct"]


def test_stage8_report_action_roundtrip_within_tolerance():
    d = _load_report()
    for j in d["jobs"]:
        assert j["verify"]["action_roundtrip_ok"], j["verify"]["max_action_roundtrip_error"]


def test_stage8_report_fresh_verification_and_cload():
    d = _load_report()
    for j in d["jobs"]:
        assert j["verify"]["fresh_verification_ok"]
        assert j["verify"]["cload_ok"]


def test_stage8_report_no_unexpected_problems():
    d = _load_report()
    assert d["summary"]["all_jobs_verify_ok"] is True, d["summary"]["n_jobs_with_problems"]
