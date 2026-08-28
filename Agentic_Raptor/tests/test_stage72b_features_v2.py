"""Stage 7.2B: POST_SAC_FEATURES_V2 -- mechanics tests. No SPICE."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.ranking import features_v2 as fv2
from agentic_raptor.ranking.types import SurrogatePrediction

REPORT_PATH = Path("artifacts/publication_v3/stage7_2b_dpo_repair/STAGE7_2B_REPORT.json")
SPEC = {"gain_target_db": 60.0, "phase_margin_target_deg": 45.0,
       "load_capacitance_pf": 100.0, "ugbw_target_hz": 1e5}


def _pred():
    return SurrogatePrediction(topology_hash="t", sizing_manifest_hash="m",
                               normalized_margins={"gain": 0.1, "pm": -0.2})


def _branch_rows(n=6):
    return [{"step": i, "gain_db": 65.0 + i, "pm_deg": 40.0 + i, "ugbw_hz": 9e4,
            "reward": -0.1 + i * 0.05} for i in range(n)]


# ---------------------------------------------------------------------------
# Section 1/4: deterministic construction, versioned schema
# ---------------------------------------------------------------------------
def test_feature_names_v2_deterministic_and_versioned():
    assert fv2.FEATURE_DIM_V2 == len(fv2.FEATURE_NAMES_V2)
    assert len(fv2.FEATURE_PROVENANCE_V2) == fv2.FEATURE_DIM_V2
    # old 11-feature schema untouched
    from agentic_raptor.ranking.model import FEATURE_DIM, FEATURE_NAMES
    assert FEATURE_DIM == 11
    assert fv2.FEATURE_NAMES_V2[:11] == FEATURE_NAMES


def test_features_v2_is_deterministic():
    row = _branch_rows()
    a = fv2.features_v2(SPEC, _pred(), row[-1], row, "2s_rc", arm="combined")
    b = fv2.features_v2(SPEC, _pred(), row[-1], row, "2s_rc", arm="combined")
    assert a == b


# ---------------------------------------------------------------------------
# Section 2: no final-verification fields possible -- structural check
# ---------------------------------------------------------------------------
def test_no_final_verification_terms_in_feature_names():
    banned = ("final_verification", "fom", "pvt", "authoritative_call")
    for name in fv2.FEATURE_NAMES_V2:
        for b in banned:
            assert b not in name.lower()


def test_candidate_measured_features_only_reads_pre_selection_fields():
    """candidate_measured_features must not accept/require an
    AuthoritativeSpiceOutcome-shaped 'final_verification' object -- only a
    sizing-loop row (gain_db/pm_deg/ugbw_hz)."""
    import inspect
    sig = inspect.signature(fv2.candidate_measured_features)
    assert list(sig.parameters) == ["row", "spec"]


# ---------------------------------------------------------------------------
# Section 3/10: provenance -- measured/reconstructed/missing handled honestly
# ---------------------------------------------------------------------------
def test_missing_candidate_row_yields_explicit_zero_with_availability_flag():
    feats = dict(zip(fv2.FEATURE_NAMES_V2,
                     fv2.features_v2(SPEC, _pred(), None, None, None, arm="combined")))
    assert feats["has_candidate_measurement"] == 0.0
    assert feats["has_trajectory"] == 0.0
    assert feats["has_topology_family"] == 0.0
    assert feats["measured_gain_margin"] == 0.0


def test_present_candidate_row_sets_availability_flags():
    row = _branch_rows()
    feats = dict(zip(fv2.FEATURE_NAMES_V2,
                     fv2.features_v2(SPEC, _pred(), row[-1], row, "2s_rc", arm="combined")))
    assert feats["has_candidate_measurement"] == 1.0
    assert feats["has_trajectory"] == 1.0
    assert feats["has_topology_family"] == 1.0


def test_reconstructed_stability_matches_documented_pm_sign_precedent():
    row = {"gain_db": 70.0, "pm_deg": 30.0, "ugbw_hz": 1e5}
    feats = fv2.candidate_measured_features(row, SPEC)
    assert feats["measured_stable_reconstructed"] == 1.0
    row_unstable = {"gain_db": 70.0, "pm_deg": -5.0, "ugbw_hz": 1e5}
    feats_u = fv2.candidate_measured_features(row_unstable, SPEC)
    assert feats_u["measured_stable_reconstructed"] == 0.0


# ---------------------------------------------------------------------------
# Section 7: normalized margins, not raw magnitudes
# ---------------------------------------------------------------------------
def test_measured_margins_are_normalized_not_raw():
    row = {"gain_db": 80.0, "pm_deg": 90.0, "ugbw_hz": 1e5}
    feats = fv2.candidate_measured_features(row, SPEC)
    # gain margin = (80-60)/20 = 1.0, not the raw 20 dB
    assert feats["measured_gain_margin"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Section 6: trajectory summary correctness
# ---------------------------------------------------------------------------
def test_trajectory_summary_captures_start_final_best():
    rows = [{"step": 0, "gain_db": 50.0, "pm_deg": 20.0, "ugbw_hz": 5e4, "reward": -1.0},
           {"step": 1, "gain_db": 70.0, "pm_deg": 50.0, "ugbw_hz": 1.2e5, "reward": 0.5}]
    summary = fv2.trajectory_summary(rows, SPEC)
    assert summary["has_trajectory"] == 1.0
    assert summary["traj_n_evaluations"] == pytest.approx(2 / 32.0)
    # improved from step0 to step1 -> distance should decrease (positive improvement)
    assert summary["traj_absolute_improvement"] >= 0


def test_trajectory_summary_valid_spice_fraction():
    rows = [{"step": 0, "gain_db": 50.0, "pm_deg": 20.0, "ugbw_hz": 5e4, "reward": -1.0},
           {"step": 1, "gain_db": None, "pm_deg": None, "ugbw_hz": None, "reward": -2.0}]
    summary = fv2.trajectory_summary(rows, SPEC)
    assert summary["traj_valid_spice_fraction"] == pytest.approx(0.5)


def test_trajectory_summary_empty_is_all_zero():
    summary = fv2.trajectory_summary(None, SPEC)
    assert all(v == 0.0 for v in summary.values())
    summary2 = fv2.trajectory_summary([], SPEC)
    assert all(v == 0.0 for v in summary2.values())


# ---------------------------------------------------------------------------
# Section 9: topology structural features -- small, principled, no ID embedding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family,stages,comp", [
    ("2s_none", 2, "comp_none"), ("2s_miller", 2, "comp_miller"),
    ("2s_rc", 2, "comp_rc"), ("3s_miller", 3, "comp_miller"), ("3s_rc", 3, "comp_rc")])
def test_parse_family_covers_all_five_corpus_families(family, stages, comp):
    parsed = fv2.parse_family(family)
    assert parsed["stage_count"] == pytest.approx(stages / 5.0)
    assert parsed[comp] == 1.0
    assert parsed["has_topology_family"] == 1.0
    others = {"comp_none", "comp_miller", "comp_rc"} - {comp}
    for o in others:
        assert parsed[o] == 0.0


def test_parse_family_unknown_string_is_missing_not_guessed():
    parsed = fv2.parse_family("totally_unknown_family")
    assert parsed["has_topology_family"] == 0.0
    assert parsed["stage_count"] == 0.0


# ---------------------------------------------------------------------------
# Feature ablation arms zero out the correct groups
# ---------------------------------------------------------------------------
def test_predicted_only_arm_zeros_measured_and_trajectory_and_topology():
    row = _branch_rows()
    feats = dict(zip(fv2.FEATURE_NAMES_V2,
                     fv2.features_v2(SPEC, _pred(), row[-1], row, "2s_rc", arm="predicted_only")))
    assert feats["has_candidate_measurement"] == 0.0
    assert feats["has_trajectory"] == 0.0
    assert feats["has_topology_family"] == 0.0
    assert feats["worst_violation"] != 0.0 or feats["margin_pm"] != 0.0   # v1 features intact


def test_measured_only_arm_zeros_predicted_features():
    row = _branch_rows()
    feats = dict(zip(fv2.FEATURE_NAMES_V2,
                     fv2.features_v2(SPEC, _pred(), row[-1], row, "2s_rc", arm="measured_only")))
    for name in fv2.FEATURE_NAMES_V2[:11]:
        assert feats[name] == 0.0
    assert feats["has_candidate_measurement"] == 1.0


# ---------------------------------------------------------------------------
# Report-derived tests (skipped if not run)
# ---------------------------------------------------------------------------
def _load_report():
    if not REPORT_PATH.is_file():
        pytest.skip("Stage 7.2B not yet run in this checkout")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_report_split_reproduces_stage_7_2a_dev_set():
    d = _load_report()
    assert d["dev_coverage"]["n_ranker_authority"] == 135
    assert d["dev_coverage"]["n_specs"] == 18


def test_report_promotion_gate_checks_disagreement_and_catastrophic():
    d = _load_report()
    if d["promotion"]["final_classification"] != "DPO_REJUSTIFIED":
        pytest.skip("no promoted candidate to check in this run")
    checks = d["promotion"]["checks"]
    assert "disagreement_wins_gt_losses" in checks
    assert "catastrophic_errors_negligible" in checks
    assert checks["disagreement_wins_gt_losses"] is True
    assert checks["catastrophic_errors_negligible"] is True


def test_report_selected_model_not_single_run_dependent():
    d = _load_report()
    sel = d.get("selected")
    if not sel:
        pytest.skip("no selected model")
    de = d["ablation_results"][sel]["dev_eval"]
    assert de["n_runs_ranker_authority"] >= 3
    per_run = de.get("per_run_accuracy", {})
    if per_run:
        # improvement must not come from a single run being perfect while
        # everything else is at/below chance
        n_strong = sum(1 for v in per_run.values() if v >= 0.6)
        assert n_strong >= 3
