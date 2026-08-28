"""Stage 7.2B: training/evaluation script mechanics."""
from __future__ import annotations

import torch

import run_stage72b_dpo_repair as repair
from agentic_raptor.ranking import features_v2 as fv2
from agentic_raptor.ranking.types import SurrogatePrediction

SPEC = {"gain_target_db": 60.0, "phase_margin_target_deg": 45.0,
       "load_capacitance_pf": 100.0, "ugbw_target_hz": 1e5}


def _pred(th="t"):
    return SurrogatePrediction(topology_hash=th, sizing_manifest_hash="m",
                               normalized_margins={"gain": 0.1, "pm": -0.1})


def _row(source="mined"):
    branch_rows = [{"step": i, "gain_db": 60.0 + i, "pm_deg": 40.0 + i, "ugbw_hz": 9e4,
                   "reward": -0.2 + i * 0.05} for i in range(4)]
    return {"originating_run_id": "r1", "spec_hash": "s1", "spec": SPEC,
           "pred_a": _pred("ta"), "pred_b": _pred("tb"),
           "candidate_row_a": {"gain_db": 65.0, "pm_deg": 45.0, "ugbw_hz": 1e5},
           "candidate_row_b": {"gain_db": 55.0, "pm_deg": 35.0, "ugbw_hz": 8e4},
           "branch_rows_a": branch_rows, "branch_rows_b": branch_rows,
           "family_a": "2s_rc", "family_b": "2s_none",
           "authoritative_winner": "A", "ranker_authority": True,
           "feasibility_category": "one_feasible_one_infeasible",
           "informative": True, "source": source}


# ---------------------------------------------------------------------------
# Section 4: reconstructing a sizing-loop-shaped row from a saved outcome
# ---------------------------------------------------------------------------
def test_pseudo_row_from_outcome_reads_achieved_values():
    outcome = {"constraints": {
        "gain_db": {"achieved": 71.5}, "phase_margin_deg": {"achieved": 48.0},
        "ugbw_hz": {"achieved": 1.1e5}}}
    row = repair._pseudo_row_from_outcome(outcome)
    assert row == {"gain_db": 71.5, "pm_deg": 48.0, "ugbw_hz": 1.1e5}


# ---------------------------------------------------------------------------
# Section 12: features built at branch level for both sides
# ---------------------------------------------------------------------------
def test_row_features_dispatches_by_side_and_arm():
    row = _row()
    fa = repair._row_features(row, "a", "combined")
    fb = repair._row_features(row, "b", "combined")
    assert len(fa) == fv2.FEATURE_DIM_V2 == len(fb)
    assert fa != fb   # distinct candidates must not produce identical features


# ---------------------------------------------------------------------------
# Model building matches features_v2's width
# ---------------------------------------------------------------------------
def test_build_model_input_dim_matches_features_v2():
    net = repair.build_model(8)
    first_layer = next(net.children())
    assert first_layer.in_features == fv2.FEATURE_DIM_V2


# ---------------------------------------------------------------------------
# Section 25: evaluate() computes disagreement/catastrophic accounting
# ---------------------------------------------------------------------------
def test_evaluate_reports_disagreement_and_catastrophic_fields():
    torch.manual_seed(0)
    net = repair.build_model(None)
    mean = [0.0] * fv2.FEATURE_DIM_V2
    std = [1.0] * fv2.FEATURE_DIM_V2
    result = repair.evaluate([_row()], net, mean, std, "combined")
    for key in ("disagreement_wins_vs_neutral", "disagreement_losses_vs_neutral",
               "disagreement_ties_vs_neutral", "net_decision_gain_vs_neutral",
               "catastrophic_error_count", "per_run_accuracy"):
        assert key in result


def test_evaluate_net_gain_equals_wins_minus_losses():
    torch.manual_seed(1)
    net = repair.build_model(8)
    mean, std = [0.0] * fv2.FEATURE_DIM_V2, [1.0] * fv2.FEATURE_DIM_V2
    rows = [_row() for _ in range(5)]
    result = repair.evaluate(rows, net, mean, std, "combined")
    assert (result["net_decision_gain_vs_neutral"]
           == result["disagreement_wins_vs_neutral"] - result["disagreement_losses_vs_neutral"])


# ---------------------------------------------------------------------------
# Section 16/17: identical split rule and weighting rule as Stage 7.2A
# ---------------------------------------------------------------------------
def test_split_train_dev_matches_stage72a_hash_rule():
    import run_stage71_dpo_repair as s71
    pool_a = [{"spec_hash": f"spec_{i}"} for i in range(20)]
    pool_b = [{"spec_hash": f"spec_{i}"} for i in range(20)]
    _, dev_a, dev_specs_a = s71.split_train_dev(pool_a, dev_fraction=repair.DEV_FRACTION)
    _, dev_b, dev_specs_b = repair.split_train_dev(pool_b, dev_fraction=repair.DEV_FRACTION)
    assert dev_specs_a == dev_specs_b


def test_compute_pair_weights_run_normalization():
    rows = [{"originating_run_id": "r1", "ranker_authority": True}] * 3
    weights = repair.compute_pair_weights(rows)
    assert abs(sum(weights) - repair.TARGET_WEIGHT_PER_RUN) < 1e-6


# ---------------------------------------------------------------------------
# Monotonicity probe operates on the V2 width
# ---------------------------------------------------------------------------
def test_monotonicity_probe_v2_runs_on_correct_width():
    net = repair.build_model(None)
    result = repair._monotonicity_probe_v2(net)
    assert "strictly_decreasing" in result
    assert len(result["probe_scores"]) == len(result["probe_values"])
