"""Post-SAC ranker: typed separation, tri-state safety, learned level 2.

Anchors the defects the architecture review found:
  * measured SPICE outcomes were relabelled as surrogate predictions;
  * the learned model was reachable only on an exact tuple tie;
  * `model=None` silently produced a "DPO" decision;
  * unknown values were treated as good;
  * `startswith("verified")` accepted `verified_unstable`.
"""

from __future__ import annotations

import pytest

from agentic_raptor.ranking import (AuthoritativeSpiceOutcome, PostSACDesign,
                                    PredictionLeakage, RankerCheckpointMissing,
                                    RankerInputError, SurrogatePrediction,
                                    compare, hard_safety_tier,
                                    measured_preference, tri_state)

SPEC = {"spec_id": "S1", "gain_target_db": 80.0,
        "phase_margin_target_deg": 60.0}


def design(label, h, sized=True):
    return PostSACDesign(
        label=label, spec_id="S1", llm_proposal_id=f"p_{label}",
        canonical_graph_hash=h, topology_signature="2s_none",
        topology_family="2s_none",
        sizing_vector={"s1_w": 1.0} if sized else {},
        sizing_manifest_hash=f"m_{label}", sizing_spice_calls=8,
        sizing_spice_call_ids=[f"sz_{label}_1"])


def pred(h, *, margins=None, op=0.9, stab=0.9, unc=0.2):
    return SurrogatePrediction(
        topology_hash=h, sizing_manifest_hash=f"m_{h}",
        gain_db=85.0, pm_deg=65.0,
        normalized_margins=margins if margins is not None
        else {"gain": 0.25, "pm": 0.1},
        operating_point_probability=op, stability_probability=stab,
        predictive_uncertainty=unc, surrogate_checkpoint_hash="sur1")


class _Model:
    """Stand-in for the DPO-trained ranker."""

    def __init__(self, prefer="A"):
        self.prefer = prefer
        self.calls = 0

    def score(self, spec, d, p):
        self.calls += 1
        return 1.0 if d.label == self.prefer else 0.0


class _BrokenModel:
    def score(self, spec, d, p):
        raise RuntimeError("checkpoint corrupt")


# --------------------------- typed separation --------------------------------
def test_prediction_cannot_be_authoritative():
    with pytest.raises(PredictionLeakage):
        SurrogatePrediction(topology_hash="h", sizing_manifest_hash="m",
                            authoritative=True)


def test_prediction_cannot_carry_a_spice_result_id():
    with pytest.raises(PredictionLeakage):
        SurrogatePrediction(topology_hash="h", sizing_manifest_hash="m",
                            spice_result_id="call_1")


def test_violation_is_non_negative_and_zero_means_satisfied():
    assert pred("h", margins={"gain": 0.5, "pm": 0.2}
                ).worst_predicted_violation == 0.0
    assert pred("h", margins={"gain": 0.5, "pm": -0.3}
                ).worst_predicted_violation == pytest.approx(0.3)


def test_unknown_margins_do_not_mean_feasible():
    assert SurrogatePrediction(topology_hash="h",
                               sizing_manifest_hash="m").predicted_feasible \
        is None


# ------------------------- tri-state safety tier -----------------------------
def test_tri_state_orders_good_unknown_bad():
    assert tri_state(True) == 0 and tri_state(None) == 1
    assert tri_state(False) == 2
    assert tri_state(True) < tri_state(None) < tri_state(False)


def test_unknown_ranks_below_known_good():
    known = pred("h1")
    unknown = pred("h2", op=None, stab=None)
    assert hard_safety_tier(known) < hard_safety_tier(unknown)


def test_infeasible_loses_to_feasible_on_the_safety_gate():
    out = compare(design("A", "h1"), design("B", "h2"),
                  pred("h1", margins={"gain": 0.2, "pm": 0.2}),
                  pred("h2", margins={"gain": -0.9, "pm": 0.2}),
                  SPEC, model=_Model("B"))
    assert out["selected_design"] == "A"
    assert out["decision_basis"] == "hard_safety_gate"


# --------------------------- learned level 2 ---------------------------------
def test_learned_ranker_decides_inside_the_same_safety_tier():
    """The previous build reached the model only on an exact tuple tie."""
    m = _Model("B")
    out = compare(design("A", "h1"), design("B", "h2"),
                  pred("h1", margins={"gain": 0.30, "pm": 0.11}),
                  pred("h2", margins={"gain": 0.25, "pm": 0.09}),
                  SPEC, model=m)
    assert out["decision_basis"] == "dpo_ranker"
    assert out["selected_design"] == "B"
    assert m.calls == 2


def test_missing_ranker_checkpoint_fails_loudly():
    with pytest.raises(RankerCheckpointMissing,
                       match="POST_SAC_RANKER_CHECKPOINT_MISSING"):
        compare(design("A", "h1"), design("B", "h2"),
                pred("h1"), pred("h2"), SPEC, model=None)


def test_explicit_baseline_arm_is_allowed_and_labelled():
    """No DPO checkpoint is permitted ONLY under a named baseline arm, and
    the decision must not be reported as a DPO decision."""
    out = compare(design("A", "h1"), design("B", "h2"),
                  pred("h1", margins={"gain": 0.3, "pm": 0.3}, unc=0.1),
                  pred("h2", margins={"gain": 0.3, "pm": 0.3}, unc=0.8),
                  SPEC, model=None, ranker_arm="deterministic_feasibility")
    assert out["decision_basis"] == "explicit_baseline"
    assert out["ranker_arm"] == "deterministic_feasibility"
    assert out["selected_design"] == "A"      # lower uncertainty wins


def test_ranker_failure_is_logged_not_silently_zero():
    out = compare(design("A", "h1"), design("B", "h2"),
                  pred("h1"), pred("h2"), SPEC, model=_BrokenModel())
    assert out["ranker_error"] is not None
    assert "checkpoint corrupt" in out["ranker_error"]
    assert out["decision_basis"] == "deterministic_tie"


# ------------------------------ input guards ---------------------------------
def test_unsized_design_is_rejected():
    with pytest.raises(RankerInputError, match="UNSIZED"):
        compare(design("A", "h1"), design("B", "h2", sized=False),
                pred("h1"), pred("h2"), SPEC, model=_Model())


def test_prediction_must_belong_to_its_design():
    with pytest.raises(RankerInputError, match="provenance"):
        compare(design("A", "h1"), design("B", "h2"),
                pred("hX"), pred("h2"), SPEC, model=_Model())


def test_backup_is_the_unselected_design():
    out = compare(design("A", "h1"), design("B", "h2"),
                  pred("h1"), pred("h2"), SPEC, model=_Model("A"))
    assert out["selected_design"] == "A" and out["backup_design"] == "B"
    assert out["backup_topology_hash"] == "h2"


# --------------------------- measured preference -----------------------------
def outcome(h, **kw):
    base = dict(call_id=f"c_{h}", topology_hash=h,
                sizing_manifest_hash=f"m_{h}", netlist_hash=f"n_{h}",
                mode="final_verification", exact_spec_pass=False,
                operating_point_valid=True, spice_converged=True,
                verified_stable=True, hard_constraints_passed=4,
                normalized_distance_to_feasibility=0.3)
    base.update(kw)
    return AuthoritativeSpiceOutcome(**base)


def test_verified_unstable_is_never_treated_as_stable():
    """`startswith('verified')` accepted verified_unstable."""
    stable = outcome("h1", verified_stable=True,
                     stability_status="verified_stable")
    unstable = outcome("h2", verified_stable=False,
                       stability_status="verified_unstable")
    w, _ = measured_preference(stable, unstable)
    assert w == "A"


def test_exact_pass_beats_failure():
    w, _ = measured_preference(outcome("h1", exact_spec_pass=True),
                               outcome("h2", exact_spec_pass=False))
    assert w == "A"


def test_distance_decides_when_both_fail():
    w, _ = measured_preference(
        outcome("h1", normalized_distance_to_feasibility=0.6),
        outcome("h2", normalized_distance_to_feasibility=0.2))
    assert w == "B"


def test_identical_outcomes_tie():
    w, reason = measured_preference(outcome("h1"), outcome("h2"))
    assert w is None and reason == "tie"
