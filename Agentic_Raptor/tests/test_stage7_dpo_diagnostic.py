"""Stage 7: FULL_DPO vs NO_LEARNED_DPO diagnostic -- mechanics/bookkeeping
tests. No SPICE needed: exercises the reconstruction, categorisation and
scoring bookkeeping in run_stage7_dpo_diagnostic.py plus the leakage/label
guards in agentic_raptor.ranking.post_sac / .model / .types it depends on.
"""
from __future__ import annotations

import math

import pytest

from agentic_raptor.ranking.post_sac import PostSACDesign, _provenance_ok
from agentic_raptor.ranking.types import (AuthoritativeSpiceOutcome,
                                          SurrogatePrediction)


def _design(label="A", graph_hash="h_a"):
    return PostSACDesign(label=label, spec_id="S", llm_proposal_id="p1",
                         canonical_graph_hash=graph_hash,
                         topology_signature="fam", topology_family="fam",
                         sizing_vector={"s1_w": 1.0}, sizing_manifest_hash="m1",
                         sac_trajectory_id="t1", sizing_spice_call_ids=["c1"])


def _outcome(call_id="final:1", topology_hash="h_a", passed=True, spec_id="S"):
    return AuthoritativeSpiceOutcome(
        call_id=call_id, topology_hash=topology_hash, sizing_manifest_hash="m1",
        netlist_hash="n1", mode="final_verification", exact_spec_pass=passed,
        operating_point_valid=True, spice_converged=True, verified_stable=True,
        spec_id=spec_id, spec_hash="sh1", normalized_distance_to_feasibility=0.0)


# ---------------------------------------------------------------------------
# Reconstruction fidelity (asdict -> dataclass round trip used to replay
# frozen trusted pairs without any new SPICE calls)
# ---------------------------------------------------------------------------
def test_design_asdict_roundtrip_reproduces_canonical_hash():
    from dataclasses import asdict, fields
    d = _design()
    names = {f.name for f in fields(PostSACDesign)}
    rebuilt = PostSACDesign(**{k: v for k, v in asdict(d).items() if k in names})
    assert rebuilt.canonical_graph_hash == d.canonical_graph_hash
    assert rebuilt.topology_hash == d.canonical_graph_hash


def test_outcome_asdict_roundtrip_stays_authoritative():
    from dataclasses import asdict, fields
    o = _outcome()
    names = {f.name for f in fields(AuthoritativeSpiceOutcome)}
    rebuilt = AuthoritativeSpiceOutcome(**{k: v for k, v in asdict(o).items() if k in names})
    assert rebuilt.authoritative is True
    assert rebuilt.exact_spec_pass == o.exact_spec_pass


# ---------------------------------------------------------------------------
# Hard-pair categorisation (Section 21)
# ---------------------------------------------------------------------------
def test_hard_pair_category_classification():
    import run_stage7_dpo_diagnostic as diag

    both_feasible = diag._hard_pair_category(_outcome(passed=True), _outcome(passed=True))
    assert both_feasible == "both_feasible"

    one_one = diag._hard_pair_category(_outcome(passed=True), _outcome(passed=False))
    assert one_one == "one_feasible_one_infeasible"

    close_a = AuthoritativeSpiceOutcome(**{**vars(_outcome(passed=False)),
                                           "normalized_distance_to_feasibility": 0.1})
    close_b = AuthoritativeSpiceOutcome(**{**vars(_outcome(passed=False)),
                                           "normalized_distance_to_feasibility": 0.15})
    assert diag._hard_pair_category(close_a, close_b) == "both_infeasible_close"

    far_a = AuthoritativeSpiceOutcome(**{**vars(_outcome(passed=False)),
                                         "normalized_distance_to_feasibility": 0.9})
    far_b = AuthoritativeSpiceOutcome(**{**vars(_outcome(passed=False)),
                                         "normalized_distance_to_feasibility": 0.3})
    assert diag._hard_pair_category(far_a, far_b) == "both_infeasible_far"


# ---------------------------------------------------------------------------
# Protected-pair rejection (Section 26 leakage audit)
# ---------------------------------------------------------------------------
def test_provenance_ok_rejects_a_protected_evaluation_spec(monkeypatch):
    import agentic_raptor.publication.eval_sets as es
    monkeypatch.setattr(es, "excluded_context_ids", lambda: {"PROTECTED_SPEC"})
    d = _design()
    o = _outcome(spec_id="PROTECTED_SPEC")
    problems = _provenance_ok(d, o, spec_id="PROTECTED_SPEC")
    assert "spec_in_protected_evaluation_set" in problems


def test_provenance_ok_flags_reused_sizing_call_as_final():
    d = _design()
    o = _outcome(call_id="c1")   # same id as d.sizing_spice_call_ids
    problems = _provenance_ok(d, o, spec_id="S")
    assert "final_call_id_is_a_sizing_call_id" in problems


def test_provenance_ok_accepts_a_clean_pair():
    d = _design()
    o = _outcome(call_id="final:new")
    assert _provenance_ok(d, o, spec_id="S") == []


# ---------------------------------------------------------------------------
# Authoritative outcome scoring uses ONLY measured fields (Section 27)
# ---------------------------------------------------------------------------
def test_measured_preference_ignores_predictions_entirely():
    """measured_preference's signature structurally cannot see a
    SurrogatePrediction -- it only accepts two AuthoritativeSpiceOutcome
    objects, so a surrogate's (possibly wrong) guess cannot influence the
    training label regardless of how confident it was."""
    from agentic_raptor.ranking.post_sac import measured_preference
    import inspect
    sig = inspect.signature(measured_preference)
    assert list(sig.parameters) == ["a", "b"]
    for p in sig.parameters.values():
        ann = p.annotation
        assert ann is inspect._empty or "SurrogatePrediction" not in str(ann)
    winner, reason = measured_preference(_outcome(passed=True), _outcome(passed=False))
    assert winner == "A" and reason == "measured_hierarchy"


def test_pair_label_hierarchy_feasible_beats_infeasible_regardless_of_fom():
    """Section 14: an infeasible high-FoM candidate must never outrank a
    feasible one. B is infeasible but has a far better (lower) distance and
    could look attractive on a FoM-only metric; A must still win because it
    passed spec."""
    from agentic_raptor.ranking.post_sac import measured_preference
    a = _outcome(call_id="a", passed=True)
    b = AuthoritativeSpiceOutcome(**{**vars(_outcome(call_id="b", passed=False)),
                                     "normalized_distance_to_feasibility": 0.001})
    winner, _ = measured_preference(a, b)
    assert winner == "A"


# ---------------------------------------------------------------------------
# No future-authoritative feature leakage (Section 16)
# ---------------------------------------------------------------------------
def test_ranker_features_are_derivable_from_prediction_alone():
    from agentic_raptor.ranking.model import features
    pred = SurrogatePrediction(topology_hash="h", sizing_manifest_hash="m",
                               normalized_margins={"gain": 0.1, "pm": -0.2})
    spec = {"gain_target_db": 60.0, "phase_margin_target_deg": 45.0}
    feats = features(spec, pred)
    assert all(math.isfinite(x) for x in feats)


def test_surrogate_prediction_structurally_cannot_hold_a_spice_result():
    from agentic_raptor.ranking.types import PredictionLeakage
    with pytest.raises(PredictionLeakage):
        SurrogatePrediction(topology_hash="h", sizing_manifest_hash="m",
                            spice_result_id="some_call_id")


# ---------------------------------------------------------------------------
# DPO decision-share / conditional-accuracy bookkeeping (Section 23)
# ---------------------------------------------------------------------------
def _fake_result(basis, correct, is_train=True):
    return {"full_dpo": {"decision_basis": basis, "correct": correct,
                         "selected": "A", "score_margin": 0.3,
                         "selected_is_feasible": True},
           "no_learned_dpo": {"decision_basis": "hard_safety_gate", "correct": True,
                              "selected": "A", "score_margin": None,
                              "selected_is_feasible": True},
           "is_train_pair": is_train, "informative": True,
           "hard_pair_category": "one_feasible_one_infeasible",
           "consistent_with_stored_label": True}


def test_dpo_decision_share_and_conditional_accuracy():
    import run_stage7_dpo_diagnostic as diag

    results = ([_fake_result("hard_safety_gate", True)] * 6
              + [_fake_result("dpo_ranker", True)] * 3
              + [_fake_result("dpo_ranker", False)] * 1)
    summary = diag.summarize(results)
    assert summary["hard_gate_only_decision_share"] == pytest.approx(0.6)
    assert summary["dpo_decision_share"] == pytest.approx(0.4)
    # 3 correct of 4 dpo_ranker-decided
    assert summary["dpo_conditional_accuracy"] == pytest.approx(0.75)
