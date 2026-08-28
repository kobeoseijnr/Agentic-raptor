"""STAGE 9B GATE REPAIR (2026-08-14): measured-evidence-first ranking.

Anchors the two defects the F0-F3 diagnosis measured:
  * the surrogate-driven hard gate decided 12/12 runs (DPO 0) and chose
    the authoritative-worse branch in 6/8 both-verified decisions -- it
    even vetoed a branch whose REAL sizing-time SPICE pass was recorded
    (predicted PM -9.3 deg vs 58.9 measured);
  * `dpo_margin_threshold` was referenced in compare()'s body but absent
    from its signature: every model scoring raised NameError, silently
    caught into "deterministic_tie" -- DPO disabled at Level 2.
Zero SPICE; same fixture pattern as test_post_sac_ranker.py.
"""
from __future__ import annotations

import inspect

import pytest

from agentic_raptor.ranking import PostSACDesign, SurrogatePrediction, compare
from agentic_raptor.ranking.post_sac import MEASURED_DIST_BAND, _measured_tier

SPEC = {"spec_id": "S1", "gain_target_db": 80.0,
        "phase_margin_target_deg": 60.0}


def design(label, h):
    return PostSACDesign(
        label=label, spec_id="S1", llm_proposal_id=f"p_{label}",
        canonical_graph_hash=h, topology_signature="2s_none",
        topology_family="2s_none", sizing_vector={"s1_w": 1.0},
        sizing_manifest_hash=f"m_{label}", sizing_spice_calls=8,
        sizing_spice_call_ids=[f"sz_{label}_1"])


def pred(h, *, stab=0.9, margins=None, unc=0.2):
    return SurrogatePrediction(
        topology_hash=h, sizing_manifest_hash=f"m_{h}",
        gain_db=85.0, pm_deg=65.0,
        normalized_margins=margins if margins is not None
        else {"gain": 0.25, "pm": 0.1},
        operating_point_probability=None, stability_probability=stab,
        predictive_uncertainty=unc, surrogate_checkpoint_hash="sur1")


def ev(*, n=16, passed=False, best=None):
    return {"n_measured": n, "exact_spec_pass": passed,
            "best_distance": best, "source": "sizing_spice_stage6"}


class _Model:
    def __init__(self, prefer="A", margin=1.0):
        self.prefer, self.margin = prefer, margin

    def score(self, spec, d, p):
        return self.margin if d.label == self.prefer else 0.0


AB = lambda: (design("A", "hA"), design("B", "hB"))


# ---------------------------------------------------------------------------
# measured tier semantics
# ---------------------------------------------------------------------------
def test_measured_tier_tristate():
    assert _measured_tier(ev(passed=True)) == 0
    assert _measured_tier(ev(passed=False)) == 2
    assert _measured_tier(ev(n=0)) == 1
    assert _measured_tier(None) == 1


# ---------------------------------------------------------------------------
# Level 0: a real measured pass beats surrogate opinion
# ---------------------------------------------------------------------------
def test_real_pass_cannot_be_vetoed_by_surrogate_stability():
    # the exact F1-idx001 shape: B passed in sizing, surrogate calls B
    # unstable (0.011) and prefers A (0.993)
    a, b = AB()
    out = compare(a, b, pred("hA", stab=0.993, margins={"gain": -0.1}),
                  pred("hB", stab=0.011, margins={"gain": -0.2}),
                  SPEC, model=None, ranker_arm="explicit_baseline",
                  measured_a=ev(passed=False, best=0.11),
                  measured_b=ev(passed=True, best=0.0),
                  gate_mode="measured_first")
    assert out["selected_design"] == "B"
    assert out["decision_basis"] == "measured_evidence_gate"
    assert out["deciding_level"] == "measured_spec_pass"


def test_both_passed_goes_to_learned_ranker_not_surrogate():
    a, b = AB()
    out = compare(a, b, pred("hA", stab=0.99), pred("hB", stab=0.01),
                  SPEC, model=_Model(prefer="B"),
                  measured_a=ev(passed=True, best=0.0),
                  measured_b=ev(passed=True, best=0.0),
                  gate_mode="measured_first")
    assert out["selected_design"] == "B"      # surrogate stab gap ignored
    assert out["decision_basis"] == "dpo_ranker"


def test_neither_passed_clear_distance_gap_decides():
    a, b = AB()
    out = compare(a, b, pred("hA", stab=0.01), pred("hB", stab=0.99),
                  SPEC, model=None, ranker_arm="explicit_baseline",
                  measured_a=ev(best=0.04), measured_b=ev(best=0.27),
                  gate_mode="measured_first")
    assert out["selected_design"] == "A"
    assert out["decision_basis"] == "measured_distance_gate"


def test_neither_passed_close_call_goes_to_learned_ranker():
    a, b = AB()
    gap = MEASURED_DIST_BAND * 0.5
    out = compare(a, b, pred("hA", stab=0.01), pred("hB", stab=0.99),
                  SPEC, model=_Model(prefer="A"),
                  measured_a=ev(best=0.10), measured_b=ev(best=0.10 + gap),
                  gate_mode="measured_first")
    assert out["decision_basis"] == "dpo_ranker"
    assert out["selected_design"] == "A"


def test_no_measurements_falls_back_to_surrogate_gate():
    a, b = AB()
    out = compare(a, b, pred("hA", stab=0.9), pred("hB", stab=0.1),
                  SPEC, model=None, ranker_arm="explicit_baseline",
                  measured_a=ev(n=0), measured_b=ev(n=0),
                  gate_mode="measured_first")
    assert out["decision_basis"] == "hard_safety_gate"


def test_default_gate_mode_is_byte_identical_old_behavior():
    a, b = AB()
    kw = dict(model=None, ranker_arm="explicit_baseline",
              measured_a=ev(passed=True, best=0.0),
              measured_b=ev(passed=False, best=0.5))
    old = compare(a, b, pred("hA", stab=0.1), pred("hB", stab=0.9), SPEC, **kw)
    # default mode ignores measured evidence entirely: surrogate gate rules
    assert old["decision_basis"] == "hard_safety_gate"
    assert old["selected_design"] == "B"
    assert "measured_tier_A" not in old


# ---------------------------------------------------------------------------
# the dpo_margin_threshold signature fix
# ---------------------------------------------------------------------------
def test_dpo_threshold_is_a_real_parameter_and_model_scoring_works():
    assert "dpo_margin_threshold" in inspect.signature(compare).parameters
    a, b = AB()
    out = compare(a, b, pred("hA"), pred("hB"), SPEC, model=_Model("A", 1.0))
    assert out["ranker_error"] is None        # no NameError swallowed
    assert out["decision_basis"] == "dpo_ranker"


def test_confidence_gate_fires_below_threshold():
    a, b = AB()
    # equal safety tiers (so the model is reachable) but distinguishable
    # deterministic scores via uncertainty (identical preds would end as
    # "deterministic_tie")
    out = compare(a, b, pred("hA"), pred("hB", unc=0.6), SPEC,
                  model=_Model("A", margin=0.2), dpo_margin_threshold=0.4574)
    assert out["dpo_gate_fallback"] is True
    assert out["decision_basis"] == "dpo_low_confidence_deterministic_fallback"
    out2 = compare(a, b, pred("hA"), pred("hB"), SPEC,
                   model=_Model("A", margin=0.9), dpo_margin_threshold=0.4574)
    assert out2["decision_basis"] == "dpo_ranker"


# ---------------------------------------------------------------------------
# pipeline wiring
# ---------------------------------------------------------------------------
def test_run_pipeline_wires_evidence_threshold_and_gate_mode():
    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    sig = inspect.signature(v2.run_pipeline)
    # STAGE 9B PROMOTION (2026-08-15): measured-first IS the live default
    # (user decision after F4 3/6 vs F0 2/6 + 53%->71% selection accuracy).
    # compare()'s own default stays "surrogate" so non-pipeline callers are
    # byte-identical unless they opt in.
    assert sig.parameters["gate_mode"].default == "measured_first"
    from agentic_raptor.ranking.post_sac import compare as _cmp
    assert inspect.signature(_cmp).parameters["gate_mode"].default == "surrogate"
    assert "dpo_margin_threshold=_dpo_threshold" in src
    assert "measured_a=ev_a" in src and "measured_b=ev_b" in src
    assert "gate_mode=gate_mode" in src
    assert "_measured_evidence(sza)" in src and "_measured_evidence(szb)" in src
