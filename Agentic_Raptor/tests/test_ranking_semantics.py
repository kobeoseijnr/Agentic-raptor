"""Parts 2/3/5/6/7: prediction coverage, checkpoint identity, missingness.

Each test anchors a specific defect found in the architecture review, so a
regression reintroduces a named failure rather than a vague one.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agentic_raptor.ranking import (AuthoritativeSpiceOutcome,
                                    SurrogatePrediction, checkpoint_sha256,
                                    measured_preference, predict_post_sac,
                                    required_constraint_names,
                                    state_dict_sha256)

SPEC = {"spec_id": "S1", "gain_target_db": 80.0,
        "phase_margin_target_deg": 60.0, "ugbw_target_hz": 1e6}


# ------------------------- Part 3: partial coverage --------------------------
def test_required_constraints_come_from_the_spec():
    assert required_constraint_names(SPEC) == ("gain", "pm", "ugbw")
    assert required_constraint_names({"gain_target_db": 1.0}) == ("gain",)
    assert required_constraint_names({}) == ()


def test_missing_required_constraint_gives_unknown_not_pass():
    """gain and PM pass, UGBW unpredicted -> feasibility is UNKNOWN."""
    p = SurrogatePrediction(topology_hash="h", sizing_manifest_hash="m",
                            normalized_margins={"gain": 0.5, "pm": 0.2})
    assert p.predicted_feasible_for(SPEC) is None
    assert p.missing_constraints(SPEC) == ("ugbw",)


def test_full_coverage_can_report_feasible():
    p = SurrogatePrediction(topology_hash="h", sizing_manifest_hash="m",
                            normalized_margins={"gain": 0.5, "pm": 0.2,
                                                "ugbw": 0.1})
    assert p.predicted_feasible_for(SPEC) is True


def test_full_coverage_with_a_violation_is_infeasible():
    p = SurrogatePrediction(topology_hash="h", sizing_manifest_hash="m",
                            normalized_margins={"gain": 0.5, "pm": -0.2,
                                                "ugbw": 0.1})
    assert p.predicted_feasible_for(SPEC) is False


def test_spec_with_power_budget_requires_a_power_prediction():
    spec = dict(SPEC, power_target_w=1e-3)
    p = SurrogatePrediction(topology_hash="h", sizing_manifest_hash="m",
                            normalized_margins={"gain": 0.5, "pm": 0.2,
                                                "ugbw": 0.1})
    assert "power" in required_constraint_names(spec)
    assert p.predicted_feasible_for(spec) is None


# ------------------------- Part 5: checkpoint identity -----------------------
def test_state_dict_hash_changes_when_any_parameter_changes():
    """Tensor-sum hashing collides under permutation; SHA-256 must not."""
    import torch
    a = {"w": torch.tensor([1.0, 2.0, 3.0])}
    b = {"w": torch.tensor([3.0, 2.0, 1.0])}      # same sum, different model
    c = {"w": torch.tensor([1.0, 2.0, 3.0])}
    assert state_dict_sha256(a) != state_dict_sha256(b)
    assert state_dict_sha256(a) == state_dict_sha256(c)


def test_checkpoint_sha256_reads_file_bytes(tmp_path: Path):
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"weights-v1")
    h1 = checkpoint_sha256(f)
    f.write_bytes(b"weights-v2")
    assert h1 != checkpoint_sha256(f)
    assert checkpoint_sha256(tmp_path / "missing.pt") is None
    assert len(h1) == 64


# --------------------------- Part 2: surrogate -------------------------------
def test_surrogate_never_invokes_ngspice():
    src = inspect.getsource(predict_post_sac)
    for banned in ("discover_ngspice", "measure(", "exe", "subprocess"):
        assert banned not in src, f"surrogate references {banned!r}"


def test_surrogate_returns_unknown_for_unmodelled_metrics():
    p = predict_post_sac(SPEC, "h1", "2s_none", {"s1_w": 1.0}, "m1")
    for field in ("ugbw_hz", "power_w", "area_um2"):
        assert getattr(p, field) is None, f"{field} was fabricated"
    assert p.operating_point_probability is None
    assert p.source == "surrogate" and p.authoritative is False


def test_surrogate_uncertainty_is_not_a_hard_coded_constant():
    src = inspect.getsource(predict_post_sac)
    assert "predictive_uncertainty = 0.25" not in src
    assert "mc_dropout" in src


# ------------------------- Part 7: missingness ordering ----------------------
def outcome(h, **kw):
    base = dict(call_id=f"c_{h}", topology_hash=h,
                sizing_manifest_hash=f"m_{h}", netlist_hash=f"n_{h}",
                mode="final_verification", spec_id="S1", spec_hash="sh1",
                exact_spec_pass=True, operating_point_valid=True,
                spice_converged=True, verified_stable=True,
                hard_constraints_passed=5,
                normalized_distance_to_feasibility=0.0)
    base.update(kw)
    return AuthoritativeSpiceOutcome(**base)


def test_unknown_power_does_not_beat_measured_power():
    """`power_w if not None else 0.0` made unmeasured look optimal."""
    w, _ = measured_preference(outcome("h1", power_w=None),
                               outcome("h2", power_w=1e-4))
    assert w == "B"


def test_lower_measured_power_wins_when_both_known():
    w, _ = measured_preference(outcome("h1", power_w=5e-4),
                               outcome("h2", power_w=1e-4))
    assert w == "B"


def test_unknown_area_does_not_beat_measured_area():
    w, _ = measured_preference(outcome("h1", power_w=1e-4, area_um2=None),
                               outcome("h2", power_w=1e-4, area_um2=500.0))
    assert w == "B"


def test_higher_robustness_wins():
    w, _ = measured_preference(
        outcome("h1", power_w=1e-4, area_um2=100.0, robustness=0.6),
        outcome("h2", power_w=1e-4, area_um2=100.0, robustness=0.95))
    assert w == "B"


def test_secondary_objectives_never_override_exact_pass():
    """A failing design with perfect power must not win."""
    w, _ = measured_preference(
        outcome("h1", exact_spec_pass=True, power_w=9e-3),
        outcome("h2", exact_spec_pass=False, power_w=1e-9,
                normalized_distance_to_feasibility=0.4))
    assert w == "A"
