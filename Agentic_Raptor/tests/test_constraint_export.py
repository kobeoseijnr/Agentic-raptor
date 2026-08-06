"""Fix 5: an exact-pass verdict spans five hard constraints, so every one of
them must be exported with target, achieved, margin and verdict -- and a
failure must name its cause. Rows that satisfied gain and PM yet still read
exact_pass=False previously gave the reader nothing to go on.
"""

from __future__ import annotations

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

from agentic_raptor.mb_sac.spec_sizing import postsizing_outcome

SPEC = {"gain_target_db": 80.0, "phase_margin_target_deg": 60.0,
        "load_capacitance_pf": 200.0, "ugbw_target_hz": 1e6}


def meas(**kw):
    base = {"gain_db": 85.0, "pm_deg": 65.0, "ugbw_hz": 5e6, "power_w": 4e-5,
            "stable": True, "stability": "verified_stable",
            "electrical": "electrically_functional", "op_valid": True}
    base.update(kw)
    return base


def test_all_constraints_exported_with_target_achieved_margin():
    o = postsizing_outcome(meas(), SPEC)
    c = o["constraints"]
    for key in ("gain_db", "phase_margin_deg", "ugbw_hz", "power_w",
                "area_um2"):
        assert key in c, f"{key} missing from constraint export"
        for field in ("target", "achieved", "margin", "passed"):
            assert field in c[key], f"{key}.{field} missing"
    for key in ("operating_point_valid", "spice_converged",
                "stability_status", "exact_spec_pass",
                "exact_failure_reason", "worst_failing_constraint",
                "normalized_distance_to_feasibility"):
        assert key in o, f"{key} missing from outcome"


def test_passing_row_has_no_failure_reason():
    o = postsizing_outcome(meas(), SPEC)
    assert o["exact_spec_pass"] is True
    assert o["exact_failure_reason"] is None
    assert o["worst_failing_constraint"] is None


def test_gain_and_pm_pass_but_ugbw_fails_names_ugbw():
    """The exact case that used to be unexplainable in the tables."""
    o = postsizing_outcome(meas(ugbw_hz=1e4), SPEC)
    assert o["exact_spec_pass"] is False
    assert o["constraints"]["gain_db"]["passed"] is True
    assert o["constraints"]["phase_margin_deg"]["passed"] is True
    assert o["constraints"]["ugbw_hz"]["passed"] is False
    assert "UGBW below target" in o["exact_failure_reason"]
    assert o["worst_failing_constraint"] == "ugbw"


def test_each_failure_mode_is_named():
    cases = [
        (meas(gain_db=60.0), "gain below target"),
        (meas(pm_deg=10.0), "PM below target"),
        (meas(ugbw_hz=1e3), "UGBW below target"),
        (meas(stable=False, stability="verified_unstable"), "unstable"),
        (meas(op_valid=False), "invalid operating point"),
    ]
    for m, expected in cases:
        o = postsizing_outcome(m, SPEC)
        assert o["exact_spec_pass"] is False
        assert expected in (o["exact_failure_reason"] or ""), \
            f"expected {expected!r}, got {o['exact_failure_reason']!r}"


def test_unbudgeted_constraints_are_not_silently_passed():
    """No power/area budget in the corpus: must read 'not applicable',
    never 'passed'."""
    o = postsizing_outcome(meas(), SPEC)
    for key in ("power_w", "area_um2"):
        assert o["constraints"][key]["applicable"] is False
        assert o["constraints"][key]["passed"] is None


def test_worst_constraint_is_the_largest_normalised_violation():
    o = postsizing_outcome(meas(gain_db=40.0, pm_deg=59.0), SPEC)
    # gain misses by 40 dB (40/20 = 2.0); PM misses by 1 deg (1/45 = 0.02)
    assert o["worst_failing_constraint"] == "gain"
