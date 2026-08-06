"""Analytical validation of the measurement library (no fabricated data —
synthetic transfer functions with closed-form expected values)."""

from __future__ import annotations

import numpy as np
import pytest

from agentic_raptor.electrical.measurements import (
    find_unity_crossings,
    measure_all,
    measure_gain,
    measure_phase_margin,
    measure_ugbw,
)

F = np.logspace(0, 9, 2000)
W = 2j * np.pi * F


def single_pole(a0: float, fp: float) -> np.ndarray:
    return a0 / (1 + W / (2 * np.pi * fp))


def two_pole(a0: float, fp1: float, fp2: float) -> np.ndarray:
    return a0 / ((1 + W / (2 * np.pi * fp1)) * (1 + W / (2 * np.pi * fp2)))


def test_single_pole_gain_ugbw_pm90():
    h = single_pole(1000.0, 1e3)  # 60 dB, UGBW ≈ 1 MHz, PM ≈ 90°
    r = measure_all(F, h)
    assert r["dc_gain_db"].status == "verified"
    assert r["dc_gain_db"].value == pytest.approx(60.0, abs=0.01)
    assert r["ugbw_hz"].status == "verified"
    assert r["ugbw_hz"].value == pytest.approx(1e6, rel=0.01)
    assert r["phase_margin_deg"].status == "verified"
    assert r["phase_margin_deg"].value == pytest.approx(90.0, abs=1.0)


@pytest.mark.parametrize(("fp2_over_ugbw", "expected_pm"), [(1 / np.sqrt(2), 45.0), (1.5, 60.0)])
def test_two_pole_known_pm(fp2_over_ugbw, expected_pm):
    # Closed-form: crossing shifts with fp2; PM = 90° − atan(f_x/fp2) where
    # (fu0/f_x)·(1+(f_x/fp2)²)^-1/2 = 1. fp2 = fu0/√2 → PM 45°; fp2 = 1.5·fu0 → PM 60°.
    a0, fp1 = 1e4, 1e2  # fu0 = a0·fp1 = 1e6
    fu = a0 * fp1
    h = two_pole(a0, fp1, fp2_over_ugbw * fu)
    r = measure_phase_margin(F, h)
    assert r.status == "verified"
    assert r.value == pytest.approx(expected_pm, abs=2.0)


def test_no_unity_crossing():
    h = single_pole(0.5, 1e3)  # max gain < 1 → no crossing
    r = measure_ugbw(F, h)
    assert r.status == "unsupported" and r.failure_reason == "no_unity_crossing"
    pm = measure_phase_margin(F, h)
    assert pm.value is None and pm.confidence == 0.0


def test_multiple_crossings_ambiguous():
    # magnitude dips below 1 then rises above then falls: 3 crossings
    h = single_pole(100.0, 1e3) * (1 + (W / (2 * np.pi * 3e5)) ** 2 * 0 + 0)  # base
    notch = (1 + W / (2 * np.pi * 2e4)) / (1 + W / (2 * np.pi * 2e6))
    h = single_pole(100.0, 1e3) / notch * notch  # keep simple: construct explicit mags
    mag = np.abs(single_pole(100.0, 1e3))
    bump = 1 + 5 * np.exp(-((np.log10(F) - 6.5) ** 2) / 0.01)
    h = mag * bump * np.exp(1j * np.angle(single_pole(100.0, 1e3)))
    crossings = find_unity_crossings(F, h)
    if len([c for c in crossings if c["crossing_type"] in ("down", "exact")]) > 1:
        r = measure_ugbw(F, h)
        assert r.status == "ambiguous" and r.failure_reason == "multiple_crossings"
        pm = measure_phase_margin(F, h)
        assert pm.value is None
        assert pm.metadata.get("measurement_status") == "ambiguous_phase_margin"


def test_unstable_system_negative_pm_is_measured():
    h = two_pole(1e5, 1e2, 1e3) / (1 + W / (2 * np.pi * 1e3))  # 3 low poles → PM < 0
    r = measure_phase_margin(F, h)
    assert r.status == "verified" and r.value is not None and r.value < 0


def test_wrapped_phase_input_handled():
    h = two_pole(1e4, 1e2, 1e5)
    # simulate simulator-wrapped phase: rebuild H from wrapped angles
    wrapped = np.angle(h)  # np.angle is inherently wrapped to (-pi, pi]
    h_wrapped = np.abs(h) * np.exp(1j * wrapped)
    a = measure_phase_margin(F, h)
    b = measure_phase_margin(F, h_wrapped)
    assert a.status == b.status == "verified"
    assert a.value == pytest.approx(b.value, abs=0.1)


def test_nonfinite_rejected():
    h = single_pole(1000.0, 1e3)
    h[100] = np.nan + 1j * np.nan
    for r in measure_all(F, h).values():
        assert r.status == "measurement_failed" and r.value is None and r.confidence == 0.0


def test_confidence_zero_for_untrusted():
    h = single_pole(0.5, 1e3)
    for r in measure_all(F, h).values():
        if r.status != "verified":
            assert r.confidence == 0.0


def test_interpolated_crossing_accuracy():
    h = single_pole(1000.0, 1e3)
    coarse = np.logspace(0, 9, 40)  # very coarse sweep
    hc = 1000.0 / (1 + 2j * np.pi * coarse / (2 * np.pi * 1e3))
    r = measure_ugbw(coarse, hc)
    assert r.status == "verified"
    assert r.value == pytest.approx(1e6, rel=0.05), "log-interpolation must recover crossing"
