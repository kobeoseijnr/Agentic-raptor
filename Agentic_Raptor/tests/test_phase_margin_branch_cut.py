"""Phase-margin extraction must survive the +-180 deg branch cut.

An INVERTING amplifier has a DC phase of exactly +-180 deg, which is the
branch cut of np.angle (principal value in (-pi, pi]). Rounding in the
imaginary part alone decides whether the DC sample comes back as +179.9 or
-175.4 for the same circuit; np.unwrap then propagates that arbitrary choice
as a ~360 deg offset through every later sample.

Measured consequence before the fix: a 3-stage nested-Miller amplifier
produced delta = +92.65 deg and pm = +272.65, which the ">= 180" guard
discarded as "measurement_instability". Its true phase margin was about
-87 deg. A real, reportable instability was silently converted into
"unmeasurable" -- the worst possible failure mode for a measurement, because
it looks like missing data rather than a bad circuit.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentic_raptor.electrical.measurements import measure_phase_margin


def _two_pole(freq, dc_gain, f1, f2, invert=True):
    """Minimum-phase two-pole response; `invert` puts DC on the branch cut."""
    s = 1j * freq
    h = dc_gain / ((1 + s / f1) * (1 + s / f2))
    return -h if invert else h


FREQ = np.logspace(-1, 9, 1001)


def test_inverting_and_noninverting_give_the_same_phase_margin():
    """The sign convention at DC must not change the measured stability."""
    inv = measure_phase_margin(FREQ, _two_pole(FREQ, 3e3, 1e2, 1e6, True))
    non = measure_phase_margin(FREQ, _two_pole(FREQ, 3e3, 1e2, 1e6, False))
    assert inv.value is not None, f"inverting discarded: {inv.failure_reason}"
    assert non.value is not None
    assert inv.value == pytest.approx(non.value, abs=1.0)


def test_branch_cut_start_does_not_discard_the_measurement():
    """Rotating the whole response must not change the phase margin."""
    h = _two_pole(FREQ, 3e3, 1e2, 1e6, invert=True)
    base = measure_phase_margin(FREQ, h)
    assert base.value is not None
    # nudge DC across the branch cut, as float noise does in practice
    for eps in (1e-9, -1e-9, 1e-7, -1e-7):
        rotated = h * np.exp(1j * eps)
        r = measure_phase_margin(FREQ, rotated)
        assert r.value is not None, (
            f"eps={eps} discarded the measurement: {r.failure_reason}")
        assert r.value == pytest.approx(base.value, abs=1.0)


def test_accumulated_phase_is_never_positive():
    """A minimum-phase amplifier cannot gain phase as frequency rises, so a
    reported PM above 180 deg is an artefact, not a measurement."""
    for invert in (True, False):
        r = measure_phase_margin(FREQ, _two_pole(FREQ, 1e4, 10.0, 1e5, invert))
        if r.value is not None:
            assert r.value < 180.0


def test_unstable_circuit_reports_a_negative_margin_not_none():
    """Three poles below unity gain: unstable, and it must SAY so."""
    s = 1j * FREQ
    h = -1e5 / ((1 + s / 1e2) * (1 + s / 3e2) * (1 + s / 1e3))
    r = measure_phase_margin(FREQ, h)
    assert r.value is not None, (
        f"instability reported as unmeasurable: {r.failure_reason}")
    assert r.value < 0.0
