"""Stage 3B.2 adjudication invariants (deterministic synthetic TFs)."""

import numpy as np
import pytest

from agentic_raptor.electrical.adjudication import compare_polarity
from agentic_raptor.electrical.measurements import measure_phase_margin

F = np.logspace(0, 9, 1500)
W = 2j * np.pi * F


def _two_pole(a0, fp1, fp2):
    return a0 / ((1 + W / (2 * np.pi * fp1)) * (1 + W / (2 * np.pi * fp2)))


def test_pure_sign_flip_detected_as_180():
    h = _two_pole(1e4, 1e2, 1.5e6)
    c = compare_polarity(F, h, F, -h)
    assert c["magnitude_identical"] and c["phase_shift_approx_180"]


def test_non_polarity_difference_rejected():
    h = _two_pole(1e4, 1e2, 1.5e6)
    assert not compare_polarity(F, h, F, 2 * h)["magnitude_identical"]
    other = _two_pole(1e4, 1e2, 1.5e5)
    assert not compare_polarity(F, h, F, other)["phase_shift_approx_180"]


def test_unstable_pm_invariant_under_sign_flip():
    unstable = 1e5 / ((1 + W / (2 * np.pi * 1e2)) * (1 + W / (2 * np.pi * 1e3)) ** 2)
    a, b = measure_phase_margin(F, unstable), measure_phase_margin(F, -unstable)
    assert a.status == b.status == "verified"
    assert a.value < 0
    assert a.value == pytest.approx(b.value, abs=1.0), "PM must be convention-independent"


def test_stable_pm_positive():
    stable = 1e3 / (1 + W / (2 * np.pi * 1e3))
    assert measure_phase_margin(F, stable).value > 85


def test_level4_child_records_link_parents():
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "datasets/simulation_memory/adjudication_runs.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert rows and all(r["parent_run_id"] and r["run_id"] != r["parent_run_id"] for r in rows)
    assert all(r["stability_status"] in
               ("verified_stable", "verified_unstable", "polarity_mismatch",
                "ambiguous", "unsupported", "insufficient_information") for r in rows)
    assert all("reasoning" in r for r in rows if r["stability_status"] == "verified_unstable")
