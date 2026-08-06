"""Fix 2: the hard stage filter must never remove an electrically valid
family. Regression anchor is the measured counter-example that killed the
rule -- spec 004_t_hard_topology_0002, structure 2s_none, 82.74 dB / 64.85
deg, the ONLY exact pass in the 40-circuit stage-rule battery, and a
candidate the hard rule rejects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

from run_puct_ablation import (CORPUS_CLASSES, MIN_PRIOR, compatible_classes,
                               tier_classes, topology_priors)

_ROOT = Path(__file__).resolve().parents[1]

#: the measured counter-example (artifacts/publication/stage_rule_check)
HARD_SPEC = {"gain_target_db": 80.42, "phase_margin_target_deg": 60.0,
             "load_capacitance_pf": 200.0, "ugbw_target_hz": 1e4}
PASSING_STRUCTURE = "2s_none"


def test_hard_rule_would_have_rejected_the_only_passing_candidate():
    """Documents WHY the gate had to go."""
    assert PASSING_STRUCTURE not in tier_classes(HARD_SPEC)


def test_passing_candidate_is_searchable_now():
    assert PASSING_STRUCTURE in compatible_classes(HARD_SPEC)
    assert topology_priors(HARD_SPEC)[PASSING_STRUCTURE] > 0.0


def test_no_family_is_ever_excluded_by_prior():
    for gain in (20.0, 45.0, 70.0, 95.0, 130.0):
        for pm in (40.0, 45.0, 55.0, 60.0, 70.0):
            spec = {"gain_target_db": gain, "phase_margin_target_deg": pm,
                    "load_capacitance_pf": 100.0, "ugbw_target_hz": 1e5}
            pri = topology_priors(spec)
            assert set(pri) == set(CORPUS_CLASSES)
            assert all(v >= MIN_PRIOR for v in pri.values()), pri
            assert abs(sum(pri.values()) - 1.0) < 1e-6


def test_prior_still_leans_the_right_way():
    """Soft does not mean uninformative."""
    low = topology_priors({"gain_target_db": 40.0,
                           "phase_margin_target_deg": 45.0,
                           "load_capacitance_pf": 100.0,
                           "ugbw_target_hz": 1e5})
    high = topology_priors({"gain_target_db": 120.0,
                            "phase_margin_target_deg": 45.0,
                            "load_capacitance_pf": 100.0,
                            "ugbw_target_hz": 1e5})
    assert high["3s_miller"] > low["3s_miller"]
    assert high["2s_none"] < low["2s_none"]
    tight = topology_priors({"gain_target_db": 70.0,
                             "phase_margin_target_deg": 65.0,
                             "load_capacitance_pf": 100.0,
                             "ugbw_target_hz": 1e5})
    loose = topology_priors({"gain_target_db": 70.0,
                             "phase_margin_target_deg": 40.0,
                             "load_capacitance_pf": 100.0,
                             "ugbw_target_hz": 1e5})
    assert tight["3s_miller"] > loose["3s_miller"]


def test_gated_control_arm_is_preserved():
    """The old hard rule must remain available as a negative control."""
    gated = topology_priors(HARD_SPEC, gated=True)
    assert gated[PASSING_STRUCTURE] == 0.0
    assert compatible_classes(HARD_SPEC, tier_gated=True) == \
        tier_classes(HARD_SPEC)


def test_measured_battery_confirms_the_counter_example():
    """If the recorded battery is present, the anchor must match it."""
    f = _ROOT / "artifacts/publication/stage_rule_check/rows.jsonl"
    if not f.is_file():
        pytest.skip("stage-rule battery not run")
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    passes = [r for r in rows if r["exact_pass"]]
    assert passes, "battery recorded no exact pass"
    for r in passes:
        assert r["cls"] == PASSING_STRUCTURE
        assert r["rule_says_ok"] is False, \
            "the passing candidate should be one the hard rule rejected"
