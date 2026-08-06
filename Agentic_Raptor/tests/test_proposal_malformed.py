"""A malformed proposal is invalid, never fatal.

The schema says stages/compensation/feedback_paths hold objects. A proposer
sampled up the diversity ladder (T=1.5) also emits them as lists of bare
strings. proposal_dict_valid used to call .get() on those and raise
AttributeError, which aborted a whole diversity campaign four contexts in.

Rejecting is deliberate: coercing "miller" into {"type": "miller"} would
credit the model for structure it did not actually emit and would inflate
the diversity metric.
"""
import pytest

from agentic_raptor.llm_dpo import proposal_dict_valid

GOOD = {"stages": [{"block": "five_transistor_first_stage"},
                   {"block": "cs_gain_stage"}],
        "bias_roles": ["tail_current_source"],
        "compensation": [{"type": "miller_cap"}]}


def test_well_formed_proposal_still_valid():
    ok, reasons = proposal_dict_valid(GOOD)
    assert ok, reasons


@pytest.mark.parametrize("field,bad", [
    ("compensation", ["miller"]),
    ("stages", ["five_transistor_first_stage"]),
    ("feedback_paths", ["vout->vinn"]),
    ("compensation", "miller"),          # bare string, not even a list
    ("stages", [None]),
    ("compensation", [42]),
])
def test_malformed_field_is_invalid_not_a_crash(field, bad):
    ok, reasons = proposal_dict_valid(dict(GOOD, **{field: bad}))
    assert ok is False
    assert any(r.startswith(f"malformed_{field}") for r in reasons), reasons


def test_malformed_compensation_does_not_reach_allowlist_lookup():
    """The original crash was inside the allow-list branch, which only runs
    when nothing else has already failed."""
    ok, reasons = proposal_dict_valid(dict(GOOD, compensation=["miller"]))
    assert ok is False
    assert not any("combo_not_electrically_verified" in r for r in reasons)
