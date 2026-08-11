"""Section 7 (Stage 5 Campaign 01B): audit_replay()'s revised degeneracy
policy -- "same terminal topology alone" must no longer be automatic
corruption; genuine corruption (bad pi, NaN/Inf, missing provenance,
duplicate RNG seeds, impossible SELECT sequences, zero legal-action
diversity) and true pathological collapse (single trajectory/terminal/
action across MULTIPLE distinct specs despite exploration being enabled)
remain hard rejects. Uses plain synthetic replay rows -- audit_replay()
operates purely on row dicts, no LLM/SPICE/torch needed, so these run fast.
"""
from __future__ import annotations

import run_alphazero_campaign as camp


def _row(*, spec_hash="s1", seed=0, step=0, action="a_edit_ADD_VERIFIED_STAGE",
        z=0.5, terminal_hash="th1", pi=None, rng_seed=1, call_id="call1",
        edit_depth=None):
    return {
        "spec_hash": spec_hash, "seed": seed, "step": step,
        "selected_action_id": action, "z": z,
        "terminal_topology_hash": terminal_hash,
        "pi": pi if pi is not None else {action: 1.0},
        "episode_rng_seed": rng_seed,
        "terminal_authoritative_call_id": call_id,
        "legal_action_ids": [action, "a_term"],
        "edit_depth": edit_depth if edit_depth is not None else step,
    }


def test_same_terminal_topology_alone_is_not_a_hard_reject():
    """The exact CAMPAIGN_01_ATTEMPT_1 shape -- one terminal topology
    dominating -- but across only ONE spec/episode is not by itself
    corruption; low diversity is a warning now, not a hard reject."""
    rows = [_row(spec_hash="s1", seed=0, step=i, terminal_hash="th1", rng_seed=1)
           for i in range(3)]
    result = camp.audit_replay(rows)
    assert result["degenerate"] is False
    assert result["hard_reject_reasons"] == []


def test_low_diversity_across_multiple_specs_is_a_warning_not_hard_reject():
    """Two DIFFERENT specs land on the same terminal/action (plausible for
    an early, lightly-trained generation) but trajectories/terminals are
    NOT perfectly single-valued in a way that indicates corruption --
    still just a warning as long as it's not a full pathological collapse."""
    rows = [
        _row(spec_hash="s1", seed=0, step=0, terminal_hash="th1",
            action="a_edit_ADD_VERIFIED_STAGE", z=0.3, rng_seed=1),
        _row(spec_hash="s1", seed=0, step=1, terminal_hash="th1",
            action="a_term", z=0.3, rng_seed=1),
        _row(spec_hash="s2", seed=0, step=0, terminal_hash="th1",
            action="a_edit_ADD_VERIFIED_STAGE", z=0.7, rng_seed=2),
        _row(spec_hash="s2", seed=0, step=1, terminal_hash="th1",
            action="a_term", z=0.7, rng_seed=2),
    ]
    result = camp.audit_replay(rows, exploration_enabled=True)
    # single trajectory across 2 specs AND single terminal AND only 2
    # distinct actions used (a_edit_*, a_term) -- not single-action, so
    # not the pathological triple-collapse; must be a warning, not reject.
    assert result["degenerate"] is False
    assert any("dominates" in w for w in result["warnings"])


def test_pathological_collapse_across_specs_with_exploration_is_hard_reject():
    """The TRUE CAMPAIGN_01_ATTEMPT_1 pattern reproduced exactly: a single
    trajectory, single terminal, AND single action id used across multiple
    distinct specs, even though exploration was (notionally) enabled --
    this remains a hard reject."""
    rows = [
        _row(spec_hash="s1", seed=0, step=0, terminal_hash="th1",
            action="a_edit_ADD_VERIFIED_STAGE", z=0.3, rng_seed=1),
        _row(spec_hash="s2", seed=0, step=0, terminal_hash="th1",
            action="a_edit_ADD_VERIFIED_STAGE", z=0.7, rng_seed=2),
        _row(spec_hash="s3", seed=0, step=0, terminal_hash="th1",
            action="a_edit_ADD_VERIFIED_STAGE", z=-0.2, rng_seed=3),
    ]
    result = camp.audit_replay(rows, exploration_enabled=True)
    assert result["degenerate"] is True
    assert any("pathological collapse" in r for r in result["hard_reject_reasons"])


def test_corrupted_pi_is_a_hard_reject():
    rows = [_row(pi={"a_edit_ADD_VERIFIED_STAGE": 0.3})]  # sums to 0.3, not 1.0
    result = camp.audit_replay(rows)
    assert result["degenerate"] is True
    assert any("corrupted" in r for r in result["hard_reject_reasons"])


def test_missing_provenance_is_a_hard_reject():
    rows = [_row(call_id=None)]
    result = camp.audit_replay(rows)
    assert result["degenerate"] is True
    assert any("provenance" in r for r in result["hard_reject_reasons"])


def test_duplicate_rng_seed_across_distinct_episodes_is_a_hard_reject():
    """Section 3/7: two DIFFERENT (spec_hash, seed) episodes sharing the
    same episode_rng_seed indicates the RNG-derivation bug reappeared."""
    rows = [
        _row(spec_hash="s1", seed=0, step=0, rng_seed=42),
        _row(spec_hash="s2", seed=0, step=0, rng_seed=42),
    ]
    result = camp.audit_replay(rows)
    assert result["degenerate"] is True
    assert any("episode_rng_seed" in r for r in result["hard_reject_reasons"])


def test_impossible_select_sequence_is_a_hard_reject():
    """Section 1/7: more than one SELECT action in a single trajectory, or
    a SELECT occurring after the first action, is structurally impossible
    under the fixed action space and is treated as corrupted data if it
    somehow appears in replay."""
    rows = [
        _row(spec_hash="s1", seed=0, step=0, action="a_sel_p00", rng_seed=1,
            pi={"a_sel_p00": 1.0}),
        _row(spec_hash="s1", seed=0, step=1, action="a_sel_p01", rng_seed=1,
            pi={"a_sel_p01": 1.0}),
    ]
    result = camp.audit_replay(rows)
    assert result["degenerate"] is True
    assert any("impossible action sequence" in r or "after seed commitment" in r
              for r in result["hard_reject_reasons"])


def test_all_z_identical_across_multiple_specs_is_a_hard_reject():
    """Real SPICE sizing should vary by spec even when the same
    structural recipe is applied (CAMPAIGN_01_ATTEMPT_1's own real data
    proved this) -- identical z across genuinely different specs is more
    consistent with a stale/cached measurement bug than legitimate data."""
    rows = [
        _row(spec_hash="s1", seed=0, z=0.42, rng_seed=1),
        _row(spec_hash="s2", seed=0, z=0.42, rng_seed=2),
        _row(spec_hash="s3", seed=0, z=0.42, rng_seed=3),
    ]
    result = camp.audit_replay(rows)
    assert result["degenerate"] is True
    assert any("all z identical" in r for r in result["hard_reject_reasons"])


def test_all_z_identical_for_a_single_spec_is_not_flagged():
    """A single spec/episode's own rows all sharing z (backfilled from one
    terminal evaluation, by design -- see build_replay_rows) is normal,
    not corruption."""
    rows = [_row(spec_hash="s1", seed=0, step=i, z=0.5, rng_seed=1) for i in range(3)]
    result = camp.audit_replay(rows)
    assert result["degenerate"] is False


def test_no_rows_is_a_hard_reject():
    result = camp.audit_replay([])
    assert result["degenerate"] is True
    assert result["hard_reject_reasons"] == ["no replay rows collected"]
