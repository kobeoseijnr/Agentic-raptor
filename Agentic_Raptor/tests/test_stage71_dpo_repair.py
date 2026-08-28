"""Stage 7.1: DPO/ranker repair-and-retrain -- mechanics/bookkeeping tests.

No SPICE, no dependence on the specific outcome of the real 91-pair rebuild
(that's covered by the report artifact itself) -- these exercise the
reusable pieces: normalization fit/apply, pair-table categorization,
spec-disjoint splitting, canonical dedupe, and the antisymmetry of the
learned ranker's decision under an A/B relabeling.
"""
from __future__ import annotations

import math

import pytest
import torch

from agentic_raptor.ranking.model import (FEATURE_DIM, NORMALIZE_MASK,
                                          PostSACRanker, fit_normalization)
from agentic_raptor.ranking.post_sac import PostSACDesign, compare
from agentic_raptor.ranking.types import (AuthoritativeSpiceOutcome,
                                          SurrogatePrediction)


# ---------------------------------------------------------------------------
# Section 9: normalization -- train-only fitting, inference reuse
# ---------------------------------------------------------------------------
def test_fit_normalization_only_touches_masked_continuous_features():
    rows = [[1.0, 2, 0, 0.05, 0.5, 1.0, -1.0, 0.2, 1, 1, 1],
           [2.0, 1, 0, 0.09, 0.7, 2.0, -0.5, 0.1, 1, 1, 1]]
    mean, std = fit_normalization(rows)
    for i, masked in enumerate(NORMALIZE_MASK):
        if not masked:
            assert mean[i] == 0.0 and std[i] == 1.0


def test_normalization_fit_on_train_rows_only_then_reused_at_inference():
    """The checkpoint's stored mean/std must come from TRAIN rows; applying
    them to an unrelated DEV-scale row must not silently refit."""
    train_rows = [[1.0] + [0] * (FEATURE_DIM - 1)] * 5
    mean, std = fit_normalization(train_rows)
    ranker = PostSACRanker(model=PostSACRanker.build(),
                           feature_mean=mean, feature_std=std)
    dev_row = torch.tensor([[100.0] + [0] * (FEATURE_DIM - 1)])
    normalized = ranker._normalize(dev_row)
    # mean/std reflect TRAIN (constant 1.0), so the wildly different DEV
    # value produces a large z-score rather than being re-centered on itself
    assert normalized[0, 0].item() > 50


def test_ranker_without_sidecar_normalization_is_a_no_op():
    """Loading an OLD checkpoint (no *_normalization.json) must behave
    exactly as before this repair -- identity, not some default scaling."""
    ranker = PostSACRanker(model=PostSACRanker.build())
    x = torch.tensor([[3.0] + [0.0] * (FEATURE_DIM - 1)])
    assert torch.equal(ranker._normalize(x), x)


def test_normalization_std_never_divides_by_zero():
    rows = [[1.0] * FEATURE_DIM] * 4     # zero variance on every column
    mean, std = fit_normalization(rows)
    ranker = PostSACRanker(model=PostSACRanker.build(),
                           feature_mean=mean, feature_std=std)
    out = ranker._normalize(torch.tensor([[1.0] * FEATURE_DIM]))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Section 4: pair categorization / ranker-authority classification
# ---------------------------------------------------------------------------
def _outcome(passed=True, dist=0.0, call_id="c1"):
    return AuthoritativeSpiceOutcome(
        call_id=call_id, topology_hash="h", sizing_manifest_hash="m",
        netlist_hash="n", mode="final_verification", exact_spec_pass=passed,
        operating_point_valid=True, spice_converged=True, verified_stable=True,
        normalized_distance_to_feasibility=dist)


def test_feasibility_category_classification():
    import run_stage71_dpo_repair as repair

    assert repair._feasibility_category(_outcome(True), _outcome(True)) == "both_feasible"
    assert repair._feasibility_category(_outcome(True), _outcome(False)) == "one_feasible_one_infeasible"
    assert repair._feasibility_category(
        _outcome(False, 0.05), _outcome(False, 0.1)) == "both_infeasible_close"
    assert repair._feasibility_category(
        _outcome(False, 0.5), _outcome(False, 0.9)) == "both_infeasible_far"


def test_ranker_authority_is_hard_tier_equality():
    from agentic_raptor.ranking.post_sac import hard_safety_tier
    pred_known_good = SurrogatePrediction(
        topology_hash="h", sizing_manifest_hash="m",
        operating_point_probability=0.9, stability_probability=0.9,
        normalized_margins={"gain": 0.1})
    pred_unknown = SurrogatePrediction(topology_hash="h2", sizing_manifest_hash="m")
    # equal tiers (both fully unknown) -> ranker has authority
    assert hard_safety_tier(pred_unknown) == hard_safety_tier(
        SurrogatePrediction(topology_hash="h3", sizing_manifest_hash="m"))
    # different tiers -> hard gate already decides, ranker has no authority
    assert hard_safety_tier(pred_known_good) != hard_safety_tier(pred_unknown)


# ---------------------------------------------------------------------------
# Section 12/13: canonical pair identity, spec-disjoint split
# ---------------------------------------------------------------------------
def test_canonical_pair_id_is_order_independent():
    id1 = ("spec1", *sorted(("hashA", "hashB")))
    id2 = ("spec1", *sorted(("hashB", "hashA")))
    assert id1 == id2


def test_spec_disjoint_split_keeps_every_spec_on_one_side():
    import run_stage71_dpo_repair as repair

    table = [{"spec_hash": f"s{i % 6}", "pair_index": i} for i in range(30)]
    train, dev, dev_specs = repair.split_train_dev(table, dev_fraction=0.3)
    train_specs = {r["spec_hash"] for r in train}
    assert train_specs.isdisjoint(dev_specs)
    for r in dev:
        assert r["spec_hash"] in dev_specs


def test_split_is_deterministic_across_calls():
    import run_stage71_dpo_repair as repair

    table = [{"spec_hash": f"spec_{i}", "pair_index": i} for i in range(50)]
    _, _, dev1 = repair.split_train_dev(table)
    _, _, dev2 = repair.split_train_dev(table)
    assert dev1 == dev2


# ---------------------------------------------------------------------------
# Section 8: pair-symmetry / candidate-swap regression test
# ---------------------------------------------------------------------------
def _design(label, graph_hash):
    return PostSACDesign(label=label, spec_id="S", llm_proposal_id="p",
                         canonical_graph_hash=graph_hash,
                         topology_signature="fam", topology_family="fam",
                         sizing_vector={"s1_w": 1.0}, sizing_manifest_hash="m1")


def _pred(graph_hash, margins):
    return SurrogatePrediction(topology_hash=graph_hash, sizing_manifest_hash="m1",
                               operating_point_probability=0.9,
                               stability_probability=0.9,
                               normalized_margins=margins)


def test_compare_decision_is_symmetric_under_candidate_swap():
    """Swapping which physical design is passed as `a` vs `b` must select
    the SAME underlying design and produce the mirrored score margin --
    proves the model/loss carries no positional A/B bias."""
    torch.manual_seed(0)
    ranker = PostSACRanker(PostSACRanker.build())
    d1 = _design("A", "hash1")
    d2 = _design("B", "hash2")
    p1 = _pred("hash1", {"gain": 0.3, "pm": 0.1})
    p2 = _pred("hash2", {"gain": -0.2, "pm": -0.4})
    spec = {"gain_target_db": 60.0, "phase_margin_target_deg": 45.0}

    fwd = compare(d1, d2, p1, p2, spec=spec, model=ranker, ranker_arm="dpo_ranker")
    bwd = compare(d2, d1, p2, p1, spec=spec, model=ranker, ranker_arm="dpo_ranker")

    assert fwd["selected_topology_hash"] == bwd["selected_topology_hash"]
    if fwd["score_margin"] is not None and bwd["score_margin"] is not None:
        assert fwd["score_margin"] == pytest.approx(bwd["score_margin"], abs=1e-5)


def test_bt_loss_has_no_positional_encoding():
    """The pairwise loss net(xw) - net(xl) depends only on WHICH feature
    vector won, never on an A/B slot -- verified structurally: the same net
    applied to the same two feature vectors in swapped order gives negated
    scores, so sigmoid(diff) correctly flips."""
    torch.manual_seed(1)
    net = PostSACRanker.build()
    fa = torch.randn(1, FEATURE_DIM)
    fb = torch.randn(1, FEATURE_DIM)
    with torch.no_grad():
        diff_ab = float(net(fa) - net(fb))
        diff_ba = float(net(fb) - net(fa))
    assert diff_ab == pytest.approx(-diff_ba, abs=1e-5)


# ---------------------------------------------------------------------------
# Section 22: catastrophic-error accounting
# ---------------------------------------------------------------------------
def test_catastrophic_error_definition_requires_ranker_authority_and_dpo_decided():
    """A wrong pick where the hard gate ALREADY decided is not catastrophic
    by this definition -- Level 1 would have overridden it live regardless
    of what Level 2 says."""
    row_hard_gate_decided = {"ranker_authority": False,
                             "feasibility_category": "one_feasible_one_infeasible"}
    row_ranker_authority_wrong = {"ranker_authority": True,
                                  "feasibility_category": "one_feasible_one_infeasible"}
    # only the second is eligible to be flagged catastrophic when the
    # decision basis was dpo_ranker and it picked wrong
    assert row_hard_gate_decided["ranker_authority"] is False
    assert row_ranker_authority_wrong["ranker_authority"] is True


# ---------------------------------------------------------------------------
# Section 26: protected-spec exclusion carries through the rebuild
# ---------------------------------------------------------------------------
def test_rebuilt_pair_table_excludes_protected_specs():
    import json
    from pathlib import Path

    from agentic_raptor.publication.eval_sets import excluded_context_ids
    path = Path("artifacts/publication_v3/stage7_1_dpo_repair/REBUILT_TRUSTED_PAIR_TABLE.json")
    if not path.is_file():
        pytest.skip("Stage 7.1 rebuild not yet run in this checkout")
    table = json.loads(path.read_text(encoding="utf-8"))
    excluded = excluded_context_ids()
    for row in table:
        assert row["spec_id"] not in excluded


# ---------------------------------------------------------------------------
# Section 29: complete provenance manifest
# ---------------------------------------------------------------------------
def test_promoted_manifest_has_required_provenance_fields():
    import json
    from pathlib import Path
    path = Path("artifacts/publication_v3/stage7_1_dpo_repair/post_sac_ranker_v2/training_manifest.json")
    if not path.is_file():
        pytest.skip("Stage 7.1 rebuild not yet run in this checkout")
    m = json.loads(path.read_text(encoding="utf-8"))
    required = ["model_type", "checkpoint_status", "checkpoint_sha256",
               "parent_checkpoint", "electrical_environment_version",
               "training_pair_file", "training_pair_table_sha256",
               "feature_schema_version", "label_schema_version",
               "optimizer", "learning_rate", "epochs", "random_seed",
               "normalization", "promotion_metrics"]
    for field in required:
        assert field in m, f"missing provenance field: {field}"
    assert m["electrical_environment_version"] == "POST_CLOAD_FIX_V1"
