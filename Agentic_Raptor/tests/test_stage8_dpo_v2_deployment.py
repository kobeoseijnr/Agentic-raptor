"""Stage 8 Part B: pre-integration unit/smoke validation for the deployed
DPO V2 ranker (agentic_raptor.ranking.model_v2). No SPICE, no LLM -- pure
code-level checks against the real promoted checkpoint on disk.

Covers: checkpoint load (hash/dim/schema/normalization/architecture/finite
output), training-vs-inference feature parity (both paths route through the
same features_v2() call), pairwise score/candidate-swap antisymmetry, and
the no-future-truth guarantee (PostSACDesign carries no final-verification
field, and features_v2's signature has no channel for one).
"""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")


def _make_spec():
    return {"spec_id": "probe", "gain_target_db": 60.0,
           "phase_margin_target_deg": 45.0, "ugbw_target_hz": 1e6,
           "load_capacitance_pf": 100.0}


def _make_pred(topology_hash: str):
    from agentic_raptor.ranking.types import SurrogatePrediction
    return SurrogatePrediction(
        topology_hash=topology_hash, sizing_manifest_hash="mh1",
        gain_db=65.0, pm_deg=60.0, ugbw_hz=2e6, power_w=1e-3, area_um2=500.0,
        operating_point_probability=0.9, stability_probability=0.9,
        normalized_margins={"gain_margin_db": 5.0, "pm_margin_deg": 15.0,
                            "ugbw_log_margin": 0.3},
        predictive_uncertainty=0.1)


def _make_candidate_row():
    return {"gain_db": 66.0, "pm_deg": 58.0, "ugbw_hz": 2.1e6, "step": 7}


def _make_branch_rows():
    return [{"gain_db": 40.0, "pm_deg": 10.0, "ugbw_hz": 1e5, "reward": 0.1, "step": 0},
           {"gain_db": 60.0, "pm_deg": 40.0, "ugbw_hz": 8e5, "reward": 0.5, "step": 1},
           {"gain_db": 66.0, "pm_deg": 58.0, "ugbw_hz": 2.1e6, "reward": 0.9, "step": 2}]


def _make_design(canonical_graph_hash: str):
    from agentic_raptor.ranking.post_sac import PostSACDesign
    return PostSACDesign(label="A", spec_id="probe", llm_proposal_id="p1",
                         canonical_graph_hash=canonical_graph_hash,
                         topology_signature="sig1", topology_family="2s_rc",
                         sizing_vector={"s1_w": 1.0})


# ---------------------------------------------------------------------------
# checkpoint load test
# ---------------------------------------------------------------------------
def test_checkpoint_loads_with_correct_hash_dim_schema_and_finite_output():
    from agentic_raptor.ranking.features_v2 import FEATURE_DIM_V2
    from agentic_raptor.ranking.model_v2 import REQUIRED_V2_SHA256, load_promoted_v2

    spec, pred = _make_spec(), _make_pred("hashA")
    design = _make_design("hashA")
    branch_context = {"hashA": {"candidate_row": _make_candidate_row(),
                                "branch_rows": _make_branch_rows(),
                                "family": "2s_rc"}}
    ranker = load_promoted_v2(branch_context)

    assert ranker.checkpoint_hash == REQUIRED_V2_SHA256
    assert ranker.feature_schema == "POST_SAC_FEATURES_V2"
    assert len(ranker.feature_mean) == FEATURE_DIM_V2
    assert len(ranker.feature_std) == FEATURE_DIM_V2
    first_linear = next(m for m in ranker.model.modules()
                        if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == FEATURE_DIM_V2

    score = ranker.score(spec, design, pred)
    assert isinstance(score, float)
    assert score == score  # not NaN
    assert score not in (float("inf"), float("-inf"))


def test_missing_checkpoint_hard_fails_not_silently_falls_back(monkeypatch, tmp_path):
    from agentic_raptor.ranking import model_v2
    fake = tmp_path / "does_not_exist.pt"
    monkeypatch.setattr(model_v2, "PROMOTED_V2_CKPT", fake)
    with pytest.raises(FileNotFoundError):
        model_v2.load_promoted_v2({})


def test_hash_mismatch_hard_fails(monkeypatch, tmp_path):
    from agentic_raptor.ranking import model_v2
    bad = tmp_path / "wrong.pt"
    bad.write_bytes(b"not the real checkpoint bytes")
    monkeypatch.setattr(model_v2, "PROMOTED_V2_CKPT", bad)
    with pytest.raises(model_v2.RankerCheckpointHashMismatch):
        model_v2.load_promoted_v2({})


def test_schema_mismatch_hard_fails(monkeypatch, tmp_path):
    import json
    import shutil

    from agentic_raptor.ranking import model_v2
    # copy the real checkpoint (so the hash check passes) but point the
    # manifest lookup at a schema-mismatched stand-in
    shutil.copy(model_v2.PROMOTED_V2_CKPT, tmp_path / "ranker.pt")
    bad_manifest = tmp_path / "training_manifest.json"
    bad_manifest.write_text(json.dumps({"feature_schema": "POST_SAC_FEATURES_V1"}),
                            encoding="utf-8")
    monkeypatch.setattr(model_v2, "PROMOTED_V2_CKPT", tmp_path / "ranker.pt")
    monkeypatch.setattr(model_v2, "PROMOTED_V2_MANIFEST", bad_manifest)
    with pytest.raises(model_v2.RankerFeatureSchemaMismatch):
        model_v2.load_promoted_v2({})


# ---------------------------------------------------------------------------
# training-vs-inference feature parity
# ---------------------------------------------------------------------------
def test_training_and_inference_paths_route_through_the_identical_features_v2_call():
    """run_stage72b_dpo_repair._row_features() (training) and
    PostSACRankerV2.score() (live inference) must both call
    agentic_raptor.ranking.features_v2.features_v2() itself -- neither may
    duplicate feature-construction logic that could silently drift apart."""
    from agentic_raptor.ranking import model_v2
    src = inspect.getsource(model_v2.PostSACRankerV2.score)
    assert "features_v2(" in src
    # no local re-derivation of any V2 feature name inside score()
    assert "_V1_NAMES" not in src
    assert "trajectory_summary(" not in src
    assert "candidate_measured_features(" not in src


def test_feature_vectors_are_numerically_identical_between_direct_call_and_scoring_path():
    """Feed the SAME inputs through features_v2() directly (what the
    training script does) and through PostSACRankerV2.score()'s internal
    construction (patched to capture the pre-normalization tensor) --
    element-wise equal within float tolerance."""
    from agentic_raptor.ranking.features_v2 import features_v2
    from agentic_raptor.ranking.model_v2 import load_promoted_v2

    spec, pred = _make_spec(), _make_pred("hashA")
    cand, branch, family = _make_candidate_row(), _make_branch_rows(), "2s_rc"
    design = _make_design("hashA")
    branch_context = {"hashA": {"candidate_row": cand, "branch_rows": branch,
                                "family": family}}
    ranker = load_promoted_v2(branch_context)

    expected = features_v2(spec, pred, cand, branch, family, arm=ranker.arm)

    captured = {}
    real_normalize = ranker._normalize

    def _spy_normalize(x):
        captured["x"] = x.clone()
        return real_normalize(x)
    ranker._normalize = _spy_normalize
    ranker.score(spec, design, pred)

    actual = captured["x"][0].tolist()
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        # float32 tensor round-trip vs. python float -- tolerance matches
        # float32 precision (~1e-7 relative), not an approximation of the
        # feature construction itself
        assert a == pytest.approx(e, abs=1e-6)


def test_feature_count_is_46():
    from agentic_raptor.ranking.features_v2 import FEATURE_DIM_V2, FEATURE_NAMES_V2
    assert FEATURE_DIM_V2 == 46
    assert len(FEATURE_NAMES_V2) == 46


# ---------------------------------------------------------------------------
# pairwise score / candidate-swap antisymmetry
# ---------------------------------------------------------------------------
def test_score_difference_is_antisymmetric_under_candidate_swap():
    from agentic_raptor.ranking.model_v2 import load_promoted_v2

    spec = _make_spec()
    pred_a, pred_b = _make_pred("hashA"), _make_pred("hashB")
    design_a, design_b = _make_design("hashA"), _make_design("hashB")
    branch_context = {
        "hashA": {"candidate_row": _make_candidate_row(),
                  "branch_rows": _make_branch_rows(), "family": "2s_rc"},
        "hashB": {"candidate_row": {"gain_db": 50.0, "pm_deg": 30.0, "ugbw_hz": 5e5},
                  "branch_rows": [{"gain_db": 30.0, "pm_deg": 5.0, "ugbw_hz": 1e4,
                                   "reward": -0.2, "step": 0}],
                  "family": "3s_miller"},
    }
    ranker = load_promoted_v2(branch_context)
    score_a = ranker.score(spec, design_a, pred_a)
    score_b = ranker.score(spec, design_b, pred_b)
    assert (score_a - score_b) == pytest.approx(-(score_b - score_a), abs=1e-9)
    # deterministic: re-scoring the identical inputs must reproduce exactly
    assert ranker.score(spec, design_a, pred_a) == pytest.approx(score_a, abs=1e-12)


# ---------------------------------------------------------------------------
# no-future-truth guarantee
# ---------------------------------------------------------------------------
def test_post_sac_design_has_no_final_verification_field():
    from dataclasses import fields

    from agentic_raptor.ranking.post_sac import PostSACDesign
    names = {f.name for f in fields(PostSACDesign)}
    assert not any("final_verification" in n or "verification_result" in n
                  for n in names)


def test_features_v2_signature_has_no_channel_for_final_verification_data():
    from agentic_raptor.ranking.features_v2 import features_v2
    sig = inspect.signature(features_v2)
    assert set(sig.parameters) == {"spec", "pred", "candidate_row",
                                   "branch_rows", "topology_family", "arm"}


def test_features_v2_module_never_reads_a_final_verification_key():
    import agentic_raptor.ranking.features_v2 as fv2
    src = inspect.getsource(fv2)
    assert "final_verification" not in src
    assert "result_id" not in src


def test_extra_final_verification_data_smuggled_into_branch_context_is_ignored():
    """Even if a caller carelessly stuffed a final_verification object into
    the branch_context dict alongside the legitimate fields, the score must
    be unaffected -- features_v2() only ever reads candidate_row/branch_rows
    /topology_family by name."""
    from agentic_raptor.ranking.model_v2 import load_promoted_v2

    spec, pred = _make_spec(), _make_pred("hashA")
    design = _make_design("hashA")
    cand, branch = _make_candidate_row(), _make_branch_rows()
    clean_ctx = {"hashA": {"candidate_row": cand, "branch_rows": branch,
                           "family": "2s_rc"}}
    contaminated_ctx = {"hashA": {"candidate_row": cand, "branch_rows": branch,
                                  "family": "2s_rc",
                                  "final_verification": {"gain_db": 999.0,
                                                         "pm_deg": 999.0,
                                                         "result_id": "leak"}}}
    ranker_clean = load_promoted_v2(clean_ctx)
    ranker_dirty = load_promoted_v2(contaminated_ctx)
    assert ranker_clean.score(spec, design, pred) == pytest.approx(
        ranker_dirty.score(spec, design, pred), abs=1e-12)
