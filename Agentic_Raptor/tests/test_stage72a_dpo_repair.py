"""Stage 7.2A: model-capacity experiment + weighting mechanics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import run_stage72a_dpo_repair as repair

REPORT_PATH = Path("artifacts/publication_v3/stage7_2a_dpo_repair/STAGE7_2A_REPORT.json")


# ---------------------------------------------------------------------------
# Section 23: three model capacities all build and run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hidden,expected_params", [(None, 12), (8, 105), (32, 417)])
def test_build_model_produces_expected_param_count(hidden, expected_params):
    net = repair.build_model(hidden)
    n = sum(p.numel() for p in net.parameters())
    assert n == expected_params
    x = torch.randn(3, 11)
    out = net(x)
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()


def test_linear_model_has_no_hidden_layer():
    net = repair.build_model(None)
    assert len(list(net.children())) == 1


# ---------------------------------------------------------------------------
# Section 20: pairwise antisymmetry for each capacity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hidden", [None, 8, 32])
def test_pairwise_score_antisymmetric(hidden):
    torch.manual_seed(0)
    net = repair.build_model(hidden)
    fa = torch.randn(1, 11)
    fb = torch.randn(1, 11)
    with torch.no_grad():
        diff_ab = float(net(fa) - net(fb))
        diff_ba = float(net(fb) - net(fa))
    assert diff_ab == pytest.approx(-diff_ba, abs=1e-5)


# ---------------------------------------------------------------------------
# Section 19: per-run weighting -- total weight per run stays ~constant
# ---------------------------------------------------------------------------
def _fake_row(run_id, ranker_authority=True):
    return {"originating_run_id": run_id, "ranker_authority": ranker_authority}


def test_compute_pair_weights_normalizes_by_run_size():
    rows = ([_fake_row("run_a")] * 1 + [_fake_row("run_b")] * 10)
    weights = repair.compute_pair_weights(rows)
    total_a = sum(w for r, w in zip(rows, weights) if r["originating_run_id"] == "run_a")
    total_b = sum(w for r, w in zip(rows, weights) if r["originating_run_id"] == "run_b")
    assert total_a == pytest.approx(repair.TARGET_WEIGHT_PER_RUN, abs=1e-6)
    assert total_b == pytest.approx(repair.TARGET_WEIGHT_PER_RUN, abs=1e-6)


def test_compute_pair_weights_applies_safety_decided_discount():
    rows = [_fake_row("run_a", ranker_authority=True),
           _fake_row("run_b", ranker_authority=False)]
    weights = repair.compute_pair_weights(rows)
    # both runs contribute exactly 1 pair, so run-normalization is 1.0 for
    # each -- the remaining difference must be exactly the safety-decided
    # discount factor
    assert weights[0] == pytest.approx(repair.TARGET_WEIGHT_PER_RUN)
    assert weights[1] == pytest.approx(repair.TARGET_WEIGHT_PER_RUN * repair.SAFETY_DECIDED_WEIGHT)


# ---------------------------------------------------------------------------
# Section 17/18: spec-disjoint split, runs never cross train/dev
# ---------------------------------------------------------------------------
def test_split_train_dev_keeps_every_spec_and_its_runs_on_one_side():
    pool = [{"spec_hash": f"s{i % 5}", "originating_run_id": f"s{i % 5}:{i}"}
           for i in range(40)]
    train, dev, dev_specs = repair.split_train_dev(pool, dev_fraction=0.4)
    train_specs = {r["spec_hash"] for r in train}
    assert train_specs.isdisjoint(dev_specs)
    train_runs = {r["originating_run_id"] for r in train}
    dev_runs = {r["originating_run_id"] for r in dev}
    assert train_runs.isdisjoint(dev_runs)


def test_split_is_deterministic():
    pool = [{"spec_hash": f"spec_{i}", "originating_run_id": f"r{i}"} for i in range(30)]
    _, _, dev1 = repair.split_train_dev(pool)
    _, _, dev2 = repair.split_train_dev(pool)
    assert dev1 == dev2


# ---------------------------------------------------------------------------
# Section 25: deterministic checkpoint reload
# ---------------------------------------------------------------------------
def test_checkpoint_reload_reproduces_identical_scores(tmp_path):
    torch.manual_seed(0)
    net = repair.build_model(8)
    x = torch.randn(5, 11)
    with torch.no_grad():
        before = net(x).clone()
    ckpt = tmp_path / "ranker.pt"
    torch.save(net.state_dict(), ckpt)
    reloaded = repair.build_model(8)
    reloaded.load_state_dict(torch.load(ckpt, weights_only=True))
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(x)
    assert torch.allclose(before, after)


# ---------------------------------------------------------------------------
# Section 33: unified pool construction (structural, not the real 2300-pair run)
# ---------------------------------------------------------------------------
def test_mined_to_unified_preserves_sizing_vector():
    p = {"pair_id": ("s1", "a:1", "b:1"), "originating_run_id": "s1:0:0",
        "spec_hash": "s1", "spec": {"spec_id": "x"}, "spec_id": "x",
        "candidate_id_a": "tA:h1", "candidate_id_b": "tB:h1",
        "topology_hash_a": "tA", "topology_hash_b": "tB",
        "knobs_a": {"s1_w": 1.5}, "knobs_b": {"s1_w": 2.0},
        "prediction_a": {"topology_hash": "tA", "sizing_manifest_hash": "m"},
        "prediction_b": {"topology_hash": "tB", "sizing_manifest_hash": "m"},
        "outcome_a": {"exact_spec_pass": False, "normalized_distance_to_feasibility": 0.1},
        "outcome_b": {"exact_spec_pass": False, "normalized_distance_to_feasibility": 0.2},
        "authoritative_winner": "A"}
    u = repair.mined_to_unified(p)
    assert u["design_a"]["sizing_vector"] == {"s1_w": 1.5}
    assert u["design_b"]["sizing_vector"] == {"s1_w": 2.0}
    assert u["source"] == "mined"


# ---------------------------------------------------------------------------
# Report-derived (skipped if the real run hasn't happened)
# ---------------------------------------------------------------------------
def test_report_disagreement_and_catastrophic_accounted_per_model():
    if not REPORT_PATH.is_file():
        pytest.skip("Stage 7.2A training not yet run in this checkout")
    d = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    for name, m in d["model_results"].items():
        de = m["dev_eval"]
        assert de["disagreement_wins_vs_neutral"] >= 0
        assert de["disagreement_losses_vs_neutral"] >= 0
        assert de["catastrophic_error_count"] >= 0
        assert (de["net_decision_gain_vs_neutral"]
               == de["disagreement_wins_vs_neutral"] - de["disagreement_losses_vs_neutral"])
