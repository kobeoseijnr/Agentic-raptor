"""Pre-SPICE DPO: hierarchy, leakage guards, ranking, selection, checkpoints, pipeline order."""

from __future__ import annotations

from pathlib import Path
from random import Random

import pytest

from agentic_raptor.core.candidate import CircuitCandidate
from agentic_raptor.core.types import GenerationSource
from agentic_raptor.dpo import (
    FEATURE_DIM,
    DPOConfig,
    DPOLeakageError,
    DPORanker,
    OutcomeRecord,
    OutcomeStore,
    build_candidate_features,
    build_pairs,
    compare,
    select_candidate,
    split_pairs,
)
from agentic_raptor.utils.config import AgenticConfig

_SMOKE = Path(__file__).resolve().parents[1] / "configs" / "experiments" / "smoke_test.yaml"


def _features(candidate, feasibility=0.5, margins=None, vector=None):
    return build_candidate_features(
        candidate,
        sizing_vector=vector or [0.0],
        predicted_margins=margins or {"gain_db": 0.1},
        predicted_feasibility=feasibility,
    )


def _outcome(candidate, *, success=True, passed=True, pvt=None, worst=0.1, fom=0.5,
             calls=3, cffp=None, runtime=1.0, edits=0) -> OutcomeRecord:
    f = _features(candidate)
    f.edit_count = edits
    return OutcomeRecord(
        features=f, dpo_score=None, dpo_rank=None, selection_reason="t",
        spice_success=success, passed_spec=passed,
        constraint_margins={"gain_db": worst}, pvt_pass_rate=pvt, fom=fom,
        runtime_s=runtime, spice_calls_total=calls, calls_to_first_pass=cffp,
    )


@pytest.fixture
def candidate(ota_graph, spec):
    return CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)


def test_leakage_guard_rejects_simulated_candidate(candidate):
    candidate.simulation_result = {"success": True}
    with pytest.raises(DPOLeakageError):
        _features(candidate)


def test_feature_vector_dim(candidate):
    assert len(_features(candidate).feature_vector()) == FEATURE_DIM


def test_preference_hierarchy(candidate):
    c = candidate
    assert compare(_outcome(c, success=True), _outcome(c, success=False))[1] == "a"
    assert compare(_outcome(c, passed=True), _outcome(c, passed=False))[1] == "b"
    assert compare(_outcome(c, pvt=1.0), _outcome(c, pvt=0.5))[1] == "c"
    assert compare(_outcome(c, worst=0.5), _outcome(c, worst=0.1))[1] == "d"
    # e: single-dimension dominance (fom better, all else equal) IS Pareto dominance
    assert compare(_outcome(c, fom=0.9), _outcome(c, fom=0.5))[1] == "e"
    # f: opposing dimensions (better fom vs fewer calls) → no dominance → FoM decides
    assert compare(_outcome(c, fom=0.9, calls=5), _outcome(c, fom=0.5, calls=3))[1] == "f"
    # g: totals equal, calls-to-first-pass differs
    assert compare(_outcome(c, cffp=2), _outcome(c, cffp=8))[1] == "g"
    verdict, rule = compare(_outcome(c, success=True), _outcome(c, success=False))
    assert verdict == -1  # chosen first


def test_ambiguous_pairs_excluded(candidate):
    a, b = _outcome(candidate, worst=0.100), _outcome(candidate, worst=0.101)
    verdict, _ = compare(a, b, tie_margin=0.02)
    assert verdict == 0
    assert build_pairs([a, b]) == []
    both_failed = [_outcome(candidate, success=False), _outcome(candidate, success=False)]
    assert build_pairs(both_failed) == []


def test_pairs_and_group_safe_split(candidate):
    records = [_outcome(candidate, worst=w) for w in (0.5, 0.2, -0.3)]
    pairs = build_pairs(records)
    assert pairs and all(p.confidence > 0 for p in pairs)
    train, val, test = split_pairs(pairs)
    assert len(train) + len(val) + len(test) == len(pairs)


def test_ranker_trains_and_orders(candidate):
    config = DPOConfig(enabled=True, minimum_pairs_before_training=1, epochs=40, seed=3)
    ranker = DPORanker(FEATURE_DIM, config)
    good = _outcome(candidate, worst=0.5, fom=0.9)
    bad = _outcome(candidate, success=False)
    bad.features.predicted_feasibility = 0.0
    bad.features.validation_valid = False
    pairs = build_pairs([good, bad])
    assert pairs
    report = ranker.train_on_pairs(pairs * 4)
    assert "loss" in report
    assert ranker.pair_accuracy(pairs) == 1.0, "trained ranker must order the training pair"


def test_ranking_deterministic_under_seed(candidate):
    pool = [_features(candidate, feasibility=f, vector=[f]) for f in (0.1, 0.5, 0.9)]
    r1 = DPORanker(FEATURE_DIM, DPOConfig(seed=7)).rank(pool)
    r2 = DPORanker(FEATURE_DIM, DPOConfig(seed=7)).rank(pool)
    assert [x[0].pool_candidate_id for x in r1] == [x[0].pool_candidate_id for x in r2]


def test_exploration_candidates_selectable(candidate):
    config = DPOConfig(enabled=True, exploration_fraction=1.0, seed=0)
    ranker = DPORanker(FEATURE_DIM, config)
    pool = [_features(candidate, feasibility=f, vector=[f]) for f in (0.1, 0.9)]
    result = select_candidate(pool, ranker, config, Random(0))
    assert result.dpo_rank != 0 and "exploration" in result.reason


def test_selector_not_dpo_alone(candidate):
    config = DPOConfig(enabled=True, exploration_fraction=0.0)
    ranker = DPORanker(FEATURE_DIM, config)
    pool = [_features(candidate, feasibility=f, vector=[f]) for f in (0.2, 0.8)]
    result = select_candidate(pool, ranker, config, Random(1))
    assert "feasibility" in result.reason  # combined score, never DPO-only


def test_checkpoint_roundtrip(tmp_path, candidate):
    config = DPOConfig(seed=1)
    ranker = DPORanker(FEATURE_DIM, config)
    f = _features(candidate)
    before = ranker.score(f)
    path = ranker.save(tmp_path / "dpo.pt")
    other = DPORanker(FEATURE_DIM, DPOConfig(seed=99))
    other.load(path)
    assert other.score(f) == pytest.approx(before)


def test_outcome_store_updates_dataset(tmp_path, candidate):
    store = OutcomeStore(tmp_path / "outcomes.jsonl")
    store.add(_outcome(candidate))
    restored = OutcomeStore(tmp_path / "outcomes.jsonl")
    assert len(restored) == 1


@pytest.fixture(scope="module")
def dpo_episode(tmp_path_factory):
    config = AgenticConfig.from_yaml(_SMOKE)
    out = tmp_path_factory.mktemp("dpo_ep")
    config.output_dir = str(out)
    config.logging.decision_log = str(out / "d.jsonl")
    config.dpo.enabled = True
    config.dpo.update_interval_episodes = 1
    config.dpo.minimum_pairs_before_training = 1
    config._base_dir = str(_SMOKE.parent)  # type: ignore[attr-defined]
    from agentic_raptor.coordinator.coordinator import AgenticCoordinator

    coordinator = AgenticCoordinator(config)
    return coordinator, coordinator.run_episode()


def test_dpo_called_before_spice(dpo_episode):
    _coordinator, result = dpo_episode
    decisions = [d["decision"] for d in result.decisions]
    assert "DPO_SELECTION" in decisions
    assert decisions.index("DPO_SELECTION") < decisions.index("SPICE_RESULT"), (
        "DPO must rank the pool before SPICE simulation"
    )
    assert result.update_report["dpo"]["outcomes"] >= 1


def test_disabled_dpo_reproduces_original_pipeline(tmp_path):
    config = AgenticConfig.from_yaml(_SMOKE)
    config.output_dir = str(tmp_path)
    config.logging.decision_log = str(tmp_path / "d.jsonl")
    config._base_dir = str(_SMOKE.parent)  # type: ignore[attr-defined]
    assert config.dpo.enabled is False
    from agentic_raptor.coordinator.coordinator import AgenticCoordinator

    result = AgenticCoordinator(config).run_episode()
    assert all(d["decision"] != "DPO_SELECTION" for d in result.decisions)
    assert "dpo" not in result.update_report
    assert result.final_state == "TERMINATE"
