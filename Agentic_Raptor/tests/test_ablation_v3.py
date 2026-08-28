"""Tests for the A0-A9 ablation framework: config isolation, A1 retrieval,
A2 RAG bypass, A4 diversity diagnostics, A5 pool preservation, A6 non-RL
sizing, A7 surrogate-off SAC, A8 baseline ranker, A9 GenerationState,
preflight/checkpoint validation, statistics, and architecture cleanliness.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

SPEC = {"spec_id": "test_spec", "gain_target_db": 40.0,
       "phase_margin_target_deg": 45.0, "ugbw_target_hz": 1e6,
       "load_capacitance_pf": 100.0, "technology": "sky130"}


def _real_graph():
    """A real, sizeable DeviceCircuitGraph -- same construction path
    size_and_predict uses (`_realise(obj)`), not the higher-level
    CircuitGraph from TopologyRegistry (no `.devices`, incompatible with
    apply_knobs/measure)."""
    import json

    from run_puct_ablation import _realise
    from run_raptor_v2 import ROOT
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    rec = next(r for r in corpus["records"] if r["split"] == "train")
    obj = json.loads(rec["response"])
    return _realise(obj)


# =============================== A0-A9 configs ===================================
def test_a0_full_enables_every_component():
    """Stage 8 (2026-08-12, second deployment): A0_FULL carries the learned
    DPO ranker again -- Stage 7.2B re-justified it (POST_SAC_FEATURES_V2,
    DPO_REJUSTIFIED: 77.80% vs the deterministic selector's 74.66%
    run-grouped DEV ranker-authority accuracy, 77 wins/45 losses/0
    catastrophic errors), reversing Stage 7.1's earlier
    LEARNED_DPO_NOT_JUSTIFIED finding. The hard safety gate itself is
    untouched and still applies ahead of the learned ranker (see
    test_a8_hard_safety_gate_still_applies_without_dpo)."""
    from agentic_raptor.publication.ablation_v3 import A0_FULL
    kw = A0_FULL.to_run_pipeline_kwargs()
    assert kw["use_llm"] and kw["use_rag"]
    assert A0_FULL.use_sft
    assert kw["conditioning"] == "exclusion"
    assert kw["search"] == "bandit_top2"  # 2026-08-15: bandit replaces AlphaZero as FULL selector
    assert kw["sizing_method"] == "sac"
    assert kw["use_surrogate"] and kw["use_sizing_ranker"]
    assert kw["ranker_mode"] == "dpo"
    assert A0_FULL.use_dpo is True
    assert A0_FULL.retired is False


def test_a1_config_disables_only_llm():
    from agentic_raptor.publication.ablation_v3 import A0_FULL, A1_NO_LLM
    kw0, kw1 = A0_FULL.to_run_pipeline_kwargs(), A1_NO_LLM.to_run_pipeline_kwargs()
    diff = {k for k in kw0 if kw0[k] != kw1[k]}
    assert diff == {"use_llm"}


def test_a1_never_calls_llm_generation_functions(monkeypatch):
    """LLM generation must never be invoked when use_llm=False -- patch both
    generation entry points to explode if called."""
    import agentic_raptor.llm_dpo.stage3e4 as st34

    def _explode(*a, **kw):
        raise AssertionError("LLM generation function called under use_llm=False")
    monkeypatch.setattr(st34, "propose_diverse", _explode)
    monkeypatch.setattr(st34, "propose_diverse_excl", _explode)

    import json
    from run_raptor_v2 import ROOT, propose_and_validate
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    rec = next(r for r in corpus["records"] if r["split"] == "train")
    spec = dict(SPEC, spec_id=rec["context_id"], topology_id=rec["topology_id"])
    res = propose_and_validate(None, None, rec["prompt"], target_k=5,
                               use_llm=False, spec=spec)
    assert res["candidates"]
    assert all(c["source"] == "retrieval" for c in res["candidates"])
    hashes = [c["canonical_graph_hash"] for c in res["candidates"]]
    assert len(hashes) == len(set(hashes))


def test_a1_retrieval_never_returns_the_specs_own_topology():
    from run_raptor_v2 import retrieve_topology_candidates
    res = retrieve_topology_candidates(
        SPEC, target_k=5, exclude_topology_id="topology_0002", seed=0)
    assert all(c["retrieved_from_topology_id"] != "topology_0002"
              for c in res["candidates"])


def test_a2_config_disables_only_rag():
    from agentic_raptor.publication.ablation_v3 import A0_FULL, A2_NO_RAG
    kw0, kw2 = A0_FULL.to_run_pipeline_kwargs(), A2_NO_RAG.to_run_pipeline_kwargs()
    assert {k for k in kw0 if kw0[k] != kw2[k]} == {"use_rag"}


def test_a2_use_rag_false_never_calls_retrieve(monkeypatch):
    import run_raptor_v2 as v2

    def _explode(*a, **kw):
        raise AssertionError("retrieve() called under use_rag=False")
    monkeypatch.setattr(v2, "retrieve", _explode)
    out = v2.rag_stage(SPEC, "the raw prompt", use_rag=False)
    assert out == {"records": [], "retrieval_ids": [], "prompt": "the raw prompt"}


def test_a2_use_rag_true_does_call_retrieve(monkeypatch):
    import run_raptor_v2 as v2
    called = []
    monkeypatch.setattr(v2, "retrieve",
                        lambda spec, prompt, memory_path=None: called.append(1) or
                        {"records": [], "retrieval_ids": [], "prompt": prompt})
    v2.rag_stage(SPEC, "p", use_rag=True)
    assert called == [1]


def test_a3_config_disables_sft_and_requires_base_proposer():
    from run_ablation_v3 import _proposer_requirement

    from agentic_raptor.publication.ablation_v3 import A0_FULL, A3_NO_SFT
    assert A3_NO_SFT.use_sft is False
    assert _proposer_requirement(A0_FULL) == "sft"
    assert _proposer_requirement(A3_NO_SFT) == "base"
    kw0, kw3 = A0_FULL.to_run_pipeline_kwargs(), A3_NO_SFT.to_run_pipeline_kwargs()
    assert kw0 == kw3   # use_sft is NOT a run_pipeline kwarg -- it changes
                        # which adapter gets loaded, handled by the driver


def test_a4_config_disables_only_exclusion_conditioning():
    from agentic_raptor.publication.ablation_v3 import A0_FULL, A4_NO_DIVERSITY
    kw0, kw4 = A0_FULL.to_run_pipeline_kwargs(), A4_NO_DIVERSITY.to_run_pipeline_kwargs()
    assert {k for k in kw0 if kw0[k] != kw4[k]} == {"conditioning"}
    assert kw4["conditioning"] == "temperature"


def test_a4_novelty_is_not_claimed_from_hash_difference_alone():
    """Two candidates with different content but the SAME family hash must
    be 'near_duplicate', never 'structurally_distinct' -- novelty needs a
    genuinely different family, not just a different hash string."""
    from agentic_raptor.publication.topology_diagnostics import classify_novelty
    obj_a = {"stages": [{"block": "x"}], "compensation": [], "output_buffer": False,
            "local_feedback": False, "ports": ["a"]}
    obj_b = {"stages": [{"block": "y"}], "compensation": [], "output_buffer": False,
            "local_feedback": False, "ports": ["b"]}
    from agentic_raptor.llm_dpo.stage3e4 import variant_hash
    fam = variant_hash(obj_a)
    assert fam == variant_hash(obj_b)          # same family, different content
    cat = classify_novelty(obj_b, fam, known_family_hashes={fam},
                           known_content_hashes=set())
    assert cat == "near_duplicate"


def test_a5_config_couples_mcts_and_puct_and_forbids_splitting():
    from agentic_raptor.publication.ablation_v3 import AblationConfig, A5_NO_MCTS_PUCT
    assert A5_NO_MCTS_PUCT.use_mcts is False and A5_NO_MCTS_PUCT.use_puct is False
    assert A5_NO_MCTS_PUCT.to_run_pipeline_kwargs()["search"] == "none"
    with pytest.raises(ValueError):
        AblationConfig("AX", "x", "x", use_mcts=True, use_puct=False)


def test_a5_same_validated_pool_size_regardless_of_search():
    """A0 and A5 must receive the exact same validated candidate pool --
    A5/NO_ALPHAZERO's direct-prior selection must not drop or add
    candidates, only reorder them. (2026-08-11: root-level PUCT retired;
    search="none" now dispatches to direct_prior_select_two(), which
    IS what this test exercises -- see run_raptor_v2.run_pipeline's
    stage5_alphazero branch.)"""
    from agentic_raptor.topology_rl.alphazero import direct_prior_select_two
    candidates = [
        {"llm_proposal_id": f"p{i:02d}",
         "canonical_graph_hash": f"h{i}",
         "canonical_family": "2s_none" if i % 2 == 0 else "3s_rc",
         "obj": {"stages": [{}] * (2 if i % 2 == 0 else 3)},
         "source": "llm"}
        for i in range(5)]
    none_sel = direct_prior_select_two(list(candidates), SPEC)
    assert none_sel["search"] == "a5_direct_prior"
    assert len(none_sel["ranked"]) == len(candidates) == 5
    assert len(none_sel["selected"]) == 2


def test_a6_config_uses_non_rl_sizer_not_random():
    from agentic_raptor.publication.ablation_v3 import A6_NO_SAC
    assert A6_NO_SAC.use_sac is False
    assert A6_NO_SAC.sizing_method == "tpe_lite"
    assert A6_NO_SAC.sizing_method != "random", (
        "A6 must not be a deliberately weak random-search baseline")


def test_a6_non_rl_sizer_produces_real_measured_results(tmp_path):
    """Real ngspice, tiny budget: confirms _non_rl_size returns sac_size's
    exact contract (outcome/best/results/spice_calls/seed)."""
    from run_raptor_v2 import _non_rl_size
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    exe = discover_ngspice()
    if not exe:
        pytest.skip("ngspice not available")
    g = _real_graph()
    costs = new_costs()
    sz = _non_rl_size("tpe_lite", "t_a6_test", g, SPEC, exe, tmp_path, costs, 3, 0)
    assert sz["spice_calls"] == 3 == len(sz["results"])
    assert sz["sac_algorithm"] == "non_rl_tpe_lite"
    assert sz["best"] is not None
    assert sz["outcome"]["hard_constraints_total"] > 0


def test_a7_config_disables_surrogate_and_sizing_ranker():
    from agentic_raptor.publication.ablation_v3 import A0_FULL, A7_NO_SURROGATE
    assert A7_NO_SURROGATE.use_surrogate is False
    assert A7_NO_SURROGATE.use_sizing_ranker is False
    kw0, kw7 = A0_FULL.to_run_pipeline_kwargs(), A7_NO_SURROGATE.to_run_pipeline_kwargs()
    assert kw7["sizing_method"] == "sac"          # SAC remains ENABLED
    assert {k for k in kw0 if kw0[k] != kw7[k]} == {"use_surrogate", "use_sizing_ranker"}


def test_a7_sac_without_surrogate_still_uses_real_spice_every_step(tmp_path):
    """A7 isolates the model-based part, not SAC itself: every step must
    still be a real ngspice call, at the identical count as A0 (budget)."""
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mb_sac.spec_sizing import sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    exe = discover_ngspice()
    if not exe:
        pytest.skip("ngspice not available")
    g = _real_graph()
    costs = new_costs()
    budget = 3
    sz = sac_size("t_a7_test", g, SPEC, exe, tmp_path, costs, budget=budget,
                  seed=0, persist=False, use_surrogate=False, use_ranker=False)
    assert sz["spice_calls"] == budget == len(sz["results"])


# =============================== A8 baseline ranker ===============================
def test_a8_config_is_meaningful_again_and_un_retired():
    """Stage 8 (second deployment): A8 is un-retired -- now that A0/FULL
    carries a re-justified learned DPO (Stage 7.2B), A8 once again removes
    something real: the ONLY kwarg difference from A0 is ranker_mode."""
    from agentic_raptor.publication.ablation_v3 import A0_FULL, A8_NO_DPO
    kw0, kw8 = A0_FULL.to_run_pipeline_kwargs(), A8_NO_DPO.to_run_pipeline_kwargs()
    assert {k for k in kw0 if kw0[k] != kw8[k]} == {"ranker_mode"}
    assert kw8["ranker_mode"] == "deterministic"
    assert A8_NO_DPO.retired is False


def test_a8_hard_safety_gate_still_applies_without_dpo():
    """Level 1 (hard safety) must decide even when model=None -- A8 removes
    the LEARNED ranker, never the safety gate."""
    from agentic_raptor.ranking.post_sac import compare
    from agentic_raptor.ranking.types import SurrogatePrediction

    good = SurrogatePrediction(
        topology_hash="ha", sizing_manifest_hash="m",
        operating_point_probability=0.99, stability_probability=0.99,
        normalized_margins={"gain": 0.5, "pm": 0.5})
    bad = SurrogatePrediction(
        topology_hash="hb", sizing_manifest_hash="m",
        operating_point_probability=0.01, stability_probability=0.01,
        normalized_margins={"gain": -0.9, "pm": -0.9})
    da = _fake_design("A", "ha")
    db = _fake_design("B", "hb")
    decision = compare(da, db, good, bad, spec=SPEC, model=None,
                      ranker_arm="explicit_baseline")
    assert decision["decision_basis"] == "hard_safety_gate"
    assert decision["selected_design"] == "A"


def _fake_design(label, topo_hash):
    from agentic_raptor.ranking import PostSACDesign
    return PostSACDesign(label=label, spec_id="s", llm_proposal_id="p0",
                         canonical_graph_hash=topo_hash,
                         topology_signature="fam", topology_family="fam",
                         sizing_vector={"s1w": 1.0}, sizing_manifest_hash="m",
                         sac_trajectory_id="t", action_space_version="",
                         reward_version="r", sizing_budget=8,
                         sizing_spice_calls=1, sizing_spice_call_ids=[],
                         puct_rank=0, puct_visits=1, final_netlist_hash="n")


# =============================== A9 GenerationState =================================
def test_a9_interrupted_generation_resumes(tmp_path):
    from agentic_raptor.publication.generation_state import (
        checkpoint_progress, resume_or_start)
    st = resume_or_start(tmp_path, "adaptive")
    st.mark_run_processed("r1")
    checkpoint_progress(st, tmp_path)
    st2 = resume_or_start(tmp_path, "adaptive")
    assert st2.generation_id == st.generation_id
    assert st2.processed_run_ids == ["r1"]


def test_a9_duplicate_harvest_prevented(tmp_path):
    from agentic_raptor.publication.generation_state import resume_or_start
    st = resume_or_start(tmp_path, "adaptive")
    assert st.mark_run_processed("r1") is True
    assert st.mark_run_processed("r1") is False


def test_a9_incomplete_generation_cannot_publish(tmp_path):
    from agentic_raptor.publication.generation_state import (
        publish_atomic, resume_or_start)
    st = resume_or_start(tmp_path, "adaptive")
    with pytest.raises(ValueError):
        publish_atomic(st, tmp_path)


def test_a9_static_mode_maps_to_frozen_learning_mode():
    from agentic_raptor.publication.ablation_v3 import A9_STATIC
    assert A9_STATIC.learning_mode == "static"


def test_a9_adaptive_and_static_g0_hashes_identical_by_construction():
    """Both lineages must start G0 from the SAME accepted-component hashes
    -- since accepted_component_hashes() is a pure function of files on
    disk, calling it twice (once per lineage) yields identical results."""
    from agentic_raptor.publication.generation_state import \
        accepted_component_hashes
    kwargs = dict(
        rag_memory_path=Path("artifacts/publication_v2/selfimprove/rag_memory_v2.jsonl"),
        sft_adapter_path=Path("artifacts/publication_v2/proposer_repair/sft_adapter_diverse"),
        puct_ckpt_path=Path("artifacts/stage3e1/policy_value_ep0.pt"),
        dpo_ckpt_path=Path("artifacts/publication_v2/post_sac_ranker/ranker.pt"),
        evaluation_set_hash="fixed_eval_hash")
    h_adaptive = accepted_component_hashes(**kwargs)
    h_static = accepted_component_hashes(**kwargs)
    assert h_adaptive == h_static


def test_a9_learning_mode_static_never_persists_ranker_pairs(tmp_path, monkeypatch):
    """learning_mode='static' must behave exactly like 'frozen' for the
    record_pair persistence gate -- static is not allowed to mutate shared
    learning state either."""
    from agentic_raptor.ranking.post_sac import TRUSTED, record_pair
    monkeypatch.setattr("agentic_raptor.ranking.post_sac.QUEUE", tmp_path)
    monkeypatch.setattr("agentic_raptor.ranking.post_sac.TRUSTED", tmp_path / "trusted_pairs.jsonl")
    da = _fake_design("A", "ha")
    db = _fake_design("B", "hb")
    rec = record_pair(SPEC, da, db, None, None, persist=False)
    assert not (tmp_path / "trusted_pairs.jsonl").is_file()
    assert rec["status"] == "provisional_incomplete_provenance"


# =============================== preflight / checkpoint validation ==================
def test_checkpoint_validation_rejects_missing_metadata(tmp_path):
    import torch

    from agentic_raptor.publication.checkpoint_validation import validate
    p = tmp_path / "fake.pt"
    torch.save({"encoder": {}, "heads": {}, "meta": {"schema_version": "x"}}, p)
    v = validate(p)
    assert v.ok is False
    assert "missing_fields" in v.reason


def test_checkpoint_validation_accepts_after_mark_validated(tmp_path):
    import torch

    from agentic_raptor.publication.checkpoint_validation import (
        mark_validated, validate)
    p = tmp_path / "fake.pt"
    torch.save({"encoder": {}, "heads": {}, "meta": {}}, p)
    mark_validated(p, post_vcm_fix=True, training_data_hash="abc123")
    v = validate(p)
    assert v.ok is True


def test_checkpoint_validation_rejects_explicit_false(tmp_path):
    import torch

    from agentic_raptor.publication.checkpoint_validation import validate
    p = tmp_path / "fake.pt"
    torch.save({"encoder": {}, "heads": {},
               "meta": {"post_vcm_fix": False, "validated": False,
                        "training_data_hash": "x"}}, p)
    v = validate(p)
    assert v.ok is False


def test_preflight_real_repo_produces_computed_not_fabricated_result():
    from agentic_raptor.publication.preflight import run_preflight
    r = run_preflight(paper_mode=False)
    names = {c["name"] for c in r["checks"]}
    # 2026-08-11: root-level PUCT retired -- "puct_value_checkpoint_post_
    # vcm_validated" replaced by "old_root_puct_checkpoint_absent" (must be
    # True post-cutover) and "alphazero_promoted_checkpoint" (expected
    # False/NOT READY until a real training campaign promotes one).
    assert {"rag_populated", "old_root_puct_checkpoint_absent",
           "alphazero_promoted_checkpoint", "fom_v1_calculation",
           "leakage_tests"} <= names
    # every check must have actually run (a bool, not None/missing)
    assert all(isinstance(c["ok"], bool) for c in r["checks"])


def test_preflight_paper_mode_refuses_on_blocker(monkeypatch):
    from agentic_raptor.publication import preflight as pf

    def _fail():
        return {"name": "x", "ok": False, "detail": "forced failure",
               "critical": True}
    monkeypatch.setattr(pf, "ALL_CHECKS", (_fail,))
    with pytest.raises(SystemExit):
        pf.run_preflight(paper_mode=True)


def test_preflight_paper_mode_passes_when_no_blockers(monkeypatch):
    from agentic_raptor.publication import preflight as pf

    def _pass():
        return {"name": "x", "ok": True, "detail": "ok", "critical": True}
    monkeypatch.setattr(pf, "ALL_CHECKS", (_pass,))
    r = pf.run_preflight(paper_mode=True)     # must NOT raise
    assert r["ready"] is True


# =============================== RAG freeze ==========================================
def test_rag_freeze_no_manifest_is_not_frozen(tmp_path):
    from agentic_raptor.publication.rag_freeze import verify
    src = tmp_path / "rag_memory_v2.jsonl"
    src.write_text('{"stability": "verified_stable", "context_id": "c0"}\n',
                   encoding="utf-8")
    v = verify(src)
    assert v["frozen"] is False and v["drifted"] is False


def test_rag_freeze_matches_immediately_after_freezing(tmp_path):
    from agentic_raptor.publication.rag_freeze import freeze_rag_snapshot, verify
    src = tmp_path / "rag_memory_v2.jsonl"
    src.write_text('{"stability": "verified_stable", "context_id": "c0"}\n'
                   '{"stability": "verified_stable", "context_id": "c1"}\n',
                   encoding="utf-8")
    m = freeze_rag_snapshot(src)
    assert m["record_count"] == 2 and m["usable_record_count"] == 2
    v = verify(src)
    assert v["frozen"] is True and v["drifted"] is False


def test_rag_freeze_detects_drift_after_append(tmp_path):
    from agentic_raptor.publication.rag_freeze import freeze_rag_snapshot, verify
    src = tmp_path / "rag_memory_v2.jsonl"
    src.write_text('{"stability": "verified_stable", "context_id": "c0"}\n',
                   encoding="utf-8")
    freeze_rag_snapshot(src)
    with src.open("a", encoding="utf-8") as f:
        f.write('{"stability": "verified_stable", "context_id": "c1"}\n')
    v = verify(src)
    assert v["frozen"] is True and v["drifted"] is True


def test_rag_freeze_counts_only_stable_records_as_usable(tmp_path):
    from agentic_raptor.publication.rag_freeze import freeze_rag_snapshot
    src = tmp_path / "rag_memory_v2.jsonl"
    src.write_text('{"stability": "verified_stable", "context_id": "c0"}\n'
                   '{"stability": null, "context_id": "c1"}\n',
                   encoding="utf-8")
    m = freeze_rag_snapshot(src)
    assert m["record_count"] == 2 and m["usable_record_count"] == 1


def test_preflight_check_rag_frozen_uses_real_verify():
    from agentic_raptor.publication.preflight import check_rag_frozen
    c = check_rag_frozen()
    assert set(c) == {"name", "ok", "detail", "critical"}
    assert c["name"] == "rag_frozen"


# =============================== statistics ==========================================
def test_statistics_paired_comparison_never_reports_only_p_value():
    from agentic_raptor.publication.ablation_stats import compare_paired
    cmp = compare_paired("fom_value", [1.0, 1.2, 0.9], [0.5, 0.6, 0.4])
    assert cmp.mean_delta is not None
    assert cmp.bootstrap_ci95 != (None, None)
    assert cmp.permutation_p_value is not None
    assert cmp.effect_size_cohens_d is not None


def test_statistics_sufficient_n_flag_matches_pilot_floor():
    from agentic_raptor.publication.ablation_stats import (
        MIN_PAIRS_FOR_CLAIM, compare_paired)
    assert MIN_PAIRS_FOR_CLAIM == 3          # the agreed pilot seed count
    two = compare_paired("m", [1.0, 2.0], [0.5, 0.5])
    three = compare_paired("m", [1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
    assert two.sufficient_n is False
    assert three.sufficient_n is True


def test_statistics_align_pairs_matches_by_spec_and_seed_not_position():
    from agentic_raptor.publication.ablation_stats import align_pairs
    rows_a = [{"spec_id": "s1", "pipeline_seed": 0, "v": 10},
             {"spec_id": "s2", "pipeline_seed": 0, "v": 20}]
    rows_b = [{"spec_id": "s2", "pipeline_seed": 0, "v": 2},
             {"spec_id": "s1", "pipeline_seed": 0, "v": 1}]
    va, vb = align_pairs(rows_a, rows_b, "v")
    assert va == [10, 20] and vb == [1, 2]


# =============================== architecture / spice accounting =====================
def test_new_ablation_modules_have_no_legacy_raptor_reference():
    from agentic_raptor.publication import (ablation_stats, ablation_v3,
                                            checkpoint_validation,
                                            generation_state, preflight,
                                            rag_freeze, topology_diagnostics)
    for mod in (ablation_v3, ablation_stats, checkpoint_validation,
               generation_state, preflight, topology_diagnostics, rag_freeze):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "RAPTOR_Legacy" not in src


def test_a0_a8_share_identical_budget_hash():
    """FAIRNESS: A0-A8 inherit the SAME base ExperimentBudget -- none of the
    ten currently override a budget field."""
    from agentic_raptor.publication.ablation_v3 import COMPONENT_ABLATIONS
    hashes = {c.budget.hash() for c in COMPONENT_ABLATIONS.values()}
    assert len(hashes) == 1


def test_yaml_configs_round_trip_to_identical_config_hash(tmp_path):
    from agentic_raptor.publication.ablation_v3 import (PRIMARY_EXPERIMENTS,
                                                         config_from_yaml_dict,
                                                         dump_yaml_configs)
    paths = dump_yaml_configs(tmp_path)
    import yaml
    for p in paths:
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        cfg = config_from_yaml_dict(d)
        assert cfg.config_hash() == PRIMARY_EXPERIMENTS[cfg.ablation_id].config_hash()
