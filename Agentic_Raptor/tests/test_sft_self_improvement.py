"""SFT self-improvement loop (2026-08-11): queue -> eligibility -> dataset
-> LoRA generation -> validation -> promotion/rollback.

Covers the runtime path this feature actually adds on top of the
already-correct agentic_raptor.selfimprove_v2.streams.harvest_run /
sft_admission_reasons (see that module's tests for admission-time
coverage; this file exercises the SECOND, independent eligibility pass,
plus everything downstream of an admitted row: dataset assembly,
generation checkpointing, lineage guards, and the explicit training
trigger's promote/reject branching).
"""
from __future__ import annotations

import json

import pytest

from agentic_raptor.publication.artifact_provenance import POST_CLOAD_FIX_V1
from agentic_raptor.selfimprove_v2 import sft_self_improvement as si

OBJ = {"stages": [{"block": "five_transistor_first_stage", "role": "input_stage",
                   "outputs": ["s1out"]},
                  {"block": "cs_gain_stage", "role": "gain_stage",
                   "outputs": ["vout"]}],
      "ports": ["gnda", "vdda", "vinn", "vinp", "vout"],
      "bias_roles": ["bias_mirror"], "compensation": [], "output_buffer": False,
      "local_feedback": False, "feedback_paths": [], "polarity": "vinp_noninverting"}

SPEC = {"gain_target_db": 40.0, "phase_margin_target_deg": 45.0}


def make_row(spec_id="ctx0", spec_hash="h0", *, canonical_graph_hash="gA",
            req_cl=1e-10, sim_cl=1e-10, mode="final_verification",
            pass_=True, env=POST_CLOAD_FIX_V1, admitted=True, spec_index=0,
            call_id="final:ctx0:A:123", family="2s_none"):
    return {
        "generation_spec_id": spec_id, "split": "train", "seed": 0,
        "spec_hash": spec_hash, "spec_index": spec_index,
        "requested_c_load_f": req_cl, "electrical_environment_version": env,
        "branch": "A", "spec": {**SPEC, "spec_id": spec_id},
        "canonical_graph_hash": canonical_graph_hash, "family": family,
        "obj": OBJ, "obj_hash": "objhashA",
        "exact_spec_pass": pass_, "distance": 0.0 if pass_ else 0.4,
        "budget": 12, "verification_spice_call_id": call_id,
        "verification_mode": mode, "simulated_c_load_f": sim_cl,
        "gain_db": 44.0, "pm_deg": 50.0, "ugbw_hz": 1e6, "idd_a": 1e-4,
        "admitted": admitted, "rejected_because": []}


# ---------------------------------------------------------------------------
# Section 19 checklist, in order
# ---------------------------------------------------------------------------
def test_verified_exact_spec_pass_enters_eligible_pool():
    row = make_row()
    assert si.sft_eligible(row, protected_context_ids=set()) == []


def test_failed_authoritative_design_does_not_enter():
    row = make_row(pass_=False)
    reasons = si.sft_eligible(row, protected_context_ids=set())
    assert "not_exact_spec_pass" in reasons


def test_surrogate_only_prediction_does_not_reach_sft_queue():
    """harvest_run() only ever populates sft_queue from `authoritative`
    outcomes -- a branch with only a `prediction` (surrogate) and no
    `authoritative` measurement must produce an EMPTY sft_queue, proving
    the existing admission gate (Section 1) already excludes this, not
    just the new eligibility pass."""
    from agentic_raptor.selfimprove_v2.streams import harvest_run
    hv = {"spec": {"spec_id": "ctx0"}, "spec_hash": "h0", "spec_index": 0,
         "seed": 0, "requested_c_load_f": 1e-10,
         "candidates": [{"llm_proposal_id": "p00", "canonical_graph_hash": "gA",
                         "canonical_family": "2s_none", "obj": OBJ}],
         "branches": {
             "A": {"design": {"llm_proposal_id": "p00",
                              "canonical_graph_hash": "gA",
                              "topology_family": "2s_none"},
                   "prediction": {"gain_db": 44.0}, "authoritative": None},
             "B": {}}}
    out = harvest_run(hv, split="train", protected_ids=set())
    assert out["sft_queue"] == []


def test_dpo_preference_without_authoritative_pass_does_not_enter():
    """A ranker `selected_design` choice alone, with neither branch
    authoritatively measured, must not reach sft_queue OR
    proposer_dpo_pairs (both require real `authoritative` outcomes)."""
    from agentic_raptor.selfimprove_v2.streams import harvest_run
    hv = {"spec": {"spec_id": "ctx0"}, "spec_hash": "h0", "spec_index": 0,
         "seed": 0, "requested_c_load_f": 1e-10,
         "candidates": [{"llm_proposal_id": "p00", "canonical_graph_hash": "gA",
                         "canonical_family": "2s_none", "obj": OBJ}],
         "ranker": {"selected_design": "A"},
         "branches": {
             "A": {"design": {"llm_proposal_id": "p00",
                              "canonical_graph_hash": "gA",
                              "topology_family": "2s_none"},
                   "authoritative": None},
             "B": {"design": {"llm_proposal_id": "p00"}, "authoritative": None}}}
    out = harvest_run(hv, split="train", protected_ids=set())
    assert out["sft_queue"] == []
    assert out["proposer_dpo_pairs"] == []


def test_reused_sizing_spice_call_fails_provenance():
    row = make_row(mode="sizing")
    reasons = si.sft_eligible(row, protected_context_ids=set())
    assert any("verification_call_reused_from_sizing" in r for r in reasons)


def test_protected_evaluation_spec_is_rejected():
    row = make_row(spec_id="ctx_protected")
    reasons = si.sft_eligible(row, protected_context_ids={"ctx_protected"})
    assert "protected_evaluation_record" in reasons


def test_pre_cload_fix_record_is_rejected():
    row = make_row(env="PRE_CLOAD_FIX")
    reasons = si.sft_eligible(row, protected_context_ids=set())
    assert any(r.startswith("not_post_cload_fix_v1") for r in reasons)


def test_requested_simulated_load_mismatch_is_rejected():
    row = make_row(req_cl=1e-10, sim_cl=2e-10)
    reasons = si.sft_eligible(row, protected_context_ids=set())
    assert any("unexplained_load_override" in r for r in reasons)


def test_canonical_duplicates_are_deduplicated():
    rows = [make_row(spec_id="ctx0", spec_hash="h0", canonical_graph_hash="gA"),
           make_row(spec_id="ctx0", spec_hash="h0", canonical_graph_hash="gA"),
           make_row(spec_id="ctx1", spec_hash="h1", canonical_graph_hash="gA")]
    kept, stats = si.deduplicate(rows)
    assert stats == {"raw": 3, "unique_spec_topology_pairs": 2,
                     "unique_topology_hashes": 1, "duplicates_removed": 1,
                     "topology_repetition_counts": {"gA": 3}}
    assert len(kept) == 2


def test_family_balancing_behaves_deterministically():
    import random
    rows = [make_row(spec_id=f"ctx{i}", spec_hash=f"h{i}",
                     canonical_graph_hash=f"g{i}", family="2s_none")
           for i in range(10)]
    r1, p1 = si.balance_by_family(rows, per_family_cap=3)
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    r2, p2 = si.balance_by_family(shuffled, per_family_cap=3)
    assert [r["spec_hash"] for r in r1] == [r["spec_hash"] for r in r2]
    assert p1 == p2
    assert len(r1) == 3
    assert p1["dropped_for_balance"] == 7


def test_original_structural_replay_is_included_and_clean():
    base_corpus = {"records": [
        {"context_id": "c0", "split": "train",
         "prompt": "### SPEC x\n### KNOWN 2stage verified_stable pm=44deg\n"
                   "### BLOCKS a\n### PROPOSAL",
         "response": "{}"}]}
    replay = si.build_replay_examples(base_corpus)
    assert len(replay) == 1
    assert "### KNOWN" not in replay[0]["prompt"]
    assert replay[0]["target_source"] == "structural_replay_clean"

    ds = si.build_dataset([], base_corpus, generation_id="G1",
                          protected_context_ids=set())
    assert ds["manifest"]["structural_replay_included"] is True
    assert ds["manifest"]["structural_replay_count"] == 1
    assert ds["manifest"]["total_records"] == 1


def test_stale_self_improvement_runs_jsonl_is_not_consumed():
    """Structural: neither this module nor the training trigger script
    constructs a PATH to the archived L4 file (Section 4). Both modules'
    docstrings mention the filename in prose, explaining exactly why it is
    excluded -- that's the point, not a violation -- so this checks for
    the actual path-construction pattern every real reader of that file
    uses (agentic_raptor.llm_dpo.rag.L4_FILE / stage3e4.py's l4p / __init__
    .py's l4), not the bare filename string."""
    import inspect

    import train_sft_self_improvement as trig
    needle = '"datasets/simulation_memory/self_improvement_runs.jsonl"'
    for mod in (si, trig):
        src = inspect.getsource(mod)
        assert needle not in src, (
            f"{mod.__name__} constructs a path to the stale archived L4 file")


def test_g0_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "GENERATIONS_ROOT", tmp_path / "generations")
    si.write_generation_manifest("G0", {"generation_id": "G0"})
    with pytest.raises(RuntimeError, match="immutable"):
        si.write_generation_manifest("G0", {"generation_id": "G0", "x": 2})


def test_g1_receives_correct_parent_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "GENERATIONS_ROOT", tmp_path / "generations")
    g0 = si.ensure_g0_manifest(base_adapter=tmp_path / "no_such_adapter",
                               base_corpus_path=tmp_path / "no_such_corpus.json")
    assert g0["generation_id"] == "G0"
    assert g0["parent_generation"] is None
    assert g0["immutable"] is True
    assert si.next_generation_id("G0") == "G1"
    assert si.next_generation_id("G1") == "G2"
    # a second call is idempotent, not a second (failed) write attempt
    g0_again = si.ensure_g0_manifest(base_adapter=tmp_path / "no_such_adapter",
                                     base_corpus_path=tmp_path / "no_such_corpus.json")
    assert g0_again == g0


def test_a0_a8_never_imports_sft_training():
    """Structural guarantee (Section 16): the frozen A0-A8 ablation runner
    never imports this module or the training trigger script -- verified
    directly on its source, not inferred."""
    import inspect

    import run_ablation_v3
    src = inspect.getsource(run_ablation_v3)
    assert "sft_self_improvement" not in src
    assert "train_sft_self_improvement" not in src
    # and the docstring explicitly says so
    assert "separate" in (run_ablation_v3.__doc__ or "").lower()


def test_a9_adaptive_lineage_can_trigger_generation_update():
    si.assert_lineage_may_train("adaptive")     # must not raise


def test_a9_static_lineage_cannot_update():
    with pytest.raises(RuntimeError, match="static"):
        si.assert_lineage_may_train("static")


def test_a0_a8_cannot_trigger_sft_training_via_cli():
    """The trigger script itself refuses any lineage other than
    'adaptive' before doing anything else -- this is the in-process half
    of the guard an A0-A8-style caller would hit if it ever DID invoke
    this script (which test_a0_a8_never_imports_sft_training already
    shows it structurally cannot)."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "train_sft_self_improvement.py", "--lineage", "static"],
        cwd=si.ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert "static" in r.stderr


def test_failed_promotion_keeps_previous_generation_active(tmp_path, monkeypatch):
    """End-to-end (heavily mocked at the model/SPICE boundary) run of the
    trigger script's decision logic: a candidate that fails the
    capability gate must be written as rejected=True with adapter_path
    None, and ACTIVE_ADAPTIVE.txt must not be created/changed -- the
    parent generation implicitly remains what any caller should use."""
    _run_trigger_smoke(tmp_path, monkeypatch, make_capability_probe_fail=True)
    manifest = si.read_generation_manifest("G1")
    assert manifest["rejected"] is True
    assert manifest["adapter_path"] is None
    assert not (si.GENERATIONS_ROOT / "ACTIVE_ADAPTIVE.txt").is_file()


def test_successful_promotion_activates_new_generation(tmp_path, monkeypatch):
    _run_trigger_smoke(tmp_path, monkeypatch, make_capability_probe_fail=False)
    manifest = si.read_generation_manifest("G1")
    assert manifest["rejected"] is False
    assert manifest["adapter_path"] is not None
    active = si.GENERATIONS_ROOT / "ACTIVE_ADAPTIVE.txt"
    assert active.is_file()
    assert active.read_text(encoding="utf-8").strip() == "G1"


def _run_trigger_smoke(tmp_path, monkeypatch, *, make_capability_probe_fail: bool):
    """Shared harness for the two promotion-outcome tests above. Mocks
    every heavy/model/SPICE boundary; exercises the REAL dataset build,
    eligibility filter, generation manifest, and promote/reject logic in
    train_sft_self_improvement.main()."""
    import sys

    import train_sft_self_improvement as trig
    from agentic_raptor.llm_dpo import stage3e4 as sft_mod
    from agentic_raptor.publication import spec_registry

    gens_root = tmp_path / "generations"
    monkeypatch.setattr(si, "GENERATIONS_ROOT", gens_root)

    base_corpus_path = tmp_path / "corpus_diverse.json"
    base_corpus_path.write_text(json.dumps({"records": [
        {"context_id": "ctx0", "split": "train",
         "prompt": "### SPEC gain>=40dB pm>=45.0deg cl=100pF ugbw>=1e+06Hz "
                   "tech=sky130\n### RAG rag_l2_topology_0000\n"
                   "### KNOWN 2stage verified_stable pm=44deg\n"
                   "### BLOCKS a,b\n### FORBIDDEN x\n### PROPOSAL",
         "response": "{}", "variant_hash": "gOLD"},
        {"context_id": "ctx_held", "split": "train",
         "prompt": "### SPEC gain>=50dB pm>=55.0deg cl=100pF ugbw>=1e+06Hz "
                   "tech=sky130\n### RAG rag_l2_topology_0001\n"
                   "### BLOCKS a,b\n### FORBIDDEN x\n### PROPOSAL",
         "response": "{}", "variant_hash": "gOLD2"}]}), encoding="utf-8")

    si.write_generation_manifest("G0", {
        "generation_id": "G0", "parent_generation": None,
        "adapter_path": str(tmp_path / "g0_adapter"),
        "training_dataset_path": str(base_corpus_path),
        "immutable": True, "rejected": False})

    si_root = tmp_path / "selfimprove"
    (si_root / "gen_000" / "streams").mkdir(parents=True)
    row = make_row(spec_id="ctx0", spec_hash="h0", canonical_graph_hash="gNEW",
                   spec_index=0)
    (si_root / "gen_000" / "streams" / "sft_queue.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8")

    monkeypatch.setattr("agentic_raptor.publication.eval_sets.excluded_context_ids",
                        lambda *a, **k: set())
    monkeypatch.setattr(
        "agentic_raptor.publication.eval_sets.excluded_evaluation_context_ids",
        lambda *a, **k: set())

    def fake_run_sft(*, steps, seed, corpus_path, out_dir, lr=3e-4):
        out_dir = type(corpus_path)(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "adapter_model.bin").write_text("fake", encoding="utf-8")
        return {"steps": steps, "seed": seed, "lr": lr,
               "loss_first_last": [1.0, 0.5], "checkpoint": str(out_dir),
               "wall_clock_s": 0.1, "model_id": "fake-model",
               "corpus": str(corpus_path), "train_records": 1, "gpu": "cpu"}
    monkeypatch.setattr(sft_mod, "run_sft", fake_run_sft)

    fake_registry = {"entries": [{
        "context_id": "ctx_held", "split": "train",
        "parsed_spec": {**SPEC, "spec_id": "ctx_held"},
        "spec_hash": "h_held", "spec_index": 1, "difficulty_tier": "easy"}]}
    monkeypatch.setattr(spec_registry, "build", lambda *a, **k: fake_registry)

    good = {"schema_graph_construction_success_rate": 1.0,
           "mean_valid_at_k": 1.0, "mean_unique_at_k": 1.0,
           "family_diversity_count": 2, "run_rehit_rate": 0.0,
           "novel_valid_yield": 1, "n_specs": 1, "per_spec": []}
    bad = {**good, "schema_graph_construction_success_rate": 0.1,
          "family_diversity_count": 0, "mean_valid_at_k": 0.0}
    calls = {"n": 0}

    def fake_capability_probe(adapter, eval_items, **kw):
        calls["n"] += 1
        # candidate is called first (adapter_out); parent second
        return (bad if (make_capability_probe_fail and calls["n"] == 1) else good)
    monkeypatch.setattr(si, "capability_probe", fake_capability_probe)

    argv = ["train_sft_self_improvement.py", "--lineage", "adaptive",
           "--parent-generation", "G0", "--output-generation", "G1",
           "--si-root", str(si_root), "--steps", "5",
           "--capability-eval-specs", "1", "--downstream-eval-specs", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    trig.main()
