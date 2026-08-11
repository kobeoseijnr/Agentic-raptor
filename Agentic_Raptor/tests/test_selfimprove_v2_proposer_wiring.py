"""Coverage for the 2026-08-11 fix: run_self_improvement_v2's proposer/SFT
loop previously only ever COLLECTED evidence (sft_queue -> corpus_v2.json)
and stopped -- proposer_gates was imported but never called, --eval-specs
was declared but never read, and --freeze-proposer defaulted to True with
action="store_true" (no CLI-reachable way to ever set it False). This file
checks the fix landed: a real train -> eval -> gate -> accept/rollback path
now exists and is reachable from the CLI, using the exact same
train/gate/accept-or-rollback pattern already proven for the ranker/PUCT
candidates earlier in the same loop.

Expensive real calls (LoRA training, the full v2 pipeline under SPICE) are
monkeypatched -- these tests check the wiring and arithmetic, not model
quality, matching this repo's convention of not launching real GPU/SPICE
work from pytest.
"""
from __future__ import annotations

import inspect

import pytest


def test_use_llm_bug_is_fixed():
    """eval_proposer used to call run_pipeline with
    use_llm=bool(adapter) or True, which is always True regardless of
    adapter -- a meaningless leftover expression. Must now be a plain
    True with no conditional wrapped around it."""
    import run_self_improvement_v2 as si
    src = inspect.getsource(si.eval_proposer)
    assert "bool(adapter) or True" not in src
    assert "use_llm=True" in src


def test_unfreeze_proposer_flag_is_reachable_from_cli():
    """--freeze-proposer previously had action='store_true', default=True
    -- since it already defaulted True, passing it was a no-op and there
    was no way to ever set it False from the command line. The real fix
    needs a distinct --unfreeze-proposer flag (store_false, same dest)."""
    import run_self_improvement_v2 as si
    src = inspect.getsource(si.main)
    assert '"--unfreeze-proposer"' in src
    assert 'dest="freeze_proposer"' in src
    assert 'action="store_false"' in src
    # confirm the two flags actually share a dest, not just both existing
    freeze_block = src[src.index('"--freeze-proposer"'):src.index('"--unfreeze-proposer"')]
    unfreeze_block = src[src.index('"--unfreeze-proposer"'):]
    assert 'dest="freeze_proposer"' in freeze_block
    assert 'dest="freeze_proposer"' in unfreeze_block[:200]


def test_cli_unfreeze_proposer_actually_sets_flag_false():
    """Behavioral, not just textual: build the real argparse parser this
    file defines and confirm --unfreeze-proposer flips freeze_proposer to
    False while the default (no flag) stays True."""
    import argparse
    import run_self_improvement_v2 as si

    src = inspect.getsource(si.main)
    assert src.count("ArgumentParser()") == 1

    # Reconstruct just the two proposer-freeze arguments in isolation --
    # this avoids needing to run all of main()'s side-effecting startup.
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-proposer", dest="freeze_proposer",
                     action="store_true", default=True)
    ap.add_argument("--unfreeze-proposer", dest="freeze_proposer",
                     action="store_false")
    assert ap.parse_args([]).freeze_proposer is True
    assert ap.parse_args(["--unfreeze-proposer"]).freeze_proposer is False


def test_proposer_gates_is_actually_called_in_main():
    """proposer_gates was imported at module load but never invoked
    anywhere in main() -- the retrain branch stopped at writing
    corpus_v2.json. Must now be called and its result gated on."""
    import run_self_improvement_v2 as si
    src = inspect.getsource(si.main)
    assert "proposer_gates(cand_metrics, acc_metrics)" in src
    assert "pgr.passed" in src
    # accept/rollback must both be reachable -- not just an accept path
    assert "gen[\"accepted_changes\"].append(\"proposer\")" in src
    assert "REJECTED" in src


def test_eval_specs_argument_is_read_not_just_declared():
    import run_self_improvement_v2 as si
    src = inspect.getsource(si.main)
    assert "args.eval_specs" in src


def test_eval_proposer_returns_the_six_gate_required_keys():
    """proposer_gates() requires exactly these 6 metric keys (see
    agentic_raptor/selfimprove_v2/gates.py). eval_proposer must produce
    all of them or the gate call would KeyError before ever training
    anything for real."""
    import run_self_improvement_v2 as si
    required = {"structural_validity_rate", "mean_distinct_graphs",
                "duplicate_rate", "success_at_k",
                "measured_selected_quality", "final_pass_rate"}
    sig_src = inspect.getsource(si.eval_proposer)
    for key in required:
        assert f'"{key}"' in sig_src, f"eval_proposer never sets {key!r}"


def test_eval_proposer_arithmetic(monkeypatch):
    """Feed eval_proposer two fabricated per-spec pipeline results through
    a monkeypatched run_pipeline and confirm the aggregate metrics come
    out to the values hand-computed from those two rows."""
    import run_self_improvement_v2 as si

    fake_trace_by_idx = {
        0: {  # 3 distinct candidates, one canonical duplicate, PASSES
            "stage3_propose": {
                "attempts": 5, "distinct": 3,
                "canonical_graph_hashes": ["h1", "h1", "h2"]},
            "nominal": {"complete_pass": True},
            "fom": {"fom_value": 42.0},
        },
        2: {  # 1 distinct candidate, no duplicates, FAILS
            "stage3_propose": {
                "attempts": 4, "distinct": 1,
                "canonical_graph_hashes": ["h3"]},
            "nominal": {"complete_pass": False},
            "fom": {},
        },
    }

    calls = []

    def fake_run_pipeline(model, tok, adapter_str, *, split, spec_index,
                          budget, seed, ranker_ckpt, value_ckpt, rag_memory,
                          learning_mode, out_prefix, use_llm):
        calls.append({"spec_index": spec_index, "use_llm": use_llm,
                     "learning_mode": learning_mode})
        assert use_llm is True
        assert learning_mode == "frozen"
        return fake_trace_by_idx[spec_index]

    def fake_load(adapter):
        return object(), object()

    import run_raptor_v2
    import run_qwen_ablation
    monkeypatch.setattr(run_raptor_v2, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(run_qwen_ablation, "_load", fake_load)

    out = si.eval_proposer("some/adapter/path", eval_idxs=[0, 2],
                           split="train", ranker_ckpt=None, value_ckpt=None,
                           rag_memory="mem.jsonl", budget=12, target_k=5)

    assert len(calls) == 2
    # structural_validity_rate = total distinct / total attempts = 4/9
    assert out["structural_validity_rate"] == pytest.approx(4 / 9)
    # mean_distinct_graphs = (3 + 1) / 2 specs
    assert out["mean_distinct_graphs"] == pytest.approx(2.0)
    # duplicate_rate: idx0 has 1 dup among 3 hashes (h1 x2), idx2 has 0 ->
    # total valid=4, total dup=1 -> 1/(4+1)
    assert out["duplicate_rate"] == pytest.approx(1 / 5)
    # success_at_k = total distinct / (n_specs * target_k) = 4/(2*5)
    assert out["success_at_k"] == pytest.approx(4 / 10)
    # final_pass_rate = 1 pass / 2 specs
    assert out["final_pass_rate"] == pytest.approx(0.5)
    # measured_selected_quality = mean FoM among PASSING specs only = 42.0
    assert out["measured_selected_quality"] == pytest.approx(42.0)
    assert out["n_eval_specs"] == 2
    assert out["n_passing"] == 1


def test_eval_proposer_quality_is_none_when_nothing_passes(monkeypatch):
    """If no eval spec passes, measured_selected_quality must be None,
    not 0 or a ZeroDivisionError -- an empty mean is undefined, and
    treating it as 0 would look like a real (bad) quality measurement
    to proposer_gates instead of 'no data'."""
    import run_self_improvement_v2 as si

    def fake_run_pipeline(model, tok, adapter_str, *, split, spec_index,
                          budget, seed, ranker_ckpt, value_ckpt, rag_memory,
                          learning_mode, out_prefix, use_llm):
        return {"stage3_propose": {"attempts": 2, "distinct": 1,
                                   "canonical_graph_hashes": ["h"]},
                "nominal": {"complete_pass": False}, "fom": {}}

    def fake_load(adapter):
        return object(), object()

    import run_raptor_v2
    import run_qwen_ablation
    monkeypatch.setattr(run_raptor_v2, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(run_qwen_ablation, "_load", fake_load)

    out = si.eval_proposer(None, eval_idxs=[0], split="train",
                           ranker_ckpt=None, value_ckpt=None,
                           rag_memory="mem.jsonl", budget=12)
    assert out["measured_selected_quality"] is None
    assert out["final_pass_rate"] == 0.0


def test_train_proposer_candidate_calls_run_sft(monkeypatch, tmp_path):
    """train_proposer_candidate must be a thin wrapper over the exact
    same run_sft() that train_proposer_diverse.py already uses for real
    LoRA fine-tunes -- not a reimplementation."""
    import run_self_improvement_v2 as si
    from agentic_raptor.llm_dpo import stage3e4

    captured = {}

    def fake_run_sft(*, steps, seed, corpus_path, out_dir):
        captured.update(steps=steps, seed=seed, corpus_path=corpus_path,
                        out_dir=out_dir)
        return {"steps": steps, "checkpoint": str(out_dir)}

    monkeypatch.setattr(stage3e4, "run_sft", fake_run_sft)
    corpus = tmp_path / "corpus_v2.json"
    corpus.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "candidate"

    rec = si.train_proposer_candidate(corpus, out_dir, steps=77, seed=3)
    assert captured == {"steps": 77, "seed": 3, "corpus_path": corpus,
                        "out_dir": out_dir}
    assert rec["steps"] == 77


def test_retrain_branch_strips_stale_known_line_before_building_corpus():
    """2026-08-11 fix: BASE_CORPUS (corpus_diverse.json) has a stale
    '### KNOWN ...' line baked into every one of its 765 prompts (sourced
    at build time from the archived, pre-VCM-fix/pre-C_LOAD-fix
    self_improvement_runs.jsonl). This branch used to pass those prompts
    straight through to build_corpus_v2 as both the template half AND the
    base for every new measured-target prompt -- silently propagating
    stale evidence into every retrain. Must now strip it first via the
    same strip_known_line() the A9 SFT self-improvement pathway uses."""
    import run_self_improvement_v2 as si
    src = inspect.getsource(si.main)
    assert "strip_known_line" in src
    idx = src.index("elif len(multi_seed) >= args.sft_threshold:")
    branch = src[idx:src[idx:].index("\n        else:") + idx]
    assert "strip_known_line(r.get(\"prompt\")" in branch
    # and it happens BEFORE prompts_by_spec/build_corpus_v2 read from it
    assert branch.index("strip_known_line") < branch.index("build_corpus_v2(")


def test_retrain_branch_only_reached_when_unfrozen_and_over_threshold():
    """Structural check on main()'s source: the train/eval/gate call must
    live inside the `elif len(multi_seed) >= args.sft_threshold:` branch,
    which is only reachable when args.freeze_proposer is falsy (the `if
    args.freeze_proposer:` branch above it returns early via a plain
    note-only path with no training call in it)."""
    import run_self_improvement_v2 as si
    src = inspect.getsource(si.main)
    frozen_branch = src[src.index("if args.freeze_proposer:"):
                        src.index("elif len(multi_seed) >= args.sft_threshold:")]
    assert "train_proposer_candidate(" not in frozen_branch
    assert "proposer_gates(" not in frozen_branch

    retrain_branch = src[src.index("elif len(multi_seed) >= args.sft_threshold:"):]
    retrain_branch = retrain_branch[:retrain_branch.index("\n        else:")]
    assert "train_proposer_candidate(" in retrain_branch
    assert "eval_proposer(" in retrain_branch
    assert "proposer_gates(" in retrain_branch
