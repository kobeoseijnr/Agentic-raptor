"""Explicit SFT self-improvement generation trigger -- A9 adaptive lineage
only (Sections 15/16/17).

    verified NGSPICE success (sft_queue.jsonl, one or more generations of
    run_self_improvement_v2.py's own run/harvest loop)
        -> filter / dedup / balance / quality-tier
           (agentic_raptor.selfimprove_v2.sft_self_improvement.build_dataset)
        -> versioned dataset (artifacts/.../generations/<Gk>/dataset/)
        -> LoRA training (agentic_raptor.llm_dpo.stage3e4.run_sft)
        -> validation: cheap capability probe (Section 12) + a SMALL
           real-SPICE downstream check (Section 13)
        -> promote (write generations/<Gk>/manifest.json,
           adapter_path set, ACTIVE_ADAPTIVE.txt updated) or reject
           (manifest written with rejected=True, candidate adapter kept
           on disk for analysis, previous generation stays active)

The runtime (run_self_improvement_v2.py, in self-improvement mode) collects
sft_queue rows CONTINUOUSLY. Model weights only ever change when THIS
script is invoked explicitly -- nothing here runs off a harvest event, so
every experiment stays reproducible (Section 15).

Refuses immediately, before touching any file or weight, for
--lineage static or anything other than "adaptive" (Section 16/17):
A0-A8 ablation code (run_ablation_v3.py) never imports this module or
script at all -- that is the structural guarantee, verified by
tests/test_sft_self_improvement.py::test_a0_a8_never_imports_sft_training.
This flag is the second, in-process guard for callers that DO reach it.

Usage (small smoke run, matching Section 20 -- do not scale this up
without review):

  python train_sft_self_improvement.py --lineage adaptive \\
      --parent-generation G0 --output-generation G1 \\
      --steps 40 --downstream-eval-specs 2 --budget 12

Real G0 -> G1 production run (once the smoke result is reviewed):

  python train_sft_self_improvement.py --lineage adaptive \\
      --parent-generation G0 --output-generation G1 \\
      --steps 1200 --capability-eval-specs 6 --downstream-eval-specs 4 \\
      --budget 16
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentic_raptor.selfimprove_v2.sft_self_improvement import (
    DEFAULT_PER_FAMILY_CAP, DEFAULT_PER_SPEC_CAP, POST_CLOAD_FIX_V1)

ROOT = Path(__file__).resolve().parent


def collect_queue_rows(si_root: Path, generations: list | None) -> list:
    """Reads sft_queue.jsonl from every gen_NNN directory under si_root (or
    only the listed indices). Deliberately reads NOTHING else -- in
    particular never datasets/simulation_memory/self_improvement_runs.jsonl
    (archived, pre-VCM-fix/pre-C_LOAD-fix; see the module docstring in
    agentic_raptor.selfimprove_v2.sft_self_improvement)."""
    from agentic_raptor.selfimprove_v2.streams import read
    rows = []
    for gd in sorted(si_root.glob("gen_*")):
        try:
            gid = int(gd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if generations is not None and gid not in generations:
            continue
        rows.extend(read(gd / "streams" / "sft_queue.jsonl"))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lineage", default="adaptive", choices=("adaptive", "static"),
                    help="only 'adaptive' may train; 'static' always refuses "
                         "(Section 16/17)")
    ap.add_argument("--parent-generation", default=None,
                    help="e.g. G0, G1, ...; default G0")
    ap.add_argument("--output-generation", default=None,
                    help="default: next after --parent-generation")
    ap.add_argument("--si-root", default=None,
                    help="root containing gen_NNN/streams/sft_queue.jsonl; "
                         "default run_self_improvement_v2.SI_REAL")
    ap.add_argument("--generations", default=None,
                    help="comma list of gen_NNN indices to harvest "
                         "sft_queue from; default ALL")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="conservative continual-SFT rate -- G0 itself was "
                         "trained from scratch at 3e-4; see run_sft()'s "
                         "docstring for why a later generation uses a "
                         "lower rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-family-cap", type=int, default=DEFAULT_PER_FAMILY_CAP)
    ap.add_argument("--per-spec-cap", type=int, default=DEFAULT_PER_SPEC_CAP)
    ap.add_argument("--rag-memory", default=None,
                    help="POST_CLOAD_FIX_V1 RAG memory file for fresh "
                         "'### KNOWN' evidence; default run_raptor_v2."
                         "RAG_MEMORY_V2")
    ap.add_argument("--capability-eval-specs", type=int, default=6,
                    help="train-split specs (excluded: protected + this "
                         "generation's own harvest range) for the cheap, "
                         "no-SPICE Section 12 capability probe")
    ap.add_argument("--downstream-eval-specs", type=int, default=6,
                    help="Section 13's real-SPICE check -- kept SMALL "
                         "deliberately (Section 20): genuinely SPICE-costly")
    ap.add_argument("--budget", type=int, default=32,
                    help="per-branch sizing budget for the downstream check; "
                         "32 == the production ablation budget (2026-08-22)")
    ap.add_argument("--target-k", type=int, default=5)
    ap.add_argument("--ranker-ckpt", default=None)
    ap.add_argument("--value-ckpt", default=None)
    ap.add_argument("--max-relative-capability-regression", type=float,
                    default=0.10)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report the dataset, then stop before "
                         "training touches any weights")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an already-PROMOTED "
                         "--output-generation (refused by default -- "
                         "choose a new generation id instead)")
    ap.add_argument("--profile", choices=("stock", "tier2"), default="stock",
                    help="tier2: train the tier-2 lineage's proposer chain "
                         "(G0 = production tier-2 adapter + mixed corpus); "
                         "downstream check on tier2_train specs (2026-08-22)")
    args = ap.parse_args()

    from agentic_raptor.selfimprove_v2 import sft_self_improvement as si
    if args.profile == "tier2":
        import run_a9_generations as a9
        prof = a9.PROFILES["tier2"]
        cfg = si.configure_profile(
            prof["sft_generations_root"], prof["adapter"],
            (ROOT / "artifacts/publication_v3/tier2/corpus_tier2_mixed.json"))
        print(f"profile tier2: {cfg}", flush=True)
        if not args.si_root:
            args.si_root = str(prof["root"] / "data" / args.lineage)

    # ---- Section 16/17: the in-process lineage guard, before ANYTHING
    # else runs (no dataset build, no file write, no model load).
    si.assert_lineage_may_train(args.lineage)

    si.ensure_g0_manifest()
    parent_gen = args.parent_generation or "G0"
    parent = si.read_generation_manifest(parent_gen)
    if parent is None:
        raise SystemExit(f"parent generation {parent_gen!r} has no "
                         "manifest -- train it first or use G0")
    output_gen = args.output_generation or si.next_generation_id(parent_gen)
    if output_gen == "G0":
        raise SystemExit("refusing to train into G0 -- G0 is immutable")
    existing_output = si.read_generation_manifest(output_gen)
    if (existing_output is not None and not existing_output.get("rejected")
            and not args.force):
        raise SystemExit(
            f"{output_gen} already exists and was PROMOTED -- pass --force "
            "to overwrite, or choose a different --output-generation")

    if args.si_root:
        si_root = Path(args.si_root)
    else:
        import run_self_improvement_v2 as rsi
        si_root = rsi.SI_REAL
    gens = ([int(x) for x in args.generations.split(",")]
           if args.generations else None)
    # 2026-08-22: the A9 data root is LINEAGE-SCOPED (si_root/<lineage>/
    # gen_NNN). A root with no gen_* of its own but a <lineage>/gen_* child
    # resolves to that child; a root with no generations at all is a hard
    # error -- the earlier silent "no eligible examples" masked a wrong
    # --si-root and skipped a real 36-row harvest.
    if not list(si_root.glob("gen_*")) and list((si_root / args.lineage).glob("gen_*")):
        si_root = si_root / args.lineage
        print(f"si-root resolved to lineage directory: {si_root}", flush=True)
    if not list(si_root.glob("gen_*")):
        raise SystemExit(f"ERROR: no gen_* directories under {si_root} -- "
                         f"pass --si-root <a9_root>/data/{args.lineage}")
    queue_rows = collect_queue_rows(si_root, gens)
    print(f"harvested {len(queue_rows)} raw sft_queue rows from {si_root} "
         f"(generations={gens or 'all'})", flush=True)

    from agentic_raptor.publication.eval_sets import (
        excluded_context_ids, excluded_evaluation_context_ids)
    protected_ids = set(excluded_context_ids())
    protected_eval_ids = set(excluded_evaluation_context_ids())

    parent_adapter = parent.get("adapter_path")
    base_corpus_path = Path(parent.get("training_dataset_path") or si.BASE_CORPUS)
    base_corpus = json.loads(base_corpus_path.read_text(encoding="utf-8"))

    dataset = si.build_dataset(
        queue_rows, base_corpus, generation_id=output_gen,
        protected_context_ids=protected_ids,
        protected_evaluation_context_ids=protected_eval_ids,
        rag_memory_path=args.rag_memory,
        per_family_cap=args.per_family_cap, per_spec_cap=args.per_spec_cap,
        proposer_checkpoint=parent_adapter, adapter_generation=parent_gen)
    print(json.dumps(dataset["manifest"], indent=1, default=str), flush=True)

    gen_dir = si.generation_dir(output_gen)
    written = si.write_versioned_dataset(dataset, gen_dir / "dataset")
    print(f"dataset written: {written['corpus_path']} "
         f"(hash {written['corpus_hash']})", flush=True)

    if dataset["manifest"]["new_verified_count"] == 0:
        print("no eligible new verified examples in this harvest range -- "
             "nothing to train on; stopping before touching any weights")
        return

    if args.dry_run:
        print("--dry-run: stopping before training")
        return

    # ---- train --------------------------------------------------------
    from agentic_raptor.llm_dpo.stage3e4 import run_sft
    adapter_out = gen_dir / "adapter"
    train_rec = run_sft(steps=args.steps, seed=args.seed,
                        corpus_path=written["corpus_path"],
                        out_dir=adapter_out, lr=args.lr)
    print(f"trained {output_gen}: {train_rec['steps']} steps, "
         f"loss {train_rec['loss_first_last']}, "
         f"{train_rec['wall_clock_s']}s", flush=True)

    # ---- validate: Section 12, cheap capability probe -------------------
    from agentic_raptor.publication.spec_registry import build as build_registry
    reg = build_registry()
    # harvest keys are (split, spec_index): indices collide across splits
    harvested_keys = {(r.get("split") or "train", r.get("spec_index"))
                      for r in queue_rows if r.get("spec_index") is not None}
    harvested_idxs = {ix for sp, ix in harvested_keys if sp == "train"}
    cap_entries = [e for e in reg["entries"] if e["split"] == "train"
                  and e["context_id"] not in protected_ids and e["parsed_spec"]
                  and e["spec_index"] not in harvested_idxs]
    cap_entries = sorted(cap_entries, key=lambda e: e["spec_index"]
                         )[:args.capability_eval_specs]
    prompts_by_spec = {r["context_id"]: r["prompt"]
                       for r in base_corpus.get("records") or []}
    eval_items = []
    for e in cap_entries:
        bp = prompts_by_spec.get(e["context_id"])
        if not bp:
            continue
        eval_items.append({
            "spec": e["parsed_spec"], "spec_hash": e["spec_hash"],
            "prompt": si.build_clean_prompt(e["parsed_spec"], bp,
                                            rag_memory_path=args.rag_memory)})
    known_hashes = {r.get("variant_hash") for r in base_corpus.get("records") or []
                    if r.get("variant_hash")}
    print(f"capability probe on {len(eval_items)} specs held out from this "
         "harvest range...", flush=True)
    cand_cap = si.capability_probe(str(adapter_out), eval_items,
                                   target_k=args.target_k, seed=args.seed,
                                   known_family_hashes=known_hashes)
    parent_cap = si.capability_probe(parent_adapter, eval_items,
                                     target_k=args.target_k, seed=args.seed,
                                     known_family_hashes=known_hashes)
    cap_gate = si.capability_gate(
        cand_cap, parent_cap,
        max_relative_regression=args.max_relative_capability_regression)
    print(f"capability gate: {'PASS' if cap_gate['passed'] else 'FAIL'} "
         f"{cap_gate['failures']}", flush=True)

    # ---- validate: Section 13, small real-SPICE downstream check --------
    downstream_gate = {"passed": True, "failures": [], "skipped": True}
    cand_downstream = parent_downstream = None
    # 2026-08-22: downstream specs are STRATIFIED across the eligible list
    # (evenly spaced by spec_index) instead of "the first N" -- the first N
    # were the lowest-index, easiest specs, all at pass-rate ceiling, so the
    # gate could not see improvement. Stratification spans the difficulty
    # range of the harvest-disjoint TRAIN specs.
    all_elig = sorted([e for e in reg["entries"] if e["split"] == "train"
                       and e["context_id"] not in protected_ids and e["parsed_spec"]
                       and e["spec_index"] not in harvested_idxs],
                      key=lambda e: e["spec_index"])
    n_ds = min(args.downstream_eval_specs, len(all_elig))
    stride = max(1, len(all_elig) // max(1, n_ds))
    downstream_idxs = [all_elig[i * stride]["spec_index"] for i in range(n_ds)]
    downstream_split = "train"
    if args.profile == "tier2":
        # the tier-2 profile's signal lives on tier2_train specs: evaluate
        # downstream on harvest-disjoint tier2_train indices (the sealed
        # tier2_heldout exam is never used here)
        from agentic_raptor.publication.tier2 import load_tier2_specs
        t2_harvested = {ix for sp, ix in harvested_keys if sp == "tier2_train"}
        t2_idx = [i for i in range(len(load_tier2_specs("tier2_train")))
                  if i not in t2_harvested]
        n_t2 = min(args.downstream_eval_specs, len(t2_idx))
        stride2 = max(1, len(t2_idx) // max(1, n_t2))
        downstream_idxs = [t2_idx[i * stride2] for i in range(n_t2)]
        downstream_split = "tier2_train"
        print(f"profile tier2: downstream check on tier2_train indices "
              f"{downstream_idxs} (harvest-disjoint)", flush=True)
    if downstream_idxs:
        from run_self_improvement_v2 import eval_proposer
        from agentic_raptor.selfimprove_v2 import proposer_gates
        import run_raptor_v2 as v2
        print(f"downstream real-SPICE check on {len(downstream_idxs)} specs "
             f"(budget={args.budget}) -- genuinely SPICE-costly", flush=True)
        common = dict(
            eval_idxs=downstream_idxs, split=downstream_split,
            ranker_ckpt=args.ranker_ckpt, value_ckpt=args.value_ckpt,
            rag_memory=(args.rag_memory or str(v2.RAG_MEMORY_V2)),
            budget=args.budget, target_k=args.target_k, seed=args.seed)
        cand_downstream = eval_proposer(str(adapter_out), **common)
        parent_downstream = eval_proposer(parent_adapter, **common)
        pgr = proposer_gates(cand_downstream, parent_downstream)
        downstream_gate = {**pgr.as_dict(), "skipped": False}
        print(f"downstream proposer_gates: "
             f"{'PASS' if pgr.passed else 'FAIL'} {pgr.failures}", flush=True)

    promoted = cap_gate["passed"] and downstream_gate["passed"]

    manifest = {
        "generation_id": output_gen, "parent_generation": parent_gen,
        "base_model": train_rec.get("model_id"),
        "source_adapter": parent_adapter,
        "adapter_path": str(adapter_out) if promoted else None,
        "candidate_adapter_path": str(adapter_out),
        "training_dataset_path": str(written["corpus_path"]),
        "training_dataset_hash": written["corpus_hash"],
        "n_examples": dataset["manifest"]["total_records"],
        "structural_replay_count": dataset["manifest"]["structural_replay_count"],
        "new_verified_count": dataset["manifest"]["new_verified_count"],
        "family_distribution":
            dataset["manifest"]["balance_policy"]["family_distribution_after"],
        "spec_distribution":
            dataset["manifest"]["balance_policy"]["spec_distribution_after"],
        "quality_tier_counts": dataset["manifest"]["quality_tier_counts"],
        "dedup": dataset["manifest"]["dedup"],
        "training_hyperparameters": {
            "steps": args.steps, "lr": args.lr, "lora_r": 8, "lora_alpha": 16,
            "lora_dropout": 0.05, "effective_batch_size": 1,
            "optimizer": "AdamW"},
        "seed": args.seed,
        "electrical_environment_version": POST_CLOAD_FIX_V1,
        "checkpoint_hash": None,
        "training_record": train_rec,
        "capability_probe": {"candidate": cand_cap, "parent": parent_cap,
                             "gate": cap_gate},
        "downstream_probe": {"candidate": cand_downstream,
                             "parent": parent_downstream,
                             "gate": downstream_gate},
        "dataset_manifest_path": str(written["manifest_path"]),
        "rejected": not promoted,
        "immutable": False,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    if promoted:
        from agentic_raptor.ranking.types import directory_sha256
        manifest["checkpoint_hash"] = directory_sha256(str(adapter_out))
        si.write_generation_manifest(output_gen, manifest)
        (si.GENERATIONS_ROOT / "ACTIVE_ADAPTIVE.txt").write_text(
            output_gen, encoding="utf-8")
        print(f"PROMOTED: {output_gen} is now the active adaptive-lineage "
             "SFT generation", flush=True)
    else:
        manifest["rejection_reasons"] = (cap_gate["failures"]
                                         + downstream_gate.get("failures", []))
        si.write_generation_manifest(output_gen, manifest)
        print(f"REJECTED: {output_gen} failed validation "
             f"({manifest['rejection_reasons']}) -- {parent_gen} remains "
             f"active. Candidate preserved at {adapter_out} for analysis.",
             flush=True)


if __name__ == "__main__":
    main()
