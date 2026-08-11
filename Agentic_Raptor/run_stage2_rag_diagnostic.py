"""Stage 2 diagnostic: does RAG improve topology proposal quality BEFORE
downstream PUCT/SAC/DPO ever runs?

Deliberately stops at the proposer/validator boundary -- rag_stage() +
propose_and_validate() only, exactly as instructed ("Do NOT invoke PUCT, SAC
or DPO to determine the Stage-2 conclusion"). No SPICE calls happen here at
all, so this is cheap to run at a real diagnostic scale (unlike the SPICE-
bound stages).

For every spec/seed pair, runs WITH_RAG and NO_RAG under otherwise IDENTICAL
conditions (same spec, same seed0, same SFT checkpoint, same target_k,
same validator, same conditioning="exclusion") and records everything the
Stage 2 spec asks for: retrieved record identity/rank/family/spec/
performance, Valid@K, Unique@K, family diversity, graph-distance diversity,
parser/schema success, and whether the generated candidate SET differs
between the two conditions.

Run:  python run_stage2_rag_diagnostic.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v3/stage2_rag_diagnostic"
DEFAULT_ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"


def graph_distance(obj_a: dict, obj_b: dict) -> int:
    """Coarse structural distance: number of differing (stages, comp, buf,
    fb) attributes -- 0..4. Not a full graph-edit distance, but real and
    cheap, and enough to separate "same family" from "different family"
    candidates the way canonical_family alone (2 families -> "different")
    understates."""
    from agentic_raptor.llm_dpo.integrity import compensation_class
    sa = len(obj_a.get("stages", [])); sb = len(obj_b.get("stages", []))
    ca = compensation_class(obj_a); cb = compensation_class(obj_b)
    ba = bool(obj_a.get("output_buffer")); bb = bool(obj_b.get("output_buffer"))
    fa = bool(obj_a.get("local_feedback")); fb = bool(obj_b.get("local_feedback"))
    return int(sa != sb) + int(ca != cb) + int(ba != bb) + int(fa != fb)


def run_condition(model, tok, adapter_str, rag_prompt, spec, seed, target_k=5):
    from run_raptor_v2 import propose_and_validate
    t0 = time.time()
    prop = propose_and_validate(model, tok, rag_prompt, target_k=target_k,
                                conditioning="exclusion", seed0=seed,
                                use_llm=True, spec=spec)
    seconds = round(time.time() - t0, 1)
    cands = prop["candidates"]
    families = [c["canonical_family"] for c in cands]
    hashes = [c["canonical_graph_hash"] for c in cands]
    # pairwise graph distance among the returned set -- mean and min, since
    # "5 distinct hashes" can still be superficial if they're all 1-apart
    dists = [graph_distance(cands[i]["obj"], cands[j]["obj"])
            for i in range(len(cands)) for j in range(i + 1, len(cands))]
    return {
        "seconds": seconds,
        "attempts": prop["attempts"], "max_temperature": prop["max_temperature"],
        "distinct": prop["distinct"], "target_k": prop["target_k"],
        "select_k": prop["select_k"],
        "candidate_generation_status": prop["candidate_generation_status"],
        "valid_at_k": prop["distinct"] / target_k,     # every returned candidate already parsed/validated
        "unique_at_k": len(set(hashes)) / max(1, len(hashes)),
        "family_diversity_count": len(set(families)),
        "families": families, "hashes": hashes,
        "mean_pairwise_graph_distance": (sum(dists) / len(dists)) if dists else None,
        "min_pairwise_graph_distance": min(dists) if dists else None,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=int, default=6,
                    help="diagnostic specs, 2 per difficulty tier")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--split", default="train",
                    help="train (default) -- heldout has only 2 unprotected "
                         "specs total (27/29 consumed by the frozen "
                         "evaluation set), too few to stratify by tier. "
                         "train has zero overlap with protected eval specs "
                         "(verified: disjoint context_id sets) and 85 "
                         "unprotected specs evenly spread across tiers.")
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    from agentic_raptor.publication.spec_registry import build as build_registry
    from agentic_raptor.publication.eval_sets import excluded_context_ids
    from run_qwen_ablation import _load
    from run_raptor_v2 import rag_stage
    from agentic_raptor.publication.preflight import CLEAN_RAG_PATH

    protected = excluded_context_ids()
    reg = build_registry()
    entries = [e for e in reg["entries"] if e["split"] == args.split
              and e["context_id"] not in protected and e["parsed_spec"]]
    by_tier: dict[str, list] = {}
    for e in entries:
        by_tier.setdefault(e["difficulty_tier"], []).append(e)
    per_tier = max(1, args.specs // 3)
    diag = []
    for tier in ("easy", "medium", "hard"):
        pool = sorted(by_tier.get(tier, []), key=lambda e: e["spec_index"])
        diag.extend(pool[:per_tier])
    # Safety fallback: if a tier's unprotected pool was too small (this is
    # EXACTLY what silently produced a 2-spec, single-tier diagnostic set
    # from split="heldout" -- 27/29 records were protected), top up from
    # whichever tier has spare unprotected specs rather than quietly
    # running a narrower, unstratified comparison than requested.
    if len(diag) < args.specs:
        used = {e["spec_hash"] for e in diag}
        spare = sorted((e for e in entries if e["spec_hash"] not in used),
                       key=lambda e: e["spec_index"])
        diag.extend(spare[:args.specs - len(diag)])
    diag = diag[:args.specs]
    if args.num_workers > 1:
        diag = [e for i, e in enumerate(diag) if i % args.num_workers == args.worker_id]
    print(f"diagnostic specs ({len(diag)}, unprotected, split={args.split!r}, "
         f"worker {args.worker_id}/{args.num_workers}):")
    for e in diag:
        print(f"  idx={e['spec_index']:3} tier={e['difficulty_tier']:6} "
             f"hash={e['spec_hash']} {e['context_id']}")
    print()

    import json as _json
    corpus = _json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    pool_recs = [r for r in corpus["records"] if r["split"] == args.split]

    import torch
    torch.set_num_threads(max(1, 32 // max(1, args.num_workers)))
    adapter = Path(args.adapter)
    tok, model = _load(str(adapter))
    print("model loaded\n", flush=True)

    out_file = (OUT / f"results_w{args.worker_id}.jsonl" if args.num_workers > 1
               else OUT / "results.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for e in diag:
        idx = e["spec_index"]
        rec = pool_recs[idx]
        spec = e["parsed_spec"]
        spec = dict(spec, spec_id=e["context_id"], spec_hash=e["spec_hash"],
                   topology_id=rec.get("topology_id"))
        for seed in seeds:
            t0 = time.time()
            rag = rag_stage(spec, rec["prompt"], use_rag=True,
                            memory_path=str(CLEAN_RAG_PATH))
            no_rag = rag_stage(spec, rec["prompt"], use_rag=False)
            with_rag = run_condition(model, tok, str(adapter), rag["prompt"], spec, seed)
            no_rag_res = run_condition(model, tok, str(adapter), no_rag["prompt"], spec, seed)
            same_set = set(with_rag["hashes"]) == set(no_rag_res["hashes"])
            row = {
                "spec_index": idx, "spec_hash": e["spec_hash"],
                "context_id": e["context_id"], "difficulty_tier": e["difficulty_tier"],
                "seed": seed,
                "retrieval": {
                    "retrieved_ids": rag["retrieval_ids"],
                    "n_records": len(rag["records"]),
                    "records": rag["records"],   # id, line, stages, pm, gain_db, outcome, rank-order preserved
                },
                "with_rag": with_rag, "no_rag": no_rag_res,
                "candidate_set_changed_by_rag": not same_set,
                "elapsed_s": round(time.time() - t0, 1)}
            results.append(row)
            out_file.open("a", encoding="utf-8").write(
                json.dumps(row, default=str) + "\n")
            print(f"idx={idx:3} tier={e['difficulty_tier']:6} seed={seed} "
                 f"retrieved={len(rag['records'])} "
                 f"WITH_RAG(distinct={with_rag['distinct']},fam={with_rag['family_diversity_count']}) "
                 f"NO_RAG(distinct={no_rag_res['distinct']},fam={no_rag_res['family_diversity_count']}) "
                 f"changed={row['candidate_set_changed_by_rag']} "
                 f"{row['elapsed_s']:.0f}s", flush=True)

    print(f"\ndone: {len(results)} paired comparisons -> {out_file}")


if __name__ == "__main__":
    main()
