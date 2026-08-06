"""Can the proposer emit >= 5 DISTINCT topologies per specification?

The intended architecture is: LLM proposes 5 candidates, PUCT scores them and
forwards the top 2 to sizing. That only works if the proposer can actually
cover the design space for a single spec.

Measured problem: at the production setting (4 samples, T=0.8, top_p=0.92) a
converged proposer emits ONE structure class per spec -- 40 samples across a
generation collapsed to 3 unique structures overall. Preference training
sharpened the distribution to a point where fixed-temperature sampling cannot
surface alternatives.

This measures two things on real specs:

  1. BASELINE  -- the production sampler, as a reference point;
  2. LADDER    -- propose_diverse(), which escalates temperature only until
                  `target_k` distinct valid structures are found.

Everything counted is the model's own output. If the ladder cannot reach 5,
that is the finding: the proposer cannot enumerate the space, and the
architecture must either fix decoding or admit the pool is code-supplied.

Run:  python check_diversity.py [--specs 6] [--target 5] [--arm L8]
"""
import argparse
import json
import time

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.publication import PUB, ROOT

OUT = PUB.parent / "publication_v2" / "diversity"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=int, default=6)
    ap.add_argument("--target", type=int, default=5)
    ap.add_argument("--arm", default="L8", help="checkpoint arm to probe")
    ap.add_argument("--baseline-k", type=int, default=4,
                    help="samples for the production-setting baseline")
    args = ap.parse_args()
    import torch
    from run_qwen_ablation import _load, resolve_arms
    from agentic_raptor.llm_dpo.stage3e4 import (generate, propose_diverse,
                                                 variant_hash)
    adapter = resolve_arms()[args.arm]["adapter"]
    assert adapter, f"{args.arm} checkpoint missing"
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    # train split: measuring the proposer, not evaluating a checkpoint
    specs = [r for r in corpus["records"] if r["split"] == "train"][:args.specs]
    tok, model = _load(adapter)

    rows = []
    for r in specs:
        t0 = time.time()
        base = set()
        for s in range(args.baseline_k):
            c = generate(model, tok, r["prompt"], sample_seed=s)
            if c["valid"]:
                base.add(c["graph_hash"])
        d = propose_diverse(model, tok, r["prompt"], target_k=args.target)
        classes = sorted({f"{len(c['obj']['stages'])}s_"
                          + ig.compensation_class(c["obj"])
                          for c in d["candidates"]})
        row = {"context_id": r["context_id"],
               "baseline_distinct": len(base),
               "ladder_distinct": d["distinct"],
               "reached_target": d["reached_target"],
               "attempts": d["attempts"],
               "max_temperature": d["max_temperature"],
               "classes": classes,
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"{r['context_id'][:26]:26} baseline={row['baseline_distinct']} "
              f"ladder={row['ladder_distinct']}/{args.target} "
              f"attempts={row['attempts']} Tmax={row['max_temperature']} "
              f"{classes}")
    del model
    torch.cuda.empty_cache()

    n = len(rows)
    reached = sum(1 for r in rows if r["reached_target"])
    doc = {"arm": args.arm, "specs": n, "target_k": args.target,
           "baseline_mean_distinct": round(
               sum(r["baseline_distinct"] for r in rows) / n, 2),
           "ladder_mean_distinct": round(
               sum(r["ladder_distinct"] for r in rows) / n, 2),
           "specs_reaching_target": reached,
           "share_reaching_target": round(reached / n, 3),
           "mean_attempts": round(sum(r["attempts"] for r in rows) / n, 1),
           "verdict": ("proposer CAN supply the candidate pool"
                       if reached == n else
                       "proposer CANNOT reliably reach the target -- the "
                       "pool is not model-supplied and must not be described "
                       "as such"),
           "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(doc, indent=1),
                                      encoding="utf-8")
    print(f"\nbaseline mean distinct : {doc['baseline_mean_distinct']}")
    print(f"ladder   mean distinct : {doc['ladder_mean_distinct']}")
    print(f"specs reaching {args.target}      : {reached}/{n}")
    print(f"mean attempts          : {doc['mean_attempts']}")
    print(f"\nVERDICT: {doc['verdict']}")


if __name__ == "__main__":
    main()
