"""Bootstrap the post-SAC DPO ranker: collect trusted pairs, then train.

A preference model has to be fitted to MEASURED preferences. There is no way
to skip the collection step: pairs are only trusted when BOTH designs were
verified by ngspice with complete provenance, matching spec ids and distinct
call ids. So the order is fixed --

    calibration runs  ->  trusted pairs  ->  train  ->  stage 8 is DPO

Draws specs from the TRAIN split. 27 of 29 heldout specs sit in the protected
evaluation set and their pairs are (correctly) refused as training data;
training on heldout would contaminate evaluation anyway.

Consecutive spec indices in the corpus are often the SAME context_id, so the
default step is 2 to spread across distinct specifications.

Each run is independent: one failure (e.g. the proposer returning a single
graph) is recorded and skipped rather than aborting the batch.

Run:  python run_ranker_bootstrap.py --runs 12 [--budget 16] [--skip-train]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"


def n_trusted() -> int:
    from agentic_raptor.ranking.post_sac import trusted_pairs
    return len([p for p in trusted_pairs() if p.get("status") == "trusted"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=2,
                    help="consecutive indices are usually the same context_id")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--split", default="train")
    ap.add_argument("--adapter", default=str(ADAPTER))
    ap.add_argument("--min-pairs", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    start_pairs = n_trusted()
    print(f"trusted pairs before : {start_pairs}")
    print(f"plan                 : {args.runs} runs on '{args.split}', "
          f"indices {args.start}..{args.start + (args.runs-1)*args.step} "
          f"step {args.step}, budget {args.budget}\n", flush=True)

    log = []
    t0 = time.time()
    for n in range(args.runs):
        idx = args.start + n * args.step
        t1 = time.time()
        proc = subprocess.run(
            [py, str(ROOT / "run_raptor_v2.py"),
             "--split", args.split, "--spec-index", str(idx),
             "--adapter", args.adapter, "--calibrate",
             "--budget", str(args.budget)],
            capture_output=True, text=True, cwd=str(ROOT))
        have = n_trusted()
        ok = proc.returncode == 0
        # the last line of stderr carries the ArchitectureViolation message
        why = ""
        if not ok:
            tail = [x for x in (proc.stderr or "").strip().splitlines() if x]
            why = tail[-1][:110] if tail else f"exit {proc.returncode}"
        log.append({"index": idx, "ok": ok, "trusted_total": have,
                    "seconds": round(time.time() - t1, 1), "note": why})
        print(f"[{n+1:>2}/{args.runs}] idx={idx:<3} "
              f"{'OK ' if ok else 'FAIL'} "
              f"trusted={have:<3} {log[-1]['seconds']:>6.0f}s {why}",
              flush=True)

    total = n_trusted()
    print(f"\ncollected {total - start_pairs} new trusted pairs "
          f"({total} total) in {(time.time()-t0)/60:.0f} min")
    (ROOT / "artifacts/publication_v2/post_sac_ranker").mkdir(
        parents=True, exist_ok=True)
    (ROOT / "artifacts/publication_v2/post_sac_ranker"
     / "bootstrap_log.json").write_text(
        json.dumps({"runs": log, "trusted_before": start_pairs,
                    "trusted_after": total}, indent=1), encoding="utf-8")

    if args.skip_train:
        return
    if total < args.min_pairs:
        print(f"\nNOT training: {total} < {args.min_pairs} trusted pairs.\n"
              f"Run more:  python run_ranker_bootstrap.py --runs 8 "
              f"--start {args.start + args.runs*args.step}")
        return
    print("\n--- training the post-SAC ranker ---", flush=True)
    subprocess.run([py, str(ROOT / "train_post_sac_ranker.py"),
                    "--min-pairs", str(args.min_pairs)], cwd=str(ROOT))


if __name__ == "__main__":
    main()
