"""Ablation over the CANONICAL v2 pipeline (run_raptor_v2.run_pipeline).

This is NOT run_ablation.py. That script drives run_self_improvement.py,
which uses the old per-candidate PUCT and never calls the exclusion-
conditioned proposer, the one-root search or the post-SAC ranker. Results
from the two must never be pooled, and only these may be described as
ablations of the v2 architecture.

Arms -- each removes exactly ONE repaired component, so its contribution is
attributable:

  full            production configuration
  no_exclusion    proposer samples by temperature only (no "propose a
                  DIFFERENT one" conditioning). Isolates the corpus repair.
  no_search       PUCT removed; the two designs are chosen by structural
                  prior alone. Isolates the one-root search.
  baseline_ranker deterministic scorer instead of the trained DPO ranker.
                  Isolates the learned ranker.

Evaluated on the HELDOUT split. The ranker was trained on train-split pairs
only, so heldout is untouched by its fitting. blindtest stays frozen.

The model is loaded ONCE and reused across every run: reloading a 4B model
per run costs more wall-clock than the pipeline it wraps.

Run:  python run_ablation_v2.py --specs 6 --seeds 1
      python run_ablation_v2.py --arms full,no_search --specs 4
"""
import argparse
import csv
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v2/ablation_v2"
ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"

ARMS = {
    "full":            dict(conditioning="exclusion", search="one_root",
                            ranker_mode="dpo"),
    "no_exclusion":    dict(conditioning="temperature", search="one_root",
                            ranker_mode="dpo"),
    "no_search":       dict(conditioning="exclusion", search="none",
                            ranker_mode="dpo"),
    "baseline_ranker": dict(conditioning="exclusion", search="one_root",
                            ranker_mode="baseline"),
}


def row_from_trace(t: dict, arm: str, spec_index: int, seed: int,
                   seconds: float) -> dict:
    s3 = t.get("stage3_propose") or {}
    # 2026-08-11: run_pipeline's trace key is now "stage5_alphazero" (the
    # retired root-PUCT selector was replaced) -- "stage5_puct" is read as
    # a fallback so historical trace files produced before the migration
    # still parse.
    s5 = t.get("stage5_alphazero") or t.get("stage5_puct") or {}
    s8 = t.get("stage8_ranker") or {}
    s9 = t.get("stage9_verification") or {}
    s11 = t.get("stage11_feedback") or {}
    return {
        "arm": arm, "spec_index": spec_index, "seed": seed,
        "spec_id": (t.get("stage1_spec") or {}).get("spec_id"),
        "result": t.get("result"),
        "distinct_proposals": s3.get("distinct"),
        "distinct_families": s3.get("distinct_family_count"),
        "proposal_attempts": s3.get("attempts"),
        "reached_target_k": s3.get("reached_target_k"),
        "single_root": s5.get("single_root"),
        "distinct_visit_counts": s5.get("distinct_visit_counts"),
        "ranker_arm": s8.get("ranker_arm"),
        "decision_basis": s8.get("decision_basis"),
        "deciding_level": s8.get("deciding_level"),
        "low_confidence": s8.get("low_confidence"),
        # the outcome that matters: did the SELECTED design meet the spec
        "selected_exact_pass": s9.get("selected_exact_pass"),
        "selected_failure_reason": s9.get("selected_failure_reason"),
        "sizing_spice_calls": s9.get("sizing_spice_calls"),
        "verification_spice_calls": s9.get("verification_spice_calls"),
        "ranker_correct": s11.get("ranker_correct"),
        "pair_status": s11.get("ranker_pair_status"),
        "seconds": round(seconds, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--specs", type=int, default=6,
                    help="how many heldout spec indices to evaluate")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=2,
                    help="consecutive corpus rows often share a context_id")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--sizing-repeats", type=int, default=1,
                    help="size each selected branch this many times (best "
                         "attempt kept) -- see run_raptor_v2.py for why. "
                         "Default 1 keeps existing behaviour; costs roughly "
                         "N x the sizing time per run")
    ap.add_argument("--adapter", default=str(ADAPTER))
    ap.add_argument("--calibrate", action="store_true",
                    help="verify BOTH designs every run (slower; only needed "
                         "to grow ranker training data, which must come from "
                         "the TRAIN split anyway)")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a in ARMS]
    if not arms:
        raise SystemExit(f"no valid arms in {args.arms!r}; choose from "
                         f"{sorted(ARMS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    idxs = [args.start + i * args.step for i in range(args.specs)]

    import run_raptor_v2 as v2
    from run_qwen_ablation import _load

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"not a peft adapter directory: {adapter}")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"arms   : {arms}")
    print(f"specs  : {idxs} on '{args.split}'")
    print(f"seeds  : {seeds}")
    print(f"total  : {len(arms)*len(idxs)*len(seeds)} runs\n", flush=True)

    tok, model = _load(str(adapter))      # ONCE
    rows, n, total = [], 0, len(arms) * len(idxs) * len(seeds)
    t_all = time.time()
    for seed in seeds:
        for idx in idxs:
            for arm in arms:
                n += 1
                cfg = ARMS[arm]
                t0 = time.time()
                try:
                    tr = v2.run_pipeline(
                        model, tok, str(adapter), split=args.split,
                        spec_index=idx, budget=args.budget,
                        calibrate=args.calibrate, seed=seed,
                        sizing_repeats=args.sizing_repeats,
                        out_prefix=f"ABL_{arm}_s{seed}", **cfg)
                    row = row_from_trace(tr, arm, idx, seed, time.time() - t0)
                except Exception as exc:
                    # one spec failing (e.g. proposer returns a single graph)
                    # must not lose the batch; record WHY and continue
                    row = {"arm": arm, "spec_index": idx, "seed": seed,
                           "result": f"ERROR: {type(exc).__name__}: "
                                     f"{str(exc)[:120]}",
                           "seconds": round(time.time() - t0, 1)}
                    (OUT / "errors.log").open("a", encoding="utf-8").write(
                        f"\n=== {arm} idx={idx} seed={seed}\n"
                        + traceback.format_exc())
                rows.append(row)
                print(f"[{n:>3}/{total}] {arm:<16} idx={idx:<3} seed={seed} "
                      f"pass={row.get('selected_exact_pass')} "
                      f"distinct={row.get('distinct_proposals')} "
                      f"basis={row.get('decision_basis')} "
                      f"{row['seconds']:>6.0f}s "
                      f"{row['result'] if str(row.get('result','')).startswith('ERROR') else ''}",
                      flush=True)

    # ---------------- aggregate ------------------------------------------
    def agg(arm):
        rs = [r for r in rows if r["arm"] == arm and r.get("result") == "OK"]
        if not rs:
            return {"runs": 0}
        def mean(k):
            vs = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            return round(sum(vs) / len(vs), 3) if vs else None
        passes = [r for r in rs if r.get("selected_exact_pass")]
        return {"runs": len(rs),
                "errors": len([r for r in rows if r["arm"] == arm
                               and r.get("result") != "OK"]),
                "spec_pass_rate": round(len(passes) / len(rs), 3),
                "mean_distinct_proposals": mean("distinct_proposals"),
                "mean_distinct_families": mean("distinct_families"),
                "mean_proposal_attempts": mean("proposal_attempts"),
                "mean_distinct_visit_counts": mean("distinct_visit_counts"),
                "decided_at_learned_level": sum(
                    1 for r in rs if r.get("deciding_level") == "learned"),
                "mean_seconds": mean("seconds")}

    summary = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "pipeline": "run_raptor_v2.run_pipeline (canonical v2)",
               "note": ("NOT comparable with run_ablation.py results, which "
                        "drive the older self-improvement architecture"),
               "split": args.split, "budget": args.budget,
               "adapter": str(adapter), "seeds": seeds, "spec_indices": idxs,
               "arms": {a: agg(a) for a in arms},
               "wall_clock_min": round((time.time() - t_all) / 60, 1)}
    (OUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with (OUT / "runs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    print(f"\n{'arm':<18}{'runs':>5}{'pass':>7}{'distinct':>10}"
          f"{'families':>10}{'learned':>9}")
    for a in arms:
        s = summary["arms"][a]
        if not s.get("runs"):
            print(f"{a:<18}{0:>5}      -         -         -        -")
            continue
        print(f"{a:<18}{s['runs']:>5}{s['spec_pass_rate']:>7.2f}"
              f"{s['mean_distinct_proposals']:>10}"
              f"{s['mean_distinct_families']:>10}"
              f"{s['decided_at_learned_level']:>9}")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
