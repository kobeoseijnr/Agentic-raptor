"""A0-A8 component-ablation driver (paired evaluation, frozen learning_mode).

NOT the A9 self-improvement study -- that is a separate longitudinal design
driven by agentic_raptor/publication/generation_state.py, never mixed into
this A0-A8 paired sweep (Part: IMPORTANT EXPERIMENTAL SEPARATION).

Every arm shares the SAME ExperimentBudget and the SAME (spec_id, seed)
draws (Part: PAIRED EVALUATION / FAIRNESS) -- the loop nests seed -> spec ->
arm so all arms run back-to-back against an identical spec/seed pair.

Default seeds are exactly [0, 1, 2] -- the agreed pilot: do not expand to 5
seeds until pilot results and variance have been reviewed.

Run:
    python run_ablation_v3.py --arms A0,A5,A8 --specs 3 --seeds 0,1,2
    python run_ablation_v3.py --paper-mode          # preflight only, refuses
                                                     # to run if blocked
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v3/ablation_v3"
DEFAULT_ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"


def _proposer_requirement(cfg) -> str:
    """"none" (A1: never touches the LLM) | "base" (A3) | "sft" (everyone else)."""
    if not cfg.use_llm:
        return "none"
    return "sft" if cfg.use_sft else "base"


def main():
    from agentic_raptor.publication.ablation_v3 import (COMPONENT_ABLATIONS,
                                                         build_result_record)
    from agentic_raptor.publication.preflight import run_preflight
    from agentic_raptor.publication.spec_registry import build as build_registry
    from agentic_raptor.publication.spec_registry import lookup as lookup_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(COMPONENT_ABLATIONS),
                    help="comma list from A0..A8 (A9 is a separate driver)")
    ap.add_argument("--specs", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--seeds", default="0,1,2",
                    help="pilot default -- do not expand until pilot "
                         "variance has been reviewed")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--rag-memory", default=None,
                    help="override which RAG memory file every run reads. "
                         "Defaults to the CLEAN, frozen, provenance-"
                         "verified-only snapshot (rag_memory_v2_clean.jsonl) "
                         "-- pass an explicit path (e.g. the raw "
                         "rag_memory_v2.jsonl) only if you deliberately "
                         "want the mixed-provenance file instead.")
    ap.add_argument("--pvt", action="store_true",
                    help="also run the frozen PVT sweep on every eligible "
                         "nominal pass (real cost multiplier; off by "
                         "default for smoke/dev runs)")
    ap.add_argument("--pvt-corners", default="tt",
                    help="comma-separated process corners for --pvt, e.g. "
                         "tt,ff,ss,sf,fs. Only corners the configured PDK "
                         "actually ships are accepted. Default is 'tt' "
                         "only (nominal-only, cheapest); pass the full "
                         "set for a real process-robustness sweep.")
    ap.add_argument("--pvt-voltages", default="1.8",
                    help="comma-separated ABSOLUTE supply voltages (V) for "
                         "--pvt, e.g. 1.62,1.8,1.98")
    ap.add_argument("--pvt-temps-c", default="27",
                    help="comma-separated ABSOLUTE temperatures (deg C) "
                         "for --pvt, e.g. -40,27,85")
    ap.add_argument("--paper-mode", action="store_true",
                    help="run the full preflight check first and REFUSE to "
                         "execute anything if a blocker is found")
    args = ap.parse_args()

    if args.rag_memory is None:
        from agentic_raptor.publication.preflight import CLEAN_RAG_PATH
        args.rag_memory = str(CLEAN_RAG_PATH)

    arms = [a for a in args.arms.split(",") if a in COMPONENT_ABLATIONS]
    if not arms:
        raise SystemExit(f"no valid arms in {args.arms!r}; choose from "
                         f"{sorted(COMPONENT_ABLATIONS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    idxs = [args.start + i * args.step for i in range(args.specs)]
    registry = build_registry()

    preflight = run_preflight(paper_mode=args.paper_mode)
    print(f"preflight: ready={preflight['ready']} "
         f"blockers={preflight['blockers']}\n")

    OUT.mkdir(parents=True, exist_ok=True)
    results_path = OUT / f"results_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    errors_path = OUT / "errors.log"

    import run_raptor_v2 as v2
    from run_qwen_ablation import _load

    # Group requested arms by which proposer they need, so a 4B model is
    # loaded AT MOST TWICE (base + SFT) regardless of how many arms run,
    # and never at all if every requested arm is A1 (no LLM).
    groups: dict[str, list[str]] = {}
    for aid in arms:
        groups.setdefault(_proposer_requirement(COMPONENT_ABLATIONS[aid]), []).append(aid)

    total = len(arms) * len(idxs) * len(seeds)
    print(f"arms  : {arms}")
    print(f"specs : {idxs} on {args.split!r}")
    print(f"seeds : {seeds}  (pilot default is exactly [0,1,2])")
    print(f"total : {total} runs -> {results_path}\n", flush=True)

    # ONE PvtConfig, built once and reused for every run -- Part: FAIRNESS
    # requires the SAME frozen PVT protocol across all comparable arms, not
    # a config that could vary run to run.
    pvt_cfg = None
    if args.pvt:
        from agentic_raptor.electrical.pvt_eval import PvtConfig
        pvt_cfg = PvtConfig(
            enabled=True,
            process_corners=tuple(c.strip()
                                  for c in args.pvt_corners.split(",")),
            supply_voltages=tuple(float(v)
                                  for v in args.pvt_voltages.split(",")),
            temperatures_c=tuple(float(t)
                                 for t in args.pvt_temps_c.split(",")))
        print(f"PVT   : ENABLED -- corners={pvt_cfg.process_corners} "
             f"voltages={pvt_cfg.supply_voltages} "
             f"temps_c={pvt_cfg.temperatures_c} "
             f"({len(pvt_cfg.process_corners) * len(pvt_cfg.supply_voltages) * len(pvt_cfg.temperatures_c)} "
             f"corners/run, added to every eligible nominal pass)\n")

    n = 0
    t_all = time.time()
    with results_path.open("a", encoding="utf-8") as out_f:
        for requirement, group_arms in groups.items():
            if requirement == "none":
                model, tok, adapter_str = None, None, ""
            elif requirement == "base":
                tok, model = _load(None)     # _load returns (tok, model)
                adapter_str = ""
            else:
                tok, model = _load(str(args.adapter))
                adapter_str = str(args.adapter)
            for seed in seeds:
                for idx in idxs:
                    for aid in group_arms:
                        n += 1
                        cfg = COMPONENT_ABLATIONS[aid]
                        kwargs = cfg.to_run_pipeline_kwargs()
                        budget = kwargs.pop("budget")
                        spec_h = (lookup_spec(args.split, idx, registry) or {}).get("spec_hash")
                        t0 = time.time()
                        try:
                            tr = v2.run_pipeline(
                                model, tok, adapter_str, split=args.split,
                                spec_index=idx, budget=budget, seed=seed,
                                out_prefix=f"ABLv3_{aid}_s{seed}",
                                pvt_config=pvt_cfg, rag_memory=args.rag_memory,
                                **kwargs)
                            row = build_result_record(
                                tr, cfg,
                                experiment_id=f"{aid}_{args.split}_{idx}_{seed}",
                                pipeline_seed=seed, spec_index=idx,
                                spec_hash=spec_h)
                            # Stage 1.5: paper mode must hard-fail on an
                            # UNEXPLAINED requested-vs-simulated C_LOAD
                            # mismatch (an explicit, reasoned override is
                            # fine and never trips this).
                            if args.paper_mode and (row.get("nominal") or {}).get(
                                    "c_load_unexplained_mismatch"):
                                raise SystemExit(
                                    "PAPER MODE: unexplained C_LOAD mismatch on "
                                    f"{row['experiment_id']} -- requested="
                                    f"{row['nominal'].get('requested_c_load_f')} "
                                    f"simulated={row['nominal'].get('simulated_c_load_f')}")
                        except Exception as exc:
                            row = {"experiment_id": f"{aid}_{args.split}_{idx}_{seed}",
                                  "ablation_id": aid, "spec_index": idx,
                                  "spec_hash": spec_h,
                                  "pipeline_seed": seed,
                                  "trace_result": f"ERROR: {type(exc).__name__}: {str(exc)[:160]}"}
                            with errors_path.open("a", encoding="utf-8") as ef:
                                ef.write(f"\n=== {aid} idx={idx} seed={seed}\n"
                                        + traceback.format_exc())
                        row["seconds"] = round(time.time() - t0, 1)
                        out_f.write(json.dumps(row, default=str) + "\n")
                        out_f.flush()
                        print(f"[{n:>4}/{total}] {aid:<3} idx={idx:<3} seed={seed} "
                             f"pass={row.get('nominal', {}).get('complete_pass')} "
                             f"fom={row.get('fom', {}).get('fom_value')} "
                             f"{row['seconds']:>6.0f}s "
                             f"{row.get('trace_result', 'OK') if str(row.get('trace_result','')).startswith('ERROR') else ''}",
                             flush=True)

    print(f"\ndone: {n} runs in {round(time.time() - t_all, 1)}s -> {results_path}")


if __name__ == "__main__":
    main()
