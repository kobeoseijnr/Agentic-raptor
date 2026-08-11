"""SUPERSEDED (2026-08-11, root-level PUCT retirement / TRUE_ALPHAZERO
cutover -- see artifacts/publication_v3/ROOT_LEVEL_PUCT_RETIRED.json):
this script still runs (it only calls run_raptor_v2.run_pipeline, which is
unaffected), but the pool it grows (puct_examples.jsonl, candidate ->
scalar value_target) is NOT AlphaZero replay and cannot become one --
AlphaZero requires real (state, pi, z) episode trajectories from genuine
multi-step self-play, which this data was never structured to record. It
may still be useful as VALUE-HEAD initialization evidence (see Part 0's
POST_CLOAD_FIX_V1 checkpoint rebuild, agentic_raptor.topology_rl.
value_refresh.refresh_v2/cross_validate_v2), but must never be used to
build or promote an AlphaZero generation checkpoint. Kept, not deleted --
Section 24: historical evidence, not silently discarded, but no longer
part of active FULL/paper preparation. Real AlphaZero replay collection
lives in agentic_raptor.topology_rl.alphazero (run_alphazero_episode,
build_replay_rows, write_replay_rows).

Grow the v2-native PUCT value-training pool
(artifacts/publication_v2/live_streams/puct_examples.jsonl) with real,
adaptive-mode, production-configuration (A0 full) run_raptor_v2.py runs.

This exists because the CURRENTLY DEPLOYED PUCT value checkpoint
(artifacts/stage3e1/policy_value_ep0.pt) was trained on
datasets/simulation_memory/az_value_targets.jsonl, which is written by
run_full_raptor.py / run_self_improvement.py -- the DEPRECATED pre-v2
architecture, not the canonical pipeline. That data is explicitly not
reused: this script generates a clean pool from run_raptor_v2.py only,
which also guarantees every example was measured under the current
(post-VCM-fix) electrical model, unlike the old checkpoint's provenance,
which could not be established.

Runs full-production config (conditioning=exclusion, search=one_root,
ranker_mode=dpo, sizing_method=sac -- i.e. A0), learning_mode="adaptive"
(the default -- this IS the growth mechanism), calibrate=True (verifies
BOTH designs every run, same lever the DPO ranker fix used, doubling
informative signal per real run).

Cycles through every train-split spec once (seed=0), then again (seed=1),
etc., so spec DIVERSITY is exhausted before seed-repeat diversity is added
-- matches the finding from the DPO ranker fix that boundary/hard-tier
specs contribute disproportionately more informative signal than easy ones
repeated.

Stops when EITHER the wall-clock time budget is spent OR --max-runs is
reached, whichever comes first. Safe to interrupt (Ctrl-C) and re-run --
each run is independent and idempotently appended to the stream files by
run_pipeline itself.

Run:  python run_puct_value_calibration.py --hours 24 --budget 32
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"
#: Post-launch fix (2026-08-10): LOG was the ONE output of this campaign
#: never versioned to _post_cload_v1 -- every other output was (puct_
#: examples, RAG memory, trusted_pairs, CLEAN_DATASET below). Consequence,
#: confirmed on the real 12h run: auto-resume's done_pairs read the OLD,
#: pre-C_LOAD-fix log.jsonl (119 rows from earlier this session) together
#: with the new run's rows in the SAME file, so it silently treated 7
#: specs (index 0-6) as "already done" from stale pre-repair measurements
#: and never re-measured them under the corrected environment --
#: campaign_status.json's "full_pass_complete (85/85)" was WRONG; the
#: genuine post-fix count was 78/85. The old file was NOT touched (kept as
#: PRE_CLOAD_FIX evidence); the 78 genuine rows were extracted by hand into
#: log_post_cload_v1.jsonl once. LOG now points there so this can't
#: silently recur.
LOG = ROOT / "artifacts/publication_v3/puct_value_calibration/log_post_cload_v1.jsonl"
ERRORS = ROOT / "artifacts/publication_v3/puct_value_calibration/errors_post_cload_v1.log"
#: Stage 1.6, item 2: the CLEAN, correct-load measurement dataset -- one
#: record per authoritative measurement this campaign takes, full schema
#: (spec identity, requested/simulated load, topology/sizing provenance,
#: simulator/PDK version, real electrical readings, pass/fail, FoM). This
#: campaign already produces puct_examples/rag_memory/trusted_pairs as a
#: side effect via run_pipeline's adaptive-mode harvest; this file is the
#: SEPARATE, auditable "what did we actually measure" record the same runs
#: also produce, not a duplicate of any one of those three streams.
CLEAN_DATASET = ROOT / "artifacts/publication_v3/clean_measurement_dataset/measurements_post_cload_v1.jsonl"


_SIM_CACHE: dict = {}


def _sim_version(exe: str | None) -> tuple[str | None, str | None]:
    """Cached: simulator version + PDK tt.spice hash don't change mid-run."""
    if "v" not in _SIM_CACHE:
        try:
            from agentic_raptor.electrical import (_PDK_CORNER_DIR,
                                                    discover_ngspice,
                                                    ngspice_version)
            import hashlib
            e = exe or discover_ngspice()
            _SIM_CACHE["v"] = ngspice_version(e) if e else None
            tt = _PDK_CORNER_DIR / "tt.spice"
            _SIM_CACHE["pdk"] = (hashlib.sha256(tt.read_bytes()).hexdigest()[:16]
                                if tt.is_file() else None)
        except Exception:
            _SIM_CACHE["v"], _SIM_CACHE["pdk"] = None, None
    return _SIM_CACHE["v"], _SIM_CACHE["pdk"]


def _clean_record(trace: dict, *, seed: int, exe: str | None) -> dict:
    """Extract ONE authoritative-measurement record in the Stage 1.6 §2
    schema from a completed run_pipeline() trace."""
    from agentic_raptor.publication.artifact_provenance import POST_CLOAD_FIX_V1
    sim_version, pdk_hash = _sim_version(exe)
    s1 = trace.get("stage1_spec", {})
    nom = trace.get("nominal") or {}
    prov = trace.get("provenance_chain") or {}
    fom = trace.get("fom") or {}
    su = trace.get("spice_usage") or {}
    return {
        "spec_index": s1.get("spec_index"), "spec_hash": s1.get("spec_hash"),
        "split": s1.get("split"), "seed": seed,
        "requested_c_load_f": s1.get("requested_c_load_f"),
        "simulated_c_load_f": nom.get("c_load_f"),
        "c_load_override": nom.get("c_load_override", False),
        "c_load_override_reason": nom.get("c_load_override_reason"),
        "topology_hash": prov.get("ranker_selected"),
        "sizing_manifest_hashes": prov.get("sizing_manifests"),
        "simulator": "ngspice",
        "simulator_version": sim_version,
        "pdk_corner": "tt", "pdk_tt_sha256_16": pdk_hash,
        "gain_db": nom.get("gain_db"), "pm_deg": nom.get("pm_deg"),
        "ugbw_hz": nom.get("ugbw_hz"), "idd_a": nom.get("idd_a"),
        "power_w": nom.get("power_w"),
        "complete_pass": nom.get("complete_pass"),
        "failure_reasons": nom.get("failure_reasons"),
        "specification_margins": nom.get("specification_margins"),
        "fom_value": fom.get("fom_value"), "fom_version": fom.get("fom_version"),
        "total_spice_calls": su.get("total_spice_calls"),
        "electrical_environment_version": POST_CLOAD_FIX_V1,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0,
                    help="wall-clock time budget; stops cleanly when spent")
    ap.add_argument("--max-runs", type=int, default=10_000,
                    help="hard cap regardless of time remaining")
    ap.add_argument("--budget", type=int, default=32,
                    help="sizing SPICE-call budget per branch (matches "
                         "A0's production ExperimentBudget default)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--start-idx", type=int, default=0,
                    help="manual override: resume from this spec index on "
                         "seed 0 instead of auto-detecting from LOG. Auto-"
                         "resume (default) makes this unnecessary in the "
                         "normal case -- only set it to deliberately re-"
                         "cover already-done specs, e.g. after a code fix.")
    ap.add_argument("--no-auto-resume", action="store_true",
                    help="disable auto-resume; use --start-idx exactly as "
                         "given even if LOG shows further progress")
    args = ap.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)

    import run_raptor_v2 as v2
    from run_qwen_ablation import _load

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"not a peft adapter directory: {adapter}")

    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    n_specs = len([r for r in corpus["records"] if r["split"] == args.split])
    print(f"split={args.split!r} has {n_specs} specs; cycling through all of "
         f"them per pass (seed increments each full pass)")
    print(f"budget={args.budget} calibrate=True learning_mode=adaptive "
         f"(A0 full production config)")

    # Pre-campaign audit (item 15): auto-resume from LOG rather than
    # relying on the caller to pass the right --start-idx. done_pairs is
    # every (seed, spec_index) that already has a NON-ERROR row -- an
    # errored attempt is deliberately NOT counted as done, so a transient
    # failure gets retried on relaunch rather than silently skipped
    # forever. This also makes duplicate harvesting impossible across a
    # restart: a (seed, idx) already in done_pairs is skipped outright, so
    # it can never be measured (and appended to puct_examples/rag_memory/
    # trusted_pairs) twice.
    done_pairs = set()
    if LOG.is_file() and not args.no_auto_resume:
        for line in LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not str(row.get("result", "")).startswith("ERROR") \
                    and "seed" in row and "spec_index" in row:
                done_pairs.add((row["seed"], row["spec_index"]))
    if done_pairs:
        print(f"auto-resume: {len(done_pairs)} (seed, spec_index) pairs "
             f"already completed per {LOG.relative_to(ROOT)} -- skipping them")
    print(f"time cap: {args.hours}h, run cap: {args.max_runs}\n", flush=True)

    tok, model = _load(str(adapter))     # ONCE
    deadline = time.time() + args.hours * 3600
    n = 0
    t_all = time.time()
    with LOG.open("a", encoding="utf-8") as log_f:
        for seed in range(10_000):        # bounded by deadline/max_runs in practice
            if time.time() >= deadline or n >= args.max_runs:
                break
            # --start-idx only applies to the FIRST pass (seed 0), and only
            # when auto-resume found nothing (or was disabled) -- it's the
            # manual fallback, not the normal path anymore.
            start = args.start_idx if seed == 0 else 0
            for idx in range(start, n_specs):
                if time.time() >= deadline or n >= args.max_runs:
                    break
                if (seed, idx) in done_pairs:
                    continue
                n += 1
                t0 = time.time()
                try:
                    tr = v2.run_pipeline(
                        model, tok, str(adapter), split=args.split,
                        spec_index=idx, budget=args.budget, seed=seed,
                        calibrate=True, out_prefix=f"PVCAL_s{seed}")
                    # Stage 1.6 §2: the authoritative-selected-design
                    # measurement in the Stage 1.6 clean-dataset schema.
                    # The BACKUP design's measurement (calibrate=True
                    # verifies both) is already captured with full
                    # provenance in ranker_pairs/trusted_pairs via
                    # harvest_run -- not duplicated here.
                    CLEAN_DATASET.parent.mkdir(parents=True, exist_ok=True)
                    with CLEAN_DATASET.open("a", encoding="utf-8") as cf:
                        cf.write(json.dumps(_clean_record(
                            tr, seed=seed, exe=None), default=str) + "\n")
                    row = {
                        "n": n, "spec_index": idx, "seed": seed,
                        "spec_id": tr.get("stage1_spec", {}).get("spec_id"),
                        "result": tr.get("result"),
                        "selected_exact_pass":
                        tr.get("stage9_verification", {}).get("selected_exact_pass"),
                        "both_measured":
                        tr.get("stage11_feedback", {}).get("both_measured"),
                        "routed_to_streams":
                        tr.get("stage11_feedback", {}).get("routed_to_streams"),
                        "seconds": round(time.time() - t0, 1)}
                except Exception as exc:
                    row = {"n": n, "spec_index": idx, "seed": seed,
                          "result": f"ERROR: {type(exc).__name__}: {str(exc)[:160]}",
                          "seconds": round(time.time() - t0, 1)}
                    with ERRORS.open("a", encoding="utf-8") as ef:
                        ef.write(f"\n=== n={n} idx={idx} seed={seed}\n"
                                + traceback.format_exc())
                log_f.write(json.dumps(row, default=str) + "\n")
                log_f.flush()
                # Post-launch fix (2026-08-10): done_pairs was only ever
                # populated from LOG as it existed BEFORE this invocation
                # started -- newly-completed (seed, idx) pairs were never
                # added, so the final status block below undercounted THIS
                # run's own progress. Confirmed on the real resume run: it
                # correctly filled the missing specs, but still reported
                # "partial_pass (78/85)" because it checked against the
                # stale pre-run snapshot instead of what had just been done.
                if not str(row.get("result", "")).startswith("ERROR"):
                    done_pairs.add((seed, idx))
                elapsed_h = (time.time() - t_all) / 3600
                remaining_h = max(0.0, (deadline - time.time()) / 3600)
                print(f"[{n:>5}] idx={idx:<3} seed={seed} "
                     f"pass={row.get('selected_exact_pass')} "
                     f"both={row.get('both_measured')} "
                     f"{row['seconds']:>6.0f}s "
                     f"elapsed={elapsed_h:.2f}h remaining={remaining_h:.2f}h "
                     f"{row.get('result') if str(row.get('result','')).startswith('ERROR') else ''}",
                     flush=True)

    puct_pool = ROOT / "artifacts/publication_v2/live_streams_post_cload_v1/puct_examples.jsonl"
    pool_size = sum(1 for _ in puct_pool.open(encoding="utf-8")) if puct_pool.is_file() else 0
    clean_size = (sum(1 for _ in CLEAN_DATASET.open(encoding="utf-8"))
                 if CLEAN_DATASET.is_file() else 0)

    # Pre-campaign audit (item 15): "complete" here means every distinct
    # spec_index in the split has been covered at least once -- NOT that
    # the time/run budget was never hit (this script is DESIGNED to stop
    # on budget, that's not an interruption). Written so a partial run
    # (some specs never reached) is distinguishable from a full pass
    # rather than silently looking the same as one in file listings.
    all_specs_covered = {i for i in range(n_specs)} <= {idx for (_s, idx) in done_pairs}
    status = {
        "status": "full_pass_complete" if all_specs_covered else "partial_pass",
        "runs_this_invocation": n,
        "distinct_specs_covered": len({idx for (_s, idx) in done_pairs}),
        "n_specs_in_split": n_specs,
        "total_done_pairs": len(done_pairs),
        "hours_this_invocation": round((time.time() - t_all) / 3600, 2),
        "stopped_reason": ("max_runs" if n >= args.max_runs else
                           "deadline" if time.time() >= deadline else
                           "exhausted"),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (LOG.parent / "campaign_status.json").write_text(
        json.dumps(status, indent=1), encoding="utf-8")
    print(f"\ndone: {n} runs in {round((time.time()-t_all)/3600, 2)}h")
    print(f"campaign status: {status['status']} "
         f"({status['distinct_specs_covered']}/{n_specs} distinct specs covered)")
    print(f"puct_examples.jsonl (post_cload_v1) now has {pool_size} examples")
    print(f"clean measurement dataset now has {clean_size} records "
         f"-> {CLEAN_DATASET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
