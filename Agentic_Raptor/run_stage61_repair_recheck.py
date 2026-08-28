"""Stage 6.1: small real recheck of the repaired MB-SAC action/replay path.

Does NOT re-run AlphaZero and does NOT re-freeze topologies -- it loads the
SAME FROZEN_TOPOLOGY_SET.json Stage 6 already produced (12 problems, hash
7d8c1203de536929c0e54371b0c76ce876343fede0db790026b81bddcc0a95ac) and reuses
4 representative problems from it, chosen for diversity BEFORE looking at
which arm the repair might favor:

  spec26_rank0  -- both arms passed exact spec in the original Stage 6 run
  spec41_rank1  -- both arms passed exact spec (a second, independent pass)
  spec30_rank0  -- the ONE job MB-SAC won on distance-to-feasible originally
  spec38_rank0  -- the job with MB-SAC's WORST distance-to-feasible margin

Same topology, same spec, same CLOAD, same budget (16), same TPE-lite
baseline, same reward, same verification protocol as the original Stage 6
run -- the only thing that changed is the action/replay bug fix inside
sac_size(). 2 sizing seeds per problem (seed 0 reuses the ORIGINAL Stage 6
seed, so it is directly comparable to the old saved result; seed 1 is new).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from run_stage6_mbsac_diagnostic import (STAGE6_ROOT, fresh_verify, pass_at_k,
                                         run_arm_mbsac, run_arm_no_sac_strong)

ROOT = Path(__file__).resolve().parent
RECHECK_ROOT = ROOT / "artifacts/publication_v3/stage6_1_repair_recheck"
SIZING_BUDGET = 16
RECHECK_PROBLEM_IDS = ["spec26_rank0", "spec41_rank1", "spec30_rank0", "spec38_rank0"]
RECHECK_SEEDS = [0, 1]

OLD_REPORT = STAGE6_ROOT / "STAGE6_DIAGNOSTIC_REPORT.json"


def load_frozen_problems() -> dict:
    frozen_path = STAGE6_ROOT / "FROZEN_TOPOLOGY_SET.json"
    problems = json.loads(frozen_path.read_text(encoding="utf-8"))
    return {p["problem_id"]: p for p in problems if not p.get("aborted")}


def load_old_results() -> dict:
    if not OLD_REPORT.is_file():
        return {}
    d = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    return {j["problem_id"]: j for j in d["job_results"] if j["sizing_seed"] == 0}


def run_one(problem: dict, seed: int, out_root: Path) -> dict:
    t0 = time.time()
    mbsac = run_arm_mbsac(problem, SIZING_BUDGET, seed, out_root)
    t1 = time.time()
    nosac = run_arm_no_sac_strong(problem, SIZING_BUDGET, seed, out_root)
    t2 = time.time()
    mbsac_verify = fresh_verify(problem, mbsac["best"]["knobs"], "mbsac_fixed", seed, out_root)
    nosac_verify = fresh_verify(problem, nosac["best"]["knobs"], "nosac", seed, out_root)
    return {
        "problem_id": problem["problem_id"], "spec_index": problem["spec_index"],
        "topology_hash": problem["topology_hash"], "sizing_seed": seed,
        "mbsac_fixed": {
            "spice_calls": mbsac["spice_calls"],
            "calls_to_first_exact_pass": mbsac["calls_to_first_exact_pass"],
            "exact_spec_pass": mbsac["outcome"]["exact_spec_pass"],
            "distance_to_feasible": mbsac["outcome"]["normalized_distance_to_feasibility"],
            "pass_at_k": pass_at_k(mbsac["results"], problem["spec"]),
            "runtime_s": round(t1 - t0, 1),
            "actor_params_changed": mbsac["actor_params_changed"],
            "critic_params_changed": mbsac["critic_params_changed"],
            "n_transitions_recorded": mbsac["n_transitions_recorded"],
            "n_nominal_anchor_transitions": mbsac["n_nominal_anchor_transitions"],
            "n_exploitation_tail_transitions": mbsac["n_exploitation_tail_transitions"],
            "max_action_roundtrip_error": mbsac["max_action_roundtrip_error"],
            "final_entropy_alpha": mbsac["final_entropy_alpha"],
            "verify": mbsac_verify,
        },
        "nosac": {
            "spice_calls": nosac["spice_calls"],
            "calls_to_first_exact_pass": nosac["calls_to_first_exact_pass"],
            "exact_spec_pass": nosac["outcome"]["exact_spec_pass"],
            "distance_to_feasible": nosac["outcome"]["normalized_distance_to_feasibility"],
            "pass_at_k": pass_at_k(nosac["results"], problem["spec"]),
            "runtime_s": round(t2 - t1, 1),
            "verify": nosac_verify,
        },
    }


def summarize(job_results: list) -> dict:
    def _fom(v):
        return v["fom"]["fom_value"] if v["fom"]["valid"] else None

    mbsac_pass = sum(1 for j in job_results if j["mbsac_fixed"]["verify"]["outcome"]["exact_spec_pass"])
    nosac_pass = sum(1 for j in job_results if j["nosac"]["verify"]["outcome"]["exact_spec_pass"])
    wins = losses = ties = 0
    diffs = []
    for j in job_results:
        md = j["mbsac_fixed"]["distance_to_feasible"]
        nd = j["nosac"]["distance_to_feasible"]
        md = 9.9 if md is None else md
        nd = 9.9 if nd is None else nd
        diffs.append(md - nd)
        if md < nd - 1e-9:
            wins += 1
        elif nd < md - 1e-9:
            losses += 1
        else:
            ties += 1
    import statistics as st
    mbsac_foms = [f for j in job_results if (f := _fom(j["mbsac_fixed"]["verify"])) is not None]
    nosac_foms = [f for j in job_results if (f := _fom(j["nosac"]["verify"])) is not None]
    graph_identity_ok = all(j["mbsac_fixed"]["verify"]["graph_identity_preserved"]
                            and j["nosac"]["verify"]["graph_identity_preserved"]
                            for j in job_results)
    max_roundtrip = max(j["mbsac_fixed"]["max_action_roundtrip_error"] for j in job_results)
    return {
        "n_jobs": len(job_results),
        "mbsac_fixed_exact_pass_rate": round(mbsac_pass / len(job_results), 4),
        "nosac_exact_pass_rate": round(nosac_pass / len(job_results), 4),
        "paired_wins_mbsac_closer": wins, "paired_losses_mbsac_farther": losses, "ties": ties,
        "mean_distance_diff_mbsac_minus_nosac": round(st.mean(diffs), 4),
        "mbsac_fixed_mean_fom": round(st.mean(mbsac_foms), 4) if mbsac_foms else None,
        "nosac_mean_fom": round(st.mean(nosac_foms), 4) if nosac_foms else None,
        "graph_identity_preserved_all_jobs": graph_identity_ok,
        "max_action_roundtrip_error_all_jobs": max_roundtrip,
    }


def compare_to_old(job_results: list, old_results: dict) -> list:
    """Seed-0 jobs only are directly comparable to the ORIGINAL Stage 6 run
    (which used a single seed=0 per problem)."""
    out = []
    for j in job_results:
        if j["sizing_seed"] != 0:
            continue
        old = old_results.get(j["problem_id"])
        if not old:
            continue
        out.append({
            "problem_id": j["problem_id"],
            "old_mbsac_exact_pass": old["mbsac"]["exact_spec_pass"],
            "new_mbsac_fixed_exact_pass": j["mbsac_fixed"]["verify"]["outcome"]["exact_spec_pass"],
            "old_mbsac_distance": old["mbsac"]["distance_to_feasible"],
            "new_mbsac_fixed_distance": j["mbsac_fixed"]["distance_to_feasible"],
            "old_mbsac_calls_to_first_pass": old["mbsac"]["calls_to_first_exact_pass"],
            "new_mbsac_fixed_calls_to_first_pass": j["mbsac_fixed"]["calls_to_first_exact_pass"],
            "old_mbsac_fom": old["mbsac"]["verify"]["fom"]["fom_value"],
            "new_mbsac_fixed_fom": j["mbsac_fixed"]["verify"]["fom"]["fom_value"],
            "old_mbsac_runtime_s": old["mbsac"]["runtime_s"],
            "new_mbsac_fixed_runtime_s": j["mbsac_fixed"]["runtime_s"],
        })
    return out


def main():
    problems = load_frozen_problems()
    missing = [p for p in RECHECK_PROBLEM_IDS if p not in problems]
    if missing:
        raise SystemExit(f"missing frozen problems: {missing}")
    old_results = load_old_results()

    RECHECK_ROOT.mkdir(parents=True, exist_ok=True)
    out_root = RECHECK_ROOT / "sizing_runs"
    job_results = []
    for pid in RECHECK_PROBLEM_IDS:
        problem = problems[pid]
        for seed in RECHECK_SEEDS:
            print(f"  job {pid} seed={seed} ...", flush=True)
            j = run_one(problem, seed, out_root)
            job_results.append(j)
            print(f"    mbsac_fixed: pass={j['mbsac_fixed']['exact_spec_pass']} "
                 f"dist={j['mbsac_fixed']['distance_to_feasible']} "
                 f"roundtrip_err={j['mbsac_fixed']['max_action_roundtrip_error']:.6f} | "
                 f"nosac: pass={j['nosac']['exact_spec_pass']} "
                 f"dist={j['nosac']['distance_to_feasible']}", flush=True)

    summary = summarize(job_results)
    old_vs_new = compare_to_old(job_results, old_results)
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=1), flush=True)
    print("\n=== old (pre-repair) vs new (repaired) MB-SAC, seed=0 only ===", flush=True)
    print(json.dumps(old_vs_new, indent=1), flush=True)

    report = {"recheck_problem_ids": RECHECK_PROBLEM_IDS, "recheck_seeds": RECHECK_SEEDS,
             "sizing_budget": SIZING_BUDGET, "no_sac_strong_method": "tpe_lite",
             "frozen_topology_set_sha256_reused":
             "7d8c1203de536929c0e54371b0c76ce876343fede0db790026b81bddcc0a95ac",
             "job_results": job_results, "summary": summary,
             "old_vs_new_mbsac_seed0": old_vs_new,
             "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path = RECHECK_ROOT / "STAGE6_1_RECHECK_REPORT.json"
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nStage 6.1 recheck report written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
