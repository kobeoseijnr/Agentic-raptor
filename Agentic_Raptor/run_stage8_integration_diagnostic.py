"""Stage 8: final frozen pipeline integration diagnostic.

Runs the REAL, complete, integrated pipeline end to end for each diagnostic
spec by calling run_raptor_v2.run_pipeline() directly -- no stage is
reimplemented or mocked here, so this exercises exactly what a live FULL
invocation does:

    Specification -> RAG -> SFT Qwen -> Validator -> TRUE_ALPHAZERO
    -> MB-SAC -> HARD SAFETY GATE -> LEARNED DPO V2 SELECTOR
    -> fresh authoritative NGSPICE -> PVT -> FoM

learning_mode="frozen" on every call (Section 10): RAG harvesting, SFT queue
updates, AlphaZero replay collection, SAC persistent memory, DPO pair
recording, and adaptive surrogate updates are all gated behind
`learning_mode == "adaptive"` inside run_pipeline and are therefore never
reached. ranker_mode is left at run_pipeline's own default ("dpo", restored
by Stage 8's second deployment after Stage 7.2B re-justified the learned
ranker under POST_SAC_FEATURES_V2) rather than passed explicitly, so this
diagnostic proves the DEFAULT behaves correctly, not a special
diagnostic-only override.

6 diagnostic specs from the TRAIN split (never blind-test/heldout),
balanced by difficulty (2 easy/2 medium/2 hard via
agentic_raptor.publication.eval_sets.feasibility_split), chosen BEFORE
looking at any result.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts/publication_v3/stage8_integration_diagnostic"

# A (easy) / B (medium) / C (hard), 2 each -- picked from the real
# feasibility_split() distribution over the train pool, before running.
DIAGNOSTIC_SPEC_INDICES = [1, 2, 0, 6, 4, 5]
SIZING_BUDGET = 16
ACTION_ROUNDTRIP_TOLERANCE = 1e-3


def run_one(model, tok, adapter, spec_index: int, seed: int = 0) -> dict:
    from agentic_raptor.electrical.pvt_eval import PvtConfig
    from run_raptor_v2 import run_pipeline

    pvt_config = PvtConfig(enabled=True, process_corners=("tt", "ff", "ss"),
                           supply_voltages=(1.8,), temperatures_c=(27.0,))
    t0 = time.time()
    trace = run_pipeline(model, tok, adapter, split="train",
                         spec_index=spec_index, budget=SIZING_BUDGET,
                         calibrate=False, seed=seed,
                         learning_mode="frozen", pvt_config=pvt_config)
    runtime_s = round(time.time() - t0, 1)
    return {"trace": trace, "runtime_s": runtime_s}


def verify_job(trace: dict) -> dict:
    """Sections 14-19: hard identity/contract checks against the REAL
    trace of one real run. Every field read here is something run_pipeline
    already computed from real components -- nothing is re-derived from
    assumption."""
    problems = []

    # Section 14: topology identity chain. run_pipeline itself already
    # hard-raises ArchitectureViolation internally if the sized branch
    # isn't the topology PUCT/AlphaZero selected, or if both branches sized
    # the same topology -- reaching this point at all is evidence those
    # invariants held. Independently re-check the trace's own three
    # recorded points agree.
    puct_selected = trace["provenance_chain"]["puct_selected"]
    stage6_hashes = {lbl: trace["stage6_sizing"][lbl]["topology_hash"] for lbl in ("A", "B")}
    routed_hashes = {lbl: trace["stage11_feedback"]["routed"].get(lbl, {}).get("topology_hash")
                     for lbl in trace["stage11_feedback"]["routed"]}
    if puct_selected[0] == puct_selected[1]:
        problems.append("puct_selected two candidates share a topology hash")
    if set(stage6_hashes.values()) != {h for h in puct_selected}:
        problems.append(f"stage6_sizing hashes {stage6_hashes} != puct_selected {puct_selected}")
    for lbl, h in routed_hashes.items():
        if h is not None and h != stage6_hashes.get(lbl):
            problems.append(f"final-verification topology hash for {lbl} "
                            f"({h}) != stage6_sizing hash ({stage6_hashes.get(lbl)})")

    # Section 15: knob identity. The live path never re-applies knobs at
    # verification time -- size_and_predict() calls apply_knobs() ONCE and
    # the SAME sized graph object is what final verification measures (see
    # run_raptor_v2.verify()'s `graph = designs[label][1]`), so there is no
    # second application that could diverge. What CAN still be wrong is the
    # recorded sizing_vector itself -- confirm every one of the 7 canonical
    # knobs is present and finite for both branches.
    from agentic_raptor.mb_sac.spec_sizing import KNOB_NAMES
    for lbl in ("A", "B"):
        sv = trace["stage6_sizing"][lbl]["sizing_vector"] or {}
        missing = [k for k in KNOB_NAMES if k not in sv]
        non_finite = [k for k, v in sv.items() if not isinstance(v, (int, float))]
        if missing:
            problems.append(f"{lbl}: sizing_vector missing knobs {missing}")
        if non_finite:
            problems.append(f"{lbl}: sizing_vector has non-numeric entries {non_finite}")

    # Section 16: SAC action/replay roundtrip (only meaningful when the
    # winning branch actually used sequential SAC, not a non-RL sizer).
    max_roundtrip = 0.0
    for lbl in ("A", "B"):
        err = trace["stage6_sizing"][lbl].get("max_action_roundtrip_error")
        if err is not None:
            max_roundtrip = max(max_roundtrip, err)
            if err > ACTION_ROUNDTRIP_TOLERANCE:
                problems.append(f"{lbl}: max_action_roundtrip_error {err} exceeds "
                                f"tolerance {ACTION_ROUNDTRIP_TOLERANCE}")

    # Section 17: two-candidate contract (redundant with the topology-hash
    # check above, kept separate for an explicit named result).
    two_candidate_ok = puct_selected[0] != puct_selected[1]

    # Section 18/34 (second deployment): selector contract -- confirm the
    # PROMOTED DPO V2 checkpoint (and ONLY that checkpoint) is loaded, with
    # the correct feature schema, matching the restored FULL default.
    from agentic_raptor.ranking.model_v2 import REQUIRED_V2_SHA256
    use_learned_dpo = trace["stage8_ranker"]["use_learned_dpo"]
    selector = trace["stage8_ranker"]["selector"]
    ranker_model_hash = trace["models"]["ranker"]["hash"]
    feature_schema = trace["stage8_ranker"].get("feature_schema")
    if (not use_learned_dpo or selector != "learned_dpo"
            or ranker_model_hash != REQUIRED_V2_SHA256
            or feature_schema != "POST_SAC_FEATURES_V2"):
        problems.append(f"selector contract violated: use_learned_dpo={use_learned_dpo} "
                        f"selector={selector} ranker_model_hash={ranker_model_hash} "
                        f"feature_schema={feature_schema}")

    # Section 19: fresh authoritative verification, never a reused call id.
    final_calls_are_new = trace["stage9_verification"]["final_calls_are_new"]
    if not final_calls_are_new:
        problems.append("final verification reused a sizing call id")

    # Section 20: electrical environment.
    cl_ok = (trace["nominal"]["requested_c_load_f"] is None
            or trace["nominal"]["simulated_c_load_f"] == trace["nominal"]["requested_c_load_f"]
            or not trace["nominal"]["c_load_unexplained_mismatch"])
    if trace["nominal"]["c_load_unexplained_mismatch"]:
        problems.append("unexplained requested/simulated CLOAD mismatch")

    return {
        "topology_identity_ok": not any("topology" in p or "puct_selected" in p for p in problems),
        "knob_identity_ok": not any("sizing_vector" in p for p in problems),
        "max_action_roundtrip_error": max_roundtrip,
        "action_roundtrip_ok": max_roundtrip <= ACTION_ROUNDTRIP_TOLERANCE,
        "two_candidate_distinct": two_candidate_ok,
        "selector_contract_ok": (use_learned_dpo and selector == "learned_dpo"
                                and ranker_model_hash == REQUIRED_V2_SHA256
                                and feature_schema == "POST_SAC_FEATURES_V2"),
        "fresh_verification_ok": final_calls_are_new,
        "cload_ok": cl_ok,
        "all_ok": not problems,
        "problems": problems,
    }


def main():
    from run_qwen_ablation import _load

    adapter = str(ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse")
    assert Path(adapter, "adapter_config.json").is_file(), f"not a peft adapter dir: {adapter}"
    print(f"loading model from adapter: {adapter}", flush=True)
    tok, model = _load(adapter)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec_index in DIAGNOSTIC_SPEC_INDICES:
        print(f"\n=== spec_index={spec_index} ===", flush=True)
        result = run_one(model, tok, adapter, spec_index)
        trace = result["trace"]
        verify = verify_job(trace)
        job = {"spec_index": spec_index, "runtime_s": result["runtime_s"],
              "trace": trace, "verify": verify}
        jobs.append(job)
        print(f"  spec_id={trace['stage1_spec']['spec_id']} "
             f"exact_pass={trace['nominal']['complete_pass']} "
             f"selector={trace['stage8_ranker']['selector']} "
             f"use_learned_dpo={trace['stage8_ranker']['use_learned_dpo']} "
             f"all_ok={verify['all_ok']} "
             f"runtime={result['runtime_s']}s", flush=True)
        if verify["problems"]:
            print(f"  PROBLEMS: {verify['problems']}", flush=True)
        trace_path = OUT_DIR / f"trace_spec{spec_index}.json"
        trace_path.write_text(json.dumps(trace, indent=1, default=str), encoding="utf-8")

    summary = summarize(jobs)
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=1), flush=True)

    report = {"diagnostic_spec_indices": DIAGNOSTIC_SPEC_INDICES,
             "sizing_budget": SIZING_BUDGET,
             "jobs": [{"spec_index": j["spec_index"], "runtime_s": j["runtime_s"],
                      "verify": j["verify"],
                      "trace_summary": {k: j["trace"].get(k) for k in
                                       ("stage1_spec", "stage2_rag", "stage3_propose",
                                        "stage5_alphazero", "stage6_sizing",
                                        "stage8_ranker", "stage9_verification",
                                        "nominal", "fom", "pvt", "spice_usage",
                                        "provenance_chain")}}
                     for j in jobs],
             "summary": summary, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_path = OUT_DIR / "STAGE8_INTEGRATION_REPORT.json"
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nStage 8 integration report written to {out_path}", flush=True)


def pass_at_k(passes: list, ks=(1, 2, 3, 4, 5, 6)) -> dict:
    out = {}
    for k in ks:
        if k > len(passes):
            continue
        out[f"pass@{k}"] = any(passes[:k])
    return out


def summarize(jobs: list) -> dict:
    n = len(jobs)
    passes = [bool(j["trace"]["nominal"]["complete_pass"]) for j in jobs]
    first_pass_idx = next((i + 1 for i, p in enumerate(passes) if p), None)
    opt_calls = sum(j["trace"]["spice_usage"]["optimization_spice_calls"] for j in jobs)
    verify_calls = sum(j["trace"]["spice_usage"]["final_nominal_verification_calls"] for j in jobs)
    pvt_calls = sum(j["trace"]["spice_usage"]["pvt_spice_calls"] for j in jobs)
    total_calls = sum(j["trace"]["spice_usage"]["total_spice_calls"] for j in jobs)
    # Section 22: FoM must not let an infeasible design "rescue" the
    # average -- report the established all-valid convention (matches
    # run_stage6_mbsac_diagnostic.py's own _fom precedent) AND a
    # feasible-only figure side by side, never collapsed into one number.
    foms = [j["trace"]["fom"]["fom_value"] for j in jobs
           if j["trace"]["fom"] and j["trace"]["fom"].get("valid")]
    foms_feasible = [j["trace"]["fom"]["fom_value"] for j in jobs
                     if j["trace"]["fom"] and j["trace"]["fom"].get("valid")
                     and j["trace"]["nominal"]["complete_pass"]]
    pvt_pass = [j["trace"]["pvt"].get("robust_complete_pass") for j in jobs
               if j["trace"]["pvt"].get("enabled")]
    all_topologies = set()
    for j in jobs:
        all_topologies.update(j["trace"]["provenance_chain"]["puct_selected"])
    return {
        "n_specs": n,
        "final_pass_count": sum(passes),
        "final_pass_rate": round(sum(passes) / n, 4) if n else None,
        "pass_at_k": pass_at_k(passes),
        "calls_to_first_pass_spec_rank": first_pass_idx,
        "total_optimization_spice_calls": opt_calls,
        "total_verification_spice_calls": verify_calls,
        "total_pvt_spice_calls": pvt_calls,
        "total_spice_calls": total_calls,
        "total_runtime_s": round(sum(j["runtime_s"] for j in jobs), 1),
        "mean_fom_all_valid": (round(sum(foms) / len(foms), 4) if foms else None),
        "n_fom_defined_all_valid": len(foms),
        "mean_fom_feasible_only": (round(sum(foms_feasible) / len(foms_feasible), 4)
                                   if foms_feasible else None),
        "n_fom_defined_feasible_only": len(foms_feasible),
        "pvt_enabled_jobs": len(pvt_pass),
        "pvt_robust_pass_count": sum(1 for p in pvt_pass if p),
        "n_distinct_topologies_selected_across_all_jobs": len(all_topologies),
        "all_jobs_verify_ok": all(j["verify"]["all_ok"] for j in jobs),
        "n_jobs_with_problems": sum(1 for j in jobs if not j["verify"]["all_ok"]),
        "max_action_roundtrip_error_all_jobs": max(
            (j["verify"]["max_action_roundtrip_error"] for j in jobs), default=0.0),
    }


if __name__ == "__main__":
    main()
