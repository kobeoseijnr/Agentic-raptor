"""RAPTOR ABLATION: 4 agents vs the agent-free FULL baseline.

    AG_FULL       planner + critic + supervisor + recovery
    AG_NO_PLAN    all minus Design Planner
    AG_NO_CRITIC  all minus Topology Critic
    AG_NO_SUPER   all minus Optimization Supervisor
    AG_NO_RECOV   all minus Recovery Agent
    BASELINE      agents=() -- today's FULL, byte-identical

6 arms x 9 distinct heldout specs x 1 seed = 54 jobs. FAIRNESS: every arm
runs the same spec/budget/seed configuration; agentic arms spend AT MOST
the baseline envelope (BudgetLedger-enforced) -- wins are orchestration,
never extra compute. Resumable: finished (arm, spec) pairs are skipped.
    python run_agentic_ablation.py
"""
import json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v3/agentic_ablation"
SPECS = list(range(9))
ARMS = {
    "AG_FULL": ("planner", "critic", "supervisor", "recovery"),
    "AG_NO_PLAN": ("critic", "supervisor", "recovery"),
    "AG_NO_CRITIC": ("planner", "supervisor", "recovery"),
    "AG_NO_SUPER": ("planner", "critic", "recovery"),
    "AG_NO_RECOV": ("planner", "critic", "supervisor"),
    "BASELINE": (),
}

def main():
    from agentic_raptor.electrical.pvt_eval import PvtConfig
    from run_qwen_ablation import _load
    import run_raptor_v2 as v2
    adapter = str(ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse")
    tok, model = _load(adapter)
    pvt = PvtConfig(enabled=True, process_corners=("tt", "ff", "ss"),
                    supply_voltages=(1.8,), temperatures_c=(27.0,))
    OUT.mkdir(parents=True, exist_ok=True)
    rows, done = [], set()
    prior = OUT / "AGENTIC_ROWS.jsonl"
    if prior.is_file():
        for l in prior.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l); rows.append(r); done.add((r["arm"], r["spec_index"]))
        print(f"RESUME: {len(rows)} jobs already done", flush=True)
    with prior.open("a", encoding="utf-8") as f:
        for idx in SPECS:
            for arm, ag in ARMS.items():
                if (arm, idx) in done: continue
                t0 = time.time()
                try:
                    tr = v2.run_pipeline(model, tok, adapter, split="heldout",
                                         spec_index=idx, budget=16,
                                         calibrate=False, seed=0,
                                         learning_mode="frozen",
                                         pvt_config=pvt, agents=ag,
                                         # 2026-08-30 repairs: post-pass
                                         # margin climb + mguard selection
                                         # (active only on agentic arms via
                                         # the supervisor path)
                                         margin_tail=(12 if ag else 0),
                                         out_prefix=f"AGB_{arm}")
                    s9 = tr.get("stage9_verification") or {}
                    n = tr.get("nominal") or {}
                    aginfo = tr.get("agents") or {}
                    row = {"arm": arm, "spec_index": idx,
                          "spec_id": tr["stage1_spec"]["spec_id"],
                          "pass": bool(n.get("complete_pass")),
                          "distance": s9.get("distance_to_feasibility"),
                          "pvt_robust": (tr.get("pvt") or {}).get("robust_complete_pass"),
                          "spice": tr["spice_usage"]["total_spice_calls"],
                          "opt_spice": tr["spice_usage"]["optimization_spice_calls"],
                          "plan_difficulty": ((tr.get("agent_planner") or {}).get("difficulty")),
                          "critic_rounds": ((tr.get("agent_critic") or {}).get("rounds")),
                          "supervisor_alloc": ((tr.get("agent_supervisor") or {}).get("allocation")),
                          "recovery": ((tr.get("agent_recovery") or {}).get("adopted")),
                          "banked": ((aginfo.get("ledger") or {}).get("banked")),
                          "runtime_s": round(time.time() - t0, 1)}
                except Exception as exc:
                    row = {"arm": arm, "spec_index": idx, "pass": False,
                          "error": f"{type(exc).__name__}: {exc}"[:200],
                          "runtime_s": round(time.time() - t0, 1)}
                rows.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
                print(f"{arm:12s} idx={idx}: pass={row.get('pass')} "
                     f"dist={row.get('distance')} err={row.get('error','')} "
                     f"({row['runtime_s']}s)", flush=True)
    summary = {}
    for arm in ARMS:
        ar = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"pass": sum(1 for r in ar if r.get("pass")),
                       "n": len(ar),
                       "robust": sum(1 for r in ar if r.get("pvt_robust")),
                       "errors": sum(1 for r in ar if r.get("error")),
                       "mean_opt_spice": (round(sum(r.get("opt_spice") or 0 for r in ar)
                                                / max(1, len(ar)), 1))}
    (OUT / "AGENTIC_SUMMARY.json").write_text(json.dumps(summary, indent=1),
                                              encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)

if __name__ == "__main__":
    main()
