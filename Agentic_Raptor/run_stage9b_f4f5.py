"""STAGE 9B GATE REPAIR confirmation: F4/F5 on the SAME 6 TRAIN diagnostic
specs as run_stage9b_f0f3.py (identical budget 16, frozen PVT, frozen
learning, seed 0) -- F0 rows from that run are the baseline.

  F4 = gate_mode="measured_first"            (gate repair alone)
  F5 = F4 + ranker_mode="dpo_gated"          (gate repair + confidence-gated
                                              DPO, which the repair finally
                                              makes reachable)

12 jobs. Resumable: finished (mode, spec) pairs are skipped on rerun.
    python run_stage9b_f4f5.py
"""
import json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v3/stage9b_f0f3"
SPECS = [1, 2, 0, 6, 4, 5]
MODES = {"F4": {"gate_mode": "measured_first"},
        "F5": {"gate_mode": "measured_first", "ranker_mode": "dpo_gated"}}

def main():
    from agentic_raptor.electrical.pvt_eval import PvtConfig
    from run_qwen_ablation import _load
    import run_raptor_v2 as v2
    adapter = str(ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse")
    tok, model = _load(adapter)
    pvt = PvtConfig(enabled=True, process_corners=("tt","ff","ss"),
                    supply_voltages=(1.8,), temperatures_c=(27.0,))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    done = set()
    prior = OUT / "F4F5_ROWS.jsonl"
    if prior.is_file():
        for l in prior.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l); rows.append(r); done.add((r["mode"], r["spec_index"]))
        print(f"RESUME: {len(rows)} jobs already done", flush=True)
    with prior.open("a", encoding="utf-8") as f:
        for idx in SPECS:
            for mode, kw in MODES.items():
                if (mode, idx) in done: continue
                t0 = time.time()
                tr = v2.run_pipeline(model, tok, adapter, split="train", spec_index=idx,
                                     budget=16, calibrate=False, seed=0,
                                     learning_mode="frozen", pvt_config=pvt,
                                     out_prefix=f"S9B_{mode}", **kw)
                s9 = tr.get("stage9_verification") or {}
                sr = tr.get("stage8_ranker") or {}
                row = {"mode": mode, "spec_index": idx,
                      "spec_id": tr["stage1_spec"]["spec_id"],
                      "pass": bool(tr["nominal"]["complete_pass"]),
                      "distance": s9.get("distance_to_feasibility"),
                      "basis": sr.get("decision_basis"),
                      "deciding_level": sr.get("deciding_level"),
                      "gate_mode": sr.get("gate_mode"),
                      "measured_tiers": [sr.get("measured_tier_A"),
                                         sr.get("measured_tier_B")],
                      "measured_evidence": {"A": sr.get("measured_evidence_A"),
                                            "B": sr.get("measured_evidence_B")},
                      "gate_fallback": sr.get("dpo_gate_fallback"),
                      "verified": s9.get("verified_designs"),
                      "measured_dists": {k: (v or {}).get("distance")
                                         for k, v in (s9.get("measured") or {}).items()},
                      "selected": sr.get("selected_design"),
                      "pvt_robust": (tr.get("pvt") or {}).get("robust_complete_pass"),
                      "spice": tr["spice_usage"]["total_spice_calls"],
                      "ctfp": (tr.get("stage6_sizing", {}).get(sr.get("selected_design",""), {})
                               or {}).get("calls_to_first_pass"),
                      "runtime_s": round(time.time()-t0, 1)}
                rows.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
                print(f"{mode} idx={idx}: pass={row['pass']} dist={row['distance']} "
                     f"basis={row['basis']} ({row['runtime_s']}s)", flush=True)
    # summary incl. F0 baseline from the F0-F3 run for direct comparison
    baseline = []
    f0f3 = OUT / "F0F3_ROWS.jsonl"
    if f0f3.is_file():
        baseline = [json.loads(l) for l in f0f3.read_text(encoding="utf-8").splitlines()
                    if l.strip() and json.loads(l)["mode"] == "F0"]
    summary = {}
    for mode, mr in [("F0_baseline", baseline)] + [
            (m, [r for r in rows if r["mode"] == m]) for m in MODES]:
        zs = [1.0 if r["pass"] else max(-1.0, 1-2*r["distance"]) for r in mr
              if r["distance"] is not None or r["pass"]]
        summary[mode] = {"pass": sum(r["pass"] for r in mr), "n": len(mr),
                        "mean_z": round(sum(zs)/len(zs), 4) if zs else None,
                        "robust": sum(1 for r in mr if r["pvt_robust"]),
                        "bases": sorted({r["basis"] for r in mr if r.get("basis")}),
                        "spice": sum(r["spice"] for r in mr),
                        "runtime": round(sum(r["runtime_s"] for r in mr), 1)}
    (OUT / "F4F5_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)

if __name__ == "__main__":
    main()
