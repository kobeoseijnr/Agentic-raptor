"""STAGE 9B Part 12: F0-F3 interaction test on the 6 TRAIN diagnostic specs.
F0 current FULL | F1 +champion-challenger AZ | F2 +confidence-gated DPO |
F3 both. Everything else identical (budget 16, frozen PVT, frozen learning,
enforced promoted checkpoint)."""
import json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v3/stage9b_f0f3"
SPECS = [1, 2, 0, 6, 4, 5]
MODES = {"F0": {}, "F1": {"search": "one_root_cc"},
        "F2": {"ranker_mode": "dpo_gated"},
        "F3": {"search": "one_root_cc", "ranker_mode": "dpo_gated"}}

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
    prior = OUT / "F0F3_ROWS.jsonl"
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
                      "gate_fallback": sr.get("dpo_gate_fallback"),
                      "verified": s9.get("verified_designs"),
                      "measured_dists": {k: (v or {}).get("distance")
                                         for k, v in (s9.get("measured") or {}).items()},
                      "selected": sr.get("selected_design"),
                      "champion_hash": (tr.get("stage5_alphazero") or {}).get("champion_hash"),
                      "pvt_robust": (tr.get("pvt") or {}).get("robust_complete_pass"),
                      "spice": tr["spice_usage"]["total_spice_calls"],
                      "ctfp": (tr.get("stage6_sizing", {}).get(sr.get("selected_design",""), {})
                               or {}).get("calls_to_first_pass"),
                      "runtime_s": round(time.time()-t0, 1)}
                rows.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
                print(f"{mode} idx={idx}: pass={row['pass']} dist={row['distance']} "
                     f"basis={row['basis']} fb={row['gate_fallback']} ({row['runtime_s']}s)", flush=True)
    summary = {}
    for mode in MODES:
        mr = [r for r in rows if r["mode"] == mode]
        zs = [1.0 if r["pass"] else max(-1.0, 1-2*r["distance"]) for r in mr if r["distance"] is not None or r["pass"]]
        summary[mode] = {"pass": sum(r["pass"] for r in mr), "n": len(mr),
                        "mean_z": round(sum(zs)/len(zs), 4) if zs else None,
                        "robust": sum(1 for r in mr if r["pvt_robust"]),
                        "spice": sum(r["spice"] for r in mr),
                        "runtime": round(sum(r["runtime_s"] for r in mr), 1),
                        "gate_fallbacks": sum(1 for r in mr if r.get("gate_fallback"))}
    (OUT / "F0F3_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)

if __name__ == "__main__":
    main()
