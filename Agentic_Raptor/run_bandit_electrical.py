"""BANDIT_TOP2_V1 electrical confirmation on the SAME 6 TRAIN diagnostic
specs as run_stage9b_f0f3.py (identical budget 16, frozen PVT, frozen
learning, seed 0).

BOTH ARMS RUN ON THE REPAIRED DOWNSTREAM (gate_mode="measured_first"):
the F0-F3 campaign proved the surrogate-driven gate wastes better
candidates (22/24 decision authority, 7/15 both-verified decisions
wrong), so a selector comparison on top of it would measure gate noise,
not selector quality. The controlled comparison is therefore against F4
(same repaired gate, AlphaZero selection) from run_stage9b_f4f5.py:
  G1 = search="bandit_top2"     (bandit ranks LLM proposals, no AlphaZero)
  G2 = search="bandit_top2_az"  (Option C: AlphaZero generates, bandit decides)
12 jobs total. Resumable: rerun after an interruption and finished
(mode, spec) pairs are skipped. Run AFTER run_stage9b_f4f5.py so the
F4 baseline exists:
    python run_stage9b_f4f5.py
    python run_bandit_electrical.py
"""
import json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v3/bandit_top2_v1"
SPECS = [1, 2, 0, 6, 4, 5]
MODES = {"G1": {"search": "bandit_top2", "gate_mode": "measured_first"},
        "G2": {"search": "bandit_top2_az", "gate_mode": "measured_first"}}

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
    prior = OUT / "BANDIT_ELEC_ROWS.jsonl"
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
                                     out_prefix=f"BANDIT_{mode}", **kw)
                s9 = tr.get("stage9_verification") or {}
                sr = tr.get("stage8_ranker") or {}
                s5 = tr.get("stage5_alphazero") or {}
                row = {"mode": mode, "spec_index": idx,
                      "spec_id": tr["stage1_spec"]["spec_id"],
                      "pass": bool(tr["nominal"]["complete_pass"]),
                      "distance": s9.get("distance_to_feasibility"),
                      "basis": sr.get("decision_basis"),
                      "verified": s9.get("verified_designs"),
                      "measured_dists": {k: (v or {}).get("distance")
                                         for k, v in (s9.get("measured") or {}).items()},
                      "selected": sr.get("selected_design"),
                      "weights_sha256": s5.get("weights_sha256"),
                      "pool_size": s5.get("pool_size"),
                      "az_contributed": s5.get("az_contributed"),
                      "bandit_top2": [r["canonical_graph_hash"]
                                      for r in (s5.get("ranking") or [])
                                      if r.get("selected_top2")],
                      "pvt_robust": (tr.get("pvt") or {}).get("robust_complete_pass"),
                      "spice": tr["spice_usage"]["total_spice_calls"],
                      "ctfp": (tr.get("stage6_sizing", {}).get(sr.get("selected_design",""), {})
                               or {}).get("calls_to_first_pass"),
                      "runtime_s": round(time.time()-t0, 1)}
                rows.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
                print(f"{mode} idx={idx}: pass={row['pass']} dist={row['distance']} "
                     f"pool={row['pool_size']} az={row['az_contributed']} "
                     f"({row['runtime_s']}s)", flush=True)
    # summary incl. the controlled baselines: F4 (repaired gate + AlphaZero
    # selection) and F0 (unrepaired current FULL), when their rows exist
    baselines = []
    f4 = ROOT / "artifacts/publication_v3/stage9b_f0f3/F4F5_ROWS.jsonl"
    f0 = ROOT / "artifacts/publication_v3/stage9b_f0f3/F0F3_ROWS.jsonl"
    if f4.is_file():
        baselines.append(("F4_baseline", [json.loads(l) for l in f4.read_text(encoding="utf-8").splitlines()
                                          if l.strip() and json.loads(l)["mode"] == "F4"]))
    if f0.is_file():
        baselines.append(("F0_unrepaired", [json.loads(l) for l in f0.read_text(encoding="utf-8").splitlines()
                                            if l.strip() and json.loads(l)["mode"] == "F0"]))
    summary = {}
    for mode, mr in baselines + [(m, [r for r in rows if r["mode"] == m]) for m in MODES]:
        zs = [1.0 if r["pass"] else max(-1.0, 1-2*r["distance"]) for r in mr if r["distance"] is not None or r["pass"]]
        summary[mode] = {"pass": sum(r["pass"] for r in mr), "n": len(mr),
                        "mean_z": round(sum(zs)/len(zs), 4) if zs else None,
                        "robust": sum(1 for r in mr if r["pvt_robust"]),
                        "spice": sum(r["spice"] for r in mr),
                        "runtime": round(sum(r["runtime_s"] for r in mr), 1)}
    (OUT / "BANDIT_ELEC_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)

if __name__ == "__main__":
    main()
