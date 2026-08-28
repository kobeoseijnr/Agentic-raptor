"""NULLING-BRANCH PAYOFF PROBE: one frozen FULL run on the boundary spec
(heldout idx 2, t_boundary_topology_0008 -- 0 passes in 243 jobs across
all campaigns of the collapsed-realization era). First run where a 2s_rc/
3s_rc candidate is a physically different circuit with a live rz_x knob.
    python run_boundary_probe.py          (~10 min)
"""
import json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def main():
    from agentic_raptor.electrical.pvt_eval import PvtConfig
    from run_qwen_ablation import _load
    import run_raptor_v2 as v2
    adapter = str(ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse")
    tok, model = _load(adapter)
    pvt = PvtConfig(enabled=True, process_corners=("tt", "ff", "ss"),
                    supply_voltages=(1.8,), temperatures_c=(27.0,))
    t0 = time.time()
    tr = v2.run_pipeline(model, tok, adapter, split="heldout", spec_index=2,
                         budget=16, calibrate=False, seed=0,
                         learning_mode="frozen", pvt_config=pvt,
                         out_prefix="RZPROBE")
    s9 = tr.get("stage9_verification") or {}
    sr = tr.get("stage8_ranker") or {}
    n = tr.get("nominal") or {}
    out = {"pass": bool(n.get("complete_pass")),
          "gain_db": n.get("gain_db"), "pm_deg": n.get("pm_deg"),
          "ugbw_hz": n.get("ugbw_hz"),
          "distance": s9.get("distance_to_feasibility"),
          "measured": {k: (v or {}).get("distance")
                       for k, v in (s9.get("measured") or {}).items()},
          "selected": sr.get("selected_design"), "basis": sr.get("decision_basis"),
          "pvt_robust": (tr.get("pvt") or {}).get("robust_complete_pass"),
          "runtime_s": round(time.time() - t0, 1)}
    p = ROOT / "artifacts/publication_v3/RZPROBE_RESULT.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)

if __name__ == "__main__":
    main()
