"""AUTO-CONTINUE after the tier-2 SFT retrain (2026-08-17).

Waits for sft_adapter_tier2 to finish training, runs a QUALITY GATE, and
only then launches the Tier-2 campaign:

  gate: load the new adapter, prompt it on 6 tier-2 TRAIN specs, and
        require that it PROPOSES tier-2 structures (4-stage / cascode /
        class-AB) on >= 4 of 6 -- i.e. the retrain actually taught the new
        vocabulary. A retrain that still only emits stock families is
        reported as FAILED and the campaign is NOT started.

    python run_tier2_after_sft.py        (leave running; polls every 60 s)
"""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_tier2"
REPORT = ROOT / "artifacts/publication_v3/tier2/SFT_TIER2_GATE.json"
CAMPAIGN = [sys.executable, "run_ablation_v3.py", "--paper-mode", "--pvt",
            "--tag", "TIER2", "--split", "tier2", "--specs", "18", "--seeds", "0",
            "--arms", "A0,A1,A2,A3,A4,A5,A6,A7,A8,AG_FULL",
            "--adapter", str(ADAPTER)]

def adapter_ready() -> bool:
    return (ADAPTER.is_dir() and (ADAPTER / "adapter_config.json").is_file()
            and any(ADAPTER.glob("adapter_model.*")))

def quality_gate() -> dict:
    from run_qwen_ablation import _load
    from run_raptor_v2 import propose_and_validate, ig
    from agentic_raptor.publication.tier2 import load_tier2_specs
    tok, model = _load(str(ADAPTER))
    specs = load_tier2_specs("tier2_train")[:6]
    hits, detail = 0, []
    for s in specs:
        spec = ig.parse_spec(s["prompt"]); spec["spec_id"] = s["spec_id"]
        out = propose_and_validate(model, tok, s["prompt"], target_k=5,
                                   conditioning="exclusion", seed0=0, spec=spec,
                                   stall_stop=2)
        fams = sorted({c["canonical_family"] for c in out["candidates"]})
        t2 = [f for f in fams if f.startswith("4s") or "_cas" in f or "_ab" in f]
        hits += bool(t2)
        detail.append({"spec": s["spec_id"], "families": fams, "tier2": t2})
        print(f"  {s['spec_id']}: {fams}", flush=True)
    return {"adapter": str(ADAPTER), "specs_tested": len(specs),
            "specs_with_tier2_proposals": hits, "detail": detail,
            "verdict": "PASS" if hits >= 4 else "FAIL"}

def main():
    print(f"waiting for {ADAPTER.name} ...", flush=True)
    while not adapter_ready():
        time.sleep(60)
    # let the trainer flush its final files
    time.sleep(120)
    print("adapter present -- running quality gate", flush=True)
    rep = quality_gate()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("specs_with_tier2_proposals", "verdict")}), flush=True)
    if rep["verdict"] != "PASS":
        raise SystemExit("SFT gate FAILED: retrained model does not propose tier-2 "
                         "structures -- campaign NOT started. See " + str(REPORT))
    print("=== gate PASS: launching Tier-2 campaign ===", flush=True)
    rc = subprocess.call(CAMPAIGN, cwd=str(ROOT))
    print(f"=== campaign exit code {rc} ===", flush=True)

if __name__ == "__main__":
    main()
