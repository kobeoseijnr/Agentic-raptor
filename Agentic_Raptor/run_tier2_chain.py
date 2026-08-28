"""TIER-2 CHAIN (2026-08-17): retrain the proposer on the tier-2 corpus, then
AUTOMATICALLY launch the Tier-2 ablation campaign on the new adapter.

    python run_tier2_chain.py            # both steps, back to back
    python run_tier2_chain.py --skip-train   # adapter already trained

Step 1: train_proposer_diverse.py --corpus corpus_tier2_extension.json
        --out sft_adapter_tier2 --steps 1500          (~4 h GPU)
Step 2: run_ablation_v3.py --tag TIER2 --split tier2 --specs 29 --seeds 0
        --arms A0..A8,AG_FULL --adapter sft_adapter_tier2   (~1.5 days)

Guards: step 2 refuses to start unless the adapter directory exists AND
contains adapter weights (a failed/interrupted training never silently
runs the campaign on the OLD adapter). Both steps stream to the console;
the campaign is resumable on its own if interrupted.
"""
import argparse, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "artifacts/publication_v3/tier2/corpus_tier2_extension.json"
ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_tier2"

TRAIN = [sys.executable, "train_proposer_diverse.py", "--corpus", str(CORPUS),
         "--out", str(ADAPTER), "--steps", "1500"]
CAMPAIGN = [sys.executable, "run_ablation_v3.py", "--paper-mode", "--pvt",
            "--tag", "TIER2", "--split", "tier2", "--specs", "18", "--seeds", "0",
            "--arms", "A0,A1,A2,A3,A4,A5,A6,A7,A8,AG_FULL",
            "--adapter", str(ADAPTER)]

def adapter_ok() -> bool:
    if not ADAPTER.is_dir():
        return False
    return any(ADAPTER.glob("adapter_model.*")) and (ADAPTER / "adapter_config.json").is_file()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    if not CORPUS.is_file():
        raise SystemExit(f"missing tier-2 corpus: {CORPUS}")
    if not args.skip_train:
        print("=== STEP 1/2: SFT retrain on tier-2 corpus ===", flush=True)
        t0 = time.time()
        rc = subprocess.call(TRAIN, cwd=str(ROOT))
        print(f"=== training exit code {rc} after {(time.time()-t0)/60:.1f} min ===", flush=True)
        if rc != 0:
            raise SystemExit("training FAILED -- campaign NOT started (refusing to run on the old adapter)")
    if not adapter_ok():
        raise SystemExit(f"adapter not found/incomplete at {ADAPTER} -- campaign NOT started")
    print("=== STEP 2/2: Tier-2 campaign on the new adapter ===", flush=True)
    rc = subprocess.call(CAMPAIGN, cwd=str(ROOT))
    print(f"=== campaign exit code {rc} ===", flush=True)

if __name__ == "__main__":
    main()
