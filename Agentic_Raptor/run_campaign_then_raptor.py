"""Chain runner: STEP 1 (3-generation self-improvement campaign) then, on
success, STEP 2 (full RAPTOR pipeline: RAG -> trained LLM -> MCTS -> SAC
sizing -> ngspice -> post-sizing score -> feedback channels) using the last
generation's ACCEPTED checkpoint.

Run:  python run_campaign_then_raptor.py [n_generations]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable
N = sys.argv[1] if len(sys.argv) > 1 else "3"

print(f"=== STEP 1: self-improvement campaign ({N} generations) ===",
      flush=True)
r = subprocess.run([PY, "-u", "run_self_improvement.py", N])
if r.returncode != 0:
    print(f"STEP 1 FAILED (exit {r.returncode}) — step 2 not started",
          flush=True)
    sys.exit(r.returncode)

camps = sorted(Path("artifacts/self_improvement").glob("camp_*"))
log = camps[-1] / "logs" / "generations.jsonl"
last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
accepted = last["accepted_checkpoint"]
print(f"\n=== STEP 2: full RAPTOR pipeline (MCTS + SAC) ===", flush=True)
print(f"campaign: {camps[-1].name}  accepted checkpoint: {accepted} "
      f"(hash {last['accepted_checkpoint_hash']}, "
      f"dpo_accepted={last['dpo_update_accepted']})", flush=True)
env = dict(os.environ, AGENTIC_RAPTOR_ADAPTER=accepted)
r2 = subprocess.run([PY, "-u", "run_full_raptor.py"], env=env)
print(f"\n=== CHAIN COMPLETE (step 2 exit {r2.returncode}) ===", flush=True)
sys.exit(r2.returncode)
