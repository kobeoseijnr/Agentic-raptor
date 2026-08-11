"""Publication freeze manifest: hash every component the study depends on.

No git repository exists, so the code identity is a deterministic hash over
the source tree (path + bytes, sorted). Model/adapters/checkpoints are hashed
by content; frozen splits by their manifests; the full test result is
recorded verbatim.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from agentic_raptor.llm_dpo.integrity import (SCHEMA as INTEGRITY_SCHEMA,
                                              TESTBENCH_HASH, sha_checkpoint,
                                              sha_file, sha_json)
from agentic_raptor.mb_sac.spec_sizing import (REWARD_POLICY_VERSION,
                                               SCHEMA as SIZING_SCHEMA,
                                               action_space_manifest)
from agentic_raptor.publication import FREEZE, ROOT


def source_tree_hash() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "agentic_raptor").rglob("*.py")) + \
            sorted(ROOT.glob("*.py")) + sorted((ROOT / "tests").glob("*.py")):
        if "__pycache__" in str(f):
            continue
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _hash_if(path: Path, kind: str):
    p = ROOT / path
    if kind == "dir":
        return sha_checkpoint(p) if p.is_dir() else "absent"
    return sha_file(p) if p.is_file() else "absent"


def _alphazero_checkpoint_hash() -> dict:
    """The PROMOTED AlphaZero checkpoint's hash, if one exists -- "no
    promoted checkpoint yet" is an honest, expected freeze-manifest state
    immediately after the FULL cutover migration, not an error."""
    from agentic_raptor.topology_rl.alphazero import (
        AlphaZeroSelectionError, require_promoted_az_checkpoint)
    try:
        p = require_promoted_az_checkpoint()
        return {"status": "PROMOTED", "checkpoint_hash": _hash_if(p, "file")}
    except AlphaZeroSelectionError as exc:
        return {"status": "NO_PROMOTED_CHECKPOINT", "detail": str(exc)}


def run_tests() -> dict:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       capture_output=True, text=True, cwd=str(ROOT),
                       timeout=1200)
    tail = (r.stdout or "").strip().splitlines()[-1:]
    return {"exit_code": r.returncode, "summary": tail[0] if tail else ""}


def build_manifest(latest_campaign: str | None = None,
                   with_tests: bool = True) -> dict:
    import peft
    import torch
    import transformers
    from agentic_raptor.llm_dpo import MODEL_ID
    camps = sorted((ROOT / "artifacts/self_improvement").glob("camp_*"))
    camp = (ROOT / "artifacts/self_improvement" / latest_campaign
            if latest_campaign else (camps[-1] if camps else None))
    gens = []
    if camp and (camp / "logs/generations.jsonl").is_file():
        gens = [json.loads(x) for x in
                (camp / "logs/generations.jsonl").read_text(
                    encoding="utf-8").splitlines()]
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repository": {"git": "not_a_git_repository",
                       "source_tree_hash": source_tree_hash()},
        "models": {
            "qwen_base": MODEL_ID,
            "reference_campaign": camp.name if camp else None,
            "generation_checkpoints": {
                f"gen{g['generation_id']}": {
                    "sft": g["sft"]["output_sft_hash"],
                    "dpo": (g.get("dpo") or {}).get("output_dpo_hash"),
                    "accepted": g["accepted_checkpoint_hash"],
                    "accepted_path": g["accepted_checkpoint"]}
                for g in gens},
            # 2026-08-11: root-level PUCT retired (see
            # artifacts/publication_v3/ROOT_LEVEL_PUCT_RETIRED.json) --
            # replaced by TRUE_ALPHAZERO. "absent" here for the old path is
            # the CORRECT, expected post-cutover state, not a broken freeze.
            "policy_value_checkpoint_RETIRED":
                _hash_if(Path("artifacts/stage3e1/policy_value_ep0.pt"),
                         "file"),
            "alphazero_checkpoint": _alphazero_checkpoint_hash(),
            "sizing_memory": _hash_if(Path("artifacts/sizing_memory"), "dir"),
            "vlm_adapter":
                _hash_if(Path("artifacts/stage3e4b/vlm_sft_adapter"), "dir"),
        },
        "data": {
            "corpus_hash": corpus["split_manifest"]["corpus_hash"],
            "split_hash": corpus["split_manifest"]["split_hash"],
            "frozen_validation_hash": json.loads(
                (ROOT / "artifacts/stage3e4/frozen_exam.json").read_text()
            )["frozen_exam_hash"],
            "blind_test_hash": json.loads(
                (ROOT / "artifacts/stage3e4/blind_test.json").read_text()
            )["frozen_blind_hash"],
            "l4_memory": _hash_if(Path(
                "datasets/simulation_memory/self_improvement_runs.jsonl"),
                "file"),
            "az_value_targets": _hash_if(Path(
                "datasets/simulation_memory/az_value_targets.jsonl"), "file"),
            "topology_registry": sha_json(sorted(
                p.name for p in
                (ROOT / "datasets").rglob("*registry*") if p.is_file())),
        },
        "policies": {
            "integrity_schema": INTEGRITY_SCHEMA,
            "sizing_schema": SIZING_SCHEMA,
            "reward_policy": REWARD_POLICY_VERSION,
            "action_space": action_space_manifest(),
            "testbench_hash": TESTBENCH_HASH,
            "stats_version": "pubstats.1",
        },
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "ngspice": _ngspice_version(),
            "platform": sys.platform,
        },
    }
    if with_tests:
        manifest["test_result"] = run_tests()
    FREEZE.mkdir(parents=True, exist_ok=True)
    (FREEZE / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                          encoding="utf-8")
    return manifest


def _ngspice_version() -> str:
    try:
        from agentic_raptor.electrical import discover_ngspice
        exe = discover_ngspice()
        r = subprocess.run([exe, "--version"], capture_output=True, text=True,
                           timeout=30)
        return (r.stdout or "").splitlines()[0][:80] if r.stdout else str(exe)
    except Exception as exc:
        return f"unavailable: {exc}"[:80]


if __name__ == "__main__":
    print(json.dumps(build_manifest(), indent=1))
