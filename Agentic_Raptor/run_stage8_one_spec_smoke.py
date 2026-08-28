"""Stage 8 Part B (Section 18 of the spec): ONE live spec through the ENTIRE
FULL pipeline with the deployed DPO V2 selector, before committing to the
real 6-spec integration run.

Requires, and hard-fails if any is false:
  - trace["stage8_ranker"]["use_learned_dpo"] is True
  - trace["stage8_ranker"]["selector"] == "learned_dpo"
  - trace["models"]["ranker"]["hash"] == the promoted V2 checkpoint hash
  - trace["stage8_ranker"]["feature_schema"] == "POST_SAC_FEATURES_V2"
  - run_pipeline's own internal invariants held (verify_job(), reused
    unmodified from run_stage8_integration_diagnostic.py)

Uses spec_index=3 (train split) -- deliberately NOT one of the 6 indices
DIAGNOSTIC_SPEC_INDICES reserves for the real integration run, so the smoke
probe itself never becomes (or is mistaken for) one of the 6 diagnostic
results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "artifacts/publication_v3/stage8_integration_diagnostic/ONE_SPEC_SMOKE_RESULT.json"

SMOKE_SPEC_INDEX = 3


def main():
    from run_qwen_ablation import _load
    from run_stage8_integration_diagnostic import run_one, verify_job

    adapter = str(ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse")
    assert Path(adapter, "adapter_config.json").is_file(), f"not a peft adapter dir: {adapter}"
    print(f"loading model from adapter: {adapter}", flush=True)
    tok, model = _load(adapter)

    print(f"\n=== SMOKE: spec_index={SMOKE_SPEC_INDEX} ===", flush=True)
    result = run_one(model, tok, adapter, SMOKE_SPEC_INDEX)
    trace = result["trace"]
    verify = verify_job(trace)

    from agentic_raptor.ranking.model_v2 import REQUIRED_V2_SHA256
    sr = trace["stage8_ranker"]
    checks = {
        "ranker_mode_is_dpo": sr.get("selector") == "learned_dpo",
        "use_learned_dpo_true": sr.get("use_learned_dpo") is True,
        "checkpoint_hash_matches_promoted_v2": trace["models"]["ranker"]["hash"] == REQUIRED_V2_SHA256,
        "feature_schema_is_v2": sr.get("feature_schema") == "POST_SAC_FEATURES_V2",
        "verify_job_all_ok": verify["all_ok"],
    }
    smoke_ok = all(checks.values())

    report = {"spec_index": SMOKE_SPEC_INDEX, "runtime_s": result["runtime_s"],
             "checks": checks, "smoke_ok": smoke_ok,
             "stage8_ranker": sr, "ranker_model": trace["models"]["ranker"],
             "verify": verify,
             "nominal_complete_pass": trace["nominal"]["complete_pass"]}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

    print(json.dumps(checks, indent=1), flush=True)
    print(f"\nSMOKE {'PASSED' if smoke_ok else 'FAILED'} -- report at {OUT_PATH}", flush=True)
    if not smoke_ok:
        print(f"problems: {verify['problems']}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
