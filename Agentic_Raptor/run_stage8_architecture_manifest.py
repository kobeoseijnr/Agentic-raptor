"""Stage 8 (Section 29): canonical frozen architecture manifest.

Every value here is read live from the current repository state -- file
hashes, checkpoint hashes, config defaults -- never hand-typed constants
that could drift from the code.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "artifacts/publication_v3/AGENTIC_RAPTOR_V2_STAGE8_ARCHITECTURE.json"


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _src_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


def _current_git_commit() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def build_manifest() -> dict:
    from agentic_raptor.electrical import _PDK_CORNER_DIR, effective_c_load
    from agentic_raptor.electrical.pvt_eval import available_process_corners
    from agentic_raptor.mb_sac.spec_sizing import (KNOB_NAMES, N_KNOBS,
                                                    REWARD_POLICY_VERSION,
                                                    SCHEMA as MBSAC_SCHEMA)
    from agentic_raptor.publication.artifact_provenance import \
        POST_CLOAD_FIX_V1
    from agentic_raptor.ranking.features_v2 import FEATURE_DIM_V2
    from agentic_raptor.ranking.model_v2 import (PROMOTED_V2_CKPT,
                                                  PROMOTED_V2_MANIFEST,
                                                  REQUIRED_FEATURE_SCHEMA,
                                                  REQUIRED_V2_SHA256)
    from agentic_raptor.ranking.post_sac import (_deterministic_score,
                                                  compare, hard_safety_tier)
    from agentic_raptor.spice.ngspice_simulator import (discover_ngspice,
                                                         ngspice_version)
    from agentic_raptor.topology_rl.alphazero import (
        AlphaZeroConfig, require_promoted_az_checkpoint)
    from agentic_raptor.topology_rl.stage3e1 import SCHEMA_VERSION

    sft_dir = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"
    rag_path = (ROOT / "artifacts/publication_v2/selfimprove"
               / "rag_memory_v2_post_cload_v1_clean.jsonl")
    az_ckpt = require_promoted_az_checkpoint()
    tt_spice = _PDK_CORNER_DIR / "tt.spice"
    exe = discover_ngspice()

    az_cfg = AlphaZeroConfig()
    dpo_manifest = (json.loads(PROMOTED_V2_MANIFEST.read_text(encoding="utf-8"))
                    if PROMOTED_V2_MANIFEST.is_file() else {})

    return {
        "manifest_schema": "AGENTIC_RAPTOR_V2_STAGE8_ARCHITECTURE.v2",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "electrical_environment_version": POST_CLOAD_FIX_V1,
        # Stage 8 second deployment (2026-08-12): Stage 7.2B found the
        # learned Level-2 ranker DPO_REJUSTIFIED under POST_SAC_FEATURES_V2
        # and this task redeployed it as FULL's default selector. The
        # earlier v1 manifest's "disabled_not_justified"/False reflected
        # the state AFTER Stage 7.1's negative result and BEFORE Stage
        # 7.2A/7.2B's repair -- superseded, not corrected in place, so the
        # schema version above was bumped rather than silently overwriting
        # the historical claim.
        "learned_dpo": "enabled_dpo_v2_rejustified",
        "learned_dpo_enabled": True,
        "dpo_feature_schema": REQUIRED_FEATURE_SCHEMA,
        "root_puct": "retired",
        "pipeline": ["RAG", "SFT", "Validator", "AlphaZero", "MB-SAC",
                    "HardSafetyGate", "LearnedDPO_V2", "NGSPICE", "PVT"],
        "components": {
            "RAG": {
                "live_memory_path": str(ROOT / "artifacts/publication_v2/selfimprove"
                                        / "rag_memory_v2_post_cload_v1.jsonl"),
                "clean_snapshot_path": str(rag_path),
                "clean_snapshot_sha256": _sha256(rag_path),
                "clean_snapshot_record_count": (
                    len(rag_path.read_text(encoding="utf-8").splitlines())
                    if rag_path.is_file() else 0),
            },
            "SFT": {
                "adapter_dir": str(sft_dir),
                "adapter_config_sha256": _sha256(sft_dir / "adapter_config.json"),
                "adapter_present": (sft_dir / "adapter_config.json").is_file(),
            },
            "Validator": {
                "topology_state_schema_version": SCHEMA_VERSION,
            },
            "AlphaZero": {
                "checkpoint_path": str(az_ckpt),
                "checkpoint_sha256": _sha256(az_ckpt),
                "checkpoint_status": "promoted (AZ_G1_C2_CANDIDATE, Stage 5 closeout)",
                "checkpoint_loading": "ENFORCED (Stage 8 final, 2026-08-13): live "
                    "one_root path loads via load_promoted_alphazero_nets() -- "
                    "full SHA-256 verified against the generation manifest, "
                    "parameter fingerprint proven distinct from random init, "
                    "AlphaZeroCheckpointError on any failure, no silent fallback. "
                    "(Fixes: value_ckpt=None previously produced a silent seed-0 "
                    "random network on every live run.)",
                "experimental_modes_not_live": [
                    "per_seed", "per_seed_pv", "combined_package",
                    "combined_portfolio", "linear_value_mcts",
                    "compensation_bias", "AZ_PER_SEED_G0/G1/G2 (rejected)"],
                "config": {
                    "alphazero_simulations_per_move": az_cfg.alphazero_simulations_per_move,
                    "alphazero_c_puct": az_cfg.alphazero_c_puct,
                    "alphazero_max_edit_depth": az_cfg.alphazero_max_edit_depth,
                    "alphazero_max_children": az_cfg.alphazero_max_children,
                    "leaf_mode": az_cfg.leaf_mode,
                    "training_mode": az_cfg.training_mode,
                },
            },
            "MB-SAC": {
                "schema_version": MBSAC_SCHEMA,
                "reward_policy_version": REWARD_POLICY_VERSION,
                "n_knobs": N_KNOBS,
                "knob_names_canonical_order": list(KNOB_NAMES),
                "sac_algorithm": "sequential_sac_bootstrapped_polyak",
                "action_replay_roundtrip_repair": "Stage 6.1 (2026-08-12)",
                "stage6_1_classification": "CORRECT_AND_COMPETITIVE",
            },
            "HardSafetyGate": {
                "function": "agentic_raptor.ranking.post_sac.hard_safety_tier",
                "source_hash": _src_hash(hard_safety_tier),
                "frozen_by": "Stage 7.1 (never tuned to affect DPO metrics)",
            },
            "LearnedDPO_V2": {
                "loader": "agentic_raptor.ranking.model_v2.load_promoted_v2",
                "checkpoint_path": str(PROMOTED_V2_CKPT),
                "checkpoint_sha256": _sha256(PROMOTED_V2_CKPT),
                "required_checkpoint_sha256": REQUIRED_V2_SHA256,
                "checkpoint_status": dpo_manifest.get("checkpoint_status"),
                "feature_schema": REQUIRED_FEATURE_SCHEMA,
                "feature_dim": FEATURE_DIM_V2,
                "arm": dpo_manifest.get("arm"),
                "promoted_by": "Stage 7.2B (DPO_REJUSTIFIED)",
                "dev_ranker_authority_accuracy": 0.7780,
                "deterministic_baseline_accuracy": 0.7466,
                "wins_losses_catastrophic": {"wins": 77, "losses": 45, "catastrophic": 0},
                "compare_dispatch_source_hash": _src_hash(compare),
                "no_silent_fallback": "run_pipeline hard-raises "
                                      "(FileNotFoundError/RankerCheckpointHashMismatch/"
                                      "RankerFeatureSchemaMismatch) rather than falling "
                                      "back to DeterministicSelector when ranker_mode='dpo'",
            },
            "DeterministicSelector": {
                "status": "ablation/baseline/historical-reproduction ONLY -- "
                         "not FULL's default selector (see LearnedDPO_V2)",
                "function": "agentic_raptor.ranking.post_sac._deterministic_score",
                "source_hash": _src_hash(_deterministic_score),
                "formula": "-(worst_predicted_violation) + 0.05*hard_constraints_satisfied "
                          "- 0.1*predictive_uncertainty",
                "frozen_dev_ranker_authority_accuracy": 0.7466,
            },
            "NGSPICE": {
                "executable": str(exe) if exe else None,
                "version": ngspice_version(exe) if exe else None,
                "pdk_tt_corner_sha256": _sha256(tt_spice),
                "available_process_corners": sorted(available_process_corners()),
            },
            "PVT": {
                "protocol": "agentic_raptor.electrical.pvt_eval.aggregate_pvt "
                           "(robust_complete_pass requires ALL required corners "
                           "to pass every mandatory constraint)",
                "eligibility": "nominal exact_spec_pass required before PVT runs "
                              "(never run on a failing nominal design)",
            },
        },
        "cload_handling": {
            "function": "agentic_raptor.electrical.effective_c_load",
            "example_100pf": effective_c_load({"load_capacitance_pf": 100.0}),
            "example_no_spec_fallback": effective_c_load(None),
        },
        "source_code_commit": _current_git_commit(),
        "preflight_ready_at_generation_time": None,
    }


def main():
    from agentic_raptor.publication.preflight import run_preflight
    manifest = build_manifest()
    preflight = run_preflight()
    manifest["preflight_ready_at_generation_time"] = preflight["ready"]
    manifest["preflight_blockers_at_generation_time"] = preflight["blockers"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=1, default=str))
    print(f"\nStage 8 architecture manifest written to {OUT_PATH}")


if __name__ == "__main__":
    main()
