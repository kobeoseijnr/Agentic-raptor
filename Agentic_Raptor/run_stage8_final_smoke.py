"""Stage 8 FINAL integration: one real TRAIN-domain end-to-end FULL smoke
with the hardened checkpoint loading. Uses run_pipeline's LIVE defaults
(search=one_root, ranker=dpo, value_ckpt=None -> enforced promoted loader),
learning frozen, PVT on. spec_index=3 (not one of the 6 diagnostic specs)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v3/stage8_final_smoke"
SMOKE_SPEC_INDEX = 3


def main():
    from agentic_raptor.electrical.pvt_eval import PvtConfig
    from agentic_raptor.ranking.model_v2 import REQUIRED_V2_SHA256
    from run_qwen_ablation import _load
    from run_raptor_v2 import run_pipeline

    adapter = str(ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse")
    tok, model = _load(adapter)
    t0 = time.time()
    trace = run_pipeline(model, tok, adapter, split="train",
                         spec_index=SMOKE_SPEC_INDEX, budget=16, calibrate=False,
                         seed=0, learning_mode="frozen",
                         pvt_config=PvtConfig(enabled=True,
                                              process_corners=("tt", "ff", "ss"),
                                              supply_voltages=(1.8,),
                                              temperatures_c=(27.0,)))
    runtime = round(time.time() - t0, 1)

    s5 = trace["stage5_alphazero"]
    prov = s5.get("checkpoint_provenance") or {}
    sr = trace["stage8_ranker"]
    report = {
        "spec_id": trace["stage1_spec"]["spec_id"],
        "spec_hash": trace["stage1_spec"]["spec_hash"],
        "rag_records": trace["stage2_rag"]["records"],
        "llm_candidates": trace["stage3_propose"]["distinct"],
        "llm_valid": trace["stage3_propose"]["distinct"],
        "az_checkpoint_loaded": prov.get("checkpoint_loaded"),
        "az_checkpoint_sha256": prov.get("checkpoint_sha256"),
        "az_fingerprint": prov.get("parameter_fingerprint"),
        "az_search_mode": s5["search_topology"],
        "selected_topology_hashes": trace["provenance_chain"]["puct_selected"],
        "topology_identity_ok": (
            set(trace["stage6_sizing"][l]["topology_hash"] for l in ("A", "B"))
            == set(trace["provenance_chain"]["puct_selected"])),
        "mbsac_calls": trace["spice_usage"]["optimization_spice_calls"],
        "hard_gate_decision_basis": sr["decision_basis"],
        "dpo_selector": sr["selector"],
        "dpo_checkpoint_ok": sr["ranker_checkpoint_hash"] == REQUIRED_V2_SHA256,
        "dpo_scores": {"A": sr["ranker_score_A"], "B": sr["ranker_score_B"]},
        "final_pass": trace["nominal"]["complete_pass"],
        "final_distance": trace["stage9_verification"].get("distance_to_feasibility"),
        "cload_ok": not trace["nominal"]["c_load_unexplained_mismatch"],
        "pvt_robust": trace.get("pvt", {}).get("robust_complete_pass"),
        "fom": trace["fom"],
        "runtime_s": runtime,
        "total_spice_calls": trace["spice_usage"]["total_spice_calls"],
    }
    checks = {
        "checkpoint_loaded": report["az_checkpoint_loaded"] is True,
        "checkpoint_sha_correct": (report["az_checkpoint_sha256"] or "").startswith("822a305c"),
        "search_is_live_one_root": report["az_search_mode"] == "true_alphazero_multi_depth_edit_search",
        "topology_identity": report["topology_identity_ok"],
        "dpo_v2_active": report["dpo_selector"] == "learned_dpo" and report["dpo_checkpoint_ok"],
        "cload_ok": report["cload_ok"],
    }
    report["checks"] = checks
    report["smoke_ok"] = all(checks.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STAGE8_FINAL_SMOKE.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")
    (OUT / "trace.json").write_text(json.dumps(trace, indent=1, default=str),
                                    encoding="utf-8")
    print(json.dumps(report, indent=1, default=str), flush=True)
    if not report["smoke_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
