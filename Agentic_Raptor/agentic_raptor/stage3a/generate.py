"""Stage 3A debug generation: run real episodes, capture, derive, validate.

Runs N five-transistor-OTA episodes (>=5 spec groups) through the Stage 2
coordinator with the Stage 2 verified LLM/simulator config, captures every
artifact into content-addressed raw storage (originals preserved), then builds
the four derived datasets + splits + manifests + reports.

Preference labels (correction 5): derived ONLY from deterministic validation /
real SPICE / PVT via dpo.preference_pairs.compare (rules a-i); ranker scores are
recorded as PREDICTED features, never labels; candidate pools come from sizing
evaluations independent of any trained ranker (DPO disabled during generation).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agentic_raptor.stage3a import (
    SCHEMA_VERSION,
    RawStore,
    assign_splits,
    canonical_spec_hash,
    check_leakage,
    content_id,
    duplicate_report,
    fresh_id,
    validate_records,
    write_jsonl,
)

#: five debug spec variants → five specification groups (correction 3)
DEBUG_SPEC_VARIANTS = [
    {"target_gain_db": 45.0, "target_gbw_hz": 5e6},
    {"target_gain_db": 48.0, "target_gbw_hz": 8e6},
    {"target_gain_db": 50.0, "target_gbw_hz": 1e7},
    {"target_gain_db": 42.0, "target_gbw_hz": 3e6},
    {"target_gain_db": 46.0, "target_gbw_hz": 6e6},
]

FIELD_PROVENANCE = {  # correction 7 — shipped with every manifest
    "OBSERVED": ["llm_raw_output", "validation_result", "spice metrics/log paths", "pvt results", "stdout"],
    "DERIVED": ["*_hash", "*_id", "margins", "reward*", "split", "group_key", "tier", "preference labels"],
    "PREDICTED": ["dpo_score", "predicted_*", "uncertainty", "dynamics outputs"],
}


def run_debug_generation(base_config_path: str, data_root: str, seeds: list[int], mode: str = "REAL") -> dict[str, Any]:
    from agentic_raptor.coordinator.coordinator import AgenticCoordinator
    from agentic_raptor.utils.config import AgenticConfig

    root = Path(data_root)
    raw = RawStore(root / "raw")
    runs: list[dict[str, Any]] = []
    started = time.time()

    for i, seed in enumerate(seeds):
        cfg = AgenticConfig.from_yaml(base_config_path)  # Stage 2 verified llm/spice settings
        cfg.seed = seed
        cfg.dpo.enabled = False  # correction 5: candidates gathered ranker-independently
        cfg.output_dir = str(root.parent.parent / "outputs" / "stage3a" / f"run_{seed}")
        cfg.logging.decision_log = f"{cfg.output_dir}/decisions.jsonl"
        cfg._base_dir = str(Path(base_config_path).resolve().parent)  # type: ignore[attr-defined]
        variant = DEBUG_SPEC_VARIANTS[i % len(DEBUG_SPEC_VARIANTS)]
        for k, v in variant.items():
            cfg.specification.defaults[k] = v
        if mode == "MOCK":
            cfg.spice.simulator = "mock"
            cfg.llm.generator = "mock"
            cfg.llm.real_llm_enabled = False
        coordinator = AgenticCoordinator(cfg)
        result = coordinator.run_episode()
        runs.append(_capture(result, coordinator, raw, seed, mode))

    datasets = _derive(runs, root)
    datasets["report"]["wall_clock_s"] = round(time.time() - started, 1)
    return datasets


def _capture(result, coordinator, raw: RawStore, seed: int, mode: str) -> dict[str, Any]:
    rd = result.to_dict()
    spec = (rd.get("best_candidate") or {}).get("specifications") or {}
    spec_hash = canonical_spec_hash(spec) if spec else f"spec-none-{seed}"
    graph = (rd.get("best_candidate") or {}).get("topology")
    sim = (rd.get("best_candidate") or {}).get("simulation_result")
    # correction 2: copy SPICE raw evidence (netlist+log) into permanent storage
    spice_assets = []
    if sim and sim.get("raw_output_path"):
        for p in str(sim["raw_output_path"]).split(";"):
            for f in (p, str(Path(p).parent / "circuit.cir")):
                copied = raw.copy_file("spice_outputs", f)
                if copied:
                    spice_assets.append({"sha": copied[0], "path": copied[1], "original": f})
    _digest, summary_path = raw.put_text("reward_traces", json.dumps(rd, default=str), ".json")
    return {
        "run_id": fresh_id("run"), "episode_id": rd["episode_id"], "seed": seed,
        "execution_mode": mode, "dataset_version": SCHEMA_VERSION,
        "specification": spec, "specification_id": spec_hash, "group_key": spec_hash,
        "topology_family_id": "five_transistor_ota",
        "circuit_graph": graph,
        "circuit_graph_id": content_id("graph", graph) if graph else None,
        "graph_hash": (graph or {}).get("graph_id"),
        "reached_spice": bool(rd.get("update_report", {}).get("credit", {}).get("reached_spice")),
        "spice_evaluation_id": content_id("spice", sim) if sim else None,
        "simulation_result": sim, "spice_assets": spice_assets,
        "pvt_evaluated": bool((rd.get("best_candidate") or {}).get("metadata", {}).get("pvt")),
        "final_reward": rd.get("final_reward"),
        "decisions": rd.get("decisions", []),
        "trajectory_steps": [s.to_dict() for s in coordinator.topology_buffer._steps],
        "summary_asset": summary_path,
    }


def _derive(runs: list[dict], root: Path) -> dict[str, Any]:
    split_of = assign_splits([r["group_key"] for r in runs], seed=0)
    for r in runs:
        r["split"] = split_of[r["group_key"]]

    sft, prefs, topo, sac = [], [], [], []
    for r in runs:
        base = {k: r[k] for k in ("run_id", "episode_id", "specification_id", "group_key",
                                  "topology_family_id", "split", "execution_mode", "seed")}
        if r["circuit_graph"] and r["reached_spice"]:
            sim_ok = bool((r["simulation_result"] or {}).get("success"))
            margins = (r["simulation_result"] or {}).get("constraint_margins") or {}
            feasible = sim_ok and margins and all(v >= 0 for v in margins.values())
            tier = ("TIER_3_NOMINAL_FEASIBLE" if feasible else "TIER_2_SPICE_REACHED")
            sft.append({**base, "example_id": fresh_id("sft"),
                        "target_circuit_graph": r["circuit_graph"],
                        "target_graph_hash": r["circuit_graph_id"],
                        "spice_evidence": r["spice_evaluation_id"], "quality_score": r["final_reward"],
                        "tier": tier,
                        # correction 6: multimodal provenance (text-only debug run)
                        "multimodal_assets": [{"modality": "text", "asset_id": r["specification_id"],
                                               "checksum": r["specification_id"], "dimensions": None,
                                               "preprocessing_version": SCHEMA_VERSION,
                                               "prompt_position": 0, "asset_path": r["summary_asset"]}]})
        for step in r["trajectory_steps"]:
            topo.append({**base, "topology_step_id": fresh_id("tstep"),
                         "pre_action_graph": step["graph_state"],
                         "selected_action": step["selected_action"],
                         "mcts_visit_counts_if_available": step["mcts_visit_distribution"],
                         "mcts_used": True, "reached_spice": step["metadata"].get("reached_spice"),
                         "final_reward": step.get("discounted_return"),
                         "value_target_source": "REAL_POST_SIZING_SPICE" if r["reached_spice"] else "PRE_SPICE_VALIDATION_PENALTY",
                         "spice_evaluation_id": r["spice_evaluation_id"]})
        if r["reached_spice"] and r["simulation_result"]:
            sac.append({**base, "transition_id": fresh_id("sac"),
                        "real_or_imagined": "REAL", "spice_evaluation_id": r["spice_evaluation_id"],
                        "spice_metrics_after": r["simulation_result"].get("metrics"),
                        "performance_margins_after": r["simulation_result"].get("constraint_margins"),
                        "reward": r["final_reward"], "done": True,
                        "action_parameter_ids": sorted(((r["circuit_graph"] or {}).get("nodes") or [{}])[0].keys())})
    # preferences: cross-run pairs under compare() rules (labels from evidence only)
    from agentic_raptor.dpo.preference_pairs import compare as _cmp
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            if a["split"] != b["split"]:
                continue  # no cross-split preferences
            fake = lambda r: type("O", (), {"spice_success": bool((r["simulation_result"] or {}).get("success")),
                                            "passed_spec": bool((r["simulation_result"] or {}).get("constraint_margins") and all(v >= 0 for v in (r["simulation_result"] or {}).get("constraint_margins", {}).values())),
                                            "pvt_pass_rate": None, "worst_margin": min(((r["simulation_result"] or {}).get("constraint_margins") or {"x": -1.0}).values()),
                                            "fom": r["final_reward"] or 0.0, "spice_calls_total": 1,
                                            "calls_to_first_pass": None, "runtime_s": 1.0,
                                            "features": type("F", (), {"edit_count": len(r["trajectory_steps"])})()})()
            verdict, rule = _cmp(fake(a), fake(b))
            if verdict == 0:
                continue
            chosen, other = (a, b) if verdict < 0 else (b, a)
            prefs.append({"preference_id": fresh_id("pref"), "specification_id": a["specification_id"],
                          "group_key": a["group_key"], "split": a["split"], "execution_mode": a["execution_mode"],
                          "candidate_a_graph_id": a["circuit_graph_id"], "candidate_b_graph_id": b["circuit_graph_id"],
                          "preferred_candidate": "candidate_a" if chosen is a else "candidate_b",
                          "preference_type": rule, "preference_basis": "real_spice_evidence",
                          "spice_evidence": [a["spice_evaluation_id"], b["spice_evaluation_id"]],
                          "confidence": "high" if rule in "ab" else "medium"})

    proc = root / "processed"
    counts, quarantined = {}, []
    for name, rows, req in (("sft", sft, ["target_circuit_graph", "spice_evidence"]),
                            ("search_ranker", prefs, ["preferred_candidate", "preference_basis"]),
                            ("topology_rl", topo, ["pre_action_graph", "selected_action"]),
                            ("mb_sac", sac, ["spice_evaluation_id", "real_or_imagined"])):
        ok, quar = validate_records(rows, req, real_only=all(r.get("execution_mode") == "REAL" for r in rows))
        quarantined += quar
        for split in ("train", "validation", "test"):
            n = write_jsonl(proc / name / f"{name}_{split}.jsonl", [r for r in ok if r["split"] == split])
            counts[f"{name}/{split}"] = n
        write_jsonl(proc / name / f"{name}_heldout_topology_family.jsonl", [])  # single family: empty, documented
    write_jsonl(root / "quarantine" / "quarantined.jsonl", quarantined)
    write_jsonl(proc / "unified" / "unified_design_runs.jsonl",
                [{k: v for k, v in r.items() if k not in ("decisions", "trajectory_steps")} for r in runs])

    leakage = check_leakage({"sft": sft, "search_ranker": prefs, "topology_rl": topo, "mb_sac": sac})
    dup = duplicate_report(runs, "circuit_graph_id")
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "leakage_check_report.json").write_text(json.dumps(leakage, indent=1), encoding="utf-8")
    (manifests / "duplicate_report.json").write_text(json.dumps(dup, indent=1), encoding="utf-8")
    (manifests / "schema_versions.json").write_text(json.dumps(
        {"schema_version": SCHEMA_VERSION, "field_provenance": FIELD_PROVENANCE}, indent=1), encoding="utf-8")
    verdict = "DEBUG_DATASET_VALID" if leakage["prohibited_leakage"] == 0 and not quarantined else "DEBUG_DATASET_INVALID"
    report = {"verdict": verdict, "runs": len(runs), "counts": counts,
              "reached_spice": sum(r["reached_spice"] for r in runs),
              "quarantined": len(quarantined), "leakage": leakage["prohibited_leakage"],
              "duplicate_fraction": dup["duplicate_fraction"],
              "splits_of_groups": {r["group_key"][:12]: r["split"] for r in runs}}
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "debug_generation_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return {"report": report}
