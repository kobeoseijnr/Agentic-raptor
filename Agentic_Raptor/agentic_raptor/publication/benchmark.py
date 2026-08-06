"""Fixed electrical benchmark + feasibility audit.

One frozen task list; every method and generation is evaluated on the SAME
tasks with the SAME budgets, stopping policy and measurement extractor.
The feasibility audit runs a larger reference search per unique task profile
and classifies tasks; unresolved/likely-infeasible tasks form the separate
STRESS benchmark instead of silently weakening the main one.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.publication import FREEZE, ROOT

BUDGETS = {"sizing_budget": 16, "mcts_simulations": 8,
           "total_spice_cap_per_task": 40,
           "stopping_policy": "budget_exhaustion",
           "measurement_extractor": "electrical.measure_all_v1",
           "corner": "tt", "temperature_c": 27, "supply_v": 1.8}


def _tier(spec: dict, stages: int) -> str:
    """easy/moderate/hard by required gm (UGBW x load) and stage demand."""
    gm_needed = 2 * math.pi * (spec.get("ugbw_target_hz") or 1e4) \
        * spec.get("load_capacitance_pf", 100) * 1e-12
    if gm_needed > 1e-3 or (stages == 3
                            and spec["phase_margin_target_deg"] >= 60):
        return "hard"
    if stages == 2 and (spec.get("ugbw_target_hz") or 1e4) <= 1e4:
        return "easy"
    return "moderate"


def build_benchmark() -> dict:
    """Frozen task list = every frozen-validation spec + a deduplicated
    train-split panel (for learning-curve evaluation without touching
    validation-only usage rules)."""
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    tasks, seen = [], set()
    for r in corpus["records"]:
        if r["split"] == "blindtest":
            continue                      # blind stays sealed
        spec = ig.parse_spec(r["prompt"])
        key = r["prompt"].splitlines()[0]
        if not spec or key in seen:
            continue
        seen.add(key)
        tasks.append({"task_id": f"T{len(tasks):03d}",
                      "context_id": r["context_id"], "split": r["split"],
                      "spec_line": key,
                      "spec": {k: spec[k] for k in
                               ("gain_target_db", "phase_margin_target_deg",
                                "load_capacitance_pf", "ugbw_target_hz",
                                "technology")},
                      "target_class": f"{r['stages']}s_{r['comp']}",
                      "target_family": r["topology_id"],
                      "target_variant_hash": r["variant_hash"],
                      "tier": _tier(spec, r["stages"]),
                      "budgets": BUDGETS})
    doc = {"tasks": tasks, "budgets": BUDGETS,
           "counts": {"total": len(tasks),
                      **{s: sum(1 for t in tasks if t["split"] == s)
                         for s in ("train", "heldout")},
                      **{tier: sum(1 for t in tasks if t["tier"] == tier)
                         for tier in ("easy", "moderate", "hard")}},
           "benchmark_hash": ig.sha_json([(t["spec_line"],
                                           t["target_class"])
                                          for t in tasks]),
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    FREEZE.mkdir(parents=True, exist_ok=True)
    (FREEZE / "benchmark.json").write_text(json.dumps(doc, indent=1),
                                           encoding="utf-8")
    return doc


def leakage_report() -> dict:
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    sm = corpus["split_manifest"]
    lines = {s: {r["prompt"].splitlines()[0] for r in corpus["records"]
                 if r["split"] == s}
             for s in ("train", "heldout", "blindtest")}
    rep = {"family_overlap": {
               "train_vs_validation": sorted(
                   set(sm["train_family_ids"])
                   & set(sm["validation_family_ids"])),
               "train_vs_blind": sorted(
                   set(sm["train_family_ids"])
                   & set(sm["blind_test_family_ids"])),
               "validation_vs_blind": sorted(
                   set(sm["validation_family_ids"])
                   & set(sm["blind_test_family_ids"]))},
           "spec_line_overlap": {
               "train_vs_validation": len(lines["train"] & lines["heldout"]),
               "train_vs_blind": len(lines["train"] & lines["blindtest"]),
               "validation_vs_blind": len(lines["heldout"]
                                          & lines["blindtest"])},
           "split_hashes": {
               "validation": json.loads((ROOT /
                   "artifacts/stage3e4/frozen_exam.json").read_text()
                   )["frozen_exam_hash"],
               "blind": json.loads((ROOT /
                   "artifacts/stage3e4/blind_test.json").read_text()
                   )["frozen_blind_hash"],
               "corpus": sm["corpus_hash"], "split": sm["split_hash"]},
           "clean": True}
    rep["clean"] = (not any(rep["family_overlap"].values())
                    and not any(rep["spec_line_overlap"].values()))
    (FREEZE / "leakage_report.json").write_text(json.dumps(rep, indent=1),
                                                encoding="utf-8")
    return rep


# --------------------------- feasibility audit --------------------------------
def audit_feasibility(n_ref: int = 16, include_probe_family: bool = True):
    """Reference search per unique (class, load, ugbw, gain, pm) profile.
    Classification distinguishes optimizer limits from family capacity from
    likely-infeasible physics; unresolved tasks go to the stress benchmark."""
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mapping import map_family
    from agentic_raptor.mb_sac.spec_sizing import achievability_sweep
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import (EditRejected,
                                                           apply_edit)
    bench = json.loads((FREEZE / "benchmark.json").read_text())
    exe = discover_ngspice()
    out = (ROOT / "artifacts/publication/feasibility").resolve()
    out.mkdir(parents=True, exist_ok=True)
    profiles, results = {}, []
    for t in bench["tasks"]:
        prof = (t["target_class"], t["spec"]["load_capacitance_pf"],
                t["spec"]["ugbw_target_hz"],
                round(t["spec"]["gain_target_db"]),
                t["spec"]["phase_margin_target_deg"])
        profiles.setdefault(prof, []).append(t["task_id"])
    for i, (prof, task_ids) in enumerate(sorted(profiles.items())):
        cls, cl, ugbw, gain, pm = prof
        stages = int(cls[0])
        comp = cls.split("_", 1)[1]
        spec = {"gain_target_db": float(gain),
                "phase_margin_target_deg": float(pm),
                "load_capacitance_pf": float(cl),
                "ugbw_target_hz": float(ugbw), "technology": "sky130"}

        class _S:
            topology_id = f"aud{i}"
        g, _ = map_family(_S(), {
            "topology_id": _S.topology_id, "gain_stages": stages,
            "functional_blocks": [] if comp == "none" else ["C"],
            "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
            "graph_hash": None})
        if comp in ("rc", "rc_nulling"):
            try:
                g, _a = apply_edit(g, "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE")
            except EditRejected:
                pass
        rep = achievability_sweep(_S.topology_id, g, spec, exe, out,
                                  new_costs(), n=n_ref, seed=i)
        gm_needed = 2 * math.pi * float(ugbw) * float(cl) * 1e-12
        best = rep["best_attainable_gain"]
        ok = rep["best_spec_compliant"]
        measured = [r for r in rep["results"] if r["gain_db"] is not None]
        best_ugbw = max((r["ugbw_hz"] or 0 for r in measured), default=0)
        if ok:
            klass = "known_feasible"
        elif best and best["gain_db"] >= gain and best_ugbw >= ugbw:
            klass = "optimizer_limited"    # each constraint reachable alone
        elif best and best["gain_db"] < gain:
            klass = "family_capacity_limited"
        elif best_ugbw < ugbw and gm_needed > 5e-4:
            klass = "likely_infeasible"    # gm demand beyond family reach
        elif best and best["gain_db"] >= gain:
            klass = "optimizer_limited"
        elif not measured:
            klass = "measurement_limited"
        else:
            klass = "unknown"
        results.append({"profile": {"class": cls, "load_pf": cl,
                                    "ugbw_hz": ugbw, "gain_db": gain,
                                    "pm_deg": pm},
                        "tasks": task_ids, "classification": klass,
                        "gm_needed_S": round(gm_needed, 6),
                        "best_gain": (best or {}).get("gain_db"),
                        "best_ugbw_hz": best_ugbw,
                        "spec_compliant_found": bool(ok),
                        "n_reference_sims": rep["n_sims"]})
    stress = sorted({tid for r in results
                     if r["classification"] in ("likely_infeasible",
                                                "unknown",
                                                "measurement_limited")
                     for tid in r["tasks"]})
    doc = {"profiles_audited": len(results), "results": results,
           "classification_counts": {
               k: sum(1 for r in results if r["classification"] == k)
               for k in ("known_feasible", "optimizer_limited",
                         "family_capacity_limited", "likely_infeasible",
                         "measurement_limited", "unknown")},
           "stress_benchmark_task_ids": stress,
           "main_benchmark_task_ids": sorted(
               {tid for r in results
                if r["classification"] not in ("likely_infeasible",
                                               "unknown",
                                               "measurement_limited")
                for tid in r["tasks"]}),
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (FREEZE / "feasibility_audit.json").write_text(
        json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


if __name__ == "__main__":
    print(json.dumps({"benchmark": build_benchmark()["counts"],
                      "leakage": leakage_report()}, indent=1))
