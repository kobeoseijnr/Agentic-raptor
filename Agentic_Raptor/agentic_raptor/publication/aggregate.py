"""Aggregate all saved experiment artifacts into paper tables (CSV+MD) and
the multi-seed statistics. Reads ONLY saved experiment files."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agentic_raptor.publication import FREEZE, PUB, ROOT
from agentic_raptor.publication.stats import (cliffs_delta, holm_correct,
                                              mcnemar, mean_ci,
                                              paired_bootstrap_diff, wilcoxon)


#: identity of the MAIN battery. Every battery writes campaigns into the same
#: directory, and the SE/F batteries all run arm="sft_only" seed=11 -- so
#: grouping on (arm, seed) alone silently pooled ~20 unrelated campaigns into
#: the sft_only cell and then picked one of them arbitrarily via max(). The
#: engine and channel set are what distinguish a battery, so they are part of
#: both the filter and the grouping key.
MAIN_ENGINE = "sac_v2_pretrained"
MAIN_CHANNELS = "l4,ranker,replay,search,selfearn,surrogate,value"


def campaign_matrix(channels: str | None = MAIN_CHANNELS,
                    engine: str | None = MAIN_ENGINE) -> dict:
    """Arm x seed final-generation exam metrics + per-generation curves.

    channels/engine select ONE battery; pass None to include everything
    (which mixes code generations and is almost never what you want).
    """
    rows, skipped = [], 0
    for camp in sorted((ROOT / "artifacts/self_improvement").glob("camp_*")):
        f = camp / "logs/generations.jsonl"
        if not f.is_file():
            continue
        gens = [json.loads(x) for x in
                f.read_text(encoding="utf-8").splitlines()]
        if not gens:
            continue
        arm = gens[0].get("arm", "pre_ablation")
        seed = gens[0].get("campaign_seed", 0)
        se_arm = gens[0].get("se_arm", "se6")
        channels_c = ",".join(sorted(gens[0].get("channels") or [])) or "all"
        engine_c = gens[0].get("engine", "legacy")
        if (channels is not None and channels_c != channels) or \
                (engine is not None and engine_c != engine):
            skipped += 1
            continue
        channels_row = channels_c
        for g in gens:
            e = g.get("dpo_exam") if g.get("dpo_update_accepted") \
                else g["sft_exam"]
            el = g["design"].get("electrical") or {}
            rows.append({"campaign": camp.name, "arm": arm, "seed": seed,
                         "se_arm": se_arm, "channels": channels_row,
                         "engine": engine_c,
                         "generation": g["generation_id"],
                         "valid": e["valid_rate"],
                         "match": e["spec_match_rate"],
                         "unique": e["unique_structures"],
                         "top_share": e["most_common_response_fraction"],
                         "dpo": ("accepted" if g.get("dpo_update_accepted")
                                 else "rolled_back" if "dpo" in g
                                 else "skipped"),
                         "exact_pass": el.get("exact_spec_pass_rate"),
                         "stable": el.get("stable_rate"),
                         "verified_earned":
                             g["design"]["earned_tiers"]
                             ["verified_self_earned"],
                         "spice": (el.get("postsizing_spice_calls") or 0)
                         + g["design"]["spice_calls"]})
    by_arm = {}
    for r in rows:
        # CAMPAIGN is the unit, not (arm, seed): two campaigns can legitimately
        # share an arm and seed, and collapsing them hid which one was scored
        by_arm.setdefault(r["arm"], {}).setdefault(
            (r["seed"], r["campaign"]), []).append(r)
    stats = {}
    for arm, camps in by_arm.items():
        finals = [max(gs, key=lambda g: g["generation"])
                  for gs in camps.values()]
        stats[arm] = {"seeds": sorted({s for s, _ in camps}),
                      "campaigns": sorted(c for _, c in camps),
                      "n_campaigns": len(camps),
                      "final_match": mean_ci([f["match"] for f in finals]),
                      "final_unique": mean_ci([f["unique"] for f in finals]),
                      "final_valid": mean_ci([f["valid"] for f in finals]),
                      "final_exact_pass": mean_ci(
                          [f["exact_pass"] for f in finals])}
    return {"rows": rows, "per_arm_stats": stats,
            "selection": {"channels": channels, "engine": engine,
                          "campaigns_included": len({r["campaign"]
                                                     for r in rows}),
                          "campaigns_excluded": skipped}}


def headline_tests(matrix: dict, generation: str = "final") -> dict:
    """Paired tests across arms sharing seeds. generation='final' uses the
    last generation; an int uses that generation (convergence-speed tests —
    needed because every arm reaches match 1.0 by gen 2)."""
    by = {}
    for r in matrix["rows"]:
        by.setdefault((r["arm"], r["seed"]), []).append(r)
    if generation == "final":
        finals = {k: max(v, key=lambda g: g["generation"])["match"]
                  for k, v in by.items()}
    else:
        finals = {k: next((g["match"] for g in v
                           if g["generation"] == generation), None)
                  for k, v in by.items()}
        finals = {k: v for k, v in finals.items() if v is not None}
    arms = sorted({a for a, _ in finals})
    tests = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            shared = sorted({s for x, s in finals if x == a}
                            & {s for x, s in finals if x == b})
            if len(shared) < 2:
                continue
            xa = [finals[(a, s)] for s in shared]
            xb = [finals[(b, s)] for s in shared]
            tests[f"{a}_vs_{b}"] = {
                "seeds": shared,
                "paired_bootstrap": paired_bootstrap_diff(xa, xb),
                "wilcoxon": wilcoxon(xa, xb),
                "cliffs_delta": cliffs_delta(xa, xb)}
    pvals = {k: v["wilcoxon"].get("p") for k, v in tests.items()}
    return {"tests": tests, "holm": holm_correct(pvals)}


def sizing_table() -> dict | None:
    p = PUB / "sizing_baselines.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    agg = {}
    for r in d["rows"]:
        agg.setdefault(r["method"], []).append(r)
    out = {}
    for m, rs in sorted(agg.items()):
        out[m] = {"tasks": len(rs),
                  "exact_pass": sum(r["exact_pass"] for r in rs),
                  "stable": sum(r["stable"] for r in rs),
                  "mean_distance": mean_ci([r["distance"] for r in rs]),
                  "mean_calls_to_first_pass": mean_ci(
                      [r["calls_to_first_pass"] for r in rs
                       if r["calls_to_first_pass"]]),
                  "total_spice": sum(r["spice_calls"] for r in rs),
                  "mean_runtime_s": round(
                      sum(r["runtime_s"] for r in rs) / len(rs), 1)}
    # paired McNemar on exact pass: each method vs C7
    ref = {r["task_id"]: r["exact_pass"] for r in agg.get("C7", [])}
    mcn = {}
    for m, rs in agg.items():
        if m == "C7" or not ref:
            continue
        a = [r["exact_pass"] for r in rs if r["task_id"] in ref]
        b = [ref[r["task_id"]] for r in rs if r["task_id"] in ref]
        mcn[f"{m}_vs_C7"] = mcnemar(a, b)
    return {"per_method": out, "mcnemar_vs_C7": mcn,
            "budget": d["budget"], "seed": d["seed"]}


def run() -> dict:
    doc = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "campaign_matrix": campaign_matrix()}
    doc["headline_tests"] = headline_tests(doc["campaign_matrix"])
    doc["convergence_tests_gen1"] = headline_tests(doc["campaign_matrix"], 1)
    doc["convergence_tests_gen0"] = headline_tests(doc["campaign_matrix"], 0)
    doc["sizing_baselines"] = sizing_table()
    for name in ("reward_ablation", "integrity_ablation"):
        p = PUB / f"{name}.json"
        doc[name] = json.loads(p.read_text()) if p.is_file() else None
    p = PUB / "qwen_ablation" / "SUMMARY.json"
    doc["qwen_ablation"] = json.loads(p.read_text()) if p.is_file() else None
    for extra in ("sr_battery", "budget_curves"):
        p = PUB / f"{extra}.json"
        doc[extra] = json.loads(p.read_text()) if p.is_file() else None
    p = PUB / "qwen_ablation" / "RAG_BATTERY.json"
    doc["rag_battery"] = json.loads(p.read_text()) if p.is_file() else None
    p = PUB / "puct_ablation" / "SUMMARY.json"
    doc["puct_ablation"] = json.loads(p.read_text()) if p.is_file() else None
    p = FREEZE / "feasibility_audit.json"
    doc["feasibility_audit"] = ({k: v for k, v in
                                 json.loads(p.read_text()).items()
                                 if k != "results"} if p.is_file() else None)
    (PUB / "AGGREGATE.json").write_text(json.dumps(doc, indent=1,
                                                   default=str),
                                        encoding="utf-8")
    return doc


if __name__ == "__main__":
    d = run()
    print(json.dumps(d["campaign_matrix"]["per_arm_stats"], indent=1,
                     default=str))
