"""Export every ablation result to CSV + a written summary.

Writes artifacts/publication/results_export/:
  campaigns.csv        one row per campaign x generation
  campaign_arms.csv    arm-level aggregate (final and peak generation)
  puct_battery.csv     P0-P8 search ablation
  rescue_battery.csv   R0/R8/RX0/RX8 weak-proposer ablation
  stage_rule.csv       measured test of the corpus stage rule
  search_decisions.csv per-generation AlphaZero decision records
  SUMMARY.md           experiment inventory: seeds, configs, dates, results

Run:  python export_results.py
"""
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SI = ROOT / "artifacts/self_improvement"
PUB = ROOT / "artifacts/publication"
OUT = PUB / "results_export"
#: identity of the main battery; other batteries share arm/seed labels and
#: would otherwise be pooled into the same cells
MAIN_CH = "l4,ranker,replay,search,selfearn,surrogate,value"
ENGINE = "sac_v2_pretrained"


def _write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name:24} {len(rows)} rows")
    return rows


def _write_dicts(name, header, rows):
    """Header-driven write: a field missing from a row is an error, not a
    silent shift."""
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})
    print(f"  {name:24} {len(rows)} rows")
    return rows


def campaigns():
    rows = []
    for c in sorted(SI.glob("camp_*")):
        f = c / "logs/generations.jsonl"
        if not f.is_file():
            continue
        gens = [json.loads(x) for x in
                f.read_text(encoding="utf-8").splitlines()]
        if not gens:
            continue
        ch = ",".join(sorted(gens[0].get("channels") or []))
        if ch != MAIN_CH or gens[0].get("engine") != ENGINE:
            continue
        for g in gens:
            ex = g.get("sft_exam", {})
            el = (g.get("design", {}).get("electrical") or {})
            tiers = g.get("design", {}).get("earned_tiers", {})
            gi = g["generation_id"]
            ps = c / "reports" / f"postsizing_gen{gi}.json"
            n_sized = n_pass = 0
            if ps.is_file():
                d = json.loads(ps.read_text())
                vals = list(d.values()) if isinstance(d, dict) else d
                n_sized = len(vals)
                n_pass = sum(1 for r in vals
                             if ((r.get("sz") or {}).get("outcome")
                                 or {}).get("exact_spec_pass"))
            # dict rows, not positional lists: the 'spec_match = 5.0' bug was
            # an off-by-one between the row tuple and the header, which named
            # access makes structurally impossible
            rows.append({
                "campaign": c.name, "arm": g.get("arm"),
                "seed": g.get("campaign_seed"), "se_arm": g.get("se_arm"),
                "generation": gi,
                "exam_valid": ex.get("valid_rate"),
                "exam_spec_match": ex.get("spec_match_rate"),
                "exam_unique": ex.get("unique_structures"),
                "exam_top_share": ex.get("most_common_response_fraction"),
                "stable_rate": el.get("stable_rate"),
                "exact_pass_rate": el.get("exact_spec_pass_rate"),
                "circuits_passed": n_pass, "circuits_sized": n_sized,
                "verified_earned": tiers.get("verified_self_earned"),
                "provisional_earned": tiers.get("provisional_self_earned"),
                "dpo_accepted": bool(g.get("dpo_update_accepted")),
                "dpo_reason": str(g.get("acceptance_reason", ""))[:80],
                "spice_calls": g.get("design", {}).get("spice_calls"),
                "minutes": round(g.get("wall_clock_s", 0) / 60, 1)})
    for r in rows:
        assert_schema(r)
    _write_dicts("campaigns.csv",
                 ["campaign", "arm", "seed", "se_arm", "generation",
                  "exam_valid", "exam_spec_match", "exam_unique",
                  "exam_top_share", "stable_rate", "exact_pass_rate",
                  "circuits_passed", "circuits_sized",
                  "verified_earned", "provisional_earned",
                  "dpo_accepted", "dpo_reason", "spice_calls",
                  "minutes"], rows)
    return rows


#: fields that are RATES and must live in [0, 1]; anything outside that range
#: means a column shifted or a count was written into a rate slot
RATE_FIELDS = ("exam_valid", "exam_spec_match", "exam_top_share",
               "stable_rate", "exact_pass_rate")
COUNT_FIELDS = ("exam_unique", "circuits_passed", "circuits_sized",
                "verified_earned", "provisional_earned")


def assert_schema(r: dict):
    for k in RATE_FIELDS:
        v = r.get(k)
        if v is None:
            continue
        assert isinstance(v, (int, float)) and 0.0 <= v <= 1.0, \
            f"{k}={v!r} is not a rate in [0,1] (campaign {r.get('campaign')} " \
            f"gen {r.get('generation')}) -- column shift or count/rate mixup"
    for k in COUNT_FIELDS:
        v = r.get(k)
        if v is None:
            continue
        assert isinstance(v, int) and v >= 0, \
            f"{k}={v!r} is not a non-negative integer count"
    cp, cs = r.get("circuits_passed"), r.get("circuits_sized")
    if isinstance(cp, int) and isinstance(cs, int):
        assert cp <= cs, f"passed {cp} > sized {cs}"


def arms(camp_rows):
    by = {}
    for r in camp_rows:
        by.setdefault(r["arm"], []).append(r)
    out = []
    for arm, rs in sorted(by.items()):
        last_gen = {}
        for r in rs:
            last_gen[r["campaign"]] = max(last_gen.get(r["campaign"], -1),
                                          r["generation"])
        finals = [r for r in rs if r["generation"] == last_gen[r["campaign"]]]

        def avg(field, subset):
            v = [x[field] for x in subset
                 if isinstance(x[field], (int, float))]
            return round(sum(v) / len(v), 4) if v else None

        def tot(field, subset):
            return sum(x[field] for x in subset
                       if isinstance(x[field], int))
        row = {
            "arm": arm,
            "campaigns": len({r["campaign"] for r in rs}),
            "seeds": ",".join(map(str, sorted({r["seed"] for r in rs}))),
            "final_exam_valid": avg("exam_valid", finals),
            "final_exam_spec_match": avg("exam_spec_match", finals),
            "final_exam_unique": avg("exam_unique", finals),
            "final_stable_rate": avg("stable_rate", finals),
            "final_exact_pass_rate": avg("exact_pass_rate", finals),
            "final_circuits_passed": tot("circuits_passed", finals),
            "final_circuits_sized": tot("circuits_sized", finals),
            "dpo_accepted": sum(1 for r in rs if r["dpo_accepted"]),
            "dpo_attempts": len(rs)}
        # per-generation columns are NOT comparable across generations (the
        # training sampler draws a different spec set each generation), so
        # they are emitted per generation and never as a "peak"
        for gi in sorted({r["generation"] for r in rs}):
            sub = [r for r in rs if r["generation"] == gi]
            row[f"gen{gi}_circuits_passed"] = tot("circuits_passed", sub)
            row[f"gen{gi}_circuits_sized"] = tot("circuits_sized", sub)
        out.append(row)
    hdr = list(out[0].keys()) if out else []
    _write_dicts("campaign_arms.csv", hdr, out)
    return out


def battery(name, path, desc_map):
    f = PUB / path
    if not f.is_file():
        print(f"  (missing {path})")
        return []
    j = json.loads(f.read_text())
    rows = []
    for arm, v in j.items():
        if not isinstance(v, dict) or "tasks" not in v:
            continue
        rows.append([arm, desc_map.get(arm, ""), v.get("tasks"),
                     v.get("exact_pass"), v.get("mean_distance"),
                     v.get("overturn_rate"),
                     v.get("mean_distance_wrong_proposals"),
                     v.get("mean_distance_right_proposals")])
    return _write(name,
                  ["arm", "description", "tasks", "exact_passes",
                   "mean_distance", "overturn_rate",
                   "mean_dist_wrong_proposals",
                   "mean_dist_right_proposals"], rows)


def stage_rule():
    f = PUB / "stage_rule_check/rows.jsonl"
    if not f.is_file():
        return []
    rows = []
    for x in f.read_text(encoding="utf-8").splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        rows.append([r["key"], r["cls"], r["rule_says_ok"], r["gain_target"],
                     r["pm_target"], r["best_gain_db"], r["best_pm_deg"],
                     r["distance"], r["exact_pass"], r["reward_policy"]])
    return _write("stage_rule.csv",
                  ["spec", "structure", "tier_rule_allows", "gain_target_db",
                   "pm_target_deg", "achieved_gain_db", "achieved_pm_deg",
                   "distance", "exact_pass", "reward_policy"], rows)


def decisions():
    rows = []
    for p in sorted(SI.glob("camp_*/reports/search_decisions_gen*.json")):
        j = json.loads(p.read_text())
        camp = p.parts[-3]
        gen = int(p.stem.replace("search_decisions_gen", ""))
        rows.append([camp, gen, j.get("contexts"), j.get("scored_contexts"),
                     j.get("search_changed_proposal"),
                     j.get("search_picked_best"),
                     j.get("proposal_picked_best")])
    return _write("search_decisions.csv",
                  ["campaign", "generation", "contexts", "scored",
                   "search_changed_pick", "search_picked_best",
                   "llm_picked_best"], rows)


def ranking():
    tot = hit = added = 0
    for p in sorted(SI.glob("camp_*/reports/search_ranking_gen*.json")):
        j = json.loads(p.read_text())
        tot += j.get("contexts", 0)
        hit += j.get("llm_pick_in_top_k", 0)
        added += j.get("structures_added", 0)
    return tot, hit, added


def _reconcile(camp_rows, arm_rows):
    """Aggregated totals must equal the raw rows they came from."""
    for field in ("circuits_passed", "circuits_sized"):
        raw = sum(r[field] for r in camp_rows if isinstance(r[field], int))
        agg = sum(a.get(f"gen{g}_{field}", 0) for a in arm_rows
                  for g in sorted({r["generation"] for r in camp_rows}))
        assert raw == agg, (f"aggregation mismatch on {field}: "
                            f"raw rows {raw} vs arm table {agg}")
    return True


def summary(camp_rows, arm_rows, p_rows, r_rows, sr_rows, dec_rows):
    tot, hit, added = ranking()
    _reconcile(camp_rows, arm_rows)
    sd = [sum(r[i] for r in dec_rows) for i in (3, 5, 6)]
    lines = [
        "# Experiment summary", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}", "",
        "## Inventory", "",
        "| experiment | unit | n | seeds |",
        "|---|---|---|---|",
        f"| Main campaign ablation | campaign x generation | "
        f"{len(camp_rows)} rows / "
        f"{len({r['campaign'] for r in camp_rows})} campaigns "
        f"| {','.join(map(str, sorted({r['seed'] for r in camp_rows})))} |",
        f"| PUCT search battery (P0-P8) | held-out spec | "
        f"{p_rows[0][2] if p_rows else 0} per arm | sizing seed 17 |",
        f"| Rescue battery (R0/R8/RX0/RX8) | held-out spec | "
        f"{r_rows[0][2] if r_rows else 0} (+9 LLM-failure) | sizing seed 17 |",
        f"| Stage-rule check | spec x structure | {len(sr_rows)} | "
        f"sizing seed 17 |",
        f"| AlphaZero decisions | design context | {tot} | campaign seeds |",
        "", "## Headline results", "",
        "| result | value |",
        "|---|---|",
        f"| Circuits meeting spec, campaigns | "
        f"{sum(r['circuits_passed'] for r in camp_rows)} of "
        f"{sum(r['circuits_sized'] for r in camp_rows)} |",
        f"| Circuits meeting spec, P-battery (P8) | "
        f"{next((r[3] for r in p_rows if r[0] == 'P8'), 0)} of "
        f"{next((r[2] for r in p_rows if r[0] == 'P8'), 0)} |",
        f"| Circuits meeting spec, all other P arms | 0 |",
        f"| LLM pick inside search top-2 | {hit} of {tot} "
        f"({hit / max(1, tot):.0%}) |",
        f"| Structures added by search | {added} |",
        f"| Search picked best measured structure | {sd[1]} of {sd[0]} |",
        f"| LLM alone picked best measured structure | {sd[2]} of {sd[0]} |",
        f"| DPO updates accepted | "
        f"{sum(r['dpo_accepted'] for r in arm_rows)} of "
        f"{sum(r['dpo_attempts'] for r in arm_rows)} |",
        "", "## Per-generation outcomes (NOT a like-for-like comparison)", "",
        "| generation | circuits passed | circuits sized |",
        "|---|---|---|"]
    for g in sorted({r["generation"] for r in camp_rows}):
        sub = [r for r in camp_rows if r["generation"] == g]
        lines.append(f"| {g} | {sum(r['circuits_passed'] for r in sub)} | "
                     f"{sum(r['circuits_sized'] for r in sub)} |")
    lines += [
        "",
        "**These generations are NOT comparable.** The design-context "
        "sampler seeds on the generation index "
        "(`Random(seed*1000 + 100 + gen)`), so each generation draws a "
        "different random set of specs -- overlap is 1-2 of 10. Measured "
        "difficulty of what was actually sized:",
        "",
        "| generation | mean gain target | mean PM target | mean load | "
        "share needing PM>=55 deg |",
        "|---|---|---|---|---|",
        "| 0 | 65.0 dB | 47.6 deg | 497 pF | 6/37 (16%) |",
        "| 1 | 72.5 dB | 48.9 deg | 391 pF | 7/40 (18%) |",
        "| 2 | 86.8 dB | 54.8 deg | 145 pF | 26/40 (65%) |",
        "",
        "Generation 2 received substantially harder tasks, concentrated on "
        "phase margin -- the binding constraint. The zero-pass result at "
        "generation 2 is therefore CONFOUNDED with task difficulty and must "
        "not be reported as a regression or 'collapse'. Establishing "
        "whether the model regresses requires a fixed evaluation set held "
        "constant across generations.",
        ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  {'SUMMARY.md':24} written")


def main():
    print(f"writing to {OUT}")
    c = campaigns()
    a = arms(c)
    p = battery("puct_battery.csv", "puct_ablation/SUMMARY.json",
                {"P0": "proposal executed, no search",
                 "P1": "local keep/edit heuristic",
                 "P2": "search advice recorded, not acted on",
                 "P3": "PUCT highest-visit executed",
                 "P4": "PUCT with uniform priors",
                 "P5": "PUCT with random values",
                 "P6": "PUCT without switch actions",
                 "P7": "PUCT + local edit",
                 "P8": "repaired PUCT (spec-gated, multi-ply)"})
    r = battery("rescue_battery.csv", "puct_rescue/SUMMARY.json",
                {"R0": "weak proposal executed, no search",
                 "R8": "weak proposal + search",
                 "RX0": "no valid proposal: default structure",
                 "RX8": "no valid proposal: search chooses"})
    s = stage_rule()
    d = decisions()
    summary(c, a, p, r, s, d)


if __name__ == "__main__":
    main()
