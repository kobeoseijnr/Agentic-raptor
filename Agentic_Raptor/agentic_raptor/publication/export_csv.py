"""Export every saved experiment result to CSV under
artifacts/publication/csv/ (the human-inspectable proof pack)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from agentic_raptor.publication import FREEZE, PUB, ROOT

CSV = PUB / "csv"


def _w(name, rows, fields):
    CSV.mkdir(parents=True, exist_ok=True)
    with (CSV / name).open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"{name}: {len(rows)} rows")


def _j(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) \
        if Path(p).is_file() else None


def run():
    a = _j(PUB / "AGGREGATE.json")
    if a:
        _w("campaign_matrix.csv", a["campaign_matrix"]["rows"],
           ["campaign", "arm", "seed", "se_arm", "channels", "generation",
            "valid", "match", "unique", "top_share", "dpo", "exact_pass",
            "stable", "verified_earned", "spice"])
    q = _j(PUB / "qwen_ablation/SUMMARY.json")
    if q:
        _w("qwen_ablation.csv",
           [{"arm": k, **{kk: vv for kk, vv in v.items() if kk != "rows"}}
            for k, v in sorted(q.items())],
           ["arm", "desc", "first_attempt_validity", "mean_attempts_to_valid",
            "structure_match", "topk_structure_match", "stage_accuracy",
            "comp_accuracy", "unique_structures", "top_response_share",
            "contexts", "unavailable"])
    rb = _j(PUB / "qwen_ablation/RAG_BATTERY.json")
    if rb:
        _w("rag_battery.csv",
           [{"arm": k, **v} for k, v in rb.items() if k.startswith("RAG")],
           ["arm", "first_attempt_validity", "structure_match",
            "unique_structures", "inference_s"])
    for name, fields in (
        ("sizing_baselines", ["task_id", "tier", "class", "method",
                              "exact_pass", "stable", "gain_pass", "pm_pass",
                              "ugbw_pass", "constraints_passed", "distance",
                              "calls_to_first_pass", "spice_calls",
                              "best_gain_db", "best_pm_deg", "runtime_s"]),
        ("sr_battery", ["task_id", "tier", "class", "method", "exact_pass",
                        "distance", "constraints_passed",
                        "calls_to_first_pass", "spice_calls"]),
        ("budget_curves", ["task_id", "budget", "method", "exact_pass",
                           "distance", "constraints_passed",
                           "calls_to_first_pass", "spice_calls"]),
        ("reward_ablation", ["task_id", "tier", "class", "reward",
                             "exact_pass", "constraints_passed", "distance",
                             "best_gain_db", "best_pm_deg", "pm_excess_deg",
                             "gain_margin_db", "spice_calls"])):
        d = _j(PUB / f"{name}.json")
        if d:
            _w(f"{name}.csv", d["rows"], fields)
    pa = _j(PUB / "puct_ablation/SUMMARY.json")
    if pa:
        _w("puct_ablation.csv", [{"arm": k, **v} for k, v in pa.items()],
           ["arm", "tasks", "exact_pass", "mean_distance", "overturn_rate",
            "mean_search_spice"])
    fa = _j(FREEZE / "feasibility_audit.json")
    if fa:
        _w("feasibility_audit.csv",
           [{**x["profile"], "classification": x["classification"],
             "gm_needed_S": x["gm_needed_S"],
             "best_gain": x["best_gain"] or "", "n_tasks": len(x["tasks"])}
            for x in fa["results"]],
           ["class", "load_pf", "ugbw_hz", "gain_db", "pm_deg",
            "classification", "gm_needed_S", "best_gain", "n_tasks"])
    ba = Path("artifacts/blind_eval_audit.jsonl")
    if ba.is_file():
        rows = [json.loads(x) for x in ba.read_text().splitlines()]
        _w("blind_test.csv", rows, list(rows[0].keys()))
    v = _j(ROOT / "artifacts/value_refresh/REPORT.json")
    if v:
        _w("value_net_quality.csv",
           [{"eval": "stale", **v["stale_checkpoint_eval"]},
            {"eval": "retrained", **v["retrained_eval"]}],
           ["eval", "n", "spearman", "pairwise_accuracy", "ranked_pairs"])
    print("all ->", CSV.resolve())


if __name__ == "__main__":
    run()
