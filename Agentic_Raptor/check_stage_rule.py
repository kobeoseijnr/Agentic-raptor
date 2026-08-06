"""Is the corpus stage rule supported by measurement?

The corpus encodes "gain >= 70 dB requires 3 stages", and run_puct_ablation's
compatible_classes() gates the search pool on it. A single-spec probe found
the opposite: on an 80.4 dB / 60 deg spec a 2-stage design reached 89 dB with
18.8 deg PM (distance 0.305) while every 3-stage design overshot to ~133 dB
and went unstable (best distance 0.427). If that holds across specs, then
"wrong-tier" proposals are frequently the better engineering choice, the
spec-gated pool is harmful, and counterfactual value targets built from the
rule would teach the value head something measurement contradicts.

Sizes each high-gain TRAIN spec under its rule-compatible 3-stage classes and
under the 2-stage classes, and reports which actually lands closer to
feasibility. Train split only: no held-out spec is touched.

Run:  python check_stage_rule.py [--specs 8] [--budget 12]
"""
import argparse
import json
import time

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.publication import PUB, ROOT
from run_puct_ablation import realise_class

OUT = PUB / "stage_rule_check"
TWO = ["2s_none", "2s_miller"]
THREE = ["3s_none", "3s_miller", "3s_rc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--min-gain", type=float, default=70.0)
    args = ap.parse_args()
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mb_sac.spec_sizing import sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs

    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    specs = []
    for i, r in enumerate(corpus["records"]):
        if r["split"] != "train":
            continue
        s = ig.parse_spec(r["prompt"])
        if s and s["gain_target_db"] >= args.min_gain:
            specs.append((f"{i:03d}_{r['context_id']}", s))
        if len(specs) >= args.specs:
            break
    exe = discover_ngspice()
    out = (OUT / "runs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows_f = OUT / "rows.jsonl"
    done = set()
    if rows_f.is_file():
        done = {(json.loads(x)["key"], json.loads(x)["cls"])
                for x in rows_f.read_text().splitlines() if x.strip()}
    print(f"specs >= {args.min_gain} dB: {len(specs)}")
    for key, spec in specs:
        for cls in TWO + THREE:
            if (key, cls) in done:
                continue
            t0 = time.time()
            sz = sac_size(f"sr_{key[:3]}_{cls}", realise_class(cls), spec,
                          exe, out, new_costs(), budget=args.budget,
                          seed=17, persist=False)
            o, b = sz["outcome"], sz["best"]
            row = {"key": key, "cls": cls, "rule_says_ok": cls in THREE,
                   "gain_target": spec["gain_target_db"],
                   "pm_target": spec["phase_margin_target_deg"],
                   "best_gain_db": b["gain_db"], "best_pm_deg": b["pm_deg"],
                   "distance": o["normalized_distance_to_feasibility"],
                   "constraints_passed": o["hard_constraints_passed"],
                   "exact_pass": o["exact_spec_pass"],
                   "reward_policy": sz["reward_policy"],
                   "runtime_s": round(time.time() - t0, 1)}
            with rows_f.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(f"{key[:18]:18s} {cls:10s} rule_ok={row['rule_says_ok']!s:5s} "
                  f"gain={b['gain_db']} pm={b['pm_deg']} "
                  f"dist={row['distance']}")

    rows = [json.loads(x) for x in rows_f.read_text().splitlines()
            if x.strip()]
    by_spec = {}
    for r in rows:
        by_spec.setdefault(r["key"], []).append(r)
    two_wins = three_wins = 0
    for key, rs in by_spec.items():
        t2 = [r for r in rs if not r["rule_says_ok"]]
        t3 = [r for r in rs if r["rule_says_ok"]]
        if not t2 or not t3:
            continue
        b2 = min(r["distance"] or 9.9 for r in t2)
        b3 = min(r["distance"] or 9.9 for r in t3)
        two_wins += b2 < b3
        three_wins += b3 <= b2

    def _m(rs):
        ds = [r["distance"] for r in rs if r["distance"] is not None]
        return round(sum(ds) / len(ds), 4) if ds else None
    summary = {
        "specs_compared": two_wins + three_wins,
        "two_stage_closer": two_wins,
        "three_stage_closer": three_wins,
        "mean_distance_2stage": _m([r for r in rows if not r["rule_says_ok"]]),
        "mean_distance_3stage": _m([r for r in rows if r["rule_says_ok"]]),
        "exact_passes": sum(r["exact_pass"] for r in rows),
        "reward_policy": rows[0]["reward_policy"] if rows else None,
        "rule_supported_by_measurement": three_wins > two_wins}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
