"""SAC reward ablation RW0-RW5 under equal budgets on real ngspice.

  RW0 historical scalar (stability +-1, tanh gain/pm terms, no target logic)
  RW1 unsaturated PM (margin-vector reward but PM keeps paying past target)
  RW2 PM saturates AT the target (zero cushion)
  RW3 PM saturates after the 10-degree cushion (production default)
  RW4 full target-margin vector == RW3 here (RW3 IS the margin-vector
      reward; recorded as alias so the mapping stays explicit)
  RW5 preference-conditioned multi-objective — not implemented; recorded

Behavioral requirements are enforced by tests (test_postsizing.TestReward);
this module measures end-to-end search outcomes per reward under one budget.
"""

from __future__ import annotations

import json
import time

from agentic_raptor.mb_sac.spec_sizing import (PM_CUSHION_DEG, legacy_reward,
                                               margin_vector,
                                               postsizing_outcome, sac_size,
                                               spec_reward)
from agentic_raptor.publication import FREEZE, PUB
from agentic_raptor.publication.sizing_baselines import _mk_graph


def rw1_unsaturated(meas, spec):
    r, mv = spec_reward(meas, spec, pm_cushion_deg=1e9)
    if mv["pm_margin_deg"] is not None and mv["pm_margin_deg"] > 0:
        r += 0.5 * mv["pm_margin_deg"] / 45.0        # PM excess keeps paying
    return r, mv


def rw2_saturate_at_target(meas, spec):
    return spec_reward(meas, spec, pm_cushion_deg=1e-6)


from agentic_raptor.mb_sac.hybrid_sizing import feasibility_reward

REWARDS = {"RW0": legacy_reward, "RW1": rw1_unsaturated,
           "RW2": rw2_saturate_at_target, "RW3": spec_reward,
           "RW5": feasibility_reward}


def run(task_ids: list, budget: int = 16, seed: int = 13) -> dict:
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    bench = json.loads((FREEZE / "benchmark.json").read_text())
    tasks = [t for t in bench["tasks"] if t["task_id"] in task_ids]
    exe = discover_ngspice()
    out = (PUB / "reward_ablation_runs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tasks:
        tid, g = _mk_graph(t)
        spec = dict(t["spec"])
        for name, fn in REWARDS.items():
            t0 = time.time()
            sz = sac_size(tid, g, spec, exe, out, new_costs(), budget=budget,
                          seed=seed, reward_fn=fn, persist=False)
            b, o = sz["best"], postsizing_outcome(sz["best"], spec)
            rows.append({"task_id": t["task_id"], "tier": t["tier"],
                         "class": t["target_class"], "reward": name,
                         "best_gain_db": b["gain_db"],
                         "best_pm_deg": b["pm_deg"],
                         "pm_excess_deg": (o["margin_vector"]["pm_margin_deg"]
                                           if o["margin_vector"]
                                           ["pm_margin_deg"] else None),
                         "gain_margin_db":
                             o["margin_vector"]["gain_margin_db"],
                         "exact_pass": o["exact_spec_pass"],
                         "constraints_passed": o["hard_constraints_passed"],
                         "distance":
                             o["normalized_distance_to_feasibility"],
                         "spice_calls": sz["spice_calls"],
                         "runtime_s": round(time.time() - t0, 1)})
    doc = {"budget": budget, "seed": seed,
           "rewards": {**{k: k for k in REWARDS},
                       "RW4": "alias_of_RW3_margin_vector",
                       "RW5": "feasibility_first_worst_violation_v2"},
           "pm_cushion_deg": PM_CUSHION_DEG, "rows": rows,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (PUB / "reward_ablation.json").write_text(
        json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


if __name__ == "__main__":
    import sys
    ids = (sys.argv[1].split(",") if len(sys.argv) > 1
           else ["T000", "T012", "T030", "T060"])
    d = run(ids)
    agg = {}
    for r in d["rows"]:
        agg.setdefault(r["reward"], []).append(r)
    for k, rs in sorted(agg.items()):
        print(k, "exact", sum(x["exact_pass"] for x in rs), "/", len(rs),
              "| mean dist",
              round(sum((1.0 if x["distance"] is None else x["distance"]) for x in rs) / len(rs), 3),
              "| mean pm-excess",
              round(sum(x["pm_excess_deg"] or 0 for x in rs) / len(rs), 1))
