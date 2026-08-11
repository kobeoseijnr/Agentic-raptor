"""Sizing baselines C0-C8 under EQUAL real-SPICE budgets on the frozen
benchmark. Every electrical number comes from ngspice; per-method SPICE calls
are counted; nothing is cached across methods.

Method mapping (honest labels):
  C0 nominal            — no sizing (1 call)
  C1 random             — uniform random knobs
  C2 grid               — coarse per-knob grid walk
  C3 tpe_lite           — TPE-style: sample near top-quantile of history
  C4 sac_standard       — minimal SAC (actor+critics only; no surrogate,
                          no ranker, no refinement tail)
  C5 (not implemented)  — graph-conditioned-only SAC: engine has no
                          graph-only mode; recorded as unavailable
  C6 sac_spec           — full spec-conditioned engine, cold start
                          (persist=False)
  C7 raptor_full        — C6 + persistent per-family memory warm-start
  C8 legacy             — pre-repair engine: width-only 3-knob space +
                          raw-metric reward (the original sizing engine)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from random import Random

from agentic_raptor.mb_sac.spec_sizing import (KNOB_HI, KNOB_LO, KNOB_NAMES,
                                               N_KNOBS, apply_knobs,
                                               legacy_reward, measure,
                                               postsizing_outcome, sac_size,
                                               spec_reward)
from agentic_raptor.publication import FREEZE, PUB, ROOT


def _mk_graph(task):
    from agentic_raptor.mapping import map_family
    from agentic_raptor.topology_rl.stage3e2_edits import (EditRejected,
                                                           apply_edit)
    cls = task["target_class"]
    stages, comp = int(cls[0]), cls.split("_", 1)[1]

    class _S:
        topology_id = f"bl_{task['task_id']}"
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
    return _S.topology_id, g


def _summ(results, spec, t0):
    best = max(results, key=lambda r: r["reward"]) if results else None
    o = postsizing_outcome(best, spec) if best else None
    first = next((i + 1 for i, r in enumerate(results)
                  if postsizing_outcome(r, spec)["exact_spec_pass"]), None)
    return {"spice_calls": len(results),
            "best_gain_db": best and best.get("gain_db"),
            "best_pm_deg": best and best.get("pm_deg"),
            "stable": bool(best and best.get("stable")),
            "exact_pass": bool(o and o["exact_spec_pass"]),
            "gain_pass": bool(o and o["passes"]["gain"]),
            "pm_pass": bool(o and o["passes"]["pm"]),
            "ugbw_pass": bool(o and o["passes"]["ugbw"]),
            "constraints_passed": o["hard_constraints_passed"] if o else 0,
            "distance": o["normalized_distance_to_feasibility"] if o else None,
            "calls_to_first_pass": first,
            "runtime_s": round(time.time() - t0, 1)}


def _sample_loop(tid, g, spec, exe, out, costs, budget, seed, sampler,
                 c_load_f: float | None = None):
    """Shared loop for stateless samplers; nominal point measured first."""
    from agentic_raptor.electrical import effective_c_load
    cl = effective_c_load(spec, override=c_load_f)
    rng = Random(seed)
    results = []
    for step in range(budget):
        knobs = [1.0] * N_KNOBS if step == 0 else sampler(rng, results, step)
        m = measure(tid, apply_knobs(g, knobs), exe, out, f"s{step}", costs,
                    c_load_f=cl)
        r, _mv = spec_reward(m, spec)
        results.append({**m, "reward": float(r),
                        "knobs": dict(zip(KNOB_NAMES,
                                          [round(k, 3) for k in knobs]))})
    return results


def _random(rng, results, step):
    return [lo + rng.random() * (hi - lo)
            for lo, hi in zip(KNOB_LO, KNOB_HI)]


def _grid(rng, results, step):
    levels = [(lo, (lo + hi) / 2, hi) for lo, hi in zip(KNOB_LO, KNOB_HI)]
    knobs = []
    s = step
    for lv in levels:
        knobs.append(lv[s % 3])
        s //= 3
    return knobs


def _tpe(rng, results, step):
    if len(results) < 4:
        return _random(rng, results, step)
    top = sorted(results, key=lambda r: -r["reward"])[:max(2, len(results) // 3)]
    base = rng.choice(top)["knobs"]
    return [max(lo, min(hi, float(base[k]) * (1 + 0.25 * (rng.random() - 0.5))))
            for k, lo, hi in zip(KNOB_NAMES, KNOB_LO, KNOB_HI)]


def _sac_minimal(tid, g, spec, exe, out, costs, budget, seed):
    """C4: actor+critics only — no surrogate, no ranker, no refinement."""
    import torch

    from agentic_raptor.electrical import effective_c_load
    cl = effective_c_load(spec)
    torch.manual_seed(seed)
    actor = torch.nn.Sequential(torch.nn.Linear(5, 48), torch.nn.ReLU(),
                                torch.nn.Linear(48, 2 * N_KNOBS))
    q1 = torch.nn.Sequential(torch.nn.Linear(5 + N_KNOBS, 48),
                             torch.nn.ReLU(), torch.nn.Linear(48, 1))
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-3)
    opt_c = torch.optim.Adam(q1.parameters(), lr=3e-3)
    lo, hi = torch.tensor(KNOB_LO), torch.tensor(KNOB_HI)
    results = []
    for step in range(budget):
        obs = torch.tensor([spec["gain_target_db"] / 100,
                            spec["phase_margin_target_deg"] / 90,
                            spec.get("load_capacitance_pf", 100) / 1000,
                            (budget - step) / budget, 0.0])
        if step == 0:
            a = torch.zeros(N_KNOBS)
            knobs = torch.ones(N_KNOBS)
        else:
            mu_ls = actor(obs)
            mu, ls = mu_ls[:N_KNOBS], mu_ls[N_KNOBS:].clamp(-3, 1)
            a = torch.tanh(mu + ls.exp() * torch.randn(N_KNOBS))
            knobs = (lo + (a + 1) / 2 * (hi - lo)).detach()
        m = measure(tid, apply_knobs(g, [float(x) for x in knobs]), exe, out,
                    f"c4_{step}", costs, c_load_f=cl)
        r, _mv = spec_reward(m, spec)
        results.append({**m, "reward": float(r),
                        "knobs": dict(zip(KNOB_NAMES,
                                          [round(float(x), 3)
                                           for x in knobs]))})
        qin = torch.cat([obs, a.detach()])
        lc = (q1(qin)[0] - float(r)) ** 2
        opt_c.zero_grad(); lc.backward(); opt_c.step()
        mu_ls2 = actor(obs)
        a2 = torch.tanh(mu_ls2[:N_KNOBS])
        la = -q1(torch.cat([obs, a2]))[0]
        opt_a.zero_grad(); la.backward(); opt_a.step()
    return results


LEGACY_LO = [1.0, 0.5, 1.0]     # s1_w, s2_w, cap_x — the pre-repair space
LEGACY_HI = [2.0, 1.0, 6.0]


def _legacy(tid, g, spec, exe, out, costs, budget, seed):
    """C8: width-only action space + raw-metric legacy reward."""
    from agentic_raptor.electrical import effective_c_load
    cl = effective_c_load(spec)
    rng = Random(seed)
    results = []
    for step in range(budget):
        v = ([1.0, 1.0, 1.0] if step == 0 else
             [lo + rng.random() * (hi - lo)
              for lo, hi in zip(LEGACY_LO, LEGACY_HI)])
        knobs = [v[0], v[1], 1.0, 1.0, v[2], 1.0]   # no L, no bias control
        m = measure(tid, apply_knobs(g, knobs), exe, out, f"c8_{step}", costs,
                    c_load_f=cl)
        results.append({**m, "reward": float(legacy_reward(m, spec)),
                        "knobs": dict(zip(KNOB_NAMES,
                                          [round(k, 3) for k in knobs]))})
    return results


def _one_task(args):
    """Worker: selected methods for one task (own process, own ngspice)."""
    task_id, budget, seed, methods = (args if len(args) == 4
                                      else (*args, ("C0", "C1", "C2", "C3",
                                                    "C4", "C6", "C7", "C8",
                                                    "C9", "C9s")))
    import os
    os.environ.setdefault("AGENTIC_RAPTOR_SIZING_MEMORY",
                          str((PUB / "sizing_baselines_runs"
                               / "warm_memory").resolve()))
    d = run_baselines(task_ids=[task_id], budget=budget, seed=seed,
                      _write=False, methods=tuple(methods))
    return d["rows"]


def run_baselines(task_ids: list | None = None, budget: int = 16,
                  seed: int = 11, workers: int = 1, _write: bool = True,
                  methods: tuple = ("C0", "C1", "C2", "C3", "C4", "C6",
                                    "C7", "C8", "C9", "C9s")) -> dict:
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    bench = json.loads((FREEZE / "benchmark.json").read_text())
    tasks = [t for t in bench["tasks"]
             if task_ids is None or t["task_id"] in task_ids]
    if workers > 1 and len(tasks) > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            all_rows = pool.map(_one_task,
                                [(t["task_id"], budget, seed,
                                  tuple(methods)) for t in tasks])
        rows = [r for chunk in all_rows for r in chunk]
        doc = {"budget": budget, "seed": seed, "tasks": len(tasks),
               "workers": workers,
               "methods": {"C5": "unavailable: engine has no graph-only mode"},
               "rows": rows, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        PUB.mkdir(parents=True, exist_ok=True)
        (PUB / "sizing_baselines.json").write_text(
            json.dumps(doc, indent=1, default=str), encoding="utf-8")
        return doc
    exe = discover_ngspice()
    out = (PUB / "sizing_baselines_runs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    memdir = out / "warm_memory"
    import os
    rows = []
    for t in tasks:
        tid, g = _mk_graph(t)
        spec = dict(t["spec"])
        for method in methods:
            t0 = time.time()
            costs = new_costs()
            if method == "C0":
                from agentic_raptor.electrical import effective_c_load
                m = measure(tid, g, exe, out, "c0", costs,
                           c_load_f=effective_c_load(spec))
                r, _ = spec_reward(m, spec)
                results = [{**m, "reward": float(r),
                            "knobs": dict(zip(KNOB_NAMES, [1.0] * N_KNOBS))}]
            elif method == "C1":
                results = _sample_loop(tid, g, spec, exe, out, costs, budget,
                                       seed, _random)
            elif method == "C2":
                results = _sample_loop(tid, g, spec, exe, out, costs, budget,
                                       seed, _grid)
            elif method == "C3":
                results = _sample_loop(tid, g, spec, exe, out, costs, budget,
                                       seed, _tpe)
            elif method == "C4":
                results = _sac_minimal(tid, g, spec, exe, out, costs, budget,
                                       seed)
            elif method == "C6":
                sz = sac_size(tid, g, spec, exe, out, costs, budget=budget,
                              seed=seed, persist=False)
                results = [dict(r, reward=r["reward"]) for r in sz["results"]]
            elif method == "C7":
                os.environ["AGENTIC_RAPTOR_SIZING_MEMORY"] = str(memdir)
                import importlib
                from agentic_raptor.mb_sac import spec_sizing as _ss
                importlib.reload(_ss)
                sz = _ss.sac_size(tid, g, spec, exe, out, costs,
                                  budget=budget, seed=seed,
                                  family=t["target_class"], persist=True)
                results = [dict(r, reward=r["reward"]) for r in sz["results"]]
                del os.environ["AGENTIC_RAPTOR_SIZING_MEMORY"]
                importlib.reload(_ss)
            elif method in ("C9", "C9s"):
                from agentic_raptor.mb_sac.hybrid_sizing import (c9s_size,
                                                                 hybrid_size)
                comp = t["target_class"].split("_", 1)[1]
                fn = hybrid_size if method == "C9" else c9s_size
                sz = fn(tid, g, spec, exe, out, costs,
                        budget=budget, seed=seed,
                        family=t["target_class"], comp=comp)
                results = [dict(r, reward=r["reward"])
                           for r in sz["results"]]
            else:
                results = _legacy(tid, g, spec, exe, out, costs, budget, seed)
            rows.append({"task_id": t["task_id"], "tier": t["tier"],
                         "class": t["target_class"], "method": method,
                         **_summ(results, spec, t0)})
    doc = {"budget": budget, "seed": seed, "tasks": len(tasks),
           "methods": {"C5": "unavailable: engine has no graph-only mode"},
           "rows": rows, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if _write:
        PUB.mkdir(parents=True, exist_ok=True)
        (PUB / "sizing_baselines.json").write_text(
            json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


def run_sr_battery(task_ids: list, budget: int = 16, seed: int = 11) -> dict:
    """SR0-SR5: surrogate/ranker screening ablation, equal budgets.
      SR0 neither | SR1 surrogate only | SR2 ranker only
      SR3 surrogate + GLOBAL (spec-blind) ranker
      SR4 surrogate + spec-conditioned ranker
      SR5 full policy (SR4 + persistent memory warm-start)"""
    import os
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    bench = json.loads((FREEZE / "benchmark.json").read_text())
    tasks = [t for t in bench["tasks"] if t["task_id"] in task_ids]
    exe = discover_ngspice()
    out = (PUB / "sr_battery_runs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms = {"SR0": dict(use_surrogate=False, use_ranker=False),
            "SR1": dict(use_surrogate=True, use_ranker=False),
            "SR2": dict(use_surrogate=False, use_ranker=True),
            "SR3": dict(use_surrogate=True, use_ranker=True,
                        ranker_spec_conditioned=False),
            "SR4": dict(use_surrogate=True, use_ranker=True),
            "SR5": dict(use_surrogate=True, use_ranker=True, persist=True)}
    rows = []
    memdir = out / "sr5_memory"
    for t in tasks:
        tid, g = _mk_graph(t)
        spec = dict(t["spec"])
        for arm, kw in arms.items():
            t0 = time.time()
            kw2 = dict(kw)
            if arm == "SR5":
                os.environ["AGENTIC_RAPTOR_SIZING_MEMORY"] = str(memdir)
                import importlib
                from agentic_raptor.mb_sac import spec_sizing as _ss
                importlib.reload(_ss)
                sz = _ss.sac_size(tid, g, spec, exe, out, new_costs(),
                                  budget=budget, seed=seed,
                                  family=t["target_class"], **kw2)
                del os.environ["AGENTIC_RAPTOR_SIZING_MEMORY"]
                importlib.reload(_ss)
            else:
                kw2["persist"] = False
                sz = sac_size(tid, g, spec, exe, out, new_costs(),
                              budget=budget, seed=seed, **kw2)
            results = [dict(r, reward=r["reward"]) for r in sz["results"]]
            rows.append({"task_id": t["task_id"], "tier": t["tier"],
                         "class": t["target_class"], "method": arm,
                         **_summ(results, spec, t0)})
    doc = {"budget": budget, "seed": seed, "tasks": len(tasks), "rows": rows,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (PUB / "sr_battery.json").write_text(json.dumps(doc, indent=1,
                                                    default=str),
                                         encoding="utf-8")
    return doc


def run_budget_curves(task_ids: list, budgets=(8, 16, 32, 64),
                      methods=("C2", "C3", "C6", "C9", "C9s"),
                      seed: int = 11) -> dict:
    """Exact-pass and distance vs simulator budget for representative
    methods (grid, TPE-lite, spec-SAC)."""
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    bench = json.loads((FREEZE / "benchmark.json").read_text())
    tasks = [t for t in bench["tasks"] if t["task_id"] in task_ids]
    exe = discover_ngspice()
    out = (PUB / "budget_curve_runs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tasks:
        tid, g = _mk_graph(t)
        spec = dict(t["spec"])
        for budget in budgets:
            for m in methods:
                t0 = time.time()
                costs = new_costs()
                if m == "C2":
                    results = _sample_loop(tid, g, spec, exe, out, costs,
                                           budget, seed, _grid)
                elif m == "C3":
                    results = _sample_loop(tid, g, spec, exe, out, costs,
                                           budget, seed, _tpe)
                else:
                    sz = sac_size(tid, g, spec, exe, out, costs,
                                  budget=budget, seed=seed, persist=False)
                    results = [dict(r, reward=r["reward"])
                               for r in sz["results"]]
                rows.append({"task_id": t["task_id"], "budget": budget,
                             "method": m, **_summ(results, spec, t0)})
    doc = {"budgets": list(budgets), "methods": list(methods), "seed": seed,
           "tasks": len(tasks), "rows": rows,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (PUB / "budget_curves.json").write_text(json.dumps(doc, indent=1,
                                                       default=str),
                                            encoding="utf-8")
    return doc


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--sr":
        d = run_sr_battery(sys.argv[2].split(","))
        agg = {}
        for r in d["rows"]:
            agg.setdefault(r["method"], []).append(r)
        for m, rs in sorted(agg.items()):
            print(m, "exact", sum(x["exact_pass"] for x in rs), "/", len(rs),
                  "| mean dist", round(sum((1.0 if x["distance"] is None else x["distance"]) for x in rs)
                                       / len(rs), 3))
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--budget-curves":
        d = run_budget_curves(sys.argv[2].split(","))
        for b in d["budgets"]:
            for m in d["methods"]:
                rs = [r for r in d["rows"]
                      if r["budget"] == b and r["method"] == m]
                print(f"budget {b} {m}: exact "
                      f"{sum(x['exact_pass'] for x in rs)}/{len(rs)} | dist "
                      f"{round(sum(x['distance'] or 1 for x in rs)/len(rs), 3)}")
        sys.exit(0)
    workers = 1
    argv = [a for a in sys.argv[1:]]
    if "--workers" in argv:
        i = argv.index("--workers")
        workers = int(argv[i + 1])
        del argv[i:i + 2]
    methods = ("C0", "C1", "C2", "C3", "C4", "C6", "C7", "C8", "C9", "C9s")
    if "--methods" in argv:
        i = argv.index("--methods")
        methods = tuple(argv[i + 1].split(","))
        del argv[i:i + 2]
    ids = argv[0].split(",") if argv else None
    d = run_baselines(task_ids=ids, workers=workers, methods=methods)
    agg = {}
    for r in d["rows"]:
        agg.setdefault(r["method"], []).append(r)
    for m, rs in sorted(agg.items()):
        print(m, "exact", sum(x["exact_pass"] for x in rs), "/", len(rs),
              "| mean dist", round(sum((1.0 if x["distance"] is None else x["distance"]) for x in rs)
                                   / len(rs), 3))
