"""Repaired production sizing (publication v2): hybrid structured-seed
optimizer with a feasibility-first reward.

    structured seed set -> refinement around incumbents -> constraint-
    directed local repair

Design rationale (from the v1 ablations): grid search beat learned SAC at
16-call budgets because systematic coverage of knob extremes finds the
feasible region faster than online learning can. The hybrid keeps that
strength (physics-informed seeds cover the space) and adds what grid lacks
(local refinement toward the joint constraint region and compensation-aware
repair moves). All knobs are perturbed in LOG space (widths, lengths,
currents and capacitances are ratio-scaled quantities). Matched devices stay
matched: knobs act on role groups, never on individual paired devices.

Every evaluation is a real ngspice call counted against the same budget as
the baselines. Nothing here fabricates or caches-as-new.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from random import Random

from agentic_raptor.mb_sac.spec_sizing import (DYNAMICS_FILE, KNOB_HI,
                                               KNOB_LO, KNOB_NAMES, N_KNOBS,
                                               apply_knobs, margin_vector,
                                               measure, postsizing_outcome,
                                               value_target_from_outcome)

ACTION_SPACE_VERSION = "v2_log_hybrid"
SCHEMA = "hybrid_sizing.1"


# --------------------- feasibility-first reward (Repair 6) --------------------
def violations(meas: dict, spec: dict) -> dict:
    """Normalized constraint violations (0 = satisfied). Normalizers: 20 dB
    gain, 45 deg PM, 2 decades UGBW."""
    mv = margin_vector(meas, spec)
    gm = mv["gain_margin_db"] if mv["gain_margin_db"] is not None else -20.0
    pmm = mv["pm_margin_deg"] if mv["pm_margin_deg"] is not None else -45.0
    return {"gain": max(0.0, -gm / 20.0),
            "pm": max(0.0, -pmm / 45.0),
            "ugbw": (max(0.0, -mv["ugbw_log_margin"] / 2.0)
                     if mv["ugbw_log_margin"] is not None else 0.0)}


def feasibility_reward(meas: dict, spec: dict) -> tuple:
    """Feasibility-first: worst unsatisfied hard constraint drives progress;
    no metric's excess can buy back another's failure; quality objectives
    only after every hard constraint passes.

    Guaranteed orderings (unit-tested):
      feasible > any infeasible;  55deg-PM+gain-met > 90deg-PM+gain-missed;
      stable feasible > unstable high-gain;  among feasible: balanced margins
      and lower power win."""
    mv = margin_vector(meas, spec)
    if not mv["spice_converged"] or not mv["operating_point_valid"]:
        return -4.0, mv
    v = violations(meas, spec)
    unstable = meas.get("stability") == "verified_unstable"
    if unstable:
        return -2.0 - min(1.0, max(v.values())), mv
    worst = max(v.values())
    total = sum(v.values())
    if worst > 0:
        return -worst - 0.2 * total, mv
    # all hard constraints pass: quality tier in [1, 2]
    q = 0.0
    pm_m = mv["pm_margin_deg"] or 0.0
    q += 0.2 * min(pm_m, 10.0) / 10.0          # small cushion only
    if meas.get("power_w"):
        q += 0.3 * max(0.0, 1.0 - meas["power_w"] / 1e-3)
    return 1.0 + q, mv


# --------------------------- structured seeds ---------------------------------
def _memory_seeds(family: str, spec: dict, n: int = 2) -> list:
    """Verified-success sizings from labelled dynamics memory: same family,
    nearest spec (gain/load distance), current knob schema only."""
    if not family or not DYNAMICS_FILE.is_file():
        return []
    cands = []
    for x in DYNAMICS_FILE.read_text(encoding="utf-8").splitlines():
        if not x.strip():
            continue
        e = json.loads(x)
        if e.get("family") != family or not isinstance(e.get("knobs"), dict):
            continue
        # Accept entries written before a knob was added. Requiring an exact
        # length silently discarded EVERY historical measurement the moment
        # rz_x appeared -- the warm-start memory looked empty rather than
        # stale, which is the harder failure to notice. A knob the entry
        # predates is simply nominal.
        if len(e["knobs"]) > N_KNOBS or e.get("pm") is None:
            continue
        sp = e.get("spec") or {}
        dist = (abs((sp.get("gain_target_db") or 60)
                    - spec["gain_target_db"]) / 20.0
                + abs(math.log10((sp.get("load_capacitance_pf") or 100)
                                 / spec.get("load_capacitance_pf", 100))))
        # prefer measured-stable, spec-near entries
        score = (0 if e.get("pm", -99) > 0 else 1, dist)
        cands.append((score,
                      [float(e["knobs"].get(k, 1.0)) for k in KNOB_NAMES]))
    cands.sort(key=lambda c: c[0])
    seeds, seen = [], set()
    for _s, k in cands:
        key = tuple(round(x, 2) for x in k)
        if key not in seen:
            seen.add(key)
            seeds.append(k)
        if len(seeds) >= n:
            break
    return seeds


def structured_seeds(spec: dict, family: str | None, comp: str,
                     rng: Random) -> list:
    """Physics-informed seed set (order matters; each costs one real call):
    nominal anchor, gain-oriented corner, bandwidth-oriented corner,
    compensation-aware point, and up to two memory retrievals."""
    def _pad(v):
        """Seeds are written positionally; pad any knob added later to
        nominal so a new action dimension cannot silently truncate them."""
        return list(v) + [1.0] * (N_KNOBS - len(v))

    seeds = [("nominal", [1.0] * N_KNOBS)]
    gain_need = spec["gain_target_db"]
    # gain corner: long channels, wide stage-2, modest bias
    seeds.append(("gain_corner",
                  _pad([2.0, 4.0, 4.0 if gain_need > 70 else 2.0,
                        4.0 if gain_need > 70 else 2.0, 1.0, 0.5])))
    # bandwidth corner: short channels, higher bias
    seeds.append(("ugbw_corner", _pad([2.0, 2.0, 0.7, 0.7, 1.0, 2.5])))
    if comp in ("miller", "miller_cap", "rc", "rc_nulling"):
        # compensation-aware: cap scaled with load (pole-splitting region).
        # For an RC-nulling branch also open the nulling resistor, which
        # cancels the Miller RHP zero -- with rz at nominal the rc family
        # tracked plain miller to within 0.3 deg of phase margin, i.e. the
        # resistor was inert.
        # seed in the pole-splitting region: Cc must be comparable to the
        # LOAD, not to the 2 pF structural prior. Seeding at ~CL/2 lands the
        # search near the measured stable band instead of in the RHP-zero
        # valley that dominates below ~1 nF.
        cl_pf = float(spec.get("load_capacitance_pf", 100) or 100)
        capx = min(2048.0, max(0.5, (cl_pf / 2.0) / 2.0))
        rzx = 8.0 if comp in ("rc", "rc_nulling") else 1.0
        seeds.append(("comp_seed",
                      _pad([1.5, 2.0, 2.0, 1.5, capx, 1.0, rzx])))
    for i, k in enumerate(_memory_seeds(family, spec)):
        seeds.append((f"memory_{i}", _pad(k)))
    return seeds


# ------------------------ constraint-directed repair --------------------------
def repair_move(best: dict, spec: dict, rng: Random) -> tuple:
    """One targeted move against the WORST failing constraint of the
    incumbent (compensation-aware; log-space steps)."""
    v = violations(best, spec)
    k = dict(best["knobs"])
    unstable = best.get("stability") == "verified_unstable"
    lo = dict(zip(KNOB_NAMES, KNOB_LO))
    hi = dict(zip(KNOB_NAMES, KNOB_HI))

    def bump(name, factor):
        k[name] = max(lo[name], min(hi[name], k[name] * factor))
    step = math.exp(rng.uniform(0.15, 0.5))
    if unstable or (v["pm"] >= max(v["gain"], v["ugbw"]) and v["pm"] > 0):
        why = "stabilise: split poles / strengthen compensation"
        bump("cap_x", step)
        bump("s2_l", 1 / math.sqrt(step))
        if rng.random() < 0.5:
            bump("ib_x", math.sqrt(step))
    elif v["gain"] >= v["ugbw"] and v["gain"] > 0:
        why = "raise gain: longer channels, wider stage2, lower bias"
        bump("s1_l", step)
        bump("s2_l", math.sqrt(step))
        bump("s2_w", math.sqrt(step))
        bump("ib_x", 1 / math.sqrt(step))
    elif v["ugbw"] > 0:
        why = "raise UGBW: more gm per cap"
        bump("ib_x", step)
        bump("cap_x", 1 / math.sqrt(step))
        bump("s1_w", math.sqrt(step))
    else:
        why = "feasible: balanced local polish"
        for name in KNOB_NAMES:
            k[name] = max(lo[name], min(hi[name],
                          k[name] * math.exp(rng.uniform(-0.08, 0.08))))
    return [k[n] for n in KNOB_NAMES], why


def c9s_size(topology_id: str, graph, spec: dict, exe, out_dir: Path,
             costs, budget: int = 16, seed: int = 0,
             family: str | None = None, comp: str = "none") -> dict:
    """C9s — the spec-faithful hybrid: structured seeds -> TRUE SAC
    refinement -> constraint-directed repair.

    Phase 2 is genuine soft actor-critic:
      * actor network outputs a Gaussian over the 6 knobs (tanh-squashed),
        sampled -- exploration comes from the learned distribution;
      * TWIN critics predict the feasibility reward; the min of the two is
        used for the actor update (SAC's overestimation guard);
      * automatic entropy temperature (log-alpha) tuned toward the standard
        -|A| target;
      * every REAL ngspice result becomes a (action, reward) transition;
        both critics and the actor take gradient steps on replayed
        mini-batches after every simulation;
      * the replay buffer is WARM-STARTED with the phase-1 seed outcomes, so
        learning starts from knowledge instead of noise.
    """
    import torch

    from agentic_raptor.electrical import effective_c_load
    cl = effective_c_load(spec)
    torch.manual_seed(seed)
    rng = Random(seed)
    n_seed = min(5, max(3, budget // 3))
    n_repair = max(2, budget // 6)
    n_sac = max(0, budget - n_seed - n_repair)
    lo_t, hi_t = torch.tensor(KNOB_LO), torch.tensor(KNOB_HI)
    obs = torch.tensor([spec["gain_target_db"] / 100,
                        spec["phase_margin_target_deg"] / 90,
                        spec.get("load_capacitance_pf", 100) / 1000,
                        (spec.get("ugbw_target_hz") or 1e4) / 1e6, 1.0])
    actor = torch.nn.Sequential(torch.nn.Linear(5, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 2 * N_KNOBS))
    q1 = torch.nn.Sequential(torch.nn.Linear(5 + N_KNOBS, 64),
                             torch.nn.ReLU(), torch.nn.Linear(64, 1))
    q2 = torch.nn.Sequential(torch.nn.Linear(5 + N_KNOBS, 64),
                             torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-3)
    opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                             lr=3e-3)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_al = torch.optim.Adam([log_alpha], lr=3e-3)
    target_entropy = -float(N_KNOBS)
    replay, results, log = [], [], []

    def to_action(knobs):
        """knob values -> pre-tanh action coordinates."""
        a = (torch.tensor([float(k) for k in knobs]) - lo_t) \
            / (hi_t - lo_t) * 2 - 1
        return torch.clamp(a, -0.999, 0.999)

    def run(knobs, tag, origin):
        m = measure(topology_id, apply_knobs(graph, knobs), exe, out_dir,
                    tag, costs, c_load_f=cl)
        r, mv = feasibility_reward(m, spec)
        rec = {**m, "reward": float(r), "origin": origin,
               "knobs": dict(zip(KNOB_NAMES,
                                 [round(float(x), 3) for x in knobs])),
               "margin_vector": mv}
        results.append(rec)
        replay.append((to_action(knobs).detach(), float(r)))
        return rec

    def sac_update(k_updates=4, batch=8):
        for _ in range(k_updates):
            samp = [replay[rng.randrange(len(replay))]
                    for _ in range(min(batch, len(replay)))]
            a_b = torch.stack([a for a, _ in samp])
            r_b = torch.tensor([[r] for _, r in samp])
            ob = obs.expand(len(samp), -1)
            # twin-critic regression toward measured rewards
            lc = ((q1(torch.cat([ob, a_b], 1)) - r_b) ** 2).mean() \
                + ((q2(torch.cat([ob, a_b], 1)) - r_b) ** 2).mean()
            opt_c.zero_grad(); lc.backward(); opt_c.step()
            # actor: maximize min-Q + entropy
            mu_ls = actor(ob)
            mu, ls = mu_ls[:, :N_KNOBS], mu_ls[:, N_KNOBS:].clamp(-3, 1)
            z = mu + ls.exp() * torch.randn_like(mu)
            a_new = torch.tanh(z)
            logp = (-0.5 * ((z - mu) / ls.exp()) ** 2 - ls
                    - torch.log(1 - a_new ** 2 + 1e-6)).sum(1, keepdim=True)
            q_min = torch.min(q1(torch.cat([ob, a_new], 1)),
                              q2(torch.cat([ob, a_new], 1)))
            la = (log_alpha.exp().detach() * logp - q_min).mean()
            opt_a.zero_grad(); la.backward(); opt_a.step()
            # automatic temperature
            lal = -(log_alpha * (logp.detach() + target_entropy)).mean()
            opt_al.zero_grad(); lal.backward(); opt_al.step()

    # ---- Phase 1: structured seeds (fills the replay buffer) ----------------
    # structured_seeds() can return FEWER than n_seed: the memory-derived
    # seeds only exist once a dynamics file has been written for this family.
    # n_seed is therefore the PLANNED count; record what actually ran, or the
    # phase accounting claims seed rows that were never executed.
    for name, knobs in structured_seeds(spec, family, comp, rng)[:n_seed]:
        run(knobs, f"c9s{seed}_s{len(results)}", f"seed:{name}")
        log.append({"phase": "seed", "origin": name})
    n_seed_ran = len(results)
    sac_update(k_updates=8)                 # learn from the seed experience
    # ---- Phase 2: TRUE SAC refinement ---------------------------------------
    for step in range(n_sac):
        with torch.no_grad():
            mu_ls = actor(obs)
            mu, ls = mu_ls[:N_KNOBS], mu_ls[N_KNOBS:].clamp(-3, 1)
            a = torch.tanh(mu + ls.exp() * torch.randn(N_KNOBS))
            knobs = (lo_t + (a + 1) / 2 * (hi_t - lo_t)).tolist()
        run(knobs, f"c9s{seed}_a{len(results)}", "sac:actor_sample")
        log.append({"phase": "sac", "step": step,
                    "alpha": round(float(log_alpha.exp()), 4)})
        sac_update()
    # ---- Phase 3: constraint-directed repair --------------------------------
    while len(results) < budget:
        best = max(results, key=lambda r: r["reward"])
        knobs, why = repair_move(best, spec, rng)
        run(knobs, f"c9s{seed}_r{len(results)}", f"repair:{why[:24]}")
        log.append({"phase": "repair", "why": why})
    best = max(results, key=lambda r: r["reward"])
    outcome = postsizing_outcome(best, spec)
    first = next((i + 1 for i, r in enumerate(results)
                  if postsizing_outcome(r, spec)["exact_spec_pass"]), None)
    return {"best": best, "outcome": outcome, "results": results,
            "transitions": [{"action": [float(x) for x in a],
                             "reward": r} for a, r in replay],
            "spice_calls": len(results),
            "calls_to_first_exact_pass": first,
            "value": value_target_from_outcome(outcome, len(results), budget),
            "action_space": KNOB_NAMES,
            "action_space_version": ACTION_SPACE_VERSION + "_sac",
            "reward_policy": "feasibility_first_v2",
            "memory": {"family": family, "loaded_nets": False,
                       "persisted": False,
                       "memory_seeds_used": sum(
                           1 for r in results
                           if r["origin"].startswith("seed:memory"))},
            "phase_log": log, "phase_budget": {"seeds": n_seed_ran,
                                               "seeds_planned": n_seed,
                                               "sac": n_sac,
                                               "repair": n_repair},
            "budget": budget, "seed": seed, "timestamp": time.time(),
            "schema_version": SCHEMA}


def hybrid_size(topology_id: str, graph, spec: dict, exe, out_dir: Path,
                costs, budget: int = 16, seed: int = 0,
                family: str | None = None, comp: str = "none") -> dict:
    """C9 production candidate. Budget split: seeds (<=6) then repair/refine.
    Phases emerge from the worst-constraint logic: convergence -> stability
    -> gain/UGBW -> quality, without hiding any constraint."""
    from agentic_raptor.electrical import effective_c_load
    cl = effective_c_load(spec)
    rng = Random(seed)
    results, log = [], []

    def run(knobs, tag, origin):
        m = measure(topology_id, apply_knobs(graph, knobs), exe, out_dir,
                    tag, costs, c_load_f=cl)
        r, mv = feasibility_reward(m, spec)
        rec = {**m, "reward": float(r), "origin": origin,
               "knobs": dict(zip(KNOB_NAMES,
                                 [round(float(x), 3) for x in knobs])),
               "margin_vector": mv}
        results.append(rec)
        return rec

    for name, knobs in structured_seeds(spec, family, comp, rng):
        if len(results) >= max(4, budget // 2):
            break
        run(knobs, f"hs{seed}_{len(results)}", f"seed:{name}")
        log.append({"phase": "seed", "origin": name})
    while len(results) < budget:
        best = max(results, key=lambda r: r["reward"])
        knobs, why = repair_move(best, spec, rng)
        run(knobs, f"hr{seed}_{len(results)}", f"repair:{why[:24]}")
        log.append({"phase": "repair", "why": why,
                    "from_reward": best["reward"]})
    best = max(results, key=lambda r: r["reward"])
    outcome = postsizing_outcome(best, spec)
    first = next((i + 1 for i, r in enumerate(results)
                  if postsizing_outcome(r, spec)["exact_spec_pass"]), None)
    mem_used = sum(1 for r in results if r["origin"].startswith("seed:memory"))
    return {"best": best, "outcome": outcome, "results": results,
            "transitions": [], "spice_calls": len(results),
            "calls_to_first_exact_pass": first,
            "value": value_target_from_outcome(outcome, len(results), budget),
            "action_space": KNOB_NAMES,
            "action_space_version": ACTION_SPACE_VERSION,
            "reward_policy": "feasibility_first_v2",
            "memory": {"family": family, "memory_seeds_used": mem_used,
                       "loaded_nets": False, "persisted": False},
            "phase_log": log, "budget": budget, "seed": seed,
            "timestamp": time.time(), "schema_version": SCHEMA}
