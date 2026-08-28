"""DESIGN PLANNER AGENT: spec -> StrategyPlan.

Deterministic physics + TRAIN-measured statistics; every number it uses is
either textbook electronics or a recorded TRAIN-domain measurement, and
every decision carries its rationale in the plan (no hidden thresholds).

Stage-count guidance is a SOFT prior (see StrategyPlan): the TRAIN gain
ceilings below are empirical ("2-stage observed success drops sharply
above ~95 dB"), not physical law. The planner discourages, never bans.
"""
from __future__ import annotations

import math

from agentic_raptor.agents.state import StrategyPlan

#: Best authoritatively measured gain per stage count, TRAIN domain
#: (artifacts/publication_v3/az_value_diagnostic_v1/OUTCOME_TABLE.jsonl,
#: joined by realised stage count): {2: 94.6 dB, 3: 136.4 dB}. A small
#: margin below the observed best marks where success "drops sharply".
#: REFIT 2026-08-17 from the TIER-2 forecast matrix (real ngspice, best
#: MEASURED passing gain per stage count): 3-stage 143.2 dB (class-AB output),
#: 4-stage 172.0 dB. Before this refit the table stopped at 3 stages, so on
#: 140+ dB specs the planner "preferred" 3 stages, screened OUT the 4-stage
#: candidates, and the offline replay showed AG dropping from 4/4 to 2/4
#: passes -- the exact "dataset ceiling baked in as a ban" failure the soft
#: prior was designed to avoid. Ceilings are MEASURED capability, refit as
#: evidence arrives (the A9 agent-episode audit surfaces this), promoted by
#: a human code change -- never automatically.
TRAIN_GAIN_CEILING_DB = {1: 45.0, 2: 94.6, 3: 143.2, 4: 172.0}
CEILING_MARGIN_DB = 3.0

#: gm demand ~ 2*pi*UGBW*CL. Tiers from the measured campaign record:
#: every spec at or below 1e-5 A-equivalent has passed somewhere; nothing
#: above 1e-4 has ever passed (t_boundary class).
GM_DEMAND_MEDIUM = 6.3e-5     # 2*pi * 1e5 Hz * 100 pF
GM_DEMAND_HARD = 6.3e-4       # 2*pi * 1e6 Hz * 100 pF


def plan(spec: dict) -> StrategyPlan:
    gain = float(spec.get("gain_target_db") or 40.0)
    pm = float(spec.get("phase_margin_target_deg") or 45.0)
    ugbw = float(spec.get("ugbw_target_hz") or 1e4)
    cl_f = float(spec.get("load_capacitance_pf") or 100.0) * 1e-12
    rationale = []

    # ---- stage-count prior (soft) -----------------------------------------
    preferred, discouraged = [], []
    for n, ceiling in sorted(TRAIN_GAIN_CEILING_DB.items()):
        if gain <= ceiling - CEILING_MARGIN_DB:
            preferred.append(n)
        else:
            discouraged.append(n)
    if not preferred:
        preferred = [max(TRAIN_GAIN_CEILING_DB)]
        discouraged = [n for n in TRAIN_GAIN_CEILING_DB if n != preferred[0]]
        rationale.append(
            f"gain {gain:.1f} dB exceeds every TRAIN ceiling -- prefer the "
            f"deepest cascade ({preferred[0]} stages) and treat the spec as "
            "beyond demonstrated capability")
    else:
        rationale.append(
            f"gain {gain:.1f} dB vs TRAIN ceilings "
            f"{TRAIN_GAIN_CEILING_DB}: prefer stages {preferred}, "
            f"discourage {discouraged} (SOFT prior -- a discouraged "
            "candidate survives when the pool has no alternative)")
    # bandwidth-heavy specs punish extra poles: prefer the SHALLOWEST
    # feasible cascade first
    gm_demand = 2 * math.pi * ugbw * cl_f
    preferred.sort()
    if gm_demand >= GM_DEMAND_MEDIUM:
        rationale.append(
            f"gm demand {gm_demand:.2e} (2pi*UGBW*CL): shallowest feasible "
            "cascade first -- every extra stage adds a pole at high UGBW")

    # ---- difficulty tier + budgets ---------------------------------------
    # hard when: extreme gm demand, OR gain beyond every TRAIN ceiling, OR
    # only the deepest cascade is even feasible (no fallback family exists)
    only_deepest = (preferred == [max(TRAIN_GAIN_CEILING_DB)])
    if (gm_demand >= GM_DEMAND_HARD or only_deepest or not any(
            gain <= c - CEILING_MARGIN_DB
            for c in TRAIN_GAIN_CEILING_DB.values())):
        difficulty, sizing_class, proposal_budget = "hard", "high", 6
    elif gm_demand >= GM_DEMAND_MEDIUM or gain >= 85.0 or pm >= 60.0:
        difficulty, sizing_class, proposal_budget = "medium", "normal", 5
    else:
        difficulty, sizing_class, proposal_budget = "easy", "low", 4
    rationale.append(
        f"difficulty={difficulty} (gm demand {gm_demand:.2e}, gain "
        f"{gain:.1f} dB, pm {pm:.0f} deg) -> sizing class {sizing_class}, "
        f"proposal budget {proposal_budget}")

    # ---- compensation preference ------------------------------------------
    comp = ("rc", "miller") if gm_demand >= GM_DEMAND_MEDIUM else ("miller", "rc")
    if gm_demand >= GM_DEMAND_MEDIUM:
        rationale.append(
            "high gm demand: rc/nulling compensation first (the series "
            "nulling resistor cancels the RHP zero that otherwise forces "
            "bandwidth-killing compensation caps -- measured 2.8 nF / 21 kHz "
            "pathology on the boundary class)")

    return StrategyPlan(difficulty=difficulty,
                        preferred_stages=tuple(preferred),
                        discouraged_stages=tuple(sorted(discouraged)),
                        compensation_preferences=comp,
                        proposal_budget=proposal_budget,
                        sizing_budget_class=sizing_class,
                        rationale=rationale)
