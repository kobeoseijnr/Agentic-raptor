"""Deterministic coordinator: threads DesignState through the agents.

NOT an agent itself -- it owns no decisions, only dispatch order and the
soft strategy screen. Composition per arm:

    AG-FULL : planner + critic + supervisor + recovery
    AG-x    : all minus one (the agentic ablation arms)
    baseline: agents=() -- byte-identical to the pre-agentic pipeline

The screen implements the SOFT stage prior: a plan-discouraged candidate
is dropped ONLY when at least two plan-preferred candidates exist -- the
"structurally plausible exception" path stays open by construction, so
TRAIN-data ceilings can never hard-ban discovery.
"""
from __future__ import annotations

from agentic_raptor.agents import planner as _planner
from agentic_raptor.agents.state import BudgetLedger, DesignState, StrategyPlan

VALID_AGENTS = ("planner", "critic", "supervisor", "recovery")


def make_state(spec: dict, agents: tuple, *, spice_cap: int,
               llm_attempt_cap: int) -> DesignState:
    for a in agents:
        if a not in VALID_AGENTS:
            raise ValueError(f"unknown agent {a!r}; valid: {VALID_AGENTS}")
    st = DesignState(spec=spec,
                     ledger=BudgetLedger(spice_cap=spice_cap,
                                         llm_attempt_cap=llm_attempt_cap))
    if "planner" in agents:
        st.plan = _planner.plan(spec)
    else:
        # neutral plan: downstream agents still function, nothing is
        # preferred or discouraged, budgets are baseline
        st.plan = StrategyPlan(difficulty="unknown", proposal_budget=5,
                               rationale=["planner disabled: neutral plan"])
    return st


def apply_strategy_screen(candidates: list[dict], plan: StrategyPlan,
                          state: DesignState) -> list[dict]:
    """SOFT prior: drop discouraged-stage candidates only when >= 2
    preferred-stage candidates exist. Otherwise the pool passes untouched
    (the discouraged candidates ARE the plausible exception)."""
    if not plan.preferred_stages:
        return candidates
    def stages(c):
        fam = str(c.get("canonical_family") or "")
        return int(fam[0]) if fam[:1].isdigit() else 0
    preferred = [c for c in candidates if stages(c) in plan.preferred_stages]
    if len(preferred) >= 2:
        dropped = [c["canonical_family"] for c in candidates
                   if c not in preferred]
        if dropped:
            state.interventions.append(
                {"who": "coordinator", "what": "strategy_screen",
                 "dropped_families": dropped,
                 "why": f"{len(preferred)} preferred-stage candidates "
                        "available (soft prior; nothing was banned)"})
        return preferred
    return candidates
