"""TOPOLOGY CRITIC AGENT: iterative propose -> critique -> re-prompt.

The measured gap it closes: proposal generation was completely spec-blind
-- the SAME candidate pool appeared for every specification, which is why
the selection layer never had a meaningful choice. The critic evaluates
each round of proposals against the StrategyPlan and feeds TARGETED
feedback into the next round's prompt until the pool satisfies the plan
or the ATTEMPT BUDGET (identical to the baseline's cap -- the fairness
invariant) is exhausted.

The critic is rule-based: its rules are the plan's own physics, so its
verdicts are deterministic, auditable, and cost zero extra LLM calls.
"""
from __future__ import annotations

from agentic_raptor.agents.state import BudgetLedger, DesignState, StrategyPlan


def _stages_of(candidate: dict) -> int:
    fam = str(candidate.get("canonical_family") or "")
    return int(fam[0]) if fam[:1].isdigit() else 0


def _comp_of(candidate: dict) -> str:
    fam = str(candidate.get("canonical_family") or "")
    return fam.split("_", 1)[1] if "_" in fam else "none"


def critique(candidates: list[dict], plan: StrategyPlan) -> dict:
    """Verdict on a candidate pool vs the plan. Never rejects a pool
    outright -- reports what is missing so the next round can target it."""
    n_preferred = sum(1 for c in candidates
                      if _stages_of(c) in plan.preferred_stages)
    comp_hit = any(_comp_of(c) == plan.compensation_preferences[0]
                   and _stages_of(c) in plan.preferred_stages
                   for c in candidates) if plan.compensation_preferences else True
    feedback = []
    if n_preferred < 2:
        feedback.append(
            f"need >= 2 proposals with {list(plan.preferred_stages)} gain "
            f"stages (gain target demands them); have {n_preferred}")
    if not comp_hit and plan.compensation_preferences:
        feedback.append(
            f"need a {plan.compensation_preferences[0]}-compensated proposal "
            f"at {list(plan.preferred_stages)} stages (bandwidth demand)")
    return {"satisfied": not feedback, "n_preferred_stage": n_preferred,
           "preferred_comp_present": comp_hit, "feedback": feedback}


def run_critic_loop(propose_fn, prompt: str, plan: StrategyPlan,
                    ledger: BudgetLedger, state: DesignState,
                    max_rounds: int = 3) -> dict:
    """Drive propose_fn (the UNCHANGED baseline proposer) in rounds.

    Round 1 spends most of the attempt budget on the plain prompt (the
    baseline behavior); later rounds spend the remainder with the critic's
    feedback appended to the prompt. Total attempts NEVER exceed the
    ledger's cap = the baseline's own cap.
    """
    all_out = None
    seen_hashes: set[str] = set()
    pool: list[dict] = []
    for rnd in range(max_rounds):
        remaining = ledger.llm_attempt_cap - ledger.llm_attempts_spent
        if remaining <= 0:
            break
        attempts = max(4, remaining // 2) if rnd < max_rounds - 1 else remaining
        attempts = min(attempts, remaining)
        fb = ""
        if state.critic_feedback:
            fb = ("\n### CRITIC FEEDBACK (address ALL points)\n- "
                  + "\n- ".join(state.critic_feedback[-1]["feedback"]))
        out = propose_fn(prompt + fb, plan.proposal_budget, attempts)
        ledger.spend_llm(out.get("attempts", attempts), "topology_critic",
                         f"round {rnd}")
        for c in out.get("candidates") or []:
            h = c.get("canonical_graph_hash")
            if h and h not in seen_hashes:
                seen_hashes.add(h)
                pool.append(c)
                state.candidate_provenance[h] = {"round": rnd,
                                                 "with_feedback": bool(fb)}
        all_out = out
        verdict = critique(pool, plan)
        state.critic_feedback.append({"round": rnd, **verdict})
        if verdict["satisfied"] and len(pool) >= min(plan.proposal_budget, 2):
            break
    return {"candidates": pool, "rounds": len(state.critic_feedback),
           "final_verdict": state.critic_feedback[-1] if state.critic_feedback
           else None, "last_raw": all_out}
