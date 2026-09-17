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
    # DEEP-COMP COVERAGE (2026-08-30, HELDOUT29R failure analysis): on hard
    # specs the comp requirement was satisfiable by the SHALLOWEST preferred
    # depth (a 2s_rc at an 89.5 dB target that sits at the 2-stage gain
    # ceiling), so the pool never contained the rc-compensated DEEPER
    # cascade (3s_rc) that those specs are actually passed with -- the
    # selection layer then had only 3s_miller to escalate to, which reaches
    # the gain but cannot stabilize pm/ugbw. Rule (plan physics, zero LLM
    # cost): a hard-difficulty pool must also contain the preferred
    # compensation at a depth ABOVE the minimum preferred stage count.
    # LOFO CONTROL (2026-09-06): the deep-comp rule names the preferred
    # compensation at a depth above the minimum -- on 2-stage-ceiling specs
    # that is exactly the held-out 3s_rc cascade. A leave-one-family-out
    # generalization experiment must NOT let a scripted rule re-inject the
    # family it removed, so AGR_LOFO_DISABLE_DEEPCOMP=1 disables this rule.
    # Default (unset) preserves the frozen behaviour byte-for-byte.
    import os as _os
    deep_comp_hit = True
    if (_os.environ.get("AGR_LOFO_DISABLE_DEEPCOMP") != "1"
            and plan.difficulty == "hard" and plan.compensation_preferences
            and len(plan.preferred_stages) > 1):
        dmin = min(plan.preferred_stages)
        deep_comp_hit = any(
            _comp_of(c) == plan.compensation_preferences[0]
            and _stages_of(c) > dmin
            and _stages_of(c) in plan.preferred_stages
            for c in candidates)
        if not deep_comp_hit:
            feedback.append(
                f"need a {plan.compensation_preferences[0]}-compensated "
                f"proposal at >= {dmin + 1} stages (gain target sits at the "
                f"{dmin}-stage ceiling; the deeper cascade needs the "
                f"nulling branch to hold phase margin)")
    return {"satisfied": not feedback, "n_preferred_stage": n_preferred,
           "preferred_comp_present": comp_hit,
           "deep_comp_present": deep_comp_hit, "feedback": feedback}


def run_critic_loop(propose_fn, prompt: str, plan: StrategyPlan,
                    ledger: BudgetLedger, state: DesignState,
                    max_rounds: int = 3,
                    pool_floor: int | None = None) -> dict:
    """Drive propose_fn (the UNCHANGED baseline proposer) in rounds.

    Round 1 spends most of the attempt budget on the plain prompt (the
    baseline behavior); later rounds spend the remainder with the critic's
    feedback appended to the prompt. Total attempts NEVER exceed the
    ledger's cap = the baseline's own cap.

    pool_floor (DATE step 3, 2026-09-07): the loop historically stopped as
    soon as the critic was satisfied AND the pool held min(proposal_budget,
    2) candidates -- so the agentic arm never reached target_k (HELDOUT29 R2:
    attempts=2 on every run, target_k=4). None keeps that behaviour
    byte-identical; an int keeps proposing until the pool holds that many
    distinct candidates (or the attempt cap is spent). bandit_top2 still
    sizes exactly two, so a larger pool costs LLM attempts, not SPICE.
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
        floor = (min(plan.proposal_budget, 2) if pool_floor is None
                 else pool_floor)
        if verdict["satisfied"] and len(pool) >= floor:
            break
    return {"candidates": pool, "rounds": len(state.critic_feedback),
           "final_verdict": state.critic_feedback[-1] if state.critic_feedback
           else None, "last_raw": all_out}
