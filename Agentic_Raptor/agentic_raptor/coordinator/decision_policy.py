"""Transparent rule-based decision policy (interface ready to be learned later)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from agentic_raptor.coordinator.state_machine import Decision


@dataclass
class DecisionContext:
    """Everything the policy is allowed to look at."""

    specifications_parsed: bool = False
    has_candidate: bool = False
    candidate_valid: bool = False
    validation_error_count: int = 0
    sized: bool = False
    simulated: bool = False
    last_sim_failed: bool = False          # Stage 2: hard simulation failure
    last_sim_error_type: str | None = None
    pvt_done: bool = False
    all_constraints_met: bool = False
    current_reward: float | None = None
    best_reward: float | None = None
    accept_reward_threshold: float = 0.5
    worst_margin: float | None = None
    budget_snapshot: dict[str, float] = field(default_factory=dict)
    budgets_exhausted: bool = False
    edit_budget_remaining: int = 0
    sizing_budget_remaining: int = 0
    spice_budget_remaining: int = 0
    generation_budget_remaining: int = 0
    retrieval_budget_remaining: int = 0
    retrieved_count: int = 0
    topology_edits_done: int = 0
    repeated_failures: int = 0
    max_repeated_failures: int = 3
    stagnated: bool = False


class DecisionPolicy(Protocol):
    def decide(self, context: DecisionContext) -> tuple[Decision, str]: ...


class RuleBasedDecisionPolicy:
    """Ordered, human-readable rules. Every decision returns (decision, reason)."""

    def decide(self, ctx: DecisionContext) -> tuple[Decision, str]:
        # --- hard stops first ---
        if ctx.budgets_exhausted:
            return Decision.STOP_BUDGET_EXHAUSTED, "a hard budget (edits/sizing/SPICE/runtime) is exhausted"
        if ctx.repeated_failures >= ctx.max_repeated_failures:
            return Decision.STOP_BUDGET_EXHAUSTED, (
                f"{ctx.repeated_failures} repeated failures ≥ limit {ctx.max_repeated_failures}"
            )

        # --- success path ---
        if ctx.simulated and ctx.all_constraints_met and ctx.pvt_done:
            return Decision.ACCEPT_CANDIDATE, "all constraints met at nominal and PVT evaluated"
        if ctx.simulated and ctx.all_constraints_met and not ctx.pvt_done and ctx.spice_budget_remaining > 0:
            return Decision.RUN_PVT, "nominal constraints met; verifying robustness across corners"
        if (
            ctx.simulated
            and ctx.current_reward is not None
            and ctx.current_reward >= ctx.accept_reward_threshold
            and ctx.pvt_done
        ):
            return Decision.STOP_SUCCESS, (
                f"reward {ctx.current_reward:.3f} ≥ threshold {ctx.accept_reward_threshold} with PVT done"
            )

        # --- acquisition path ---
        if not ctx.has_candidate:
            if ctx.retrieved_count == 0 and ctx.retrieval_budget_remaining > 0:
                return Decision.RETRIEVE_MORE, "no candidate and nothing retrieved yet"
            if ctx.generation_budget_remaining > 0:
                return Decision.GENERATE_NEW_TOPOLOGY, "no candidate topology available"
            return Decision.STOP_BUDGET_EXHAUSTED, "no candidate and generation budget exhausted"

        # --- repair path ---
        if not ctx.candidate_valid:
            if ctx.edit_budget_remaining > 0:
                return Decision.REPAIR_OR_EDIT_TOPOLOGY, (
                    f"candidate invalid ({ctx.validation_error_count} errors); edit budget remains"
                )
            if ctx.generation_budget_remaining > 0:
                return Decision.GENERATE_NEW_TOPOLOGY, "candidate invalid and edit budget exhausted"
            return Decision.STOP_BUDGET_EXHAUSTED, "invalid candidate; edit and generation budgets exhausted"

        # --- hard simulation failure (Stage 2): editing cannot fix convergence
        # or netlist emission problems; regenerate while budget remains ---
        if ctx.last_sim_failed and ctx.generation_budget_remaining > 0:
            return Decision.GENERATE_NEW_TOPOLOGY, (
                f"simulation failed hard ({ctx.last_sim_error_type}); regenerating topology"
            )

        # --- stagnation escape ---
        if ctx.stagnated and ctx.generation_budget_remaining > 0:
            return Decision.GENERATE_NEW_TOPOLOGY, "search stagnated; trying a fresh topology"

        # --- mandatory topology-RL pass: the architecture runs MCTS refinement
        # before sizing; every fresh candidate gets at least one planned decision
        # (which may legitimately be TERMINATE) so the trajectory is never empty ---
        if (
            not ctx.sized
            and ctx.topology_edits_done == 0
            and ctx.edit_budget_remaining > 0
        ):
            return Decision.REPAIR_OR_EDIT_TOPOLOGY, (
                "initial MCTS refinement pass before sizing (topology RL precedes MB-SAC)"
            )

        # --- improve path ---
        if not ctx.sized and ctx.sizing_budget_remaining > 0:
            return Decision.CONTINUE_SIZING, "valid topology not yet sized"
        if ctx.sized and not ctx.simulated and ctx.spice_budget_remaining > 0:
            return Decision.RUN_SPICE, "sizing proposed; evaluating with SPICE"
        if (
            ctx.simulated
            and not ctx.all_constraints_met
            and ctx.worst_margin is not None
            and ctx.worst_margin > -0.25
            and ctx.sizing_budget_remaining > 0
        ):
            return Decision.CONTINUE_SIZING, (
                f"near-feasible (worst margin {ctx.worst_margin:.3f}); refining sizing"
            )
        if ctx.simulated and not ctx.all_constraints_met and ctx.edit_budget_remaining > 0:
            return Decision.REPAIR_OR_EDIT_TOPOLOGY, (
                "constraints missed by a structural gap; editing topology"
            )
        if ctx.simulated and not ctx.pvt_done and ctx.spice_budget_remaining > 0:
            return Decision.RUN_PVT, "no better local move; gathering PVT information"

        return Decision.STOP_BUDGET_EXHAUSTED, "no productive action remains within budgets"
