"""Shared agent state: StrategyPlan, BudgetLedger, DesignState.

AGENTIC RAPTOR (2026-08-16). The four agents (Design Planner, Topology
Critic, Optimization Supervisor, Recovery) communicate ONLY through the
DesignState a deterministic coordinator threads between them -- without
shared state this would be another four-stage assembly line with each
stage renamed "agent".

FAIRNESS INVARIANT (publication-critical): BudgetLedger enforces that the
agentic arms consume AT MOST the baseline's SPICE and LLM budgets. Agents
reallocate; they never spend more. A reviewer must be unable to attribute
an agentic win to extra compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyPlan:
    """Design Planner output: a structured, inspectable plan -- explicitly
    NOT a collection of hidden if-statements inside downstream stages.

    Stage-count guidance is a SOFT prior: `discouraged_stages` lowers
    priority but never hard-prohibits -- TRAIN data ceilings are empirical
    observations, not physical law, and a hard ban would bake current
    dataset biases into the system and prevent discovery. A discouraged
    candidate is dropped only when enough preferred candidates exist
    (see coordinator.apply_strategy_screen)."""
    difficulty: str                       # "easy" | "medium" | "hard"
    preferred_stages: tuple = ()
    discouraged_stages: tuple = ()
    compensation_preferences: tuple = ()  # e.g. ("rc", "miller")
    proposal_budget: int = 5              # distinct candidates to seek
    sizing_budget_class: str = "normal"   # "low" | "normal" | "high"
    rationale: list = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class BudgetLedger:
    """Hard budget envelope. Every agent spend flows through here."""
    spice_cap: int                        # baseline total (e.g. 2 x 16)
    llm_attempt_cap: int                  # baseline proposal attempts (20)
    spice_spent: int = 0
    llm_attempts_spent: int = 0
    banked: int = 0                       # unspent calls available to Recovery
    entries: list = field(default_factory=list)

    def spend_spice(self, n: int, who: str, why: str) -> None:
        if self.spice_spent + n > self.spice_cap:
            raise BudgetExceeded(
                f"{who}: {why}: would spend {n} SPICE calls with only "
                f"{self.spice_cap - self.spice_spent} of {self.spice_cap} left")
        self.spice_spent += n
        self.entries.append({"who": who, "why": why, "spice": n})

    def spend_llm(self, n: int, who: str, why: str) -> None:
        if self.llm_attempts_spent + n > self.llm_attempt_cap:
            raise BudgetExceeded(
                f"{who}: {why}: would spend {n} LLM attempts with only "
                f"{self.llm_attempt_cap - self.llm_attempts_spent} left")
        self.llm_attempts_spent += n
        self.entries.append({"who": who, "why": why, "llm_attempts": n})

    def bank(self, n: int, who: str, why: str) -> None:
        """Move unspent-but-allocated calls into the bank. These calls were
        ALREADY charged to the cap when allocated (spend_spice) but never
        physically used -- banking records that they are available to be
        used later WITHOUT a second charge. Banking never creates budget."""
        self.banked += n
        self.entries.append({"who": who, "why": why, "banked": n})

    def spend_from_bank(self, n: int, who: str, why: str) -> int:
        """Use up to n banked calls. Returns the amount actually drawn.
        NOT charged against the cap again (it was charged at allocation);
        the physical call count still cannot exceed the cap because the
        bank only ever holds calls that were allocated and unused."""
        take = min(n, self.banked)
        self.banked -= take
        self.entries.append({"who": who, "why": why, "from_bank": take})
        return take

    def spice_remaining(self) -> int:
        return self.spice_cap - self.spice_spent

    def to_dict(self) -> dict:
        return {"spice_cap": self.spice_cap, "spice_spent": self.spice_spent,
               "llm_attempt_cap": self.llm_attempt_cap,
               "llm_attempts_spent": self.llm_attempts_spent,
               "banked": self.banked, "entries": self.entries}


class BudgetExceeded(RuntimeError):
    """An agent tried to spend beyond the baseline envelope."""


@dataclass
class DesignState:
    """The blackboard every agent reads and writes."""
    spec: dict
    plan: StrategyPlan | None = None
    ledger: BudgetLedger | None = None
    candidates: list = field(default_factory=list)
    candidate_provenance: dict = field(default_factory=dict)
    critic_feedback: list = field(default_factory=list)
    supervisor_log: list = field(default_factory=list)
    branch_probe: dict = field(default_factory=dict)     # label -> probe summary
    branch_allocation: dict = field(default_factory=dict)
    interventions: list = field(default_factory=list)
    recovery_already_used: bool = False
    recovery_log: dict = field(default_factory=dict)

    def trace(self) -> dict:
        """Everything an auditor needs, JSON-ready."""
        return {"plan": self.plan.to_dict() if self.plan else None,
               "ledger": self.ledger.to_dict() if self.ledger else None,
               "critic_feedback": self.critic_feedback,
               "supervisor_log": self.supervisor_log,
               "branch_probe": self.branch_probe,
               "branch_allocation": self.branch_allocation,
               "interventions": self.interventions,
               "recovery_already_used": self.recovery_already_used,
               "recovery_log": self.recovery_log}
