"""AGENTIC RAPTOR agents (2026-08-16).

Four agents around the frozen pipeline core, communicating through a
shared DesignState under a deterministic coordinator, spending inside a
hard baseline-equal budget envelope:

    Design Planner          spec -> StrategyPlan (soft priors, rationale)
    Topology Critic         propose -> critique -> re-prompt loop
    Optimization Supervisor probe branches, reallocate sizing budget
    Recovery Agent          one bounded second shot from banked calls
"""
from agentic_raptor.agents.coordinator import (VALID_AGENTS,
                                               apply_strategy_screen,
                                               make_state)
from agentic_raptor.agents.critic import critique, run_critic_loop
from agentic_raptor.agents.planner import plan
from agentic_raptor.agents.recovery import decide as recovery_decide
from agentic_raptor.agents.recovery import diagnose as recovery_diagnose
from agentic_raptor.agents.recovery import execute as recovery_execute
from agentic_raptor.agents.state import (BudgetExceeded, BudgetLedger,
                                         DesignState, StrategyPlan)
from agentic_raptor.agents.supervisor import (allocate, probe_verdict,
                                              supervise)

__all__ = ["VALID_AGENTS", "apply_strategy_screen", "make_state", "critique",
           "run_critic_loop", "plan", "recovery_decide", "recovery_diagnose",
           "recovery_execute", "BudgetExceeded", "BudgetLedger", "DesignState",
           "StrategyPlan", "allocate", "probe_verdict", "supervise"]
