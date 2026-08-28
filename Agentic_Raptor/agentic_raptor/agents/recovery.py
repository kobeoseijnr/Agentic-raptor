"""RECOVERY AGENT: one bounded second shot after a failed verification.

The measured gap: the pipeline was one-shot -- a near-miss (authoritative
distance 0.0044 was recorded in the campaigns) reported failure and threw
its own diagnosis away.

Contract (hard, tested):
  - runs ONLY if the final verification failed;
  - runs AT MOST once per pipeline run (state.recovery_already_used);
  - spends ONLY banked calls (savings the Supervisor/early-stop created
    inside the same envelope) -- the ledger still enforces the global cap;
  - one action from a fixed menu, then an honest final answer either way.

Menu:
  RESIZE_BACKUP  -- the failure diagnosis says the backup branch's
                    topology is at least as feasible as the selected one:
                    re-size the backup with the banked budget.
  NONE           -- banked budget too small or no action is indicated;
                    report the original failure untouched.
"""
from __future__ import annotations

from agentic_raptor.agents.state import BudgetLedger, DesignState

MIN_BANKED_TO_ACT = 4


def diagnose(measured: dict, spec: dict) -> dict:
    """Which constraint failed, by how much -- from the authoritative
    measurement only (no surrogate opinions at this stage)."""
    gaps = {}
    if measured.get("gain_db") is not None and spec.get("gain_target_db"):
        gaps["gain_db"] = round(measured["gain_db"] - spec["gain_target_db"], 2)
    if measured.get("pm_deg") is not None and spec.get("phase_margin_target_deg"):
        gaps["pm_deg"] = round(measured["pm_deg"]
                               - spec["phase_margin_target_deg"], 2)
    if measured.get("ugbw_hz") is not None and spec.get("ugbw_target_hz"):
        gaps["ugbw_ratio"] = round(measured["ugbw_hz"]
                                   / max(spec["ugbw_target_hz"], 1.0), 4)
    failing = [k for k, v in gaps.items()
               if (k == "ugbw_ratio" and v < 1.0) or (k != "ugbw_ratio" and v < 0)]
    return {"gaps": gaps, "failing": failing}


def decide(state: DesignState, ledger: BudgetLedger,
           selected_failed_distance: float | None) -> dict:
    if state.recovery_already_used:
        return {"action": "NONE", "why": "recovery already used this run"}
    if ledger.banked < MIN_BANKED_TO_ACT:
        return {"action": "NONE",
               "why": f"banked {ledger.banked} < {MIN_BANKED_TO_ACT} calls"}
    return {"action": "RESIZE_BACKUP", "budget": ledger.banked,
           "why": ("selected branch failed authoritative verification "
                   f"(distance {selected_failed_distance}); banked calls "
                   "fund one bounded re-size of the backup topology")}


def execute(state: DesignState, ledger: BudgetLedger, decision: dict,
            resize_fn) -> dict:
    """`resize_fn(budget)` re-sizes the backup branch and returns
    {"distance": float|None, "pass": bool, "spice_calls": int, ...}.
    The ledger is charged for what was actually spent."""
    if decision["action"] != "RESIZE_BACKUP":
        state.recovery_log = {"decision": decision, "executed": False}
        return state.recovery_log
    state.recovery_already_used = True
    budget = min(decision["budget"], ledger.banked)
    result = resize_fn(budget)
    spent = int(result.get("spice_calls") or 0)
    # banked calls were charged at allocation -- draw, don't double-charge
    ledger.spend_from_bank(spent, "recovery_agent", "resize backup from bank")
    state.recovery_log = {"decision": decision, "executed": True,
                          "spent": spent, "result": {k: result.get(k) for k in
                                                     ("distance", "pass")}}
    return state.recovery_log
