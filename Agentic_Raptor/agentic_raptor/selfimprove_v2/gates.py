"""Accept-or-rollback gates for every v2 checkpoint.

A retrain that makes things worse must not survive. Each gate compares a
CANDIDATE checkpoint against the currently ACCEPTED one on data the candidate
was not fitted to, and returns every failing criterion by name -- a bare
False would tell the next generation nothing about what went wrong.

The proposer carries the strictest gate because it is the only component
whose regression is invisible downstream: a proposer that collapses to one
structure still produces a pipeline run that looks entirely successful.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateResult:
    name: str
    passed: bool
    failures: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"gate": self.name, "passed": self.passed,
                "failures": self.failures, "metrics": self.metrics}


def _pct(new, old):
    if old in (None, 0) or new is None:
        return None
    return round((new - old) / abs(old), 4)


def ranker_gate(cand_acc: float | None, accepted_acc: float | None,
                n_eval: int, min_eval: int = 4) -> GateResult:
    """Held-out pairwise accuracy must not regress.

    Ties go to the INCUMBENT: with a handful of pairs, equal accuracy is
    noise, and swapping checkpoints on noise makes the lineage meaningless.
    """
    f, m = [], {"candidate_accuracy": cand_acc,
                "accepted_accuracy": accepted_acc, "eval_pairs": n_eval}
    if n_eval < min_eval:
        f.append(f"insufficient_eval_pairs:{n_eval}<{min_eval}")
    if cand_acc is None:
        f.append("candidate_not_evaluable")
    elif accepted_acc is not None and cand_acc <= accepted_acc:
        f.append(f"accuracy_not_improved:{cand_acc}<={accepted_acc}")
    return GateResult("ranker", not f, f, m)


def puct_gate(cand_loss: float | None, accepted_loss: float | None,
              n_examples: int, min_examples: int = 4) -> GateResult:
    """Held-out value loss must not regress.

    Examples with value_target None are dropped upstream; if that leaves too
    few, the gate refuses rather than accepting a model fitted to nothing.
    """
    f, m = [], {"candidate_value_loss": cand_loss,
                "accepted_value_loss": accepted_loss,
                "eval_examples": n_examples}
    if n_examples < min_examples:
        f.append(f"insufficient_eval_examples:{n_examples}<{min_examples}")
    if cand_loss is None:
        f.append("candidate_not_evaluable")
    elif accepted_loss is not None and cand_loss >= accepted_loss:
        f.append(f"value_loss_not_improved:{cand_loss}>={accepted_loss}")
    return GateResult("puct_policy_value", not f, f, m)


#: quality AND diversity, both required. A proposer can trade one for the
#: other and look better on either alone.
def proposer_gates(cand: dict, accepted: dict,
                   tol: float = 1e-9) -> GateResult:
    """Every proposer criterion, evaluated on the FROZEN held-out set.

    cand/accepted keys:
      structural_validity_rate   must not decline
      mean_distinct_graphs       must not decline
      duplicate_rate             must not increase
      success_at_k               must not decline
      measured_selected_quality  must IMPROVE  (higher is better)
      final_pass_rate            must IMPROVE
    """
    f = []
    m = {"candidate": cand, "accepted": accepted,
         "delta": {k: _pct(cand.get(k), accepted.get(k))
                   for k in set(cand) | set(accepted)}}

    def not_worse(key, higher_is_better=True):
        c, a = cand.get(key), accepted.get(key)
        if c is None:
            f.append(f"{key}:candidate_missing")
            return
        if a is None:
            return                      # no incumbent value to regress from
        if higher_is_better and c < a - tol:
            f.append(f"{key}_declined:{c}<{a}")
        if not higher_is_better and c > a + tol:
            f.append(f"{key}_increased:{c}>{a}")

    def must_improve(key):
        c, a = cand.get(key), accepted.get(key)
        if c is None:
            f.append(f"{key}:candidate_missing")
            return
        if a is not None and c <= a + tol:
            f.append(f"{key}_not_improved:{c}<={a}")

    # --- diversity / validity: must not decline ---
    not_worse("structural_validity_rate")
    not_worse("mean_distinct_graphs")
    not_worse("duplicate_rate", higher_is_better=False)
    not_worse("success_at_k")
    # --- quality: must actively improve ---
    must_improve("measured_selected_quality")
    must_improve("final_pass_rate")
    return GateResult("proposer", not f, f, m)
