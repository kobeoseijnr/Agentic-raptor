"""Post-SAC pairwise ranker: compare two SIZED designs, select one.

Not to be confused with the Bradley-Terry ranker inside `spec_sizing.py`,
which orders KNOB VECTORS during one design's sizing loop.

Two-level decision (Part 6):

  Level 1  HARD SAFETY TIER -- non-learned, categorical only. Compares
           feasibility / operating point / stability as three states
           (good < unknown < bad). UNKNOWN IS NEVER TREATED AS GOOD.
  Level 2  LEARNED DPO RANKER -- the normal selector whenever both designs
           sit in the SAME safety tier. This is the common case, which is
           the point: the previous version consulted the model only when two
           full lexicographic tuples were exactly equal, which essentially
           never happened, so the "DPO ranker" never actually ran.

Ranker inputs are SurrogatePrediction objects, which structurally cannot
carry a SPICE result -- the ranker must not know the authoritative answer
before choosing which design gets verified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentic_raptor.ranking.types import (AuthoritativeSpiceOutcome,
                                          SurrogatePrediction,
                                          assert_no_leakage)

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
QUEUE = _ROOT / "datasets/ranker_preference_queue"
TRUSTED = QUEUE / "trusted_pairs.jsonl"
PROVISIONAL = QUEUE / "provisional_pairs.jsonl"

#: documented stability thresholds (Part 6)
STABLE_P = 0.75
UNSTABLE_P = 0.25


class RankerInputError(ValueError):
    """The ranker was handed something other than two sized designs."""


class RankerCheckpointMissing(RuntimeError):
    """POST_SAC_RANKER_CHECKPOINT_MISSING"""


@dataclass
class PostSACDesign:
    """One fully sized branch. Carries NO authoritative outcome."""
    label: str
    spec_id: str
    llm_proposal_id: str | None
    canonical_graph_hash: str
    topology_signature: str
    topology_family: str
    sizing_vector: dict = field(default_factory=dict)
    sizing_manifest_hash: str | None = None
    sac_trajectory_id: str | None = None
    action_space_version: str | None = None
    reward_version: str | None = None
    sizing_budget: int = 0
    sizing_spice_calls: int = 0
    sizing_spice_call_ids: list = field(default_factory=list)
    puct_rank: int | None = None
    puct_visits: int | None = None
    puct_visit_fraction: float | None = None
    final_netlist_hash: str | None = None

    @property
    def topology_hash(self) -> str:      # canonical identity
        return self.canonical_graph_hash


def tri_state(value) -> int:
    """0 good, 1 unknown, 2 bad. Unknown never ranks as good."""
    if value is True:
        return 0
    if value is None:
        return 1
    return 2


def _stability_tier(p: float | None) -> int:
    if p is None:
        return 1                       # unknown
    if p >= STABLE_P:
        return 0                       # stable
    if p <= UNSTABLE_P:
        return 2                       # unstable
    return 1                           # uncertain


def hard_safety_tier(pred: SurrogatePrediction) -> tuple:
    """Categorical safety state only -- no continuous quantities here.

    Continuous comparison belongs to the learned ranker; mixing it in at
    this level is what previously made the model unreachable.
    """
    return (tri_state(pred.predicted_feasible),
            tri_state(None if pred.operating_point_probability is None
                      else pred.operating_point_probability >= 0.5),
            _stability_tier(pred.stability_probability))


def compare(a: PostSACDesign, b: PostSACDesign,
            pred_a: SurrogatePrediction, pred_b: SurrogatePrediction,
            spec: dict, model=None, ranker_arm: str = "dpo_ranker",
            ranker_checkpoint_hash: str | None = None) -> dict:
    """Select exactly one of two SIZED designs."""
    for d in (a, b):
        if not d.sizing_vector:
            raise RankerInputError(
                f"design {d.label} is UNSIZED -- this ranker compares sized "
                f"designs, not topology labels")
        if not d.canonical_graph_hash:
            raise RankerInputError(f"design {d.label} has no canonical hash")
    if a.canonical_graph_hash == b.canonical_graph_hash:
        raise RankerInputError("A and B share a canonical graph hash")
    for p, d in ((pred_a, a), (pred_b, b)):
        assert_no_leakage(p)
        if p.topology_hash != d.canonical_graph_hash:
            raise RankerInputError(
                f"prediction/design mismatch on {d.label}: provenance broken")

    ta, tb = hard_safety_tier(pred_a), hard_safety_tier(pred_b)
    out = {"hard_safety_tier_A": ta, "hard_safety_tier_B": tb,
           "ranker_arm": ranker_arm,
           "ranker_checkpoint_hash": ranker_checkpoint_hash,
           "ranker_score_A": None, "ranker_score_B": None,
           "score_margin": None, "ranker_error": None}

    # ---- Level 1: hard safety gate ----------------------------------------
    if ta != tb:
        winner = a if ta < tb else b
        out.update(decision_basis="hard_safety_gate",
                   deciding_level=_first_diff(ta, tb),
                   low_confidence=False)
        return _finish(out, winner, b if winner is a else a)

    # ---- Level 2: learned ranker (the normal path) ------------------------
    if model is None:
        if ranker_arm == "dpo_ranker":
            raise RankerCheckpointMissing("POST_SAC_RANKER_CHECKPOINT_MISSING")
        sa, sb = _deterministic_score(pred_a), _deterministic_score(pred_b)
        basis = "explicit_baseline"
    else:
        try:
            sa = float(model.score(spec, a, pred_a))
            sb = float(model.score(spec, b, pred_b))
            basis = "dpo_ranker"
        except Exception as exc:      # never silently return 0.0
            log.error("post-SAC ranker failed on %s: %s",
                      a.spec_id, exc, exc_info=True)
            out["ranker_error"] = f"{type(exc).__name__}: {exc}"[:200]
            sa, sb = _deterministic_score(pred_a), _deterministic_score(pred_b)
            basis = "deterministic_tie"
    out["ranker_score_A"], out["ranker_score_B"] = sa, sb
    out["score_margin"] = abs(sa - sb)
    if sa == sb:
        winner = a if a.canonical_graph_hash <= b.canonical_graph_hash else b
        basis = "deterministic_tie"
    else:
        winner = a if sa > sb else b
    out.update(decision_basis=basis, deciding_level="learned",
               low_confidence=bool(out["score_margin"] < 0.05
                                   or basis != "dpo_ranker"))
    return _finish(out, winner, b if winner is a else a)


def _finish(out: dict, winner, loser) -> dict:
    out.update(selected_design=winner.label, backup_design=loser.label,
               selected=winner, backup=loser,
               selected_topology_hash=winner.canonical_graph_hash,
               backup_topology_hash=loser.canonical_graph_hash)
    return out


_LEVELS = ("predicted_feasible", "operating_point", "stability")


def _first_diff(ta, tb):
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x != y:
            return _LEVELS[i]
    return None


def _deterministic_score(p: SurrogatePrediction) -> float:
    """Explicit baseline ordering; higher is better. Part 7 semantics."""
    wv = p.worst_predicted_violation
    wv = 9.9 if wv is None else wv
    unc = 1.0 if p.predictive_uncertainty is None else p.predictive_uncertainty
    return -(wv) + 0.05 * p.hard_constraints_satisfied() - 0.1 * unc


# ------------------------- Part 8: trusted pair data -------------------------
def _missing_last(value, *, lower_is_better: bool = True) -> tuple:
    """Sort key that ranks UNKNOWN behind any measured value.

    `power_w if power_w is not None else 0.0` made a missing power look
    optimal -- a design nobody measured would beat a design measured at any
    positive power. Missingness is now its own leading component.
    """
    if value is None:
        return (1, 0.0)
    return (0, value if lower_is_better else -value)


def measured_preference(a: AuthoritativeSpiceOutcome,
                        b: AuthoritativeSpiceOutcome) -> tuple:
    """Preference from AUTHORITATIVE outcomes only. Smaller key wins."""
    def k(o):
        return (0 if o.exact_spec_pass else 1,
                0 if o.operating_point_valid else 1,
                0 if o.verified_stable else 1,   # explicit, not startswith
                o.normalized_distance_to_feasibility
                if o.normalized_distance_to_feasibility is not None else 9.9,
                -(o.hard_constraints_passed or 0),
                # secondary objectives: only meaningful when BOTH pass, and
                # unknown never outranks measured
                _missing_last(o.power_w),
                _missing_last(o.area_um2),
                _missing_last(o.robustness, lower_is_better=False))
    ka, kb = k(a), k(b)
    if ka == kb:
        return None, "tie"
    return ("A" if ka < kb else "B"), "measured_hierarchy"


def _provenance_ok(design: PostSACDesign, out: AuthoritativeSpiceOutcome,
                   spec_id: str, spec_hash: str | None = None) -> list:
    """Every reason this pair is not trustworthy. Empty list = trusted."""
    bad = []
    if out is None:
        return ["missing_outcome"]
    if out.source != "ngspice" or not out.authoritative:
        bad.append("not_authoritative")
    if out.mode not in ("final_verification", "backup"):
        bad.append(f"wrong_mode:{out.mode}")
    if out.topology_hash != design.canonical_graph_hash:
        bad.append("topology_hash_mismatch")
    if out.sizing_manifest_hash != design.sizing_manifest_hash:
        bad.append("sizing_manifest_mismatch")
    if not out.netlist_hash:
        bad.append("missing_netlist_hash")
    if not out.call_id:
        bad.append("missing_call_id")
    if not out.spice_converged:
        bad.append("simulation_did_not_converge")
    if design.spec_id != spec_id:
        bad.append("design_spec_id_mismatch")
    if out.spec_id != spec_id:
        bad.append("outcome_spec_id_mismatch")
    if spec_hash and out.spec_hash != spec_hash:
        bad.append("outcome_spec_hash_mismatch")
    # a sizing call reused as final verification is the exact defect the
    # architecture review flagged; catch it at the data boundary too
    if out.call_id in (design.sizing_spice_call_ids or []):
        bad.append("final_call_id_is_a_sizing_call_id")
    # protected splits must never produce ranker-training data
    try:
        from agentic_raptor.publication.eval_sets import excluded_context_ids
        if spec_id in excluded_context_ids():
            bad.append("spec_in_protected_evaluation_set")
    except Exception:
        pass
    return bad


def record_pair(spec: dict, a: PostSACDesign, b: PostSACDesign,
                out_a: AuthoritativeSpiceOutcome | None,
                out_b: AuthoritativeSpiceOutcome | None,
                pred_a: SurrogatePrediction | None = None,
                pred_b: SurrogatePrediction | None = None,
                ranker_choice: str | None = None) -> dict:
    """Write a ranker-training pair.

    TRUSTED requires full provenance on BOTH sides -- `bool(outcome)` is not
    proof of authoritative measurement, which is exactly how a predicted
    result could otherwise be laundered into the training queue.
    """
    QUEUE.mkdir(parents=True, exist_ok=True)
    spec_id = spec.get("spec_id") or spec.get("context_id") or ""
    spec_hash = spec.get("spec_hash")
    prob_a = _provenance_ok(a, out_a, spec_id, spec_hash)
    prob_b = _provenance_ok(b, out_b, spec_id, spec_hash)
    if out_a is not None and out_b is not None:
        if out_a.call_id == out_b.call_id:
            prob_a.append("shared_call_id")
        if out_a.spec_hash != out_b.spec_hash:
            prob_a.append("cross_spec_pair")
    trusted = not prob_a and not prob_b
    rec = {"spec_id": spec_id, "spec": spec,
           "design_a": asdict(a), "design_b": asdict(b),
           "prediction_a": asdict(pred_a) if pred_a else None,
           "prediction_b": asdict(pred_b) if pred_b else None,
           "outcome_a": asdict(out_a) if out_a else None,
           "outcome_b": asdict(out_b) if out_b else None,
           "ranker_choice": ranker_choice,
           "provenance_problems": {"A": prob_a, "B": prob_b}}
    if trusted:
        winner, reason = measured_preference(out_a, out_b)
        if winner is None:
            rec["status"] = "dropped_tie"
            _append(PROVISIONAL, rec)
            return rec
        rec.update(chosen=winner, rejected=("B" if winner == "A" else "A"),
                   reason=reason, status="trusted",
                   ranker_correct=(ranker_choice == winner
                                   if ranker_choice else None))
        _append(TRUSTED, rec)
    else:
        rec["status"] = "provisional_incomplete_provenance"
        _append(PROVISIONAL, rec)
    return rec


def _append(path: Path, rec: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def trusted_pairs() -> list:
    if not TRUSTED.is_file():
        return []
    return [json.loads(x) for x in
            TRUSTED.read_text(encoding="utf-8").splitlines() if x.strip()]


def ranker_accuracy() -> dict:
    pairs = [p for p in trusted_pairs() if p.get("ranker_correct") is not None]
    if not pairs:
        return {"pairs": 0, "accuracy": None}
    ok = sum(1 for p in pairs if p["ranker_correct"])
    return {"pairs": len(pairs), "correct": ok,
            "accuracy": round(ok / len(pairs), 4)}
