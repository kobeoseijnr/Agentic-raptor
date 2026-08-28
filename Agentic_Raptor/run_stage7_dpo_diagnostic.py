"""Stage 7: FULL_DPO vs NO_LEARNED_DPO frozen post-SAC selector diagnostic.

Does NOT run new SPICE. Every trusted pair in
datasets/ranker_preference_queue/trusted_pairs_post_cload_v1.jsonl already
carries a REAL ngspice measurement on BOTH sides (that is what "trusted"
means -- see agentic_raptor.ranking.post_sac.record_pair/_provenance_ok),
so the authoritative ground truth this diagnostic scores against already
exists; nothing here is fabricated or re-simulated.

Two selector arms, both behind the SAME hard safety gate
(agentic_raptor.ranking.post_sac.compare, Level 1 -- categorical
feasibility/operating-point/stability tiers, never removed):

  FULL_DPO         Level 2 = the deployed learned ranker
                   (agentic_raptor.ranking.model.PostSACRanker, DEFAULT_CKPT)
  NO_LEARNED_DPO   Level 2 = the SAME predeclared deterministic baseline
                   run_raptor_v2.py already falls back to when ranker_mode
                   != "dpo" (compare(model=None) -> _deterministic_score) --
                   this is the existing A8 arm, not an invented weak one.

Frozen split: pairs are append-only in trusted_pairs_post_cload_v1.jsonl, so
the first N pairs satisfying train_post_sac_ranker.py's usability filter, in
file order, are EXACTLY the pairs_total=45 the deployed checkpoint
(artifacts/publication_v2/post_sac_ranker/ranker.pt, trained_at
2026-08-08 13:55:45) was fitted on. Everything after that is genuinely
held-out from that checkpoint's training -- never re-fit here (Stage 7 is a
frozen diagnostic, no retraining).
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRUSTED = ROOT / "datasets/ranker_preference_queue/trusted_pairs_post_cload_v1.jsonl"
OUT_DIR = ROOT / "artifacts/publication_v3/stage7_dpo_diagnostic"
N_TRAIN_PAIRS = 45          # matches training_report.json's pairs_total


def _load_usable_pairs() -> list:
    from train_post_sac_ranker import is_informative
    lines = TRUSTED.read_text(encoding="utf-8").splitlines()
    recs = [json.loads(x) for x in lines if x.strip()]
    usable = [p for p in recs if p.get("status") == "trusted" and p.get("chosen")
             and p.get("prediction_a") and p.get("prediction_b")]
    for i, p in enumerate(usable):
        p["_pair_index"] = i
        p["_is_train_pair"] = i < N_TRAIN_PAIRS
        p["_informative"] = is_informative(p)
    return usable


def _reconstruct(pair: dict):
    from agentic_raptor.ranking.post_sac import PostSACDesign
    from agentic_raptor.ranking.types import (AuthoritativeSpiceOutcome,
                                              SurrogatePrediction)

    def _filt(cls, d):
        names = {f.name for f in fields(cls)}
        return {k: v for k, v in d.items() if k in names}

    da = PostSACDesign(**_filt(PostSACDesign, pair["design_a"]))
    db = PostSACDesign(**_filt(PostSACDesign, pair["design_b"]))
    pa = SurrogatePrediction(**_filt(SurrogatePrediction, pair["prediction_a"]))
    pb = SurrogatePrediction(**_filt(SurrogatePrediction, pair["prediction_b"]))
    oa = AuthoritativeSpiceOutcome(**_filt(AuthoritativeSpiceOutcome, pair["outcome_a"]))
    ob = AuthoritativeSpiceOutcome(**_filt(AuthoritativeSpiceOutcome, pair["outcome_b"]))
    return da, db, pa, pb, oa, ob


def _hard_pair_category(oa, ob) -> str:
    fa, fb = oa.exact_spec_pass, ob.exact_spec_pass
    if fa and fb:
        return "both_feasible"
    if fa != fb:
        return "one_feasible_one_infeasible"
    da = oa.normalized_distance_to_feasibility
    db = ob.normalized_distance_to_feasibility
    close = (da is not None and db is not None and max(da, db) < 0.2)
    return "both_infeasible_close" if close else "both_infeasible_far"


def evaluate_pair(pair: dict, ranker) -> dict:
    from agentic_raptor.ranking.post_sac import compare, measured_preference

    da, db, pa, pb, oa, ob = _reconstruct(pair)
    spec = pair["spec"]

    truth_winner, truth_reason = measured_preference(oa, ob)
    # every TRUSTED pair was written with a determinate 'chosen' -- a tie
    # (winner is None) would have been routed to PROVISIONAL, never here.
    # Recomputed independently from the raw authoritative outcomes rather
    # than trusting the stored 'chosen' field (Section 27).
    assert truth_winner is not None, f"pair {pair['_pair_index']} recomputes as a tie"
    consistent_with_stored_label = (truth_winner == pair["chosen"])

    full = compare(da, db, pa, pb, spec=spec, model=ranker,
                  ranker_arm="dpo_ranker",
                  ranker_checkpoint_hash=getattr(ranker, "checkpoint_hash", None))
    neutral = compare(da, db, pa, pb, spec=spec, model=None,
                      ranker_arm="explicit_baseline")

    return {
        "pair_index": pair["_pair_index"], "spec_id": pair["spec_id"],
        "is_train_pair": pair["_is_train_pair"], "informative": pair["_informative"],
        "hard_pair_category": _hard_pair_category(oa, ob),
        "authoritative_winner": truth_winner,
        "authoritative_reason": truth_reason,
        "consistent_with_stored_label": consistent_with_stored_label,
        "full_dpo": {
            "selected": full["selected_design"], "decision_basis": full["decision_basis"],
            "deciding_level": full["deciding_level"], "score_margin": full["score_margin"],
            "correct": full["selected_design"] == truth_winner},
        "no_learned_dpo": {
            "selected": neutral["selected_design"], "decision_basis": neutral["decision_basis"],
            "deciding_level": neutral["deciding_level"], "score_margin": neutral["score_margin"],
            "correct": neutral["selected_design"] == truth_winner},
    }


def _margin_bucket(m):
    if m is None:
        return "n/a"
    for edge, label in ((0.05, "<0.05"), (0.1, "0.05-0.1"), (0.2, "0.1-0.2"), (0.5, "0.2-0.5")):
        if m < edge:
            return label
    return ">=0.5"


def summarize(results: list) -> dict:
    def _acc(rs, key):
        rs = [r for r in rs if r]
        if not rs:
            return None
        return round(sum(1 for r in rs if r[key]["correct"]) / len(rs), 4)

    all_r = results
    train_r = [r for r in results if r["is_train_pair"]]
    held_r = [r for r in results if not r["is_train_pair"]]
    info_r = [r for r in results if r["informative"]]
    info_held_r = [r for r in held_r if r["informative"]]

    dpo_decided = [r for r in results if r["full_dpo"]["decision_basis"] == "dpo_ranker"]
    hard_gate_only = [r for r in results if r["full_dpo"]["decision_basis"] == "hard_safety_gate"]
    dpo_decided_held = [r for r in held_r if r["full_dpo"]["decision_basis"] == "dpo_ranker"]

    full_feasible_sel = sum(1 for r in results
                            if r["full_dpo"].get("selected_is_feasible"))
    neutral_feasible_sel = sum(1 for r in results
                               if r["no_learned_dpo"].get("selected_is_feasible"))

    def _wlt(rs):
        w = l = t = 0
        for r in rs:
            fc, nc = r["full_dpo"]["correct"], r["no_learned_dpo"]["correct"]
            if fc and not nc:
                w += 1
            elif nc and not fc:
                l += 1
            else:
                t += 1
        return w, l, t
    wins, losses, ties = _wlt(all_r)
    wins_train, losses_train, ties_train = _wlt(train_r)
    wins_held, losses_held, ties_held = _wlt(held_r)

    buckets = {}
    for r in dpo_decided:
        b = _margin_bucket(r["full_dpo"]["score_margin"])
        buckets.setdefault(b, []).append(r["full_dpo"]["correct"])
    calibration = {b: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
                   for b, v in sorted(buckets.items())}

    by_category = {}
    for cat in ("both_feasible", "one_feasible_one_infeasible",
               "both_infeasible_close", "both_infeasible_far"):
        rs = [r for r in results if r["hard_pair_category"] == cat]
        by_category[cat] = {
            "n": len(rs),
            "full_dpo_accuracy": _acc(rs, "full_dpo"),
            "no_learned_dpo_accuracy": _acc(rs, "no_learned_dpo"),
            "n_hard_gate_decided": sum(1 for r in rs
                                       if r["full_dpo"]["decision_basis"] == "hard_safety_gate"),
            "n_dpo_ranker_decided": sum(1 for r in rs
                                        if r["full_dpo"]["decision_basis"] == "dpo_ranker")}

    return {
        "n_pairs_total": len(all_r), "n_train_pairs": len(train_r),
        "n_held_out_pairs": len(held_r), "n_informative_pairs": len(info_r),
        "n_informative_held_out_pairs": len(info_held_r),
        "n_stored_label_inconsistencies": sum(
            1 for r in results if not r["consistent_with_stored_label"]),
        "full_dpo_accuracy_all": _acc(all_r, "full_dpo"),
        "no_learned_dpo_accuracy_all": _acc(all_r, "no_learned_dpo"),
        "full_dpo_accuracy_informative": _acc(info_r, "full_dpo"),
        "no_learned_dpo_accuracy_informative": _acc(info_r, "no_learned_dpo"),
        "full_dpo_accuracy_train_pairs": _acc(train_r, "full_dpo"),
        "full_dpo_accuracy_held_out_pairs": _acc(held_r, "full_dpo"),
        "full_dpo_accuracy_held_out_informative": _acc(info_held_r, "full_dpo"),
        "no_learned_dpo_accuracy_train_pairs": _acc(train_r, "no_learned_dpo"),
        "no_learned_dpo_accuracy_held_out_pairs": _acc(held_r, "no_learned_dpo"),
        "dpo_conditional_accuracy_train_pairs": _acc(
            [r for r in train_r if r["full_dpo"]["decision_basis"] == "dpo_ranker"], "full_dpo"),
        "n_dpo_decided_train_pairs": sum(
            1 for r in train_r if r["full_dpo"]["decision_basis"] == "dpo_ranker"),
        "n_dpo_decided_held_out_pairs": sum(
            1 for r in held_r if r["full_dpo"]["decision_basis"] == "dpo_ranker"),
        "train_dev_accuracy_gap":
            (round(_acc(train_r, "full_dpo") - _acc(held_r, "full_dpo"), 4)
             if _acc(train_r, "full_dpo") is not None and _acc(held_r, "full_dpo") is not None
             else None),
        "hard_gate_only_decision_share": round(len(hard_gate_only) / len(all_r), 4),
        "dpo_decision_share": round(len(dpo_decided) / len(all_r), 4),
        "dpo_decision_share_held_out": round(len(dpo_decided_held) / max(1, len(held_r)), 4),
        "dpo_conditional_accuracy": _acc(dpo_decided, "full_dpo"),
        "dpo_conditional_accuracy_held_out": _acc(dpo_decided_held, "full_dpo"),
        "feasible_selection_rate_full_dpo": round(full_feasible_sel / len(all_r), 4),
        "feasible_selection_rate_no_learned_dpo": round(neutral_feasible_sel / len(all_r), 4),
        "full_dpo_vs_no_learned_dpo_wins": wins,
        "full_dpo_vs_no_learned_dpo_losses": losses,
        "full_dpo_vs_no_learned_dpo_ties": ties,
        "full_dpo_vs_no_learned_dpo_wins_train": wins_train,
        "full_dpo_vs_no_learned_dpo_losses_train": losses_train,
        "full_dpo_vs_no_learned_dpo_ties_train": ties_train,
        "full_dpo_vs_no_learned_dpo_wins_held_out": wins_held,
        "full_dpo_vs_no_learned_dpo_losses_held_out": losses_held,
        "full_dpo_vs_no_learned_dpo_ties_held_out": ties_held,
        "calibration_by_score_margin_bucket": calibration,
        "by_hard_pair_category": by_category,
    }


def _annotate_feasibility(pair: dict, result: dict):
    da, db, pa, pb, oa, ob = _reconstruct(pair)
    outcomes = {"A": oa, "B": ob}
    result["full_dpo"]["selected_is_feasible"] = bool(
        outcomes[result["full_dpo"]["selected"]].exact_spec_pass)
    result["no_learned_dpo"]["selected_is_feasible"] = bool(
        outcomes[result["no_learned_dpo"]["selected"]].exact_spec_pass)


def main():
    from agentic_raptor.ranking.model import DEFAULT_CKPT, PostSACRanker

    ranker = PostSACRanker.load(DEFAULT_CKPT)
    if ranker is None:
        raise SystemExit(f"active DPO checkpoint missing/unloadable: {DEFAULT_CKPT}")
    print(f"active checkpoint: {DEFAULT_CKPT}", flush=True)
    print(f"checkpoint sha256: {ranker.checkpoint_hash}", flush=True)

    pairs = _load_usable_pairs()
    print(f"usable trusted pairs: {len(pairs)} "
         f"({N_TRAIN_PAIRS} treated as checkpoint-train, "
         f"{len(pairs) - N_TRAIN_PAIRS} held-out)", flush=True)

    frozen_ids = [{"pair_index": p["_pair_index"], "spec_id": p["spec_id"],
                  "outcome_a_call_id": p["outcome_a"]["call_id"],
                  "outcome_b_call_id": p["outcome_b"]["call_id"]} for p in pairs]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_path = OUT_DIR / "FROZEN_DPO_DIAGNOSTIC_PAIR_SET.json"
    frozen_path.write_text(json.dumps(frozen_ids, indent=1), encoding="utf-8")
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    print(f"frozen pair set -> {frozen_path} (sha256={frozen_sha})", flush=True)

    results = []
    for p in pairs:
        r = evaluate_pair(p, ranker)
        _annotate_feasibility(p, r)
        results.append(r)

    summary = summarize(results)
    print("\n=== summary ===", flush=True)
    print(json.dumps(summary, indent=1), flush=True)

    report = {"frozen_pair_set_path": str(frozen_path), "frozen_pair_set_sha256": frozen_sha,
             "n_train_pairs_assumed": N_TRAIN_PAIRS,
             "active_checkpoint_path": str(DEFAULT_CKPT),
             "active_checkpoint_sha256": ranker.checkpoint_hash,
             "results": results, "summary": summary}
    out_path = OUT_DIR / "STAGE7_DPO_DIAGNOSTIC_REPORT.json"
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nStage 7 DPO diagnostic report written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
