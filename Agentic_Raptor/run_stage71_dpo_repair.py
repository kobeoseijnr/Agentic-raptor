"""Stage 7.1: repair and retrain the learned post-SAC DPO/ranker (Level 2).

Does NOT touch hard_safety_tier() (Level 1, frozen exactly as-is) and does
NOT touch MB-SAC (frozen CORRECT + COMPETITIVE per Stage 6.1). Does NOT run
new SPICE.

Primary hypothesis being tested: the deployed ranker's training problem was
poorly aligned with its live responsibility. Stage 7 found the hard gate
alone decides 72.5% of trusted pairs; the ranker only ever has genuine
decision authority on the remaining ~27.5% ("ranker-authority" pairs, where
predicted hard-safety tiers are EQUAL on both sides). Training treated every
pair equally, so the 72.5% the ranker will never actually decide dominated
its objective. This repair rebuilds the pair table with that distinction
explicit, splits by spec (not by row) into DPO_TRAIN/DPO_DEV, fits feature
normalization on TRAIN only, weights ranker-authority pairs as primary and
safety-decided pairs as reduced-weight auxiliary regularization, retrains
from a fresh initialization (not continued from the stale 45-pair
checkpoint), and selects the best of a small (R1-R4) hyperparameter sweep by
DEV ranker-authority conditional accuracy -- never by overall accuracy,
which the hard gate would dominate.

Mining existing (non-live-queue) POST_CLOAD_FIX_V1 artifacts for additional
both-feasible pairs was attempted (Section 11) and is reported honestly:
Stage 5 closeout's paired-validation report stores only aggregate
statistics per generation, no per-episode authoritative outcome+design
pairs. The Stage 6 diagnostic and Stage 6.1 recheck reports DO carry real
authoritative outcomes for two different topologies under a common spec
(exactly the pair shape needed) -- but the sizing knob VECTOR actually used
for each winning design was never persisted to those reports (only summary
metrics were saved), so a genuine, non-fabricated SurrogatePrediction
cannot be reconstructed for them without either new SPICE or reverse-
engineering knobs from a sized device_graph.json (fragile: apply_knobs'
per-role multiplier math is not cleanly invertible once physical W/L/R
clamps have been hit). Rather than risk a corrupted or misleading feature
row, zero pairs are mined from these sources into DPO_TRAIN/DPO_DEV; this
task proceeds on a rebuilt version of the existing 91 trusted pairs only.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
import time
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRUSTED = ROOT / "datasets/ranker_preference_queue/trusted_pairs_post_cload_v1.jsonl"
OUT_DIR = ROOT / "artifacts/publication_v3/stage7_1_dpo_repair"
OLD_CKPT = ROOT / "artifacts/publication_v2/post_sac_ranker/ranker.pt"

SAFETY_DECIDED_WEIGHT = 0.15   # Section 5: reduced, non-zero auxiliary weight
DEV_FRACTION = 0.22            # within the 20-25% requested range
BOTH_INFEASIBLE_CLOSE_THRESHOLD = 0.2

SWEEP = [
    {"name": "R1", "lr": 1e-3, "weight_decay": 0.1},
    {"name": "R2", "lr": 3e-4, "weight_decay": 0.1},
    {"name": "R3", "lr": 1e-4, "weight_decay": 0.1},
    {"name": "R4", "lr": 3e-4, "weight_decay": 0.01},
]
MAX_EPOCHS = 400
EVAL_EVERY = 10
SEED = 0


# ---------------------------------------------------------------------------
# Section 3: rebuild the trusted pair table
# ---------------------------------------------------------------------------
def _load_usable_pairs() -> list:
    from train_post_sac_ranker import is_informative
    lines = TRUSTED.read_text(encoding="utf-8").splitlines()
    recs = [json.loads(x) for x in lines if x.strip()]
    usable = [p for p in recs if p.get("status") == "trusted" and p.get("chosen")
             and p.get("prediction_a") and p.get("prediction_b")]
    for i, p in enumerate(usable):
        p["_pair_index"] = i
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


def _feasibility_category(oa, ob) -> str:
    fa, fb = oa.exact_spec_pass, ob.exact_spec_pass
    if fa and fb:
        return "both_feasible"
    if fa != fb:
        return "one_feasible_one_infeasible"
    da = oa.normalized_distance_to_feasibility
    db = ob.normalized_distance_to_feasibility
    close = (da is not None and db is not None
            and max(da, db) < BOTH_INFEASIBLE_CLOSE_THRESHOLD)
    return "both_infeasible_close" if close else "both_infeasible_far"


def _mine_additional_pairs_report() -> dict:
    """Section 11 -- documents what was checked and why nothing was added.
    See module docstring for the full reasoning."""
    stage5 = ROOT / "artifacts/publication_v3/stage5_closeout/PART_A_CLOSEOUT_REPORT.json"
    stage6 = ROOT / "artifacts/publication_v3/stage6_mbsac_diagnostic/STAGE6_DIAGNOSTIC_REPORT.json"
    stage61 = ROOT / "artifacts/publication_v3/stage6_1_repair_recheck/STAGE6_1_RECHECK_REPORT.json"
    return {
        "sources_checked": [
            {"path": str(stage5), "exists": stage5.is_file(),
            "finding": "paired_validation stores only PER-GENERATION AGGREGATE "
                       "statistics (mean/rate); no per-episode authoritative "
                       "outcome+design records to reconstruct a pair from"},
            {"path": str(stage6), "exists": stage6.is_file(),
            "finding": "job_results carry real AuthoritativeSpiceOutcome-shaped "
                       "verify() results for two DIFFERENT topologies under a "
                       "common spec (the right pair shape), but the winning "
                       "sizing KNOB VECTOR was never persisted to this report "
                       "-- only summary metrics -- so predict_post_sac() cannot "
                       "be run on them without either a new SPICE call or "
                       "reverse-engineering knobs from a sized device_graph.json "
                       "(not cleanly invertible once physical W/L/R clamps are "
                       "hit); also spec26 (t_validation_topology_v2_0003) is in "
                       "the protected evaluation set and would be excluded "
                       "regardless"},
            {"path": str(stage61), "exists": stage61.is_file(),
            "finding": "same limitation as stage6 (knob vector not persisted)"},
        ],
        "n_additional_pairs_mined": 0,
        "reason": "no source both (a) has authoritative outcomes for 2 distinct "
                  "topologies under one spec AND (b) persists the exact sizing "
                  "vector needed for a genuine, non-fabricated SurrogatePrediction, "
                  "without a new SPICE call. Proceeding on the rebuilt existing "
                  "91-pair set only.",
    }


def build_pair_table(pairs: list) -> list:
    from agentic_raptor.ranking.post_sac import hard_safety_tier, measured_preference

    table = []
    seen_ids = {}
    n_dupes = 0
    for p in pairs:
        da, db, pa, pb, oa, ob = _reconstruct(p)
        spec_hash = p["spec"].get("spec_hash") or p["spec_id"]
        ha, hb = da.canonical_graph_hash, db.canonical_graph_hash
        canonical_id = (spec_hash, min(ha, hb), max(ha, hb))
        if canonical_id in seen_ids:
            n_dupes += 1
            continue
        seen_ids[canonical_id] = p["_pair_index"]

        ta, tb = hard_safety_tier(pa), hard_safety_tier(pb)
        ranker_authority = (ta == tb)
        winner, reason = measured_preference(oa, ob)
        assert winner is not None, f"pair {p['_pair_index']} recomputes as a tie"

        table.append({
            "pair_index": p["_pair_index"], "canonical_id": canonical_id,
            "spec_id": p["spec_id"], "spec_hash": spec_hash,
            "topology_hash_a": ha, "topology_hash_b": hb,
            "hard_safety_tier_a": list(ta), "hard_safety_tier_b": list(tb),
            "ranker_authority": ranker_authority,
            "feasibility_category": _feasibility_category(oa, ob),
            "informative": p["_informative"],
            "authoritative_winner": winner, "authoritative_reason": reason,
            "_raw": p,
        })
    return table, n_dupes


# ---------------------------------------------------------------------------
# Section 13: split by spec hash, not by row
# ---------------------------------------------------------------------------
def split_train_dev(table: list, dev_fraction: float = DEV_FRACTION) -> tuple:
    specs = sorted({row["spec_hash"] for row in table})
    dev_specs = set()
    for s in specs:
        frac = (int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 10_000) / 10_000
        if frac < dev_fraction:
            dev_specs.add(s)
    train = [r for r in table if r["spec_hash"] not in dev_specs]
    dev = [r for r in table if r["spec_hash"] in dev_specs]
    return train, dev, dev_specs


def coverage_stats(rows: list) -> dict:
    def _n(pred):
        return sum(1 for r in rows if pred(r))
    return {
        "n_specs": len(set(r["spec_hash"] for r in rows)),
        "n_pairs": len(rows),
        "n_ranker_authority": _n(lambda r: r["ranker_authority"]),
        "n_safety_decided": _n(lambda r: not r["ranker_authority"]),
        "n_informative": _n(lambda r: r["informative"]),
        "n_both_feasible": _n(lambda r: r["feasibility_category"] == "both_feasible"),
        "n_one_feasible_one_infeasible": _n(
            lambda r: r["feasibility_category"] == "one_feasible_one_infeasible"),
        "n_both_infeasible_close": _n(
            lambda r: r["feasibility_category"] == "both_infeasible_close"),
        "n_both_infeasible_far": _n(
            lambda r: r["feasibility_category"] == "both_infeasible_far"),
    }


# ---------------------------------------------------------------------------
# Sections 7/8/9: pairwise training with ranker-authority weighting
# ---------------------------------------------------------------------------
def _xw_xl_weights(rows: list):
    from agentic_raptor.ranking.model import features
    xw, xl, w = [], [], []
    for r in rows:
        p = r["_raw"]
        da, db, pa, pb, oa, ob = _reconstruct(p)
        fa, fb = features(p["spec"], pa), features(p["spec"], pb)
        if r["authoritative_winner"] == "A":
            xw.append(fa); xl.append(fb)
        else:
            xw.append(fb); xl.append(fa)
        w.append(1.0 if r["ranker_authority"] else SAFETY_DECIDED_WEIGHT)
    return xw, xl, w


def train_one_config(train_rows: list, dev_rows: list, lr: float,
                     weight_decay: float, seed: int = SEED) -> dict:
    import torch

    from agentic_raptor.ranking.model import PostSACRanker, fit_normalization

    xw_raw, xl_raw, w = _xw_xl_weights(train_rows)
    mean, std = fit_normalization(xw_raw + xl_raw)

    torch.manual_seed(seed)
    net = PostSACRanker.build()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)
    xw = (torch.tensor(xw_raw, dtype=torch.float32) - mean_t) / std_t
    xl = (torch.tensor(xl_raw, dtype=torch.float32) - mean_t) / std_t
    wt = torch.tensor(w, dtype=torch.float32)

    def _dev_ranker_authority_accuracy(model) -> float | None:
        ra = [r for r in dev_rows if r["ranker_authority"]]
        if not ra:
            return None
        correct = 0
        for r in ra:
            xa_raw, xb_raw, _w = _xw_xl_weights([{**r, "authoritative_winner": "A"}])
            fa = torch.tensor(xa_raw, dtype=torch.float32)
            with torch.no_grad():
                sa = float(model((fa[0:1] - mean_t) / std_t))
                sb = float(model((torch.tensor(xb_raw, dtype=torch.float32)[0:1] - mean_t) / std_t))
            picked = "A" if sa > sb else ("B" if sb > sa else
                                          ("A" if r["topology_hash_a"] <= r["topology_hash_b"] else "B"))
            if picked == r["authoritative_winner"]:
                correct += 1
        return correct / len(ra)

    best_state, best_metric, best_epoch = None, -1.0, 0
    losses = []
    for ep in range(MAX_EPOCHS):
        per_pair = -torch.nn.functional.logsigmoid(net(xw) - net(xl)).squeeze(-1)
        loss = (per_pair * wt).sum() / wt.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        if ep % EVAL_EVERY == 0 or ep == MAX_EPOCHS - 1:
            m = _dev_ranker_authority_accuracy(net)
            if m is not None and m >= best_metric:
                best_metric, best_epoch = m, ep
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)

    from agentic_raptor.selfimprove_v2.gates import check_ranker_monotonicity

    class _NormalizedNet(torch.nn.Module):
        def __init__(self, inner, mean_, std_):
            super().__init__()
            self.inner = inner
            self.register_buffer("mean", torch.tensor(mean_, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std_, dtype=torch.float32).clamp(min=1e-6))

        def forward(self, x):
            return self.inner((x - self.mean) / self.std)

    mono = check_ranker_monotonicity(_NormalizedNet(net, mean, std))

    with torch.no_grad():
        train_acc = float((net(xw) > net(xl)).float().mean())

    ranker = PostSACRanker(net, feature_mean=mean, feature_std=std)
    return {
        "config": {"lr": lr, "weight_decay": weight_decay, "seed": seed},
        "net": net, "feature_mean": mean, "feature_std": std, "ranker": ranker,
        "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
        "train_pair_accuracy": round(train_acc, 4),
        "best_epoch": best_epoch,
        "dev_ranker_authority_accuracy_at_selection": (
            round(best_metric, 4) if best_metric >= 0 else None),
        "monotonicity": mono,
    }


# ---------------------------------------------------------------------------
# Sections 15/16/20-22: DEV evaluation for any ranker (OLD/NEW/each Rn)
# ---------------------------------------------------------------------------
def evaluate_on_dev(dev_rows: list, ranker) -> dict:
    from agentic_raptor.ranking.post_sac import compare

    results = []
    for r in dev_rows:
        p = r["_raw"]
        da, db, pa, pb, oa, ob = _reconstruct(p)
        spec = p["spec"]
        full_arm = "dpo_ranker" if ranker is not None else "explicit_baseline"
        full = compare(da, db, pa, pb, spec=spec, model=ranker, ranker_arm=full_arm,
                       ranker_checkpoint_hash=getattr(ranker, "checkpoint_hash", None))
        neutral = compare(da, db, pa, pb, spec=spec, model=None, ranker_arm="explicit_baseline")
        results.append({
            "canonical_id": r["canonical_id"], "ranker_authority": r["ranker_authority"],
            "informative": r["informative"], "feasibility_category": r["feasibility_category"],
            "authoritative_winner": r["authoritative_winner"],
            "full": {"selected": full["selected_design"], "basis": full["decision_basis"],
                     "score_margin": full["score_margin"],
                     "correct": full["selected_design"] == r["authoritative_winner"]},
            "neutral": {"selected": neutral["selected_design"], "basis": neutral["decision_basis"],
                       "correct": neutral["selected_design"] == r["authoritative_winner"]},
        })

    def _acc(rs):
        return round(sum(1 for x in rs if x["full"]["correct"]) / len(rs), 4) if rs else None

    all_r = results
    ra_r = [x for x in results if x["ranker_authority"]]
    info_r = [x for x in results if x["informative"]]
    dpo_decided = [x for x in results if x["full"]["basis"] == "dpo_ranker"]

    wins = losses = ties = 0
    catastrophic = 0
    for x in results:
        fc, nc = x["full"]["correct"], x["neutral"]["correct"]
        if x["full"]["selected"] != x["neutral"]["selected"]:
            if fc and not nc:
                wins += 1
            elif nc and not fc:
                losses += 1
            else:
                ties += 1
        # catastrophic (Section 22): ranker had authority, this is a real
        # feasible-vs-infeasible pair, and the learned score picked the
        # infeasible side
        if (x["ranker_authority"] and x["feasibility_category"] == "one_feasible_one_infeasible"
                and x["full"]["basis"] == "dpo_ranker" and not x["full"]["correct"]):
            catastrophic += 1

    return {
        "n_dev_pairs": len(all_r),
        "accuracy_all": _acc(all_r),
        "accuracy_ranker_authority": _acc(ra_r),
        "accuracy_informative": _acc(info_r),
        "n_ranker_authority": len(ra_r),
        "decision_share": round(len(dpo_decided) / len(all_r), 4) if all_r else None,
        "conditional_accuracy": _acc(dpo_decided),
        "n_dpo_decided": len(dpo_decided),
        "score_margins": [x["full"]["score_margin"] for x in dpo_decided
                          if x["full"]["score_margin"] is not None],
        "disagreement_wins_vs_neutral": wins,
        "disagreement_losses_vs_neutral": losses,
        "disagreement_ties_vs_neutral": ties,
        "net_decision_gain_vs_neutral": wins - losses,
        "catastrophic_error_count": catastrophic,
        "results": results,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    from agentic_raptor.ranking.model import PostSACRanker

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Section 3-4: rebuilding trusted pair table ===", flush=True)
    raw_pairs = _load_usable_pairs()
    table, n_dupes = build_pair_table(raw_pairs)
    print(f"rebuilt table: {len(table)} pairs ({n_dupes} canonical duplicates removed "
         f"from {len(raw_pairs)} raw usable pairs)", flush=True)

    mining_report = _mine_additional_pairs_report()
    print(f"Section 11 mining: {mining_report['n_additional_pairs_mined']} additional "
         f"pairs added (see report for sources checked)", flush=True)

    pair_table_path = OUT_DIR / "REBUILT_TRUSTED_PAIR_TABLE.json"
    pair_table_path.write_text(json.dumps(
        [{k: v for k, v in row.items() if k != "_raw"} for row in table],
        indent=1, default=str), encoding="utf-8")
    pair_table_sha = hashlib.sha256(pair_table_path.read_bytes()).hexdigest()
    print(f"pair table -> {pair_table_path} (sha256={pair_table_sha})", flush=True)

    print("\n=== Section 13: spec-disjoint DPO_TRAIN / DPO_DEV split ===", flush=True)
    train_rows, dev_rows, dev_specs = split_train_dev(table)
    train_cov = coverage_stats(train_rows)
    dev_cov = coverage_stats(dev_rows)
    print("TRAIN:", json.dumps(train_cov), flush=True)
    print("DEV:  ", json.dumps(dev_cov), flush=True)
    if dev_cov["n_ranker_authority"] < 5:
        print("WARNING: DEV has < 5 ranker-authority pairs -- reported honestly, "
             "conclusions from this subset will be low-confidence.", flush=True)

    print("\n=== Section 15: OLD_DPO baseline on new DEV split ===", flush=True)
    old_ranker = PostSACRanker.load(OLD_CKPT)
    old_eval = evaluate_on_dev(dev_rows, old_ranker)
    print({k: v for k, v in old_eval.items() if k != "results"}, flush=True)

    print("\n=== Section 16: NO_LEARNED_DPO baseline on new DEV split ===", flush=True)
    neutral_eval = evaluate_on_dev(dev_rows, None)
    print({k: v for k, v in neutral_eval.items() if k != "results"}, flush=True)

    print("\n=== Sections 17-18: hyperparameter sweep (fresh init each) ===", flush=True)
    sweep_results = {}
    for cfg in SWEEP:
        print(f"  training {cfg['name']}: lr={cfg['lr']} weight_decay={cfg['weight_decay']}", flush=True)
        res = train_one_config(train_rows, dev_rows, cfg["lr"], cfg["weight_decay"])
        eval_dev = evaluate_on_dev(dev_rows, res["ranker"])
        sweep_results[cfg["name"]] = {**res, "dev_eval": eval_dev}
        print(f"    train_acc={res['train_pair_accuracy']} "
             f"monotonic={res['monotonicity']['strictly_decreasing']} "
             f"dev_ranker_authority_acc={eval_dev['accuracy_ranker_authority']} "
             f"dev_conditional_acc={eval_dev['conditional_accuracy']}", flush=True)

    print("\n=== Section 20: selecting NEW_DPO ===", flush=True)
    eligible = {name: r for name, r in sweep_results.items()
               if r["monotonicity"]["strictly_decreasing"]}
    if not eligible:
        print("NO CONFIG PASSED THE MONOTONICITY GATE.", flush=True)
        selected_name = None
    else:
        def _key(item):
            name, r = item
            ra = r["dev_eval"]["accuracy_ranker_authority"] or -1
            info = r["dev_eval"]["accuracy_informative"] or -1
            return (ra, info, -r["loss_first_last"][1])
        selected_name = max(eligible.items(), key=_key)[0]
    print(f"selected: {selected_name}", flush=True)

    new_eval = sweep_results[selected_name]["dev_eval"] if selected_name else None
    new_ranker = sweep_results[selected_name]["ranker"] if selected_name else None

    print("\n=== context: OLD_DPO / NEW_DPO / NO_LEARNED_DPO on TRAIN rows "
         "(optimistic, not used for selection) ===", flush=True)
    old_train_eval = evaluate_on_dev(train_rows, old_ranker)
    neutral_train_eval = evaluate_on_dev(train_rows, None)
    new_train_eval = (evaluate_on_dev(train_rows, new_ranker) if new_ranker else None)
    print(f"  OLD_DPO train ranker-authority acc: {old_train_eval['accuracy_ranker_authority']} "
         f"(n={old_train_eval['n_ranker_authority']})", flush=True)
    print(f"  NEW_DPO train ranker-authority acc: "
         f"{new_train_eval['accuracy_ranker_authority'] if new_train_eval else None} "
         f"(n={new_train_eval['n_ranker_authority'] if new_train_eval else 0})", flush=True)
    print(f"  NO_LEARNED_DPO train ranker-authority acc: "
         f"{neutral_train_eval['accuracy_ranker_authority']} "
         f"(n={neutral_train_eval['n_ranker_authority']})", flush=True)

    print("\n=== Section 24: promotion gate ===", flush=True)
    promotion = evaluate_promotion(selected_name, sweep_results, old_eval, neutral_eval, new_eval)
    print(json.dumps(promotion, indent=1), flush=True)

    final_class, final_reason = final_classification(
        promotion, new_eval, new_train_eval, neutral_eval, neutral_train_eval)
    promotion["final_classification"] = final_class
    promotion["final_classification_reason"] = final_reason
    print(f"\nFINAL CLASSIFICATION: {final_class}\n  {final_reason}", flush=True)

    checkpoint_info = save_promoted_checkpoint(
        selected_name, sweep_results[selected_name], pair_table_sha,
        train_cov, dev_cov, dev_specs, checkpoint_status=final_class)
    print(f"\n{final_class}: candidate checkpoint saved (NOT wired as live "
         f"DEFAULT_CKPT either way) -> {checkpoint_info['checkpoint_path']}", flush=True)

    report = {
        "pair_table_path": str(pair_table_path), "pair_table_sha256": pair_table_sha,
        "n_dupes_removed": n_dupes, "mining_report": mining_report,
        "train_coverage": train_cov, "dev_coverage": dev_cov,
        "dev_spec_hashes": sorted(dev_specs),
        "old_dpo_dev_eval": {k: v for k, v in old_eval.items() if k != "results"},
        "no_learned_dpo_dev_eval": {k: v for k, v in neutral_eval.items() if k != "results"},
        "old_dpo_train_eval": {k: v for k, v in old_train_eval.items() if k != "results"},
        "no_learned_dpo_train_eval": {k: v for k, v in neutral_train_eval.items() if k != "results"},
        "new_dpo_train_eval": ({k: v for k, v in new_train_eval.items() if k != "results"}
                               if new_train_eval else None),
        "sweep_summary": {name: {"config": r["config"],
                                 "train_pair_accuracy": r["train_pair_accuracy"],
                                 "monotonicity": r["monotonicity"]["strictly_decreasing"],
                                 "dev_eval": {k: v for k, v in r["dev_eval"].items() if k != "results"}}
                          for name, r in sweep_results.items()},
        "selected_config": selected_name,
        "promotion": promotion, "checkpoint_info": checkpoint_info,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = OUT_DIR / "STAGE7_1_DPO_REPAIR_REPORT.json"
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nStage 7.1 report written to {out_path}", flush=True)


MIN_DEV_RANKER_AUTHORITY_N = 5


def evaluate_promotion(selected_name, sweep_results, old_eval, neutral_eval, new_eval) -> dict:
    checks = {}
    if selected_name is None:
        checks["monotonic"] = False
        return {"promoted": False, "checks": checks,
               "reason": "no swept config passed the monotonicity gate"}
    r = sweep_results[selected_name]
    checks["monotonic"] = r["monotonicity"]["strictly_decreasing"]
    checks["checkpoint_loads"] = r["net"] is not None
    checks["finite_scores"] = all(
        m == m and abs(m) < 1e6 for m in new_eval["score_margins"]) if new_eval["score_margins"] else True
    ra_acc = new_eval["accuracy_ranker_authority"]
    checks["ranker_authority_dev_accuracy_above_chance"] = (
        ra_acc is not None and ra_acc >= 0.6)
    checks["no_clear_regression_vs_neutral"] = (
        (new_eval["accuracy_ranker_authority"] or 0)
        >= (neutral_eval["accuracy_ranker_authority"] or 0) - 0.02)
    checks["disagreement_wins_ge_losses"] = (
        new_eval["disagreement_wins_vs_neutral"] >= new_eval["disagreement_losses_vs_neutral"])
    checks["no_catastrophic_collapse"] = new_eval["catastrophic_error_count"] <= 1
    # NOT a mechanical/behavioral pass-fail check (n is a property of the
    # frozen spec-disjoint split, not of the model) -- tracked SEPARATELY so
    # a tiny-n "pass" cannot silently read as a confident promotion. Section
    # 14 explicitly requires reporting this limitation, not hiding it.
    dev_n_adequate = new_eval["n_ranker_authority"] >= MIN_DEV_RANKER_AUTHORITY_N
    promoted = all(checks.values())
    return {"promoted": promoted, "checks": checks, "selected_config": selected_name,
           "new_dpo_ranker_authority_accuracy": ra_acc,
           "old_dpo_ranker_authority_accuracy": old_eval["accuracy_ranker_authority"],
           "neutral_ranker_authority_accuracy": neutral_eval["accuracy_ranker_authority"],
           "dev_ranker_authority_n": new_eval["n_ranker_authority"],
           "dev_ranker_authority_sample_size_adequate": dev_n_adequate,
           "confidence": ("LOW_SAMPLE_SIZE_CAVEAT" if promoted and not dev_n_adequate
                          else ("NORMAL" if promoted else "N/A"))}


def final_classification(promotion: dict, new_eval: dict | None, new_train_eval: dict | None,
                         neutral_eval: dict, neutral_train_eval: dict) -> tuple:
    """The mechanical gate (`promotion["promoted"]`) is a narrow, literal
    rule check -- it can pass on a lucky tiny-n DEV sample alone (3
    ranker-authority pairs here). This holistic step is the actual
    Section-25/26 decision: it additionally requires the SAME model to beat
    the neutral baseline on the larger (though optimistic, seen-during-
    training) TRAIN ranker-authority sample before calling the repair
    justified. A DEV-only "win" that doesn't hold up on 22 train examples is
    not treated as evidence of real improvement."""
    if not promotion["promoted"]:
        return ("LEARNED_DPO_NOT_JUSTIFIED",
               "mechanical/behavioral promotion gate did not pass; see checks")
    dev_adequate = promotion["dev_ranker_authority_sample_size_adequate"]
    train_ra = new_train_eval["accuracy_ranker_authority"] if new_train_eval else None
    neutral_train_ra = neutral_train_eval["accuracy_ranker_authority"]
    train_beats_neutral = (train_ra is not None and neutral_train_ra is not None
                           and train_ra > neutral_train_ra + 0.02)
    if dev_adequate and train_beats_neutral:
        return ("PROMOTED",
               f"DEV ranker-authority sample size adequate (n={new_eval['n_ranker_authority']}) "
               f"and NEW_DPO beats neutral on both DEV and the larger TRAIN "
               f"ranker-authority sample ({train_ra} vs {neutral_train_ra})")
    if train_ra is not None and neutral_train_ra is not None and train_ra <= neutral_train_ra:
        return ("LEARNED_DPO_NOT_JUSTIFIED",
               f"the DEV promotion gate passed only on a tiny n="
               f"{new_eval['n_ranker_authority']} ranker-authority sample; on the much "
               f"larger n={new_train_eval['n_ranker_authority'] if new_train_eval else '?'} "
               f"TRAIN ranker-authority sample the SAME model scores {train_ra} vs "
               f"neutral's {neutral_train_ra} -- ties or loses to the deterministic "
               f"baseline, so the DEV result is not treated as real evidence of "
               f"improvement (Section 25 applies)")
    return ("PROMOTED_LOW_CONFIDENCE",
           f"mechanical gate passed but DEV ranker-authority n="
           f"{new_eval['n_ranker_authority']} is below the {MIN_DEV_RANKER_AUTHORITY_N} "
           f"minimum for a confident claim; recommend more data before relying on this")


def save_promoted_checkpoint(selected_name, res, pair_table_sha, train_cov, dev_cov,
                             dev_specs, checkpoint_status: str = "CANDIDATE"):
    import torch

    out_dir = OUT_DIR / "post_sac_ranker_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ranker_v2.pt"
    torch.save(res["net"].state_dict(), ckpt_path)
    norm_path = out_dir / "ranker_v2_normalization.json"
    norm_path.write_text(json.dumps(
        {"feature_mean": res["feature_mean"], "feature_std": res["feature_std"]},
        indent=1), encoding="utf-8")
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    from agentic_raptor.ranking.model import FEATURE_NAMES
    from agentic_raptor.publication.artifact_provenance import stamp

    manifest = stamp({
        "model_type": "post_sac_dpo_ranker_v2",
        "checkpoint_status": checkpoint_status,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "normalization_path": str(norm_path),
        "parent_checkpoint": str(OLD_CKPT),
        "parent_checkpoint_note": "fresh initialization, NOT continued from parent "
                                  "(Section 17)",
        "source_code_commit": None,
        "source_code_commit_note": "repository is not under git version control "
                                   "in this checkout",
        "training_pair_file": str(TRUSTED),
        "training_pair_table_sha256": pair_table_sha,
        "train_coverage": train_cov, "dev_coverage": dev_cov,
        "dev_spec_hashes": sorted(dev_specs),
        "feature_schema_version": "post_sac_ranker_features.v1",
        "label_schema_version": "measured_preference.v1",
        "optimizer": "Adam", "learning_rate": res["config"]["lr"],
        "weight_decay": res["config"]["weight_decay"],
        "epochs": MAX_EPOCHS, "selected_epoch": res["best_epoch"],
        "batch_size": "full_batch",
        "random_seed": res["config"]["seed"],
        "feature_names": list(FEATURE_NAMES),
        "normalization": {"feature_mean": res["feature_mean"],
                          "feature_std": res["feature_std"]},
        "safety_decided_pair_weight": SAFETY_DECIDED_WEIGHT,
        "promotion_metrics": res["dev_eval"],
        "selected_sweep_config": selected_name,
        "monotonicity": res["monotonicity"],
    }, model_type="post_sac_dpo_ranker_v2", checkpoint_hash=ckpt_sha,
       validated=(checkpoint_status == "PROMOTED"))
    manifest_path = out_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return {"checkpoint_path": str(ckpt_path), "checkpoint_sha256": ckpt_sha,
           "normalization_path": str(norm_path), "manifest_path": str(manifest_path)}


if __name__ == "__main__":
    main()
