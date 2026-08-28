"""Stage 7.2B: final DPO repair experiment -- richer POST_SAC_FEATURES_V2
(agentic_raptor.ranking.features_v2), tested via a 3-arm feature ablation
(predicted_only == old 11-feature schema; measured_only == candidate +
trajectory + topology features; combined == everything) x 3 model
capacities (Linear/Small/Current), on the EXACT SAME frozen spec-disjoint
train/dev split as Stage 7.2A (same hashing rule, same pool membership ->
reproduces identically without needing to persist the split separately).

Primary metric: run-grouped DEV ranker-authority accuracy. Baseline
(frozen, not retuned): the deterministic selector's Stage 7.2A DEV authority
accuracy, 74.66%.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path

from agentic_raptor.ranking import features_v2 as fv2
from agentic_raptor.ranking import pair_mining as pm

ROOT = Path(__file__).resolve().parent
MINED_PATH = ROOT / "artifacts/publication_v3/stage7_2a_pair_mining/MINED_PAIRS.jsonl"
REPLAY_PATH = ROOT / "artifacts/publication_v2/live_streams_post_cload_v1/sac_replay.jsonl"
OUT_DIR = ROOT / "artifacts/publication_v3/stage7_2b_dpo_repair"
OLD_CKPT = ROOT / "artifacts/publication_v2/post_sac_ranker/ranker.pt"

SAFETY_DECIDED_WEIGHT = 0.15
DEV_FRACTION = 0.22             # IDENTICAL to Stage 7.2A -- reproduces the same split
TARGET_WEIGHT_PER_RUN = 1.0
MAX_EPOCHS = 400
EVAL_EVERY = 10
SEED = 0
LR, WEIGHT_DECAY = 3e-4, 0.1

FEATURE_ARMS = ("predicted_only", "measured_only", "combined")
MODEL_CAPACITIES = {"L_linear": None, "S_small": 8, "C_current": 32}

DETERMINISTIC_BASELINE_FROZEN = 0.7466   # Section 20: frozen, never retuned here


# ---------------------------------------------------------------------------
# build a run index: originating_run_id -> {"A": rows, "B": rows}
# ---------------------------------------------------------------------------
def build_run_index() -> dict:
    runs = pm.load_runs_from_replay(REPLAY_PATH)
    paired = pm.pair_runs_by_spec_seed(runs)
    return {p["originating_run_id"]: {"A": p["A"]["rows"], "B": p["B"]["rows"]}
           for p in paired}


# ---------------------------------------------------------------------------
# V2-ready unified pool: each row carries everything features_v2() needs
# ---------------------------------------------------------------------------
def _pseudo_row_from_outcome(outcome: dict) -> dict:
    """Reconstructs a sizing-loop-row-shaped dict from a saved
    postsizing_outcome()'s 'constraints' achieved values -- these are the
    SAME pre-verification measured numbers the outcome was computed from,
    not a final-verification result (mined pairs' outcome_a/b come from
    real sizing-loop rows, never from a fresh authoritative call)."""
    c = outcome["constraints"]
    return {"gain_db": c["gain_db"]["achieved"], "pm_deg": c["phase_margin_deg"]["achieved"],
           "ugbw_hz": c["ugbw_hz"]["achieved"]}


def build_v2_pool() -> tuple:
    import run_stage71_dpo_repair as s71
    from agentic_raptor.ranking.post_sac import hard_safety_tier
    from agentic_raptor.ranking.types import SurrogatePrediction

    run_index = build_run_index()

    raw_pairs = s71._load_usable_pairs()
    table, n_orig_dupes = s71.build_pair_table(raw_pairs)
    pool = []
    for row in table:
        da, db, pa, pb, oa, ob = s71._reconstruct(row["_raw"])
        pool.append({
            "pair_id": ("original", row["canonical_id"]),
            "originating_run_id": f"original:{row['pair_index']}",
            "spec_hash": row["spec_hash"], "spec": row["_raw"]["spec"],
            "pred_a": pa, "pred_b": pb,
            "candidate_row_a": None, "candidate_row_b": None,   # Section 10: MISSING, honestly
            "branch_rows_a": None, "branch_rows_b": None,       # Section 10: MISSING, honestly
            "family_a": da.topology_family, "family_b": db.topology_family,
            "authoritative_winner": row["authoritative_winner"],
            "ranker_authority": row["ranker_authority"],
            "feasibility_category": row["feasibility_category"],
            "informative": row["informative"], "source": "original",
        })

    def _pred(d):
        names = {f.name for f in fields(SurrogatePrediction)}
        return SurrogatePrediction(**{k: v for k, v in d.items() if k in names})

    n_mined = 0
    for line in MINED_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        n_mined += 1
        pa, pb = _pred(p["prediction_a"]), _pred(p["prediction_b"])
        ranker_authority = hard_safety_tier(pa) == hard_safety_tier(pb)
        fa = p["outcome_a"]["exact_spec_pass"]
        fb = p["outcome_b"]["exact_spec_pass"]
        da_ = p["outcome_a"]["normalized_distance_to_feasibility"] or 9
        db_ = p["outcome_b"]["normalized_distance_to_feasibility"] or 9
        cat = ("both_feasible" if fa and fb else
              "one_feasible_one_infeasible" if fa != fb else
              "both_infeasible_close" if max(da_, db_) < 0.2 else "both_infeasible_far")
        run = run_index.get(p["originating_run_id"], {})
        pool.append({
            "pair_id": ("mined", tuple(p["pair_id"])),
            "originating_run_id": p["originating_run_id"],
            "spec_hash": p["spec_hash"], "spec": p["spec"],
            "pred_a": pa, "pred_b": pb,
            "candidate_row_a": _pseudo_row_from_outcome(p["outcome_a"]),
            "candidate_row_b": _pseudo_row_from_outcome(p["outcome_b"]),
            "branch_rows_a": run.get("A"), "branch_rows_b": run.get("B"),
            "family_a": p.get("family_a"), "family_b": p.get("family_b"),
            "authoritative_winner": p["authoritative_winner"],
            "ranker_authority": ranker_authority, "feasibility_category": cat,
            "informative": fa != fb, "source": "mined",
        })

    seen, deduped = {}, []
    for row in pool:
        if row["pair_id"] not in seen:
            seen[row["pair_id"]] = row
            deduped.append(row)
    return deduped, {"n_original": len(table), "n_original_dupes": n_orig_dupes,
                     "n_mined": n_mined, "n_combined_dupes": len(pool) - len(deduped)}


# ---------------------------------------------------------------------------
# IDENTICAL split rule to Stage 7.2A -- reproduces the same DEV spec set
# ---------------------------------------------------------------------------
def split_train_dev(pool: list, dev_fraction: float = DEV_FRACTION) -> tuple:
    specs = sorted({row["spec_hash"] for row in pool})
    dev_specs = set()
    for s in specs:
        frac = (int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 10_000) / 10_000
        if frac < dev_fraction:
            dev_specs.add(s)
    train = [r for r in pool if r["spec_hash"] not in dev_specs]
    dev = [r for r in pool if r["spec_hash"] in dev_specs]
    return train, dev, dev_specs


def coverage_stats(rows: list) -> dict:
    def _n(pred):
        return sum(1 for r in rows if pred(r))
    return {"n_specs": len(set(r["spec_hash"] for r in rows)),
           "n_runs": len(set(r["originating_run_id"] for r in rows)),
           "n_pairs": len(rows),
           "n_ranker_authority": _n(lambda r: r["ranker_authority"]),
           "n_with_trajectory": _n(lambda r: r["branch_rows_a"] is not None
                                   and r["branch_rows_b"] is not None),
           "n_both_feasible": _n(lambda r: r["feasibility_category"] == "both_feasible")}


def compute_pair_weights(rows: list) -> list:
    from collections import Counter
    run_counts = Counter(r["originating_run_id"] for r in rows)
    return [(1.0 if r["ranker_authority"] else SAFETY_DECIDED_WEIGHT)
           * (TARGET_WEIGHT_PER_RUN / run_counts[r["originating_run_id"]]) for r in rows]


# ---------------------------------------------------------------------------
# feature construction per arm
# ---------------------------------------------------------------------------
def _row_features(row: dict, side: str, arm: str) -> list:
    pred = row[f"pred_{side}"]
    cand = row[f"candidate_row_{side}"]
    branch = row[f"branch_rows_{side}"]
    family = row[f"family_{side}"]
    return fv2.features_v2(row["spec"], pred, cand, branch, family, arm=arm)


def build_model(hidden):
    import torch
    if hidden is None:
        return torch.nn.Sequential(torch.nn.Linear(fv2.FEATURE_DIM_V2, 1))
    return torch.nn.Sequential(torch.nn.Linear(fv2.FEATURE_DIM_V2, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, 1))


def _xw_xl_weights(rows: list, weights: list, arm: str):
    xw, xl, w = [], [], []
    for r, wt in zip(rows, weights):
        fa, fb = _row_features(r, "a", arm), _row_features(r, "b", arm)
        if r["authoritative_winner"] == "A":
            xw.append(fa); xl.append(fb)
        else:
            xw.append(fb); xl.append(fa)
        w.append(wt)
    return xw, xl, w


def fit_normalization_v2(rows: list) -> tuple:
    import statistics as st
    n = fv2.FEATURE_DIM_V2
    mean, std = [0.0] * n, [1.0] * n
    for i in range(n):
        col = [r[i] for r in rows]
        mean[i] = st.mean(col)
        std[i] = (st.pstdev(col) if len(col) > 1 else 1.0) or 1.0
    return mean, std


def train_one(arm: str, hidden, train_rows: list, train_weights: list, dev_rows: list,
             seed: int = SEED) -> dict:
    import torch

    xw_raw, xl_raw, w = _xw_xl_weights(train_rows, train_weights, arm)
    mean, std = fit_normalization_v2(xw_raw + xl_raw)
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)

    torch.manual_seed(seed)
    net = build_model(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    xw = (torch.tensor(xw_raw, dtype=torch.float32) - mean_t) / std_t
    xl = (torch.tensor(xl_raw, dtype=torch.float32) - mean_t) / std_t
    wt = torch.tensor(w, dtype=torch.float32)

    ra_dev = [r for r in dev_rows if r["ranker_authority"]]

    def _dev_ra_acc(model):
        if not ra_dev:
            return None
        fa = [_row_features(r, "a", arm) for r in ra_dev]
        fb = [_row_features(r, "b", arm) for r in ra_dev]
        with torch.no_grad():
            sa = model((torch.tensor(fa, dtype=torch.float32) - mean_t) / std_t)
            sb = model((torch.tensor(fb, dtype=torch.float32) - mean_t) / std_t)
        correct = 0
        for i, r in enumerate(ra_dev):
            picked = "A" if float(sa[i]) > float(sb[i]) else (
                "B" if float(sb[i]) > float(sa[i]) else "A")
            correct += int(picked == r["authoritative_winner"])
        return correct / len(ra_dev)

    best_state, best_metric, best_epoch = None, -1.0, 0
    losses = []
    for ep in range(MAX_EPOCHS):
        per_pair = -torch.nn.functional.logsigmoid(net(xw) - net(xl)).squeeze(-1)
        loss = (per_pair * wt).sum() / wt.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        if ep % EVAL_EVERY == 0 or ep == MAX_EPOCHS - 1:
            m = _dev_ra_acc(net)
            if m is not None and m >= best_metric:
                best_metric, best_epoch = m, ep
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    class _NormalizedNet(torch.nn.Module):
        def __init__(self, inner, mean_, std_):
            super().__init__()
            self.inner = inner
            self.register_buffer("mean", torch.tensor(mean_, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std_, dtype=torch.float32).clamp(min=1e-6))

        def forward(self, x):
            return self.inner((x - self.mean) / self.std)

    mono = _monotonicity_probe_v2(_NormalizedNet(net, mean, std))
    with torch.no_grad():
        train_acc = float((net(xw) > net(xl)).float().mean())
    n_params = sum(p.numel() for p in net.parameters())
    return {"net": net, "feature_mean": mean, "feature_std": std, "n_params": n_params,
           "train_pair_accuracy": round(train_acc, 4), "best_epoch": best_epoch,
           "monotonicity": mono, "arm": arm, "hidden": hidden}


def _monotonicity_probe_v2(model) -> dict:
    """Same spirit as agentic_raptor.selfimprove_v2.gates.check_ranker_
    monotonicity, adapted to the V2 width: score must strictly decrease as
    the 'measured_distance_to_feasibility' feature (index into
    FEATURE_NAMES_V2) increases, holding everything else at a neutral
    baseline."""
    import torch
    idx = fv2.FEATURE_NAMES_V2.index("measured_distance_to_feasibility")
    base = [0.0] * fv2.FEATURE_DIM_V2
    base[fv2.FEATURE_NAMES_V2.index("has_candidate_measurement")] = 1.0
    probe_vals = (0.0, 0.1, 0.25, 0.5, 1.0)
    scores = []
    with torch.no_grad():
        for v in probe_vals:
            row = list(base); row[idx] = v
            scores.append(float(model(torch.tensor([row], dtype=torch.float32))))
    bad = [(a, b) for a, b, sa, sb in
          zip(probe_vals, probe_vals[1:], scores, scores[1:]) if sb >= sa]
    return {"probe_values": list(probe_vals), "probe_scores": [round(s, 4) for s in scores],
           "strictly_decreasing": not bad}


# ---------------------------------------------------------------------------
# evaluation -- Section 21/25: primary accuracy PLUS disagreement wins/
# losses/ties and catastrophic-error accounting against the SAME frozen
# neutral (deterministic) selector Stage 7.2A used. A high accuracy number
# alone is not sufficient for promotion (Section 25) -- both must be
# checked, not just the headline metric.
# ---------------------------------------------------------------------------
def evaluate(rows: list, net, mean, std, arm: str) -> dict:
    import torch

    from agentic_raptor.ranking.post_sac import _deterministic_score
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)

    def _score(r, side):
        f = _row_features(r, side, arm)
        with torch.no_grad():
            return float(net((torch.tensor([f], dtype=torch.float32) - mean_t) / std_t))

    results = []
    wins = losses = ties = catastrophic = 0
    for r in rows:
        sa, sb = _score(r, "a"), _score(r, "b")
        picked = "A" if sa > sb else ("B" if sb > sa else "A")
        correct = picked == r["authoritative_winner"]

        na, nb = _deterministic_score(r["pred_a"]), _deterministic_score(r["pred_b"])
        neutral_picked = "A" if na > nb else ("B" if nb > na else "A")
        neutral_correct = neutral_picked == r["authoritative_winner"]

        if picked != neutral_picked:
            if correct and not neutral_correct:
                wins += 1
            elif neutral_correct and not correct:
                losses += 1
            else:
                ties += 1
        if (r["ranker_authority"] and r["feasibility_category"] == "one_feasible_one_infeasible"
                and not correct):
            catastrophic += 1

        results.append({"originating_run_id": r["originating_run_id"],
                        "ranker_authority": r["ranker_authority"],
                        "feasibility_category": r["feasibility_category"],
                        "informative": r["informative"], "correct": correct,
                        "neutral_correct": neutral_correct,
                        "score_margin": abs(sa - sb)})

    def _acc(rs):
        return round(sum(1 for x in rs if x["correct"]) / len(rs), 4) if rs else None

    ra = [x for x in results if x["ranker_authority"]]
    from collections import defaultdict
    by_run = defaultdict(list)
    for x in ra:
        by_run[x["originating_run_id"]].append(x["correct"])
    run_accs = [sum(v) / len(v) for v in by_run.values()]
    grouped = round(sum(run_accs) / len(run_accs), 4) if run_accs else None
    return {"n_pairs": len(results), "n_ranker_authority": len(ra),
           "n_runs_ranker_authority": len(by_run),
           "accuracy_all": _acc(results), "accuracy_ranker_authority": _acc(ra),
           "accuracy_ranker_authority_run_grouped": grouped,
           "accuracy_informative": _acc([x for x in results if x["informative"]]),
           "disagreement_wins_vs_neutral": wins, "disagreement_losses_vs_neutral": losses,
           "disagreement_ties_vs_neutral": ties, "net_decision_gain_vs_neutral": wins - losses,
           "catastrophic_error_count": catastrophic,
           "per_run_accuracy": {k: round(sum(v) / len(v), 4) for k, v in by_run.items()}}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== building V2-ready pool ===", flush=True)
    pool, pool_stats = build_v2_pool()
    print(pool_stats, flush=True)
    print(f"pool size: {len(pool)}", flush=True)

    train_rows, dev_rows, dev_specs = split_train_dev(pool)
    train_cov, dev_cov = coverage_stats(train_rows), coverage_stats(dev_rows)
    print("TRAIN:", train_cov, flush=True)
    print("DEV:  ", dev_cov, flush=True)
    assert dev_cov["n_ranker_authority"] == 135, (
        f"DEV split diverged from Stage 7.2A (expected 135 ranker-authority "
        f"pairs, got {dev_cov['n_ranker_authority']}) -- not comparable to "
        f"the frozen 74.66% baseline")

    train_weights = compute_pair_weights(train_rows)

    print("\n=== feature ablation x model capacity ===", flush=True)
    all_results = {}
    for arm in FEATURE_ARMS:
        for name, hidden in MODEL_CAPACITIES.items():
            key = f"{arm}__{name}"
            print(f"  training {key} ...", flush=True)
            res = train_one(arm, hidden, train_rows, train_weights, dev_rows)
            dev_eval = evaluate(dev_rows, res["net"], res["feature_mean"], res["feature_std"], arm)
            all_results[key] = {**res, "dev_eval": dev_eval}
            print(f"    n_params={res['n_params']} train_acc={res['train_pair_accuracy']} "
                 f"monotonic={res['monotonicity']['strictly_decreasing']} "
                 f"dev_ra_grouped={dev_eval['accuracy_ranker_authority_run_grouped']}", flush=True)

    print("\n=== selection ===", flush=True)
    eligible = {k: v for k, v in all_results.items() if v["monotonicity"]["strictly_decreasing"]}
    selected_key = None
    if eligible:
        def _key(item):
            k, v = item
            return (v["dev_eval"]["accuracy_ranker_authority_run_grouped"] or -1, -v["n_params"])
        selected_key = max(eligible.items(), key=_key)[0]
    print(f"selected: {selected_key}", flush=True)

    report = {"pool_stats": pool_stats, "train_coverage": train_cov, "dev_coverage": dev_cov,
             "dev_spec_hashes": sorted(dev_specs),
             "deterministic_baseline_frozen": DETERMINISTIC_BASELINE_FROZEN,
             "feature_dim_v2": fv2.FEATURE_DIM_V2, "feature_names_v2": list(fv2.FEATURE_NAMES_V2),
             "feature_provenance_v2": list(fv2.FEATURE_PROVENANCE_V2),
             "ablation_results": {k: {"arm": v["arm"], "hidden": v["hidden"],
                                      "n_params": v["n_params"],
                                      "train_pair_accuracy": v["train_pair_accuracy"],
                                      "monotonicity": v["monotonicity"]["strictly_decreasing"],
                                      "dev_eval": v["dev_eval"]}
                                  for k, v in all_results.items()},
             "selected": selected_key, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    if selected_key:
        sel = all_results[selected_key]
        de = sel["dev_eval"]
        ra_acc = de["accuracy_ranker_authority_run_grouped"] or 0
        # Section 25: accuracy alone is NOT sufficient -- disagreement
        # wins must exceed losses and catastrophic errors must be ~0,
        # checked explicitly rather than inferred from the headline number.
        checks = {
            "monotonic": sel["monotonicity"]["strictly_decreasing"],
            "meaningfully_competitive_or_better":
            ra_acc >= DETERMINISTIC_BASELINE_FROZEN - 0.02,
            "actual_improvement_not_tie": ra_acc > DETERMINISTIC_BASELINE_FROZEN,
            "not_single_run_dependent": de["n_runs_ranker_authority"] >= 3,
            "disagreement_wins_gt_losses":
            de["disagreement_wins_vs_neutral"] > de["disagreement_losses_vs_neutral"],
            "catastrophic_errors_negligible": de["catastrophic_error_count"] <= 1,
        }
        promoted = all(checks.values())
        cls = "DPO_REJUSTIFIED" if promoted else "LEARNED_DPO_FINAL_REJECTED"
        report["promotion"] = {"checks": checks, "final_classification": cls,
                               "selected_ranker_authority_dev_accuracy_grouped": ra_acc,
                               "deterministic_baseline": DETERMINISTIC_BASELINE_FROZEN}
        print("\n=== promotion ===", flush=True)
        print(json.dumps(report["promotion"], indent=1), flush=True)
        ckpt_info = save_checkpoint(selected_key, sel, cls)
        report["checkpoint_info"] = ckpt_info
    else:
        report["promotion"] = {"final_classification": "LEARNED_DPO_FINAL_REJECTED",
                               "reason": "no arm/capacity combination passed the monotonicity gate"}

    out_path = OUT_DIR / "STAGE7_2B_REPORT.json"
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nStage 7.2B report written to {out_path}", flush=True)


def save_checkpoint(key: str, res: dict, status: str) -> dict:
    import torch
    out_dir = OUT_DIR / f"post_sac_ranker_v2_{key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "ranker.pt"
    torch.save(res["net"].state_dict(), ckpt)
    (out_dir / "normalization.json").write_text(
        json.dumps({"feature_mean": res["feature_mean"], "feature_std": res["feature_std"]},
                  indent=1), encoding="utf-8")
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    from agentic_raptor.publication.artifact_provenance import stamp
    manifest = stamp({"model_type": f"post_sac_dpo_ranker_stage72b_{key}",
                      "checkpoint_status": status, "checkpoint_sha256": ckpt_sha,
                      "feature_schema": "POST_SAC_FEATURES_V2", "arm": res["arm"],
                      "hidden": res["hidden"], "n_params": res["n_params"],
                      "monotonicity": res["monotonicity"]},
                     model_type=f"post_sac_dpo_ranker_stage72b_{key}",
                     checkpoint_hash=ckpt_sha, validated=(status == "DPO_REJUSTIFIED"))
    (out_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return {"checkpoint_path": str(ckpt), "checkpoint_sha256": ckpt_sha}


if __name__ == "__main__":
    main()
