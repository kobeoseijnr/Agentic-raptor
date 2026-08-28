"""Stage 7.2A: retrain the post-SAC DPO ranker on an EXPANDED pair pool
(original 89 rebuilt trusted pairs + cross-branch pairs mined from MB-SAC's
own already-measured sizing-loop candidates -- run_stage72a_mine_pairs.py),
testing whether fixing DATA VOLUME + MODEL CAPACITY (holding the existing
11-feature schema fixed) makes the learned ranker useful, per Stage 7.1's
LEARNED_DPO_NOT_JUSTIFIED finding.

Primary metric: RANKER_AUTHORITY_ACCURACY on spec-disjoint DEV (never
overall pair accuracy -- the hard gate already handles safety-decided
pairs). Compares three model capacities (Linear 11->1, Small 11->8->1,
Current 11->32->1) against the deterministic selector and OLD_DPO, using
BOTH pair-weighted and run/spec-grouped metrics so no single run/spec can
dominate the result (Sections 17-20, 27-32).
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
import time
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MINED_PATH = ROOT / "artifacts/publication_v3/stage7_2a_pair_mining/MINED_PAIRS.jsonl"
OUT_DIR = ROOT / "artifacts/publication_v3/stage7_2a_dpo_repair"
OLD_CKPT = ROOT / "artifacts/publication_v2/post_sac_ranker/ranker.pt"

SAFETY_DECIDED_WEIGHT = 0.15
DEV_FRACTION = 0.22
TARGET_WEIGHT_PER_RUN = 1.0   # Section 19: constant total weight per run
MAX_EPOCHS = 400
EVAL_EVERY = 10
SEED = 0

MODEL_CAPACITIES = {
    "L_linear": {"hidden": None},
    "S_small": {"hidden": 8},
    "C_current": {"hidden": 32},
}
LR, WEIGHT_DECAY = 3e-4, 0.1   # Section 26: one conservative, frozen setting


# ---------------------------------------------------------------------------
# Section 3/16: build the unified pair pool (original + mined)
# ---------------------------------------------------------------------------
def load_original_pairs() -> list:
    import run_stage71_dpo_repair as s71
    raw_pairs = s71._load_usable_pairs()
    table, n_dupes = s71.build_pair_table(raw_pairs)
    return table, n_dupes


def original_to_unified(row: dict) -> dict:
    import run_stage71_dpo_repair as s71
    from agentic_raptor.ranking.pair_mining import knob_hash

    da, db, pa, pb, oa, ob = s71._reconstruct(row["_raw"])
    cid_a = f"{da.canonical_graph_hash}:{knob_hash(da.sizing_vector or {})}"
    cid_b = f"{db.canonical_graph_hash}:{knob_hash(db.sizing_vector or {})}"
    return {
        "pair_id": ("original", row["spec_hash"], *sorted((cid_a, cid_b))),
        "originating_run_id": f"original:{row['pair_index']}",
        "spec_hash": row["spec_hash"], "spec": row["_raw"]["spec"],
        "spec_id": row["spec_id"],
        "candidate_id_a": cid_a, "candidate_id_b": cid_b,
        "pred_a_dict": asdict(pa), "pred_b_dict": asdict(pb),
        "design_a": {"label": "A", "spec_id": da.spec_id, "llm_proposal_id": da.llm_proposal_id,
                    "canonical_graph_hash": da.canonical_graph_hash,
                    "topology_signature": da.topology_signature,
                    "topology_family": da.topology_family,
                    "sizing_vector": da.sizing_vector},
        "design_b": {"label": "B", "spec_id": db.spec_id, "llm_proposal_id": db.llm_proposal_id,
                    "canonical_graph_hash": db.canonical_graph_hash,
                    "topology_signature": db.topology_signature,
                    "topology_family": db.topology_family,
                    "sizing_vector": db.sizing_vector},
        "authoritative_winner": row["authoritative_winner"],
        "ranker_authority": row["ranker_authority"],
        "feasibility_category": row["feasibility_category"],
        "informative": row["informative"],
        "source": "original",
    }


def mined_to_unified(p: dict) -> dict:
    from agentic_raptor.ranking.post_sac import hard_safety_tier
    from agentic_raptor.ranking.types import SurrogatePrediction

    def _pred(d):
        names = {f.name for f in fields(SurrogatePrediction)}
        return SurrogatePrediction(**{k: v for k, v in d.items() if k in names})

    pa, pb = _pred(p["prediction_a"]), _pred(p["prediction_b"])
    ranker_authority = hard_safety_tier(pa) == hard_safety_tier(pb)
    fa, fb = p["outcome_a"]["exact_spec_pass"], p["outcome_b"]["exact_spec_pass"]
    informative = fa != fb
    return {
        "pair_id": ("mined", *p["pair_id"]),
        "originating_run_id": p["originating_run_id"],
        "spec_hash": p["spec_hash"], "spec": p["spec"], "spec_id": p["spec_id"],
        "candidate_id_a": p["candidate_id_a"], "candidate_id_b": p["candidate_id_b"],
        "pred_a_dict": p["prediction_a"], "pred_b_dict": p["prediction_b"],
        "design_a": {"label": "A", "spec_id": p["spec_id"], "llm_proposal_id": None,
                    "canonical_graph_hash": p["topology_hash_a"],
                    "topology_signature": "mined", "topology_family": "mined",
                    "sizing_vector": p["knobs_a"]},
        "design_b": {"label": "B", "spec_id": p["spec_id"], "llm_proposal_id": None,
                    "canonical_graph_hash": p["topology_hash_b"],
                    "topology_signature": "mined", "topology_family": "mined",
                    "sizing_vector": p["knobs_b"]},
        "authoritative_winner": p["authoritative_winner"],
        "ranker_authority": ranker_authority,
        "feasibility_category": ("both_feasible" if fa and fb else
                                 "one_feasible_one_infeasible" if fa != fb else
                                 "both_infeasible_close" if max(
                                     p["outcome_a"]["normalized_distance_to_feasibility"] or 9,
                                     p["outcome_b"]["normalized_distance_to_feasibility"] or 9) < 0.2
                                 else "both_infeasible_far"),
        "informative": informative,
        "source": "mined",
    }


def build_unified_pool() -> tuple:
    original_table, n_orig_dupes = load_original_pairs()
    unified = [original_to_unified(r) for r in original_table]

    n_mined_raw = 0
    if MINED_PATH.is_file():
        lines = MINED_PATH.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            p = json.loads(line)
            n_mined_raw += 1
            unified.append(mined_to_unified(p))

    seen, deduped = {}, []
    for u in unified:
        if u["pair_id"] not in seen:
            seen[u["pair_id"]] = u
            deduped.append(u)
    n_combined_dupes = len(unified) - len(deduped)
    return deduped, {"n_original": len(original_table), "n_original_dupes": n_orig_dupes,
                     "n_mined_raw": n_mined_raw,
                     "n_combined_dupes_removed": n_combined_dupes}


# ---------------------------------------------------------------------------
# Section 17: spec-disjoint split (identical rule to Stage 7.1)
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
    return {
        "n_specs": len(set(r["spec_hash"] for r in rows)),
        "n_runs": len(set(r["originating_run_id"] for r in rows)),
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
        "n_topologies": len(set(r["design_a"]["canonical_graph_hash"] for r in rows)
                            | set(r["design_b"]["canonical_graph_hash"] for r in rows)),
    }


# ---------------------------------------------------------------------------
# Section 19: per-run weighting so one run cannot dominate total loss
# ---------------------------------------------------------------------------
def compute_pair_weights(rows: list) -> list:
    from collections import Counter
    run_counts = Counter(r["originating_run_id"] for r in rows)
    weights = []
    for r in rows:
        base = 1.0 if r["ranker_authority"] else SAFETY_DECIDED_WEIGHT
        run_norm = TARGET_WEIGHT_PER_RUN / run_counts[r["originating_run_id"]]
        weights.append(base * run_norm)
    return weights


# ---------------------------------------------------------------------------
# Section 23/24: model capacity + pairwise training
# ---------------------------------------------------------------------------
def build_model(hidden: int | None):
    import torch
    from agentic_raptor.ranking.model import FEATURE_DIM
    if hidden is None:
        return torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM, 1))
    return torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, 1))


def _xw_xl_weights(rows: list, weights: list):
    from agentic_raptor.ranking.model import features
    from agentic_raptor.ranking.types import SurrogatePrediction

    def _pred(d):
        names = {f.name for f in fields(SurrogatePrediction)}
        return SurrogatePrediction(**{k: v for k, v in d.items() if k in names})

    xw, xl, w = [], [], []
    for r, wt in zip(rows, weights):
        pa, pb = _pred(r["pred_a_dict"]), _pred(r["pred_b_dict"])
        fa, fb = features(r["spec"], pa), features(r["spec"], pb)
        if r["authoritative_winner"] == "A":
            xw.append(fa); xl.append(fb)
        else:
            xw.append(fb); xl.append(fa)
        w.append(wt)
    return xw, xl, w


def train_one_model(hidden, train_rows: list, train_weights: list,
                    dev_rows: list, seed: int = SEED) -> dict:
    import torch
    from agentic_raptor.ranking.model import fit_normalization

    xw_raw, xl_raw, w = _xw_xl_weights(train_rows, train_weights)
    mean, std = fit_normalization(xw_raw + xl_raw)

    torch.manual_seed(seed)
    net = build_model(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)
    xw = (torch.tensor(xw_raw, dtype=torch.float32) - mean_t) / std_t
    xl = (torch.tensor(xl_raw, dtype=torch.float32) - mean_t) / std_t
    wt = torch.tensor(w, dtype=torch.float32)

    ra_dev = [r for r in dev_rows if r["ranker_authority"]]

    def _dev_ra_acc(model) -> float | None:
        if not ra_dev:
            return None
        xa_raw, xb_raw, _ = _xw_xl_weights(
            [{**r, "authoritative_winner": "A"} for r in ra_dev],
            [1.0] * len(ra_dev))
        with torch.no_grad():
            sa = model((torch.tensor(xa_raw, dtype=torch.float32) - mean_t) / std_t)
            sb = model((torch.tensor(xb_raw, dtype=torch.float32) - mean_t) / std_t)
        correct = 0
        for i, r in enumerate(ra_dev):
            picked = "A" if float(sa[i]) > float(sb[i]) else (
                "B" if float(sb[i]) > float(sa[i]) else
                ("A" if r["candidate_id_a"] <= r["candidate_id_b"] else "B"))
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

    from agentic_raptor.selfimprove_v2.gates import check_ranker_monotonicity
    mono = check_ranker_monotonicity(_NormalizedNet(net, mean, std))

    with torch.no_grad():
        train_acc = float((net(xw) > net(xl)).float().mean())

    from agentic_raptor.ranking.model import PostSACRanker
    ranker = PostSACRanker(net, feature_mean=mean, feature_std=std)
    n_params = sum(p.numel() for p in net.parameters())
    return {"net": net, "feature_mean": mean, "feature_std": std, "ranker": ranker,
           "n_params": n_params, "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
           "train_pair_accuracy": round(train_acc, 4), "best_epoch": best_epoch,
           "dev_ranker_authority_accuracy_at_selection":
           round(best_metric, 4) if best_metric >= 0 else None,
           "monotonicity": mono}


# ---------------------------------------------------------------------------
# Sections 27-32: evaluation (pair-weighted AND run-grouped)
# ---------------------------------------------------------------------------
def evaluate_on_rows(rows: list, ranker) -> dict:
    from agentic_raptor.ranking.post_sac import PostSACDesign, compare
    from agentic_raptor.ranking.types import SurrogatePrediction

    def _pred(d):
        names = {f.name for f in fields(SurrogatePrediction)}
        return SurrogatePrediction(**{k: v for k, v in d.items() if k in names})

    def _design(d):
        names = {f.name for f in fields(PostSACDesign)}
        return PostSACDesign(**{k: v for k, v in d.items() if k in names})

    results = []
    for r in rows:
        da, db = _design(r["design_a"]), _design(r["design_b"])
        pa, pb = _pred(r["pred_a_dict"]), _pred(r["pred_b_dict"])
        full_arm = "dpo_ranker" if ranker is not None else "explicit_baseline"
        full = compare(da, db, pa, pb, spec=r["spec"], model=ranker, ranker_arm=full_arm,
                       ranker_checkpoint_hash=getattr(ranker, "checkpoint_hash", None))
        neutral = compare(da, db, pa, pb, spec=r["spec"], model=None, ranker_arm="explicit_baseline")
        results.append({
            "pair_id": r["pair_id"], "originating_run_id": r["originating_run_id"],
            "spec_hash": r["spec_hash"], "ranker_authority": r["ranker_authority"],
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

    all_r, ra_r, info_r = results, [x for x in results if x["ranker_authority"]], \
        [x for x in results if x["informative"]]
    dpo_decided = [x for x in results if x["full"]["basis"] == "dpo_ranker"]

    wins = losses = ties = catastrophic = 0
    for x in results:
        fc, nc = x["full"]["correct"], x["neutral"]["correct"]
        if x["full"]["selected"] != x["neutral"]["selected"]:
            if fc and not nc:
                wins += 1
            elif nc and not fc:
                losses += 1
            else:
                ties += 1
        if (x["ranker_authority"] and x["feasibility_category"] == "one_feasible_one_infeasible"
                and x["full"]["basis"] == "dpo_ranker" and not x["full"]["correct"]):
            catastrophic += 1

    # run-grouped: average PER-RUN accuracy, each run weighted equally
    from collections import defaultdict
    by_run = defaultdict(list)
    for x in ra_r:
        by_run[x["originating_run_id"]].append(x["full"]["correct"])
    run_accs = [sum(v) / len(v) for v in by_run.values()]
    grouped_ra_acc = round(sum(run_accs) / len(run_accs), 4) if run_accs else None

    return {
        "n_pairs": len(all_r), "n_ranker_authority": len(ra_r),
        "n_runs_ranker_authority": len(by_run),
        "accuracy_all": _acc(all_r), "accuracy_ranker_authority": _acc(ra_r),
        "accuracy_ranker_authority_run_grouped": grouped_ra_acc,
        "accuracy_informative": _acc(info_r),
        "decision_share": round(len(dpo_decided) / len(all_r), 4) if all_r else None,
        "conditional_accuracy": _acc(dpo_decided), "n_dpo_decided": len(dpo_decided),
        "score_margins": [x["full"]["score_margin"] for x in dpo_decided
                          if x["full"]["score_margin"] is not None],
        "disagreement_wins_vs_neutral": wins, "disagreement_losses_vs_neutral": losses,
        "disagreement_ties_vs_neutral": ties, "net_decision_gain_vs_neutral": wins - losses,
        "catastrophic_error_count": catastrophic, "results": results,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    from agentic_raptor.ranking.model import PostSACRanker

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== building unified pair pool (original + mined) ===", flush=True)
    pool, pool_stats = build_unified_pool()
    print(pool_stats, flush=True)
    print(f"unified pool size: {len(pool)}", flush=True)

    pool_path = OUT_DIR / "UNIFIED_PAIR_POOL.json"
    pool_path.write_text(json.dumps(
        [{k: v for k, v in row.items()} for row in pool], indent=1, default=str),
        encoding="utf-8")
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    print(f"pool -> {pool_path} (sha256={pool_sha})", flush=True)

    print("\n=== spec-disjoint TRAIN/DEV split ===", flush=True)
    train_rows, dev_rows, dev_specs = split_train_dev(pool)
    train_cov, dev_cov = coverage_stats(train_rows), coverage_stats(dev_rows)
    print("TRAIN:", json.dumps(train_cov), flush=True)
    print("DEV:  ", json.dumps(dev_cov), flush=True)

    print("\n=== OLD_DPO / NO_LEARNED_DPO baselines on DEV ===", flush=True)
    old_ranker = PostSACRanker.load(OLD_CKPT)
    old_eval = evaluate_on_rows(dev_rows, old_ranker)
    neutral_eval = evaluate_on_rows(dev_rows, None)
    print("OLD_DPO:", {k: v for k, v in old_eval.items() if k != "results"}, flush=True)
    print("NEUTRAL:", {k: v for k, v in neutral_eval.items() if k != "results"}, flush=True)

    train_weights = compute_pair_weights(train_rows)

    print("\n=== model capacity experiment (L/S/C) ===", flush=True)
    model_results = {}
    for name, cfg in MODEL_CAPACITIES.items():
        print(f"  training {name} (hidden={cfg['hidden']}) ...", flush=True)
        res = train_one_model(cfg["hidden"], train_rows, train_weights, dev_rows)
        dev_eval = evaluate_on_rows(dev_rows, res["ranker"])
        model_results[name] = {**res, "dev_eval": dev_eval}
        print(f"    n_params={res['n_params']} train_acc={res['train_pair_accuracy']} "
             f"monotonic={res['monotonicity']['strictly_decreasing']} "
             f"dev_ra_acc={dev_eval['accuracy_ranker_authority']} "
             f"dev_ra_acc_grouped={dev_eval['accuracy_ranker_authority_run_grouped']} "
             f"catastrophic={dev_eval['catastrophic_error_count']}", flush=True)

    print("\n=== model selection ===", flush=True)
    eligible = {name: r for name, r in model_results.items()
               if r["monotonicity"]["strictly_decreasing"]}
    selected_name = None
    if eligible:
        def _key(item):
            _, r = item
            ra = r["dev_eval"]["accuracy_ranker_authority_run_grouped"] or -1
            return (ra, -r["n_params"])   # Section 34: prefer simplest on ties
        selected_name = max(eligible.items(), key=_key)[0]
    print(f"selected: {selected_name}", flush=True)

    report = {
        "pool_stats": pool_stats, "pool_path": str(pool_path), "pool_sha256": pool_sha,
        "train_coverage": train_cov, "dev_coverage": dev_cov,
        "dev_spec_hashes": sorted(dev_specs),
        "old_dpo_dev_eval": {k: v for k, v in old_eval.items() if k != "results"},
        "no_learned_dpo_dev_eval": {k: v for k, v in neutral_eval.items() if k != "results"},
        "model_results": {name: {"n_params": r["n_params"],
                                 "train_pair_accuracy": r["train_pair_accuracy"],
                                 "monotonicity": r["monotonicity"]["strictly_decreasing"],
                                 "dev_eval": {k: v for k, v in r["dev_eval"].items() if k != "results"}}
                          for name, r in model_results.items()},
        "selected_model": selected_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if selected_name:
        sel = model_results[selected_name]
        promotion = evaluate_promotion(sel, old_eval, neutral_eval)
        report["promotion"] = promotion
        print("\n=== promotion gate ===", flush=True)
        print(json.dumps(promotion, indent=1), flush=True)
        if promotion["final_classification"] in ("PROMOTED",):
            ckpt_info = save_checkpoint(selected_name, sel, pool_sha, train_cov, dev_cov,
                                        dev_specs, status="PROMOTED")
        else:
            ckpt_info = save_checkpoint(selected_name, sel, pool_sha, train_cov, dev_cov,
                                        dev_specs, status=promotion["final_classification"])
        report["checkpoint_info"] = ckpt_info
    else:
        report["promotion"] = {"final_classification": "STAGE_7_2A_DATA_FIX_INSUFFICIENT",
                               "reason": "no model config passed the monotonicity gate"}

    out_path = OUT_DIR / "STAGE7_2A_REPORT.json"
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nStage 7.2A report written to {out_path}", flush=True)


def evaluate_promotion(sel: dict, old_eval: dict, neutral_eval: dict) -> dict:
    de = sel["dev_eval"]
    checks = {
        "monotonic": sel["monotonicity"]["strictly_decreasing"],
        "finite_scores": all(m == m and abs(m) < 1e6 for m in de["score_margins"]) if de["score_margins"] else True,
        "ranker_authority_dev_accuracy_above_chance": (de["accuracy_ranker_authority_run_grouped"] or 0) >= 0.6,
        "no_clear_regression_vs_neutral": (
            (de["accuracy_ranker_authority_run_grouped"] or 0)
            >= (neutral_eval["accuracy_ranker_authority_run_grouped"] or 0) - 0.02),
        "disagreement_wins_ge_losses": de["disagreement_wins_vs_neutral"] >= de["disagreement_losses_vs_neutral"],
        "no_catastrophic_collapse": de["catastrophic_error_count"] <= 1,
        "not_single_run_dependent": de["n_runs_ranker_authority"] >= 3,
    }
    promoted = all(checks.values())
    if not promoted:
        cls = "STAGE_7_2A_DATA_FIX_INSUFFICIENT"
    else:
        cls = "PROMOTED"
    return {"checks": checks, "final_classification": cls,
           "ranker_authority_dev_accuracy_grouped": de["accuracy_ranker_authority_run_grouped"],
           "neutral_ranker_authority_dev_accuracy_grouped":
           neutral_eval["accuracy_ranker_authority_run_grouped"],
           "old_dpo_ranker_authority_dev_accuracy_grouped":
           old_eval["accuracy_ranker_authority_run_grouped"]}


def save_checkpoint(name, res, pool_sha, train_cov, dev_cov, dev_specs, status: str) -> dict:
    import torch
    out_dir = OUT_DIR / f"post_sac_ranker_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "ranker.pt"
    torch.save(res["net"].state_dict(), ckpt)
    norm = out_dir / "ranker_normalization.json"
    norm.write_text(json.dumps({"feature_mean": res["feature_mean"],
                                "feature_std": res["feature_std"]}, indent=1), encoding="utf-8")
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    from agentic_raptor.publication.artifact_provenance import stamp
    manifest = stamp({
        "model_type": f"post_sac_dpo_ranker_stage72a_{name}",
        "checkpoint_status": status, "checkpoint_path": str(ckpt),
        "checkpoint_sha256": ckpt_sha, "normalization_path": str(norm),
        "parent_checkpoint": str(OLD_CKPT),
        "parent_checkpoint_note": "fresh initialization, not continued",
        "pool_sha256": pool_sha, "train_coverage": train_cov, "dev_coverage": dev_cov,
        "dev_spec_hashes": sorted(dev_specs), "n_params": res["n_params"],
        "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "epochs": MAX_EPOCHS,
        "selected_epoch": res["best_epoch"], "random_seed": SEED,
        "safety_decided_pair_weight": SAFETY_DECIDED_WEIGHT,
        "target_weight_per_run": TARGET_WEIGHT_PER_RUN,
        "dev_eval": {k: v for k, v in res["dev_eval"].items() if k != "results"},
        "monotonicity": res["monotonicity"],
    }, model_type=f"post_sac_dpo_ranker_stage72a_{name}", checkpoint_hash=ckpt_sha,
       validated=(status == "PROMOTED"))
    (out_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return {"checkpoint_path": str(ckpt), "checkpoint_sha256": ckpt_sha,
           "manifest_path": str(out_dir / "training_manifest.json")}


if __name__ == "__main__":
    main()
