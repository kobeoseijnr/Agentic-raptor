"""Part 8: train the post-SAC ranker on TRUSTED measured pairs.

The canonical architecture says a DPO-trained ranker selects which of the two
sized designs receives authoritative verification. No such model existed --
no checkpoint, no trainer -- so `compare()` raised RankerCheckpointMissing
and stage 8 could not run.

Objective is pairwise Bradley-Terry / DPO:

    L = -log sigmoid( s(winner) - s(loser) )

where the winner is decided by `measured_preference` over two AUTHORITATIVE
ngspice outcomes. Only pairs marked `trusted` are used: both designs measured,
complete provenance, same spec, distinct call ids, and not drawn from a
protected evaluation split. Predictions supply the features; measurements
supply only the label.

INFORMATIVE VS AMBIGUOUS PAIRS -- this is not optional, it is why the first
deployed checkpoint was backwards. A "trusted" pair is anything with complete
provenance, but most of them (22 of the first 27 collected) compared two
designs that BOTH failed the spec -- the "winner" is only whichever failed
less, via normalized_distance_to_feasibility, a much noisier signal than
pass/fail. Retraining the exact 12-pair set the deployed checkpoint used, with
increasing weight decay down to a bare LINEAR model, still learned a POSITIVE
weight on worst_violation (should be negative): that is not overfitting a
high-capacity net, it is the data-supported direction on that mix. Training
defaults to INFORMATIVE pairs only (one design passed, the other failed).
Pass --include-ambiguous to add the rest at a reduced weight if informative
pairs are too scarce; the report always states which pairs actually
contributed.

A MONOTONICITY CHECK runs after every fit and BLOCKS saving the checkpoint if
it fails: score must strictly decrease as worst_predicted_violation increases,
holding everything else fixed. A correctly-regularised linear model on the
data above still inverted, so this cannot be trusted to "usually be fine" --
it is checked every time, not assumed.

Bootstrap order matters and cannot be skipped:
    1. run_raptor_v2.py --calibrate   (verifies BOTH designs -> trusted pairs)
    2. python train_post_sac_ranker.py
    3. run_raptor_v2.py               (stage 8 now uses the learned ranker)

Run:  python train_post_sac_ranker.py [--epochs 200] [--min-pairs 8]
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def is_informative(pair: dict) -> bool:
    """One design passed and the other failed -- a real preference exists.

    A pair where both passed or both failed is not meaningless, but its label
    comes from a noisier proxy (distance-to-feasibility) than pass/fail, and
    it dominated the data that produced an inverted checkpoint.
    """
    oa, ob = pair.get("outcome_a") or {}, pair.get("outcome_b") or {}
    return bool(oa.get("exact_spec_pass")) != bool(ob.get("exact_spec_pass"))


def _ident_hash(pair: dict) -> float:
    import hashlib
    oa, ob = pair.get("outcome_a") or {}, pair.get("outcome_b") or {}
    ident = "|".join(str(x) for x in
                     (oa.get("call_id", ""), ob.get("call_id", ""),
                      pair.get("spec_id", "")))
    return (int(hashlib.sha256(ident.encode()).hexdigest()[:12], 16)
           % 10_000) / 10_000


def cross_validate_ranker(k: int = 5, pairs_file=None, weight_decay: float = 0.1,
                          epochs: int = 200, seed: int = 0) -> dict:
    """K-FOLD held-out pairwise accuracy -- the DPO ranker's deployed
    training_report.json only ever reported TRAIN-set accuracy (how well it
    fit the pairs it saw), never a held-out number. That is exactly the gap
    that let the PUCT value net's negative correlation hide behind a
    healthy-looking training loss. This closes it: every informative pair
    is scored exactly once, by a net that never trained on it.
    """
    import torch

    from agentic_raptor.ranking.model import FEATURE_DIM, PostSACRanker, features
    from agentic_raptor.ranking.post_sac import trusted_pairs
    from agentic_raptor.ranking.types import SurrogatePrediction

    def _pred(d):
        keep = {k2: v for k2, v in d.items()
               if k2 in SurrogatePrediction.__dataclass_fields__
               and k2 != "prediction_timestamp"}
        return SurrogatePrediction(**keep)

    all_pairs = [p for p in trusted_pairs(Path(pairs_file) if pairs_file else None)
                if p.get("status") == "trusted" and p.get("chosen")
                and p.get("prediction_a") and p.get("prediction_b")]
    informative = [p for p in all_pairs if is_informative(p)]
    if len(informative) < 2 * k:
        return {"skipped": f"only {len(informative)} informative pairs "
                           f"(need >= {2 * k} for {k}-fold CV)",
               "informative_total": len(informative)}
    folds = [[] for _ in range(k)]
    for p in informative:
        folds[int(_ident_hash(p) * k) % k].append(p)

    def _xw_xl(pairs):
        xw, xl = [], []
        for p in pairs:
            pa, pb = _pred(p["prediction_a"]), _pred(p["prediction_b"])
            fa, fb = features(p["spec"], pa), features(p["spec"], pb)
            if p["chosen"] == "A":
                xw.append(fa); xl.append(fb)
            else:
                xw.append(fb); xl.append(fa)
        return (torch.tensor(xw, dtype=torch.float32),
               torch.tensor(xl, dtype=torch.float32))

    correct = total = 0
    fold_reports = []
    for i in range(k):
        test_pairs = folds[i]
        train_pairs = [p for j, f in enumerate(folds) if j != i for p in f]
        if not test_pairs or not train_pairs:
            continue
        torch.manual_seed(seed + i)
        net = PostSACRanker.build()
        opt = torch.optim.Adam(net.parameters(), lr=1e-2,
                               weight_decay=weight_decay)
        xw, xl = _xw_xl(train_pairs)
        for _ep in range(epochs):
            loss = -torch.nn.functional.logsigmoid(net(xw) - net(xl)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        txw, txl = _xw_xl(test_pairs)
        with torch.no_grad():
            fold_correct = int((net(txw) > net(txl)).sum())
        correct += fold_correct
        total += len(test_pairs)
        fold_reports.append({"fold": i, "train_n": len(train_pairs),
                             "test_n": len(test_pairs),
                             "test_correct": fold_correct})
    return {"k": k, "informative_total": len(informative),
           "pooled_n": total,
           "pooled_pairwise_accuracy": round(correct / total, 4) if total else None,
           "fold_reports": fold_reports,
           "note": "every informative pair scored exactly once, by a net "
                   "that never trained on it -- pooled_n == "
                   "informative_total (minus any dropped empty folds)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=0.1,
                    help="L2 regularisation. With ~20 informative pairs, a "
                         "32-hidden-unit net has enough capacity to memorise "
                         "individual pairs into a locally-inverted shape "
                         "even at 100%% train accuracy -- measured directly: "
                         "weight_decay=0.0 failed the monotonicity check, "
                         "0.01-0.1 passed it at the SAME 100%% accuracy on "
                         "identical data. Not a data problem, a training one.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=8,
                    help="refuse to train below this many USABLE pairs; a "
                         "ranker fitted to 2 examples is noise wearing a "
                         "checkpoint's clothes")
    ap.add_argument("--include-ambiguous", action="store_true",
                    help="also train on both-pass/both-fail pairs, at "
                         "reduced weight. Off by default: this is exactly "
                         "the data that produced the first inverted "
                         "checkpoint.")
    ap.add_argument("--ambiguous-weight", type=float, default=0.2,
                    help="loss weight for ambiguous pairs when included")
    ap.add_argument("--allow-non-monotone", action="store_true",
                    help="save the checkpoint even if it fails the "
                         "monotonicity check. For debugging only -- never "
                         "for a checkpoint that will be deployed.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pairs-file", default=None,
                    help="train on THIS file instead of the live "
                         "trusted_pairs.jsonl -- e.g. a filtered, "
                         "provenance-verified-only snapshot. Does not read "
                         "or write the live file, so this is safe to run "
                         "alongside a calibration batch that's still "
                         "appending to it.")
    args = ap.parse_args()

    import torch

    from agentic_raptor.ranking.model import (DEFAULT_CKPT, FEATURE_NAMES,
                                              PostSACRanker, features)
    from agentic_raptor.ranking.post_sac import TRUSTED, trusted_pairs
    from agentic_raptor.ranking.types import SurrogatePrediction
    from agentic_raptor.selfimprove_v2.gates import check_ranker_monotonicity

    pairs_source = Path(args.pairs_file) if args.pairs_file else None
    all_pairs = [p for p in trusted_pairs(pairs_source)
                if p.get("status") == "trusted" and p.get("chosen")
                and p.get("prediction_a") and p.get("prediction_b")]
    informative = [p for p in all_pairs if is_informative(p)]
    ambiguous = [p for p in all_pairs if not is_informative(p)]
    print(f"trusted pairs file : {pairs_source or TRUSTED}")
    print(f"trusted pairs total: {len(all_pairs)}")
    print(f"  informative (pass vs fail) : {len(informative)}")
    print(f"  ambiguous (both pass/fail) : {len(ambiguous)}"
          f"{' -- INCLUDED at weight ' + str(args.ambiguous_weight) if args.include_ambiguous else ' -- EXCLUDED'}")

    used = list(informative)
    weights = [1.0] * len(informative)
    if args.include_ambiguous:
        used += ambiguous
        weights += [args.ambiguous_weight] * len(ambiguous)

    if len(used) < args.min_pairs:
        raise SystemExit(
            f"\nNOT ENOUGH USABLE TRAINING PAIRS: {len(used)} < "
            f"{args.min_pairs}\n"
            f"({len(informative)} informative, {len(ambiguous)} ambiguous"
            f"{' [not included]' if not args.include_ambiguous else ' [included]'})\n"
            "Generate more with:\n"
            "    python run_raptor_v2.py --calibrate --split train "
            "--spec-index N\n"
            "Prefer more informative pairs over --include-ambiguous: "
            "ambiguous pairs are what produced the first inverted "
            "checkpoint.")

    def _pred(d: dict) -> SurrogatePrediction:
        keep = {k: v for k, v in d.items()
                if k in SurrogatePrediction.__dataclass_fields__
                and k != "prediction_timestamp"}
        return SurrogatePrediction(**keep)

    X_w, X_l, W = [], [], []
    for p, w in zip(used, weights):
        pa, pb = _pred(p["prediction_a"]), _pred(p["prediction_b"])
        spec = p["spec"]
        fa, fb = features(spec, pa), features(spec, pb)
        if p["chosen"] == "A":
            X_w.append(fa); X_l.append(fb)
        else:
            X_w.append(fb); X_l.append(fa)
        W.append(w)

    torch.manual_seed(args.seed)
    net = PostSACRanker.build()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    xw = torch.tensor(X_w, dtype=torch.float32)
    xl = torch.tensor(X_l, dtype=torch.float32)
    w = torch.tensor(W, dtype=torch.float32)
    losses = []
    for _ep in range(args.epochs):
        per_pair = -torch.nn.functional.logsigmoid(net(xw) - net(xl)).squeeze(-1)
        loss = (per_pair * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        train_acc = float((net(xw) > net(xl)).float().mean())
        # accuracy on informative pairs ONLY, even if ambiguous ones were
        # also used to fit -- this is the number that predicts real behaviour
        if informative:
            iw = torch.tensor([features(p["spec"], _pred(p["prediction_a"]))
                               if p["chosen"] == "A" else
                               features(p["spec"], _pred(p["prediction_b"]))
                               for p in informative], dtype=torch.float32)
            il = torch.tensor([features(p["spec"], _pred(p["prediction_b"]))
                               if p["chosen"] == "A" else
                               features(p["spec"], _pred(p["prediction_a"]))
                               for p in informative], dtype=torch.float32)
            informative_acc = float((net(iw) > net(il)).float().mean())
        else:
            informative_acc = None

    mono = check_ranker_monotonicity(net)
    print(f"\nmonotonicity check: "
          f"{'PASS' if mono['strictly_decreasing'] else 'FAIL'}")
    for s in mono["non_decreasing_steps"]:
        print(f"    violated at worst_violation {s}")
    if not mono["strictly_decreasing"] and not args.allow_non_monotone:
        (ROOT / "artifacts/publication_v2/post_sac_ranker").mkdir(
            parents=True, exist_ok=True)
        (ROOT / "artifacts/publication_v2/post_sac_ranker"
         / "REJECTED_training_report.json").write_text(
            json.dumps({"reason": "monotonicity_check_failed",
                        "monotonicity": mono,
                        "pairs_used": len(used),
                        "informative_pairs": len(informative),
                        "ambiguous_pairs_included":
                        len(ambiguous) if args.include_ambiguous else 0},
                       indent=1), encoding="utf-8")
        raise SystemExit(
            "\nREFUSING TO SAVE: the trained ranker scores WORSE designs "
            "higher as worst_predicted_violation increases. This is "
            "exactly the defect that made the first deployed checkpoint "
            "lose to its own hand-written baseline. Collect more "
            "informative pairs rather than override this check.\n"
            "(--allow-non-monotone bypasses this for debugging only)")

    out = Path(args.out or DEFAULT_CKPT)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)

    import hashlib
    from agentic_raptor.publication.artifact_provenance import stamp
    data_hash = hashlib.sha256(
        json.dumps([p.get("spec_hash") or p.get("spec", {}).get("spec_hash")
                   for p in used], sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    ck_hash = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    # every pair used to train this checkpoint must itself be
    # POST_CLOAD_FIX_V1, or the checkpoint cannot honestly claim to be --
    # this is the same discipline check_electrical_environment_
    # compatibility applies at the preflight level, checked HERE too so a
    # training run fails loudly instead of producing a falsely-labeled
    # checkpoint.
    stale = [p for p in used
            if p.get("electrical_environment_version") != "POST_CLOAD_FIX_V1"]
    rec = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "objective": "pairwise Bradley-Terry / DPO",
           "pairs_total": len(all_pairs),
           "pairs_informative": len(informative),
           "pairs_ambiguous_included":
           len(ambiguous) if args.include_ambiguous else 0,
           "ambiguous_weight": args.ambiguous_weight
           if args.include_ambiguous else None,
           "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
           "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
           "train_pair_accuracy": round(train_acc, 4),
           "informative_pair_accuracy":
           round(informative_acc, 4) if informative_acc is not None else None,
           "monotonicity": mono,
           "feature_names": list(FEATURE_NAMES),
           "label_source": "measured_preference over AuthoritativeSpiceOutcome",
           "feature_source": "SurrogatePrediction only (no ngspice)",
           "checkpoint": str(out),
           "pairs_pre_cload_fix_count": len(stale)}
    rec = stamp(rec, model_type="dpo_ranker", training_data_hash=data_hash,
               checkpoint_hash=ck_hash, validated=mono["strictly_decreasing"])
    if stale:
        rec["electrical_environment_version"] = "MIXED"
        print(f"\nWARNING: {len(stale)}/{len(used)} training pairs are not "
             "POST_CLOAD_FIX_V1 -- checkpoint tagged MIXED, not "
             "POST_CLOAD_FIX_V1. Re-run against a pairs file built entirely "
             "after the C_LOAD repair.")
    (out.parent / "training_report.json").write_text(
        json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    if informative_acc is not None and informative_acc < 0.6:
        print("\nWARNING: accuracy on INFORMATIVE pairs is below 0.6 -- the "
              "features may not separate real passes from real failures. "
              "Report this rather than shipping it as a working ranker.")


if __name__ == "__main__":
    main()
