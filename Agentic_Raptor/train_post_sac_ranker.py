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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=8,
                    help="refuse to train below this many trusted pairs; a "
                         "ranker fitted to 2 examples is noise wearing a "
                         "checkpoint's clothes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    from agentic_raptor.ranking.model import (DEFAULT_CKPT, FEATURE_NAMES,
                                              PostSACRanker, features)
    from agentic_raptor.ranking.post_sac import TRUSTED, trusted_pairs
    from agentic_raptor.ranking.types import SurrogatePrediction

    pairs = [p for p in trusted_pairs()
             if p.get("status") == "trusted" and p.get("chosen")
             and p.get("prediction_a") and p.get("prediction_b")]
    print(f"trusted pairs file : {TRUSTED}")
    print(f"usable pairs       : {len(pairs)}")
    if len(pairs) < args.min_pairs:
        raise SystemExit(
            f"\nNOT ENOUGH TRUSTED PAIRS: {len(pairs)} < {args.min_pairs}\n"
            "Each pair needs BOTH designs measured by ngspice with full "
            "provenance.\nGenerate them first:\n"
            "    python run_raptor_v2.py --calibrate --spec-index N\n"
            "(--calibrate verifies both branches instead of only the "
            "selected one)")

    def _pred(d: dict) -> SurrogatePrediction:
        keep = {k: v for k, v in d.items()
                if k in SurrogatePrediction.__dataclass_fields__
                and k != "prediction_timestamp"}
        return SurrogatePrediction(**keep)

    X_w, X_l = [], []
    for p in pairs:
        pa, pb = _pred(p["prediction_a"]), _pred(p["prediction_b"])
        spec = p["spec"]
        fa, fb = features(spec, pa), features(spec, pb)
        if p["chosen"] == "A":
            X_w.append(fa); X_l.append(fb)
        else:
            X_w.append(fb); X_l.append(fa)

    torch.manual_seed(args.seed)
    net = PostSACRanker.build()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    xw = torch.tensor(X_w, dtype=torch.float32)
    xl = torch.tensor(X_l, dtype=torch.float32)
    losses = []
    for _ep in range(args.epochs):
        loss = -torch.nn.functional.logsigmoid(net(xw) - net(xl)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        train_acc = float((net(xw) > net(xl)).float().mean())

    out = Path(args.out or DEFAULT_CKPT)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)

    rec = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "objective": "pairwise Bradley-Terry / DPO",
           "pairs": len(pairs), "epochs": args.epochs, "lr": args.lr,
           "seed": args.seed,
           "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)],
           "train_pair_accuracy": round(train_acc, 4),
           "feature_names": list(FEATURE_NAMES),
           "label_source": "measured_preference over AuthoritativeSpiceOutcome",
           "feature_source": "SurrogatePrediction only (no ngspice)",
           "checkpoint": str(out)}
    (out.parent / "training_report.json").write_text(
        json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    if train_acc < 0.6:
        print("\nWARNING: train accuracy below 0.6 -- the features may not "
              "separate these pairs. Report this rather than shipping it as "
              "a working ranker.")


if __name__ == "__main__":
    main()
