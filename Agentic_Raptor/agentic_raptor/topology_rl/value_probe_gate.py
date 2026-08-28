"""ALPHAZERO VALUE ROOT-CAUSE DIAGNOSTIC, Part I (the justified repair):
the OFFLINE VALUE-PROBE GATE.

Root-cause classification (Part H, 2026-08-13): MULTIPLE_CAUSES, primary =
DESTRUCTIVE_EDIT_ACTION_SPACE + VALUE_TRAINING_FAILURE. The measured
evidence behind the VALUE_TRAINING half: a plain LINEAR probe on the
existing frozen state representation reaches DEV Spearman 0.35 / feasibility
AUROC 0.93 (physical features: 0.40 / 0.88) on spec-disjoint held-out data,
while every AlphaZero-trained value head -- live super-root, per-seed G1,
per-seed G2 -- sits at 0.22-0.24. The representation carries signal; the
value TRAINING pipeline extracts less of it than linear regression, and
every nonlinear head overfits catastrophically at the current ~200-row
data scale (MLP: TRAIN 0.77 -> DEV 0.03).

This module makes that finding operational: ANY future AlphaZero value
candidate (new head, new encoder, new training scheme) must BEAT the
frozen linear-probe reference on the frozen spec-disjoint DEV split of
AZ_VALUE_DIAGNOSTIC_V1 BEFORE any real (SPICE-spending) campaign may be
justified by it. This is the gate that would have saved the G1/G2 campaign
cost: both would have failed it.

No live component reads this module; it changes nothing about FULL.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DATASET = _ROOT / "artifacts/publication_v3/az_value_diagnostic_v1/AZ_VALUE_DIAGNOSTIC_V1.jsonl"
GATE_DIR = _ROOT / "artifacts/publication_v3/az_value_probe_gate"
DEV_FRACTION = 0.3   # must match run_azvalue_partC's frozen split rule

#: frozen reference thresholds -- the measured linear-probe DEV results
#: (Part C/D, 2026-08-13). A candidate passes only if it beats the BEST
#: reference on BOTH ranking metrics and is at least competitive on AUROC.
REFERENCE = {
    "linear_frozen_current_repr": {"dev_spearman": 0.3523,
                                   "dev_pairwise": 0.696, "dev_auroc": 0.925},
    "linear_physical_features": {"dev_spearman": 0.3985,
                                 "dev_pairwise": 0.7128, "dev_auroc": 0.8825},
    "best_alphazero_trained_head": {"dev_spearman": 0.242,
                                    "dev_pairwise": 0.6269, "dev_auroc": 0.8183},
}


def load_dev_rows() -> list[dict]:
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    dev = [r for r in rows
          if (int(hashlib.sha256(r["spec_hash"].encode()).hexdigest()[:8], 16)
              % 10_000) / 10_000 < DEV_FRACTION]
    return dev


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx > 1e-12 and sy > 1e-12 else None


def evaluate_candidate(predict_fn) -> dict:
    """predict_fn(row: dict) -> float. Rows are AZ_VALUE_DIAGNOSTIC_V1
    entries (state/state_graph/spec/...). Returns DEV metrics + gate
    verdict against the frozen references."""
    dev = load_dev_rows()
    preds = [float(predict_fn(r)) for r in dev]
    targets = [r["z"] for r in dev]
    feas = [r["feasible"] for r in dev]

    spearman = _pearson(_rank(preds), _rank(targets))
    correct = total = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            if targets[i] == targets[j]:
                continue
            total += 1
            if (preds[i] > preds[j]) == (targets[i] > targets[j]):
                correct += 1
    pairwise = correct / total if total else None
    pos = [preds[i] for i in range(len(preds)) if feas[i]]
    neg = [preds[i] for i in range(len(preds)) if not feas[i]]
    auroc = None
    if len(pos) >= 3 and len(neg) >= 3:
        wins = sum(1 for p in pos for q in neg if p > q) \
            + 0.5 * sum(1 for p in pos for q in neg if p == q)
        auroc = wins / (len(pos) * len(neg))
    # distance ranking on rows with an uncensored measured distance
    d_rows = [(preds[i], dev[i]["aux"].get("distance")) for i in range(len(dev))
             if dev[i]["aux"].get("distance") is not None]
    dist_spearman = (_pearson(_rank([p for p, _ in d_rows]),
                              _rank([-d for _, d in d_rows]))
                     if len(d_rows) >= 3 else None)

    best_ref = max(REFERENCE.values(), key=lambda r: r["dev_spearman"])
    checks = {
        "beats_reference_dev_spearman": spearman is not None
            and spearman > best_ref["dev_spearman"],
        "beats_reference_dev_pairwise": pairwise is not None
            and pairwise > best_ref["dev_pairwise"],
        "auroc_competitive": auroc is not None and auroc >= best_ref["dev_auroc"] - 0.05,
    }
    return {"n_dev": len(dev),
           "dev_spearman": round(spearman, 4) if spearman is not None else None,
           "dev_pairwise_ranking": round(pairwise, 4) if pairwise is not None else None,
           "dev_feasibility_auroc": round(auroc, 4) if auroc is not None else None,
           "dev_distance_spearman": round(dist_spearman, 4)
               if dist_spearman is not None else None,
           "reference": REFERENCE, "checks": checks,
           "gate": "PASS" if all(checks.values()) else "FAIL",
           "gate_note": ("a candidate value model must beat the best frozen "
                        "linear-probe reference on spec-disjoint DEV before any "
                        "SPICE-spending campaign may cite it as justification")}
