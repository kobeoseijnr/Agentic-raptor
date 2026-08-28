"""PAIRED BOOTSTRAP CONFIDENCE INTERVALS for ablation campaigns (2026-08-17).

Top-venue reviewers expect intervals, not point estimates, and an honest
statement of the effective sample. This module computes, per arm vs A0:

  * paired bootstrap CI (resampling SPECS, the true unit of replication --
    seeds are nested inside specs) on pass-rate delta, FoM delta (log
    scale), and SPICE-cost delta;
  * the EFFECTIVE sample: the number of specs on which the two arms'
    outcomes actually differ (byte-identical selections contribute zero
    discriminating evidence and are reported as such, not hidden);
  * a sign-test p-value on the paired wins/losses.

Determinism note: sizing is bit-reproducible at a fixed seed, so pipeline
seeds vary ONLY the LLM proposal sampling (torch.manual_seed(seed0*1000 +
attempt)). Where proposal SETS converge across seeds, per-seed rows are
identical -- correctly treated as one replicate, not three.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _outcome(r: dict) -> dict:
    n = r.get("nominal") or {}
    fom = (r.get("fom") or {}).get("fom_value")
    sp = r.get("spice") or {}
    return {"pass": bool(n.get("complete_pass")),
            "fom": fom,
            "spice": sp.get("total_calls") if isinstance(sp, dict) else None,
            "ok": r.get("trace_result") == "OK"}


def paired_table(rows: list[dict], baseline: str = "A0") -> dict:
    """{arm: {spec_key: (base_outcome, arm_outcome)}} keyed by (spec, seed)."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["spec_index"], r.get("pipeline_seed", 0))][r["ablation_id"]] = _outcome(r)
    arms = sorted({r["ablation_id"] for r in rows} - {baseline})
    out = {}
    for a in arms:
        pairs = {}
        for k, d in by.items():
            if baseline in d and a in d:
                pairs[k] = (d[baseline], d[a])
        out[a] = pairs
    return out


def bootstrap_ci(values: list[float], n_boot: int = 5000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap over the given paired deltas."""
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return (sum(values) / n, means[int(alpha / 2 * n_boot)],
            means[int((1 - alpha / 2) * n_boot) - 1])


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial p-value on paired wins vs losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)


STOCK_FAMILIES = ("1s_none", "2s_none", "2s_miller", "2s_rc",
                  "3s_none", "3s_miller", "3s_rc")


def composition_metric(rows: list[dict]) -> dict:
    """LLM-VALUE PROBE: per arm, fraction of PASSES whose winning family is
    OUTSIDE the stock template library -- direct evidence the proposer
    composed a structure retrieval could not have supplied. Rows lacking
    selected_family (pre-2026-08-17 campaigns) report None."""
    out = {}
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["ablation_id"]].append(r)
    for arm, rs in by_arm.items():
        passes = [r for r in rs if (r.get("nominal") or {}).get("complete_pass")]
        known = [r for r in passes if r.get("selected_family")]
        if not known:
            out[arm] = {"passes": len(passes), "off_library_passes": None}
            continue
        off = sum(1 for r in known
                  if r["selected_family"] not in STOCK_FAMILIES)
        out[arm] = {"passes": len(passes), "off_library_passes": off,
                    "off_library_fraction": round(off / len(known), 3)}
    return out


def analyze(results_path: Path, baseline: str = "A0") -> dict:
    rows = _rows(results_path)
    table = paired_table(rows, baseline)
    report = {"results_file": str(results_path), "baseline": baseline, "arms": {},
              "composition": composition_metric(rows)}
    for arm, pairs in table.items():
        d_pass, d_logfom, d_spice = [], [], []
        wins = losses = ties = 0
        differing = 0
        for (b, a) in pairs.values():
            if not (b["ok"] and a["ok"]):
                # arm hard-failed (e.g. A3): counts as a pass-rate loss vs
                # baseline when baseline passed; nothing else comparable
                if b["ok"] and not a["ok"]:
                    d_pass.append(-1.0 if b["pass"] else 0.0)
                    losses += 1 if b["pass"] else 0
                    differing += 1
                continue
            dp = float(a["pass"]) - float(b["pass"])
            d_pass.append(dp)
            if a["fom"] and b["fom"] and a["fom"] > 0 and b["fom"] > 0:
                d_logfom.append(math.log(a["fom"]) - math.log(b["fom"]))
            if a["spice"] is not None and b["spice"] is not None:
                d_spice.append(a["spice"] - b["spice"])
            differs = (dp != 0) or (a["fom"] != b["fom"]) or (a["spice"] != b["spice"])
            differing += int(differs)
            if dp > 0:
                wins += 1
            elif dp < 0:
                losses += 1
            else:
                ties += 1
        report["arms"][arm] = {
            "n_pairs": len(pairs),
            "effective_n_differing": differing,
            "pass_delta": bootstrap_ci(d_pass),
            "log_fom_delta": bootstrap_ci(d_logfom) if d_logfom else None,
            "spice_delta": bootstrap_ci(d_spice) if d_spice else None,
            "wins": wins, "losses": losses, "ties": ties,
            "sign_test_p": sign_test_p(wins, losses),
        }
    return report


def format_markdown(report: dict) -> str:
    lines = [f"| arm vs {report['baseline']} | n | eff. n | d-pass [95% CI] | d-logFoM [95% CI] | d-SPICE [95% CI] | W/L/T | p |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, s in report["arms"].items():
        def ci(t, fmt="{:+.2f}"):
            if not t or any(isinstance(x, float) and math.isnan(x) for x in t):
                return "-"
            return f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"
        lines.append(f"| {arm} | {s['n_pairs']} | {s['effective_n_differing']} | "
                     f"{ci(s['pass_delta'])} | {ci(s['log_fom_delta'])} | "
                     f"{ci(s['spice_delta'], '{:+.0f}')} | "
                     f"{s['wins']}/{s['losses']}/{s['ties']} | {s['sign_test_p']:.2f} |")
    return "\n".join(lines)
