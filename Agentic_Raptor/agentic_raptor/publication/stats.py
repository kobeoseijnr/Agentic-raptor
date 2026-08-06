"""Frozen statistical formulas for the publication study.

Defined and frozen BEFORE final results are viewed (experimental rule).
Paired bootstrap CIs, Wilcoxon signed-rank, McNemar, Cliff's delta effect
size, Holm-Bonferroni correction.
"""

from __future__ import annotations

import math
from random import Random

STATS_VERSION = "pubstats.1"


def mean_ci(xs: list, n_boot: int = 5000, seed: int = 0,
            alpha: float = 0.05) -> dict:
    """Bootstrap mean with 95% percentile CI."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = Random(seed)
    n = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(n)) / n
                   for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / n) ** 0.5
    med = sorted(xs)[n // 2]
    return {"n": n, "mean": round(m, 4), "median": round(med, 4),
            "sd": round(sd, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def paired_bootstrap_diff(a: list, b: list, n_boot: int = 5000,
                          seed: int = 0) -> dict:
    """CI on mean(a-b) over paired tasks; pairs with None are dropped."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"n": 0}
    rng = Random(seed)
    n = len(pairs)
    diffs = [x - y for x, y in pairs]
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n
                   for _ in range(n_boot))
    return {"n": n, "mean_diff": round(sum(diffs) / n, 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4),
                     round(boots[int(0.975 * n_boot) - 1], 4)],
            "significant_at_05": not (boots[int(0.025 * n_boot)] <= 0
                                      <= boots[int(0.975 * n_boot) - 1])}


def wilcoxon(a: list, b: list) -> dict:
    """Wilcoxon signed-rank on paired samples (scipy; NaN-safe)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None
             and x != y]
    if len(pairs) < 5:
        return {"n": len(pairs), "p": None,
                "note": "fewer than 5 non-tied pairs"}
    from scipy.stats import wilcoxon as _w
    stat, p = _w([x for x, _ in pairs], [y for _, y in pairs])
    return {"n": len(pairs), "statistic": float(stat), "p": round(float(p), 5)}


def mcnemar(a_pass: list, b_pass: list) -> dict:
    """Exact McNemar test on paired boolean exact-pass outcomes."""
    b01 = sum(1 for x, y in zip(a_pass, b_pass) if x and not y)
    b10 = sum(1 for x, y in zip(a_pass, b_pass) if y and not x)
    n = b01 + b10
    if n == 0:
        return {"discordant": 0, "p": None, "note": "no discordant pairs"}
    from scipy.stats import binomtest
    p = binomtest(min(b01, b10), n, 0.5).pvalue * 1.0
    return {"a_only": b01, "b_only": b10, "discordant": n,
            "p": round(float(p), 5)}


def cliffs_delta(a: list, b: list) -> dict:
    """Cliff's delta effect size (non-parametric)."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not a or not b:
        return {"delta": None}
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    d = (gt - lt) / (len(a) * len(b))
    mag = ("negligible" if abs(d) < 0.147 else "small" if abs(d) < 0.33
           else "medium" if abs(d) < 0.474 else "large")
    return {"delta": round(d, 4), "magnitude": mag}


def holm_correct(named_pvals: dict) -> dict:
    """Holm-Bonferroni step-down correction over a family of tests."""
    items = sorted(((k, v) for k, v in named_pvals.items() if v is not None),
                   key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[k] = {"p_raw": p, "p_holm": round(adj, 5),
                  "significant_05": adj < 0.05}
    for k, v in named_pvals.items():
        if v is None:
            out[k] = {"p_raw": None, "p_holm": None, "significant_05": None}
    return out
