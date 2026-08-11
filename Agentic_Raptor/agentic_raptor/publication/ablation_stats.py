"""Statistical aggregation across specifications and pipeline seeds (Part:
STATISTICS). Stdlib-only (random + statistics) -- no scipy dependency.

Never reports only a p-value: `compare_paired` always returns mean/median
delta, a bootstrap CI, a permutation p-value, AND an effect size together,
plus an explicit `sufficient_n` flag so a caller cannot accidentally
declare a component important from too few paired runs.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

#: below this many paired (spec, seed) observations, `sufficient_n` is
#: False -- matches the agreed 3-seed pilot floor, not an arbitrary number
MIN_PAIRS_FOR_CLAIM = 3


@dataclass
class DescriptiveStats:
    n: int
    mean: float | None
    median: float | None
    std: float | None
    ci95_lo: float | None
    ci95_hi: float | None


def describe(values: list) -> DescriptiveStats:
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return DescriptiveStats(0, None, None, None, None, None)
    mean = statistics.fmean(vals)
    median = statistics.median(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0
    se = std / (n ** 0.5) if n > 1 else 0.0
    return DescriptiveStats(n, round(mean, 6), round(median, 6),
                            round(std, 6),
                            round(mean - 1.96 * se, 6) if n > 1 else mean,
                            round(mean + 1.96 * se, 6) if n > 1 else mean)


def paired_deltas(values_a: list, values_b: list) -> list[float]:
    """Delta_i = metric_A0(spec_i,seed_i) - metric_Ax(spec_i,seed_i), pairs
    with a missing value on either side are dropped, not imputed."""
    return [float(a) - float(b) for a, b in zip(values_a, values_b)
           if a is not None and b is not None]


def bootstrap_ci(deltas: list[float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Paired bootstrap CI on the mean delta (resample pairs with
    replacement, not the two arms independently -- independence would
    destroy the pairing this whole design exists to exploit)."""
    if not deltas:
        return None, None
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[max(0, int((alpha / 2) * n_boot))]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1)]
    return round(lo, 6), round(hi, 6)


def permutation_test(deltas: list[float], n_perm: int = 2000,
                     seed: int = 0) -> float | None:
    """Paired sign-flip permutation test. Null hypothesis: each pair's sign
    is equally likely to have gone either way (no systematic A0 vs Ax
    effect). Two-sided p-value."""
    if not deltas:
        return None
    rng = random.Random(seed)
    n = len(deltas)
    observed = abs(sum(deltas))
    if observed == 0:
        return 1.0
    count = 0
    for _ in range(n_perm):
        stat = abs(sum(d if rng.random() < 0.5 else -d for d in deltas))
        if stat >= observed:
            count += 1
    return round(count / n_perm, 4)


def cohens_d_paired(deltas: list[float]) -> float | None:
    """Effect size for the paired difference (mean delta / std of deltas)."""
    if len(deltas) < 2:
        return None
    sd = statistics.stdev(deltas)
    return round(statistics.fmean(deltas) / sd, 4) if sd > 0 else None


@dataclass
class PairedComparison:
    metric: str
    n_pairs: int
    sufficient_n: bool
    mean_delta: float | None
    median_delta: float | None
    bootstrap_ci95: tuple
    permutation_p_value: float | None
    effect_size_cohens_d: float | None
    pass_rate_a: float | None
    pass_rate_b: float | None
    absolute_pp_change: float | None


def compare_paired(metric: str, values_a: list, values_b: list, *,
                   is_pass_metric: bool = False, n_boot: int = 2000,
                   n_perm: int = 2000, seed: int = 0) -> PairedComparison:
    """A0 (`values_a`) vs Ax (`values_b`), aligned by (spec_id, pipeline_seed)
    -- caller is responsible for the alignment; this function only computes
    statistics over already-paired sequences."""
    deltas = paired_deltas(values_a, values_b)
    n = len(deltas)
    lo, hi = bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
    p = permutation_test(deltas, n_perm=n_perm, seed=seed)
    d = cohens_d_paired(deltas)
    pass_a = pass_b = pp = None
    if is_pass_metric:
        ab = [bool(v) for v in values_a if v is not None]
        bb = [bool(v) for v in values_b if v is not None]
        pass_a = round(sum(ab) / len(ab), 4) if ab else None
        pass_b = round(sum(bb) / len(bb), 4) if bb else None
        if pass_a is not None and pass_b is not None:
            pp = round((pass_a - pass_b) * 100, 4)
    return PairedComparison(
        metric=metric, n_pairs=n, sufficient_n=n >= MIN_PAIRS_FOR_CLAIM,
        mean_delta=round(statistics.fmean(deltas), 6) if deltas else None,
        median_delta=round(statistics.median(deltas), 6) if deltas else None,
        bootstrap_ci95=(lo, hi), permutation_p_value=p,
        effect_size_cohens_d=d, pass_rate_a=pass_a, pass_rate_b=pass_b,
        absolute_pp_change=pp)


def align_pairs(rows_a: list[dict], rows_b: list[dict], metric_key: str, *,
                key_fields: tuple = ("spec_id", "pipeline_seed")) -> tuple[list, list]:
    """Align two arms' result rows by (spec_id, pipeline_seed) and extract
    one metric from each -- the shared alignment step every paired
    comparison in this framework needs, done once rather than per-caller."""
    idx_b = {tuple(r.get(k) for k in key_fields): r for r in rows_b}
    va, vb = [], []
    for ra in rows_a:
        key = tuple(ra.get(k) for k in key_fields)
        rb = idx_b.get(key)
        if rb is None:
            continue
        va.append(ra.get(metric_key))
        vb.append(rb.get(metric_key))
    return va, vb
