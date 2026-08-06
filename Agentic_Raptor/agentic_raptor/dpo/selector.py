"""Final pre-SPICE candidate selector.

Never selects by DPO score alone: the selection score combines DPO preference,
predicted feasibility, surrogate/dynamics uncertainty, and diversity, and an
explicit exploration quota picks a non-top candidate with configured
probability (seeded, deterministic) to avoid preference-model collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from agentic_raptor.dpo.ranker import DPOConfig, DPORanker
from agentic_raptor.dpo.schemas import CandidateFeatures


@dataclass
class SelectionResult:
    selected: CandidateFeatures
    dpo_score: float
    dpo_rank: int
    reason: str
    ranking: list[tuple[str, float, int]]  # (pool_candidate_id, score, rank)


def _diversity(candidate: CandidateFeatures, pool: list[CandidateFeatures]) -> float:
    """Mean feature distance to the rest of the pool (novelty within the pool)."""
    others = [c for c in pool if c.pool_candidate_id != candidate.pool_candidate_id]
    if not others:
        return 0.0
    v = candidate.feature_vector()
    total = 0.0
    for other in others:
        w = other.feature_vector()
        total += sum((a - b) ** 2 for a, b in zip(v, w, strict=True)) ** 0.5
    return total / len(others)


def select_candidate(
    pool: list[CandidateFeatures],
    ranker: DPORanker | None,
    config: DPOConfig,
    rng: Random,
) -> SelectionResult:
    if not pool:
        raise ValueError("empty candidate pool")
    if ranker is None or not config.enabled:
        ordered = sorted(pool, key=lambda c: (-c.predicted_feasibility, c.pool_candidate_id))
        return SelectionResult(
            ordered[0], 0.0, 0, "dpo disabled: highest predicted feasibility",
            [(c.pool_candidate_id, c.predicted_feasibility, i) for i, c in enumerate(ordered)],
        )

    ranked = ranker.rank(pool)
    ranking = [(f.pool_candidate_id, s, r) for f, s, r in ranked]

    # Exploration quota: with configured probability pick a non-top candidate.
    if len(ranked) > 1 and rng.random() < config.exploration_fraction:
        features, score, rank = ranked[rng.randrange(1, len(ranked))]
        return SelectionResult(
            features, score, rank,
            f"exploration quota ({config.exploration_fraction:.0%}): rank {rank} chosen over rank 0",
            ranking,
        )

    # Combined score — DPO never acts alone.
    diversities = {f.pool_candidate_id: _diversity(f, pool) for f, _s, _r in ranked}
    max_div = max(diversities.values()) or 1.0
    best, best_score = None, -1e18
    for features, dpo_score, rank in ranked:
        combined = (
            1.0 * dpo_score
            + 0.5 * features.predicted_feasibility
            - 0.3 * features.uncertainty
            + 0.2 * diversities[features.pool_candidate_id] / max_div
        )
        if combined > best_score:
            best, best_score, best_rank, best_dpo = features, combined, rank, dpo_score
    assert best is not None
    return SelectionResult(
        best, best_dpo, best_rank,
        f"combined score {best_score:.3f} (dpo + feasibility − uncertainty + diversity)",
        ranking,
    )
