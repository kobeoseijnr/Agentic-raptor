"""Outcome attribution: classify episodes for dataset construction.

A — successful ORIGINAL proposal (valid, ≤ light repair, passed SPICE);
B — successful CORRECTED proposal (original failed; MCTS-repaired passed);
C — unsuccessful (no verified success in budget).

A heavily repaired original is NEVER labelled a direct success (class A caps
repair burden); failed final topologies are never positive targets.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_raptor.finetuning.outcome_store import EpisodeOutcome

_LIGHT_REPAIR_EDITS = 1  # class A allows at most this many MCTS edits


@dataclass
class Attribution:
    episode_id: str
    category: str            # "A" | "B" | "C"
    original_valid: bool
    edit_distance: int       # MCTS edits applied original → final
    repair_burden: float     # edits / (edits + 1) in [0, 1)
    sizing_difficulty: float
    spice_efficiency: float
    final_worst_margin: float
    pvt_robustness: float
    confidence: float


def attribute(episode: EpisodeOutcome, budgets_sizing_steps: int = 12) -> Attribution:
    edits = episode.n_mcts_edits
    worst = min(episode.constraint_margins.values(), default=-1.0)
    if not episode.passed or episode.final_topology is None:
        category, confidence = "C", 1.0
    elif episode.original_valid and edits <= _LIGHT_REPAIR_EDITS:
        category, confidence = "A", 1.0 if edits == 0 else 0.8
    else:
        category, confidence = "B", 0.9 if edits <= 4 else 0.6
    return Attribution(
        episode_id=episode.episode_id,
        category=category,
        original_valid=episode.original_valid,
        edit_distance=edits,
        repair_burden=edits / (edits + 1),
        sizing_difficulty=min(1.0, episode.n_sizing_steps / max(1, budgets_sizing_steps)),
        spice_efficiency=1.0 / max(1, episode.spice_calls),
        final_worst_margin=worst,
        pvt_robustness=episode.pvt_pass_rate if episode.pvt_pass_rate is not None else 0.0,
        confidence=confidence,
    )
