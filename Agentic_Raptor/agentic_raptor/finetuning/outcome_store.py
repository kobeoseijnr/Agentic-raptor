"""Verified episode store + outcome attribution for one-time LoRA fine-tuning.

Separate from the pre-SPICE DPO ranker: this feeds a single supervised LoRA
run of the topology generator after enough episodes accumulate. No continual
per-episode fine-tuning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeOutcome:
    """One completed Agentic RAPTOR episode (fields per spec §3)."""

    episode_id: str
    specifications: dict[str, Any]
    operating_conditions: dict[str, Any] = field(default_factory=dict)
    rag_context: list[str] = field(default_factory=list)
    original_topology: dict[str, Any] | None = None
    original_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    mcts_edits: list[dict[str, Any]] = field(default_factory=list)
    final_topology: dict[str, Any] | None = None
    sizing_vector: list[float] = field(default_factory=list)
    spice_metrics: dict[str, float] = field(default_factory=dict)
    constraint_margins: dict[str, float] = field(default_factory=dict)
    pvt_pass_rate: float | None = None
    fom: float | None = None
    passed: bool = False
    n_mcts_edits: int = 0
    n_sizing_steps: int = 0
    spice_calls: int = 0
    calls_to_first_pass: int | None = None
    runtime_s: float = 0.0
    simulation_failures: int = 0
    model_version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeOutcome:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class EpisodeOutcomeStore:
    """Append-only JSONL store of completed episodes."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.episodes: list[EpisodeOutcome] = []
        if self.path and self.path.is_file():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.episodes.append(EpisodeOutcome.from_dict(json.loads(line)))

    def add(self, episode: EpisodeOutcome) -> None:
        self.episodes.append(episode)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.episodes)
