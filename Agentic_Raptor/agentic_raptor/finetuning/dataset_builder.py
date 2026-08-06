"""SFT dataset construction with quality tiers and group-safe splits."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from random import Random
from typing import Any

from agentic_raptor.finetuning.attribution import Attribution, attribute
from agentic_raptor.finetuning.outcome_store import EpisodeOutcome

TIERS = ("elite", "verified_success", "verified_corrected")


@dataclass
class SFTExample:
    example_id: str
    kind: str                       # "original_success" | "corrected"
    prompt: dict[str, Any]          # deployment-time inputs ONLY
    target: dict[str, Any]          # final VERIFIED topology (structured)
    spec_group: str                 # split key: specification family
    tier: str
    attribution_confidence: float

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)


def _spec_group(spec: dict[str, Any]) -> str:
    key = f"{spec.get('circuit_class')}|{spec.get('technology')}|{round(float(spec.get('supply_voltage', 0)), 1)}"
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def build_dataset(
    episodes: list[EpisodeOutcome],
    tier: str = "verified_success",
    include_corrected: bool = True,
) -> list[SFTExample]:
    """Positive targets are verified FINAL topologies only; failed episodes and
    failed finals are never targets; prompts contain no post-simulation data."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    examples: list[SFTExample] = []
    for episode in episodes:
        a: Attribution = attribute(episode)
        if a.category == "C" or episode.final_topology is None:
            continue
        if tier == "elite" and (a.final_worst_margin < 0.05 or a.pvt_robustness < 0.999 or a.edit_distance > _elite_edits()):
            continue
        base_prompt = {
            "specifications": episode.specifications,
            "rag_context": episode.rag_context,
            "operating_conditions": episode.operating_conditions,
            "instructions": "generate a complete structured circuit topology (JSON schema)",
        }
        if a.category == "A":
            examples.append(
                SFTExample(
                    example_id=f"sft-{episode.episode_id}-orig",
                    kind="original_success",
                    prompt=base_prompt,
                    target=episode.final_topology,
                    spec_group=_spec_group(episode.specifications),
                    tier=tier,
                    attribution_confidence=a.confidence,
                )
            )
        elif include_corrected and tier != "elite":
            examples.append(
                SFTExample(
                    example_id=f"sft-{episode.episode_id}-corr",
                    kind="corrected",
                    prompt={
                        **base_prompt,
                        "original_topology": episode.original_topology,
                        "failure_summary": episode.validation_errors[:5],
                    },
                    target=episode.final_topology,
                    spec_group=_spec_group(episode.specifications),
                    tier=tier,
                    attribution_confidence=a.confidence,
                )
            )
    return examples


def _elite_edits() -> int:
    return 1


@dataclass
class DatasetSplits:
    train: list[SFTExample] = field(default_factory=list)
    val: list[SFTExample] = field(default_factory=list)
    test: list[SFTExample] = field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        return {
            "train": [e.example_id for e in self.train],
            "val": [e.example_id for e in self.val],
            "test": [e.example_id for e in self.test],
            "train_groups": sorted({e.spec_group for e in self.train}),
            "val_groups": sorted({e.spec_group for e in self.val}),
            "test_groups": sorted({e.spec_group for e in self.test}),
        }


def group_safe_split(
    examples: list[SFTExample], seed: int = 0, val_fraction: float = 0.15, test_fraction: float = 0.15
) -> DatasetSplits:
    """Split by specification group — related variants never cross splits."""
    groups = sorted({e.spec_group for e in examples})
    rng = Random(seed)
    rng.shuffle(groups)
    n_test = max(1, math.ceil(len(groups) * test_fraction)) if len(groups) > 2 else 0
    n_val = max(1, math.ceil(len(groups) * val_fraction)) if len(groups) > 2 else 0
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])
    splits = DatasetSplits()
    for example in examples:
        target = (
            splits.test if example.spec_group in test_groups
            else splits.val if example.spec_group in val_groups
            else splits.train
        )
        target.append(example)
    return splits


def serialize_example(example: SFTExample) -> tuple[str, str]:
    """(prompt_text, target_text) for teacher-forced autoregressive training."""
    return (
        json.dumps(example.prompt, sort_keys=True),
        json.dumps(example.target, sort_keys=True),
    )
