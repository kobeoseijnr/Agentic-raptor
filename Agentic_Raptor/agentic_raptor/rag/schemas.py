"""Structured circuit-design experience entries (not a text-document store)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.utils.logging import utc_now_iso


def new_memory_id() -> str:
    return f"mem-{uuid.uuid4().hex[:12]}"


@dataclass
class MemoryEntry:
    """One complete engineering experience record."""

    memory_id: str
    specifications: dict[str, Any]                    # serialized DesignSpecifications
    topology: dict[str, Any] | None = None            # serialized CircuitGraph
    reusable_blocks: list[dict[str, Any]] = field(default_factory=list)
    sizing_values: dict[str, dict[str, float]] = field(default_factory=dict)
    edit_history: list[dict[str, Any]] = field(default_factory=list)
    spice_metrics: dict[str, float] = field(default_factory=dict)
    pvt_results: dict[str, Any] | None = None
    reward: float | None = None
    fom: float | None = None
    success: bool | None = None
    failure_reason: str | None = None
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    spec_embedding: list[float] = field(default_factory=list)
    topology_embedding: list[float] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def create(cls, specifications: dict[str, Any], **kwargs: Any) -> MemoryEntry:
        return cls(memory_id=new_memory_id(), specifications=specifications, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
