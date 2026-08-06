"""In-memory circuit experience store with weighted-similarity retrieval.

Deliberately simple for the first stage (per the plan: no heavy database).
JSONL persistence keeps runs reproducible; the interface is what matters —
``add`` / ``retrieve`` / ``update_outcome``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.rag.schemas import MemoryEntry


def spec_embedding(spec: DesignSpecifications) -> list[float]:
    """Deterministic normalized spec embedding (shared with retrieval)."""
    from agentic_raptor.topology_rl.policy_value_network import _spec_embedding

    return _spec_embedding(spec)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class RetrievalWeights:
    specification: float = 1.0
    circuit_class: float = 1.0
    technology: float = 0.5
    supply: float = 0.5
    success_bonus: float = 0.3
    failure_relevance: float = 0.2


@dataclass
class ScoredEntry:
    entry: MemoryEntry
    score: float
    parts: dict[str, float]


class CircuitMemory:
    """Structured engineering memory: add / retrieve / update_outcome."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.is_file():
            with self.persist_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = MemoryEntry.from_dict(json.loads(line))
                        self._entries[entry.memory_id] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: MemoryEntry) -> str:
        self._entries[entry.memory_id] = entry
        self._persist()
        return entry.memory_id

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self._entries.get(memory_id)

    def update_outcome(self, memory_id: str, **fields: object) -> bool:
        """Write back episode outcomes (reward, metrics, failure reason, ...)."""
        entry = self._entries.get(memory_id)
        if entry is None:
            return False
        for name, value in fields.items():
            if hasattr(entry, name):
                setattr(entry, name, value)
        self._persist()
        return True

    def retrieve(
        self,
        spec: DesignSpecifications,
        k: int = 5,
        include_failures: bool = True,
        weights: RetrievalWeights | None = None,
    ) -> list[ScoredEntry]:
        """Top-k entries by weighted similarity to the query specification."""
        w = weights or RetrievalWeights()
        query_embedding = spec_embedding(spec)
        scored: list[ScoredEntry] = []
        for entry in self._entries.values():
            if not include_failures and entry.success is False:
                continue
            parts: dict[str, float] = {}
            parts["specification"] = w.specification * _cosine(query_embedding, entry.spec_embedding)
            entry_class = str(entry.specifications.get("circuit_class", ""))
            parts["circuit_class"] = w.circuit_class * (1.0 if entry_class == spec.circuit_class else 0.0)
            entry_tech = str(entry.specifications.get("technology", ""))
            parts["technology"] = w.technology * (1.0 if entry_tech == spec.technology else 0.0)
            entry_vdd = entry.specifications.get("supply_voltage")
            if isinstance(entry_vdd, (int, float)) and spec.supply_voltage > 0:
                proximity = max(0.0, 1.0 - abs(float(entry_vdd) - spec.supply_voltage) / spec.supply_voltage)
                parts["supply"] = w.supply * proximity
            if entry.success is True:
                parts["success_bonus"] = w.success_bonus
            elif entry.success is False and entry.failure_reason:
                parts["failure_relevance"] = w.failure_relevance
            scored.append(ScoredEntry(entry, sum(parts.values()), parts))
        scored.sort(key=lambda s: (-s.score, s.entry.memory_id))
        return scored[: max(k, 0)]

    def _persist(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("w", encoding="utf-8") as f:
            for entry in self._entries.values():
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
