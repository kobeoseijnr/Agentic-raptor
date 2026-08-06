"""Retrieval facade + seed corpus for the smoke pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.rag.memory import CircuitMemory, ScoredEntry, spec_embedding
from agentic_raptor.rag.schemas import MemoryEntry


@dataclass
class RetrievalResult:
    entries: list[ScoredEntry]

    @property
    def memory_ids(self) -> list[str]:
        return [s.entry.memory_id for s in self.entries]

    def successes(self) -> list[MemoryEntry]:
        return [s.entry for s in self.entries if s.entry.success is True]

    def failures(self) -> list[MemoryEntry]:
        return [s.entry for s in self.entries if s.entry.success is False]

    def to_dict(self) -> dict:
        return {
            "memory_ids": self.memory_ids,
            "scores": {s.entry.memory_id: s.score for s in self.entries},
        }


class Retriever:
    def __init__(self, memory: CircuitMemory, k: int = 5, include_failures: bool = True) -> None:
        self.memory = memory
        self.k = k
        self.include_failures = include_failures

    def retrieve(self, spec: DesignSpecifications) -> RetrievalResult:
        return RetrievalResult(
            self.memory.retrieve(spec, k=self.k, include_failures=self.include_failures)
        )


def seed_memory_for_smoke(memory: CircuitMemory, spec: DesignSpecifications) -> list[str]:
    """Populate a few deterministic experience entries so retrieval is non-trivial.

    One matching success, one different-class success, one matching failure.
    """
    from agentic_raptor.topology_generation.generator import MockTopologyGenerator

    generator = MockTopologyGenerator(seed=7)
    graphs = generator.generate(spec, [], None, number_of_candidates=1)
    base_spec = spec.to_dict()

    ids: list[str] = []
    success = MemoryEntry.create(
        specifications=base_spec,
        topology=graphs[0].to_dict(),
        sizing_values={"m1": {"width_m": 4e-6, "length_m": 0.2e-6}},
        spice_metrics={"gain_db": 72.0, "gbw_hz": 9.5e7, "phase_margin_deg": 61.0, "power_w": 4.2e-4},
        reward=1.1,
        fom=180.0,
        success=True,
        generation_metadata={"origin": "seed_corpus"},
        spec_embedding=spec_embedding(spec),
    )
    ids.append(memory.add(success))

    other = dict(base_spec)
    other["circuit_class"] = "comparator"
    ids.append(
        memory.add(
            MemoryEntry.create(
                specifications=other,
                spice_metrics={"gain_db": 55.0},
                reward=0.4,
                success=True,
                generation_metadata={"origin": "seed_corpus"},
                spec_embedding=list(reversed(spec_embedding(spec))),
            )
        )
    )

    failure = MemoryEntry.create(
        specifications=base_spec,
        topology=graphs[0].to_dict(),
        spice_metrics={"gain_db": 38.0, "phase_margin_deg": 22.0},
        reward=-0.6,
        success=False,
        failure_reason="phase margin collapse without compensation capacitor",
        generation_metadata={"origin": "seed_corpus"},
        spec_embedding=spec_embedding(spec),
    )
    ids.append(memory.add(failure))
    return ids
