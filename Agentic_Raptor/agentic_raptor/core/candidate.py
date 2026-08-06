"""Circuit candidate: one topology + sizing + evaluation lifecycle record."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.core.types import CandidateStatus, GenerationSource


def new_candidate_id() -> str:
    return f"cand-{uuid.uuid4().hex[:12]}"


@dataclass
class CircuitCandidate:
    """A candidate design flowing through the agentic pipeline.

    ``validation_result`` / ``simulation_result`` hold the ``to_dict()`` form of
    the corresponding structured results (kept as dicts here to avoid circular
    imports between core and the validation/spice packages).
    """

    candidate_id: str
    topology: CircuitGraph
    specifications: DesignSpecifications
    generation_source: GenerationSource
    parent_candidate_id: str | None = None
    edit_history: list[dict[str, Any]] = field(default_factory=list)
    sizing_state: dict[str, dict[str, float]] = field(default_factory=dict)
    validation_result: dict[str, Any] | None = None
    simulation_result: dict[str, Any] | None = None
    reward: float | None = None
    status: CandidateStatus = CandidateStatus.GENERATED
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        topology: CircuitGraph,
        specifications: DesignSpecifications,
        generation_source: GenerationSource,
        parent_candidate_id: str | None = None,
        **metadata: Any,
    ) -> CircuitCandidate:
        return cls(
            candidate_id=new_candidate_id(),
            topology=topology,
            specifications=specifications,
            generation_source=generation_source,
            parent_candidate_id=parent_candidate_id,
            metadata=dict(metadata),
        )

    def fork(self, generation_source: GenerationSource = GenerationSource.EDITED) -> CircuitCandidate:
        """Child candidate with a copied topology, linked to this one."""
        return CircuitCandidate(
            candidate_id=new_candidate_id(),
            topology=self.topology.copy(),
            specifications=self.specifications,
            generation_source=generation_source,
            parent_candidate_id=self.candidate_id,
            edit_history=list(self.edit_history),
            sizing_state={k: dict(v) for k, v in self.sizing_state.items()},
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return {
            "candidate_id": self.candidate_id,
            "parent_candidate_id": self.parent_candidate_id,
            "topology": self.topology.to_dict(),
            "specifications": self.specifications.to_dict(),
            "generation_source": self.generation_source.value,
            "edit_history": to_jsonable(self.edit_history),
            "sizing_state": to_jsonable(self.sizing_state),
            "validation_result": to_jsonable(self.validation_result),
            "simulation_result": to_jsonable(self.simulation_result),
            "reward": self.reward,
            "status": self.status.value,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CircuitCandidate:
        return cls(
            candidate_id=str(data["candidate_id"]),
            parent_candidate_id=data.get("parent_candidate_id"),
            topology=CircuitGraph.from_dict(data["topology"]),
            specifications=DesignSpecifications.from_dict(data["specifications"]),
            generation_source=GenerationSource(data["generation_source"]),
            edit_history=list(data.get("edit_history") or []),
            sizing_state={k: dict(v) for k, v in (data.get("sizing_state") or {}).items()},
            validation_result=data.get("validation_result"),
            simulation_result=data.get("simulation_result"),
            reward=data.get("reward"),
            status=CandidateStatus(data.get("status", CandidateStatus.GENERATED.value)),
            metadata=dict(data.get("metadata") or {}),
        )
