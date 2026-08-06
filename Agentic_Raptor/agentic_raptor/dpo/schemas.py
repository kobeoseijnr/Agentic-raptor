"""Structured candidate representation for pre-SPICE preference ranking.

Everything here is **pre-simulation** information for the current candidate:
the feature builder refuses candidates that already carry their own SPICE
result (leakage guard). Historical SPICE outcomes live in OutcomeRecord and
are used only for DPO *training*.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.utils.exceptions import AgenticRaptorError


class DPOLeakageError(AgenticRaptorError):
    """Raised when post-simulation data would leak into ranking inputs."""


@dataclass
class CandidateFeatures:
    """Pre-SPICE view of one topology-and-sizing candidate."""

    pool_candidate_id: str
    specifications: dict[str, Any]
    spec_embedding: list[float]
    topology: dict[str, Any]                 # serialized CircuitGraph
    topology_family: str
    topology_hash: str
    graph_features: list[float]
    edit_count: int
    validation_valid: bool
    validation_warnings: int
    sizing_vector: list[float]
    sac_policy_entropy: float | None = None
    predicted_margins: dict[str, float] = field(default_factory=dict)   # surrogate/dynamics
    predicted_feasibility: float = 0.0
    predicted_fom: float = 0.0
    predicted_pvt_robustness: float = 0.0
    uncertainty: float = 0.0
    repair_burden: float = 0.0
    novelty_score: float = 0.0
    #: fraction of the simulation budget still available (spec-conditioned
    #: ranking must weigh exploration differently when the budget is nearly
    #: spent)
    remaining_budget_frac: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def feature_vector(self) -> list[float]:
        margins = sorted(self.predicted_margins.items())
        worst = min((v for _k, v in margins), default=-1.0)
        mean = sum(v for _k, v in margins) / len(margins) if margins else -1.0
        return (
            self.spec_embedding
            + self.graph_features
            + [
                1.0 if self.validation_valid else 0.0,
                float(self.validation_warnings) / 5.0,
                float(self.edit_count) / 10.0,
                max(-2.0, min(2.0, worst)),
                max(-2.0, min(2.0, mean)),
                self.predicted_feasibility,
                self.predicted_fom,
                self.predicted_pvt_robustness,
                min(2.0, self.uncertainty),
                min(2.0, self.repair_burden),
                self.novelty_score,
                self.sac_policy_entropy if self.sac_policy_entropy is not None else 0.0,
                max(0.0, min(1.0, self.remaining_budget_frac)),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateFeatures:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def build_candidate_features(candidate: Any, **kwargs: Any) -> CandidateFeatures:
    """Leakage-guarded constructor from a CircuitCandidate.

    Rejects candidates that already carry their own simulation result: the
    ranker must never see the current candidate's SPICE/PVT outcome.
    """
    if getattr(candidate, "simulation_result", None) is not None:
        raise DPOLeakageError(
            "candidate already carries a SPICE result; pre-SPICE ranking may not see it"
        )
    from agentic_raptor.rag.memory import spec_embedding
    from agentic_raptor.topology_rl.policy_value_network import graph_feature_vector

    spec = candidate.specifications
    graph = candidate.topology
    validation = candidate.validation_result or {}
    return CandidateFeatures(
        pool_candidate_id=f"pool-{uuid.uuid4().hex[:10]}",
        specifications=spec.to_dict(),
        spec_embedding=spec_embedding(spec),
        topology=graph.to_dict(),
        topology_family=graph.metadata.circuit_family or spec.circuit_class,
        topology_hash=graph.structural_hash(),
        graph_features=graph_feature_vector(graph),
        edit_count=len(candidate.edit_history),
        validation_valid=bool(validation.get("is_valid")),
        validation_warnings=sum(
            1 for i in validation.get("issues", []) if i.get("severity") == "warning"
        ),
        sizing_vector=list(kwargs.pop("sizing_vector", [])),
        **kwargs,
    )


@dataclass
class OutcomeRecord:
    """Post-SPICE record for one selected candidate (training data + metrics)."""

    features: CandidateFeatures
    dpo_score: float | None
    dpo_rank: int | None
    selection_reason: str
    spice_success: bool
    passed_spec: bool
    metrics: dict[str, float] = field(default_factory=dict)
    constraint_margins: dict[str, float] = field(default_factory=dict)
    pvt_pass_rate: float | None = None
    fom: float | None = None
    runtime_s: float = 0.0
    spice_calls_total: int = 0
    calls_to_first_pass: int | None = None
    episode_id: str = ""

    @property
    def worst_margin(self) -> float:
        return min(self.constraint_margins.values(), default=-1.0)

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeRecord:
        data = dict(data)
        data["features"] = CandidateFeatures.from_dict(data["features"])
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
