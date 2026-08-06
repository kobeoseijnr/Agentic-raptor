"""Deterministic rule-based topology validator."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.topology_validation import connectivity, rules
from agentic_raptor.topology_validation.graph_hash import compute_graph_hash
from agentic_raptor.topology_validation.rules import ValidationIssue


@dataclass
class ValidationResult:
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    graph_hash: str | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict:
        from agentic_raptor.utils.serialization import to_jsonable

        return {
            "is_valid": self.is_valid,
            "issues": to_jsonable(self.issues),
            "graph_hash": self.graph_hash,
        }

    #: Compact numeric features for policy/value network input.
    def feature_vector(self) -> list[float]:
        return [1.0 if self.is_valid else 0.0, float(len(self.errors)), float(len(self.warnings))]


class TopologyValidator:
    """Composes deterministic rules; returns structured errors and warnings.

    Duplicate node IDs and duplicate terminal connections are rejected at
    graph-construction time (:class:`CircuitGraph` raises ``GraphError``);
    unsupported device types are rejected by the :class:`DeviceType` enum.
    The validator therefore re-checks only what can go wrong on a graph that
    was successfully constructed.
    """

    def __init__(self, max_nodes: int = 200, max_edges: int = 800) -> None:
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    def validate(self, graph: CircuitGraph) -> ValidationResult:
        issues: list[ValidationIssue] = []
        issues += rules.rule_non_empty(graph)
        if not issues:  # remaining rules assume a non-empty graph
            issues += rules.rule_required_ports(graph)
            issues += rules.rule_valid_terminals(graph)
            issues += rules.rule_no_floating_devices(graph)
            issues += rules.rule_dc_floating_nets(graph)
            issues += rules.rule_supply_ground_short(graph)
            issues += rules.rule_max_graph_size(graph, self.max_nodes, self.max_edges)
            issues += connectivity.rule_output_on_active_path(graph)
            issues += connectivity.rule_disconnected_islands(graph)

        is_valid = not any(i.severity == "error" for i in issues)
        graph_hash = compute_graph_hash(graph) if graph.nodes else None
        return ValidationResult(is_valid=is_valid, issues=issues, graph_hash=graph_hash)
