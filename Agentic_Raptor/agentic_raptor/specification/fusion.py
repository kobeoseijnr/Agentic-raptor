"""Fusion of per-modality extracted fields into one canonical specification.

Behaviour:
* combines fields from all modalities;
* detects conflicting values (relative disagreement above tolerance);
* resolves conflicts by modality priority (structured > table > netlist > text > image);
* records the winning source per field in ``DesignSpecifications.source_metadata``;
* reports missing required fields — it never silently invents values
  (explicit ``defaults`` passed by the caller are recorded with source="default").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.specification.multimodal_input import MODALITY_PRIORITY
from agentic_raptor.specification.parser import ExtractedField
from agentic_raptor.utils.exceptions import SpecificationError

#: Fields that must be present after fusion (others may stay None).
REQUIRED_FIELDS: tuple[str, ...] = ("circuit_class", "technology", "supply_voltage")

_STRING_FIELDS = {"circuit_class", "technology"}


@dataclass
class SpecificationConflict:
    field_name: str
    values_by_source: dict[str, Any]
    resolved_value: Any
    resolved_source: str


@dataclass
class FusionReport:
    fields_used: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, str] = field(default_factory=dict)
    conflicts: list[SpecificationConflict] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    extracted_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)


def _values_conflict(a: Any, b: Any, rel_tol: float) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return str(a).strip().lower() != str(b).strip().lower()
    fa, fb = float(a), float(b)
    scale = max(abs(fa), abs(fb), 1e-30)
    return abs(fa - fb) / scale > rel_tol


def fuse_fields(
    extracted: list[ExtractedField],
    defaults: dict[str, Any] | None = None,
    rel_tolerance: float = 0.02,
    priority: tuple[str, ...] = MODALITY_PRIORITY,
) -> tuple[DesignSpecifications | None, FusionReport]:
    """Fuse extracted fields; returns (specifications | None, report).

    Returns ``None`` for the specification when required fields are missing —
    the report then lists them in ``missing_required``.
    """
    report = FusionReport(extracted_count=len(extracted))
    rank = {source: i for i, source in enumerate(priority)}

    by_name: dict[str, list[ExtractedField]] = {}
    for f in extracted:
        by_name.setdefault(f.name, []).append(f)

    resolved: dict[str, Any] = {}
    for name, candidates in sorted(by_name.items()):
        ordered = sorted(candidates, key=lambda f: (rank.get(f.source, len(rank)), -f.confidence))
        winner = ordered[0]
        distinct: dict[str, Any] = {}
        for f in ordered:
            if all(_values_conflict(f.value, v, rel_tolerance) for v in distinct.values()) or not distinct:
                distinct.setdefault(f.source, f.value)
        if len(distinct) > 1:
            report.conflicts.append(
                SpecificationConflict(
                    field_name=name,
                    values_by_source=distinct,
                    resolved_value=winner.value,
                    resolved_source=winner.source,
                )
            )
        resolved[name] = winner.value
        report.source_metadata[name] = winner.source

    for name, value in (defaults or {}).items():
        if name not in resolved:
            resolved[name] = value
            report.source_metadata[name] = "default"

    report.missing_required = [name for name in REQUIRED_FIELDS if name not in resolved]
    report.fields_used = dict(resolved)
    if report.missing_required:
        return None, report

    numeric = {k: v for k, v in resolved.items() if k not in _STRING_FIELDS}
    spec = DesignSpecifications(
        circuit_class=str(resolved["circuit_class"]),
        technology=str(resolved["technology"]),
        supply_voltage=float(resolved["supply_voltage"]),
        source_metadata=dict(report.source_metadata),
        **{k: float(v) for k, v in numeric.items() if k != "supply_voltage"},
    )
    return spec, report


def require_specification(
    spec: DesignSpecifications | None, report: FusionReport
) -> DesignSpecifications:
    """Raise a precise error when fusion could not produce a specification."""
    if spec is None:
        raise SpecificationError(
            "specification fusion failed; missing required fields: "
            + ", ".join(report.missing_required)
            + f" (extracted {report.extracted_count} fields total)"
        )
    return spec
