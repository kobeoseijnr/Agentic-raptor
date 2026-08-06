"""Multimodal specification agent: modalities → canonical DesignSpecifications."""

from __future__ import annotations

from typing import Any

from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.specification.fusion import (
    FusionReport,
    fuse_fields,
    require_specification,
)
from agentic_raptor.specification.multimodal_input import MultimodalDesignInput
from agentic_raptor.specification.parser import (
    ExtractedField,
    MockSchematicImageParser,
    SchematicImageParser,
    parse_netlist,
    parse_structured,
    parse_table,
    parse_text,
)
from agentic_raptor.specification.validator import SpecificationReview, review_specification

__all__ = [
    "DesignSpecifications",
    "ExtractedField",
    "FusionReport",
    "MockSchematicImageParser",
    "MultimodalDesignInput",
    "SchematicImageParser",
    "SpecificationReview",
    "parse_design_input",
    "review_specification",
]


def parse_design_input(
    design_input: MultimodalDesignInput,
    image_parser: SchematicImageParser | None = None,
    defaults: dict[str, Any] | None = None,
) -> tuple[DesignSpecifications, FusionReport, SpecificationReview]:
    """End-to-end specification agent: parse every provided modality, fuse, review.

    Raises :class:`SpecificationError` when no modality is given, a file is
    missing, or required fields cannot be resolved.
    """
    design_input.validate()
    extracted: list[ExtractedField] = []
    if design_input.structured_specification_path:
        extracted += parse_structured(design_input.structured_specification_path)
    if design_input.table_path:
        extracted += parse_table(design_input.table_path)
    if design_input.netlist_path:
        extracted += parse_netlist(design_input.netlist_path)
    if design_input.text:
        extracted += parse_text(design_input.text)
    if design_input.schematic_image_path:
        parser = image_parser or MockSchematicImageParser()
        extracted += parser.parse(design_input.schematic_image_path)

    spec, report = fuse_fields(extracted, defaults=defaults)
    spec = require_specification(spec, report)
    review = review_specification(spec)
    return spec, report, review
