"""Provider-backed schematic-image understanding.

Implements the Stage 1 ``SchematicImageParser`` protocol with a real
multimodal model behind it. Output is *multimodal topology context* — hints,
not ground truth: ambiguous elements are returned as ``unresolved`` rather
than invented, confidence rides along, and image-derived fields keep the
lowest fusion priority so they can never override explicit user values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_raptor.specification.parser import ExtractedField
from agentic_raptor.topology_generation.provider import MultimodalTopologyModel, ProviderError
from agentic_raptor.utils.exceptions import SpecificationError

_EXTRACTION_PROMPT = """\
You are an analog circuit schematic reader. Analyse the attached schematic image.
Return ONLY a JSON object with this exact shape (omit nothing; use null/[] when unknown):
{
  "circuit_class": string|null,          // e.g. "ota", "opamp", "comparator", "ldo"
  "devices": [{"label": string, "type": string}],   // type in: NMOS,PMOS,RESISTOR,CAPACITOR,CURRENT_SOURCE,VOLTAGE_SOURCE
  "blocks": [string],                    // e.g. "diff_pair", "current_mirror", "output_stage"
  "ports": {"inputs": [string], "outputs": [string], "supply": string|null, "ground": string|null},
  "component_values": [{"label": string, "value": string}],   // only values printed on the schematic
  "supply_label": string|null,           // e.g. "VDD=1.8V" only if visibly printed
  "connectivity_notes": [string],        // approximate, human-readable
  "topology_hints": [string],
  "confidence": number,                  // 0..1 overall
  "unresolved": [string]                 // elements you could NOT identify — do NOT guess
}
Do not invent connections or values that are not visible."""


@dataclass
class SchematicParseResult:
    fields: list[ExtractedField] = field(default_factory=list)
    hints: dict[str, Any] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)


@dataclass
class ProviderSchematicParser:
    """Real image parser. Satisfies the ``SchematicImageParser`` protocol."""

    model: MultimodalTopologyModel
    last_result: SchematicParseResult | None = None

    def parse(self, image_path: str) -> list[ExtractedField]:
        return self.parse_rich(image_path).fields

    def parse_rich(self, image_path: str) -> SchematicParseResult:
        path = Path(image_path)
        if not path.is_file():
            raise SpecificationError(f"schematic image not found: {image_path}")
        image_bytes = path.read_bytes()
        try:
            raw = self.model.generate_structured(_EXTRACTION_PROMPT, [image_bytes], {})
        except ProviderError as exc:
            raise SpecificationError(f"schematic parsing failed: {exc}") from exc

        confidence = float(raw.get("confidence") or 0.0)
        result = SchematicParseResult(
            hints={
                "devices": raw.get("devices") or [],
                "blocks": raw.get("blocks") or [],
                "ports": raw.get("ports") or {},
                "component_values": raw.get("component_values") or [],
                "connectivity_notes": raw.get("connectivity_notes") or [],
                "topology_hints": raw.get("topology_hints") or [],
            },
            unresolved=[str(u) for u in (raw.get("unresolved") or [])],
            confidence=confidence,
        )
        # Only fields the image can legitimately contribute become ExtractedFields;
        # numeric design targets stay with explicit modalities. Supply voltage is
        # extracted ONLY from a visibly printed label.
        if raw.get("circuit_class"):
            result.fields.append(
                ExtractedField(
                    "circuit_class", str(raw["circuit_class"]).lower(), "image",
                    detail=path.name, confidence=confidence,
                )
            )
        supply_label = raw.get("supply_label")
        if isinstance(supply_label, str) and "=" in supply_label:
            from agentic_raptor.specification.parser import parse_engineering_value

            value = parse_engineering_value(supply_label.split("=", 1)[1])
            if value is not None:
                result.fields.append(
                    ExtractedField(
                        "supply_voltage", value, "image",
                        detail=f"printed label {supply_label!r}", confidence=confidence * 0.8,
                    )
                )
        self.last_result = result
        return result

    def context_summary(self) -> dict[str, str]:
        """Feed the last parse into topology-generation prompts."""
        if self.last_result is None:
            return {}
        return {"schematic_analysis": json.dumps(self.last_result.to_dict(), indent=2)}
