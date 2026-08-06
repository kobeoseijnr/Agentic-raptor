"""Typed container for multimodal design inputs.

At least one modality must be provided; none are individually required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_raptor.utils.exceptions import SpecificationError

#: Provenance labels, ordered by default fusion priority (highest first).
MODALITY_PRIORITY: tuple[str, ...] = ("structured", "table", "netlist", "text", "image")


@dataclass
class MultimodalDesignInput:
    """One design request, possibly spread over several modalities."""

    text: str | None = None
    structured_specification_path: str | None = None
    table_path: str | None = None
    schematic_image_path: str | None = None
    netlist_path: str | None = None

    def available_modalities(self) -> list[str]:
        out: list[str] = []
        if self.structured_specification_path:
            out.append("structured")
        if self.table_path:
            out.append("table")
        if self.netlist_path:
            out.append("netlist")
        if self.text:
            out.append("text")
        if self.schematic_image_path:
            out.append("image")
        return out

    def validate(self) -> None:
        """At least one modality present; referenced files must exist."""
        if not self.available_modalities():
            raise SpecificationError("at least one input modality must be provided")
        for label, path in (
            ("structured_specification_path", self.structured_specification_path),
            ("table_path", self.table_path),
            ("schematic_image_path", self.schematic_image_path),
            ("netlist_path", self.netlist_path),
        ):
            if path is not None and not Path(path).is_file():
                raise SpecificationError(f"{label} does not exist: {path}")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "text": self.text,
            "structured_specification_path": self.structured_specification_path,
            "table_path": self.table_path,
            "schematic_image_path": self.schematic_image_path,
            "netlist_path": self.netlist_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | None]) -> MultimodalDesignInput:
        return cls(
            text=data.get("text"),
            structured_specification_path=data.get("structured_specification_path"),
            table_path=data.get("table_path"),
            schematic_image_path=data.get("schematic_image_path"),
            netlist_path=data.get("netlist_path"),
        )
