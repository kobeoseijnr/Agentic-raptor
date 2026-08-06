"""Multimodal context passed to the topology generator.

Re-exports the canonical container from the specification package (the LLM
may receive the original schematic image or netlist as extra context) and
builds compact provider-neutral context summaries.
"""

from __future__ import annotations

from pathlib import Path

from agentic_raptor.specification.multimodal_input import MultimodalDesignInput

__all__ = ["MultimodalDesignInput", "build_context_summary"]

_MAX_NETLIST_CHARS = 2000


def build_context_summary(context: MultimodalDesignInput | None) -> dict[str, str]:
    """Textual summary of the multimodal context for prompt construction.

    Image content is NOT decoded here — the image path is surfaced so a
    multimodal provider adapter can attach the actual bytes.
    """
    if context is None:
        return {}
    out: dict[str, str] = {}
    if context.text:
        out["design_brief"] = context.text.strip()
    if context.netlist_path and Path(context.netlist_path).is_file():
        netlist = Path(context.netlist_path).read_text(encoding="utf-8", errors="replace")
        if len(netlist) > _MAX_NETLIST_CHARS:
            netlist = netlist[:_MAX_NETLIST_CHARS] + "\n* [truncated]"
        out["reference_netlist"] = netlist
    if context.schematic_image_path:
        out["schematic_image_path"] = str(context.schematic_image_path)
    return out
