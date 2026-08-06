"""Conversion between the new MemoryEntry and the legacy RagMemoryItem."""

from __future__ import annotations

from typing import Any

from agentic_raptor.adapters.legacy_raptor import ensure_repo_root_on_path, legacy_available
from agentic_raptor.rag.schemas import MemoryEntry


def legacy_rag_available() -> bool:
    return legacy_available("rag.memory_schema")


def memory_entry_from_legacy(item: Any) -> MemoryEntry:
    """Map a legacy ``rag.memory_schema.RagMemoryItem`` into a MemoryEntry.

    Only fields both schemas share are mapped; the legacy item is not modified.
    """
    specs = dict(item.target_specs_json or {})
    specs.setdefault("circuit_class", item.circuit_family or "unknown")
    specs.setdefault("technology", item.technology_node or "unknown")
    if item.supply_voltage is not None:
        specs.setdefault("supply_voltage", item.supply_voltage)
    return MemoryEntry.create(
        specifications=specs,
        sizing_values={"legacy": dict(item.parameter_vector_json or {})} if item.parameter_vector_json else {},
        spice_metrics={
            k: float(v) for k, v in (item.metrics_json or {}).items() if isinstance(v, (int, float))
        },
        success=item.pass_flag if item.pass_flag is not None else item.spice_success,
        generation_metadata={
            "origin": "legacy_rag",
            "legacy_memory_id": item.memory_id,
            "legacy_memory_type": item.memory_type,
            "netlist_text_present": bool(item.netlist_text),
        },
    )


def import_legacy_items(memory: Any, items: list[Any]) -> int:
    """Bulk-import legacy items into a CircuitMemory. Returns count imported."""
    ensure_repo_root_on_path()
    count = 0
    for item in items:
        memory.add(memory_entry_from_legacy(item))
        count += 1
    return count
