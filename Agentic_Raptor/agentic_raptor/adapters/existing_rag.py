"""Guarded access to the legacy RAG memory (read-only)."""

from __future__ import annotations

from typing import Any

from agentic_raptor.adapters.legacy_raptor import ensure_repo_root_on_path, legacy_available
from agentic_raptor.rag.adapters import memory_entry_from_legacy


def is_available() -> bool:
    return legacy_available("rag.memory_schema")


def legacy_memory_item_class() -> Any:
    ensure_repo_root_on_path()
    from rag.memory_schema import RagMemoryItem  # noqa: PLC0415

    return RagMemoryItem


def convert_legacy_item(item: Any) -> Any:
    """Legacy RagMemoryItem → new MemoryEntry (no writes to legacy data)."""
    return memory_entry_from_legacy(item)
