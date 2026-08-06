"""Guarded access to the legacy surrogate ensemble (sizing-outcome estimates)."""

from __future__ import annotations

from typing import Any

from agentic_raptor.adapters.legacy_raptor import ensure_repo_root_on_path, legacy_available


def is_available() -> bool:
    return legacy_available("surrogate.surrogate_ensemble")


def surrogate_ensemble_class() -> Any:
    ensure_repo_root_on_path()
    from surrogate.surrogate_ensemble import SurrogateEnsemble  # noqa: PLC0415

    return SurrogateEnsemble
