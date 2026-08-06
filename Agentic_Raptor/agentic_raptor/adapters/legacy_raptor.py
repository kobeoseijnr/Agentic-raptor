"""Read-only bridge to the legacy RAPTOR repository.

All legacy access flows through here so that:
* the repo root is added to ``sys.path`` lazily and exactly once;
* availability is probed without hard import failures;
* nothing in the legacy tree is ever written to.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: Agentic_Raptor/agentic_raptor/adapters/legacy_raptor.py → repository root,
#: then into the separated legacy project. This is the SINGLE point of legacy
#: path discovery for every adapter (existing_mb_sac/spice/rag/surrogate,
#: legacy_netlist, sizing.adapters).
_REPO_ROOT = Path(__file__).resolve().parents[3] / "RAPTOR_Legacy"

#: Legacy top-level packages we may import from (read-only).
LEGACY_PACKAGES = (
    "mb_sac",
    "rag",
    "llm",
    "graph",
    "graph_search",
    "controller",
    "surrogate",
    "dpo",
    "topology_dpo",
)


def repo_root() -> Path:
    return _REPO_ROOT


def ensure_repo_root_on_path() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def legacy_available(module_name: str) -> bool:
    """True when the legacy module can be found (without importing it fully)."""
    top = module_name.split(".")[0]
    if not (_REPO_ROOT / top).is_dir():
        return False
    ensure_repo_root_on_path()
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def legacy_inventory() -> dict[str, bool]:
    """Availability map of the legacy packages (used by the CLI inspect command)."""
    return {name: legacy_available(name) for name in LEGACY_PACKAGES}
