"""Guarded access to the legacy SPICE execution sites.

Legacy execution: ``controller.module_adapters.run_spice_validation`` (exported
netlist path + ngspice exe + timeout) and ``mb_sac/sizing_environment.py``.
Full wiring needs typed-graph → legacy-netlist export (stage 2); this module
exposes availability checks and the callable so stage 2 is one function away.
"""

from __future__ import annotations

from typing import Any

from agentic_raptor.adapters.legacy_raptor import ensure_repo_root_on_path, legacy_available


def is_available() -> bool:
    return legacy_available("controller.module_adapters")


def get_run_spice_validation() -> Any:
    """Return the legacy ``run_spice_validation`` callable (imports lazily).

    Note: the legacy module reads hard-coded ``results/...`` paths in places;
    treat every artifact it touches as read-only.
    """
    ensure_repo_root_on_path()
    from controller.module_adapters import run_spice_validation  # noqa: PLC0415

    return run_spice_validation


def get_netlist_exporter() -> Any:
    """Legacy graph→netlist exporter used to feed ngspice (stage-2 bridge)."""
    ensure_repo_root_on_path()
    import graph.export_graph_to_netlist as exporter  # noqa: PLC0415

    return exporter
