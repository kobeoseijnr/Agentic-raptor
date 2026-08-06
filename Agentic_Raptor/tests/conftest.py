"""Shared fixtures. Adds the package root to sys.path so tests run from anywhere."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from agentic_raptor.core.specifications import DesignSpecifications  # noqa: E402
from agentic_raptor.topology_generation.generator import build_five_transistor_ota  # noqa: E402


@pytest.fixture
def spec() -> DesignSpecifications:
    return DesignSpecifications(
        circuit_class="ota",
        technology="180nm",
        supply_voltage=1.8,
        target_gain_db=70.0,
        target_gbw_hz=5e7,
        minimum_phase_margin_deg=60.0,
        maximum_power_w=1e-3,
        load_capacitance_f=1e-12,
    )


@pytest.fixture
def ota_graph():
    return build_five_transistor_ota("test-ota")


@pytest.fixture
def inputs_dir() -> Path:
    return _PKG_ROOT / "configs" / "experiments" / "inputs"
