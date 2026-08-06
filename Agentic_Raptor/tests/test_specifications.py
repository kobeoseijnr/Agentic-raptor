"""Specification model validation."""

from __future__ import annotations

import pytest

from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.utils.exceptions import SpecificationError


def test_valid_specification_roundtrip(spec):
    data = spec.to_dict()
    restored = DesignSpecifications.from_dict(data)
    assert restored.circuit_class == "ota"
    assert restored.supply_voltage == pytest.approx(1.8)
    assert restored.feature_vector() == spec.feature_vector()


@pytest.mark.parametrize(
    "overrides",
    [
        {"supply_voltage": -1.0},
        {"supply_voltage": 500.0},
        {"temperature_c": -400.0},
        {"target_gbw_hz": -5.0},
        {"minimum_phase_margin_deg": 200.0},
        {"maximum_power_w": 0.0},
        {"minimum_output_swing_v": 5.0},   # > supply
        {"common_mode_input_v": 9.9},      # > supply
        {"circuit_class": "  "},
    ],
)
def test_physically_meaningless_values_rejected(overrides):
    base = {"circuit_class": "ota", "technology": "180nm", "supply_voltage": 1.8}
    base.update(overrides)
    with pytest.raises(SpecificationError):
        DesignSpecifications(**base)


def test_feature_vector_length_stable(spec):
    assert len(spec.feature_vector()) == len(DesignSpecifications.NUMERIC_FIELDS)


def test_source_metadata_carried(spec):
    tagged = DesignSpecifications(
        circuit_class="ota",
        technology="180nm",
        supply_voltage=1.8,
        source_metadata={"supply_voltage": "structured"},
    )
    assert tagged.to_dict()["source_metadata"]["supply_voltage"] == "structured"
