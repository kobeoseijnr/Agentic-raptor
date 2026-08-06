"""Multimodal input handling, per-modality parsing, fusion, conflicts."""

from __future__ import annotations

import pytest

from agentic_raptor.specification import (
    MultimodalDesignInput,
    parse_design_input,
)
from agentic_raptor.specification.fusion import fuse_fields
from agentic_raptor.specification.parser import (
    ExtractedField,
    MockSchematicImageParser,
    parse_engineering_value,
    parse_netlist,
    parse_structured,
    parse_table,
    parse_text,
)
from agentic_raptor.utils.exceptions import SpecificationError


def test_at_least_one_modality_required():
    with pytest.raises(SpecificationError):
        MultimodalDesignInput().validate()


def test_missing_file_rejected():
    with pytest.raises(SpecificationError):
        MultimodalDesignInput(structured_specification_path="does/not/exist.yaml").validate()


def test_engineering_value_parsing():
    assert parse_engineering_value("100 MHz") == pytest.approx(1e8)
    assert parse_engineering_value("1.8V") == pytest.approx(1.8)
    assert parse_engineering_value("500uW") == pytest.approx(5e-4)
    assert parse_engineering_value("2p") == pytest.approx(2e-12)
    assert parse_engineering_value("80 dB") == pytest.approx(80.0)
    assert parse_engineering_value("not a value") is None


def test_text_parsing_extracts_specs():
    text = (
        "Design a low-power OTA in 180nm. Gain >= 80 dB, GBW >= 100 MHz, "
        "phase margin >= 60 deg, supply 1.8 V, power budget 1 mW, load cap 1 pF, "
        "slew rate 20 V/us."
    )
    fields = {f.name: f.value for f in parse_text(text)}
    assert fields["circuit_class"] == "ota"
    assert fields["technology"] == "180nm"
    assert fields["target_gain_db"] == pytest.approx(80.0)
    assert fields["target_gbw_hz"] == pytest.approx(1e8)
    assert fields["minimum_phase_margin_deg"] == pytest.approx(60.0)
    assert fields["supply_voltage"] == pytest.approx(1.8)
    assert fields["maximum_power_w"] == pytest.approx(1e-3)
    assert fields["load_capacitance_f"] == pytest.approx(1e-12)
    assert fields["minimum_slew_rate_v_per_s"] == pytest.approx(2e7)


def test_structured_parsing(inputs_dir):
    fields = {f.name: f.value for f in parse_structured(inputs_dir / "smoke_spec.yaml")}
    assert fields["circuit_class"] == "ota"
    assert fields["supply_voltage"] == pytest.approx(1.8)
    assert fields["target_gbw_hz"] == pytest.approx(6e7)


def test_table_parsing(inputs_dir):
    fields = {f.name: f.value for f in parse_table(inputs_dir / "smoke_table.csv")}
    assert fields["target_gain_db"] == pytest.approx(70.0)
    assert fields["target_gbw_hz"] == pytest.approx(5.5e7)
    assert fields["minimum_slew_rate_v_per_s"] == pytest.approx(2e7)


def test_netlist_parsing(inputs_dir):
    fields = {f.name: f.value for f in parse_netlist(inputs_dir / "smoke_reference.sp")}
    assert fields["supply_voltage"] == pytest.approx(1.8)
    assert fields["load_capacitance_f"] == pytest.approx(1e-12)
    assert fields["circuit_class"] == "ota"


def test_fusion_priority_and_conflicts():
    extracted = [
        ExtractedField("supply_voltage", 1.8, "structured"),
        ExtractedField("supply_voltage", 3.3, "text"),
        ExtractedField("circuit_class", "ota", "text"),
        ExtractedField("technology", "180nm", "text"),
    ]
    spec, report = fuse_fields(extracted)
    assert spec is not None
    assert spec.supply_voltage == pytest.approx(1.8)  # structured wins
    assert len(report.conflicts) == 1
    assert report.conflicts[0].field_name == "supply_voltage"
    assert spec.source_metadata["supply_voltage"] == "structured"


def test_fusion_reports_missing_required():
    spec, report = fuse_fields([ExtractedField("target_gain_db", 70.0, "text")])
    assert spec is None
    assert set(report.missing_required) == {"circuit_class", "technology", "supply_voltage"}


def test_end_to_end_multimodal_parse(inputs_dir):
    design_input = MultimodalDesignInput(
        text="Design a low-power OTA. Gain >= 70 dB.",
        structured_specification_path=str(inputs_dir / "smoke_spec.yaml"),
        table_path=str(inputs_dir / "smoke_table.csv"),
        netlist_path=str(inputs_dir / "smoke_reference.sp"),
    )
    spec, report, review = parse_design_input(design_input, defaults={"temperature_c": 27.0})
    assert spec.circuit_class == "ota"
    assert spec.supply_voltage == pytest.approx(1.8)
    # structured (72 dB) outranks table (70) and text (70)
    assert spec.target_gain_db == pytest.approx(72.0)
    assert spec.source_metadata["target_gain_db"] == "structured"
    assert report.extracted_count > 5
    assert review.is_plausible


def test_mock_image_parser(inputs_dir, tmp_path):
    image = tmp_path / "schematic.png"
    image.write_bytes(b"\x89PNG\r\n")
    parser = MockSchematicImageParser(
        fields=[ExtractedField("circuit_class", "ota", "image")]
    )
    fields = parser.parse(str(image))
    assert fields[0].source == "image"
