"""Per-modality specification extraction.

Each parser returns :class:`ExtractedField` records carrying value + provenance;
``fusion.py`` combines them into one canonical ``DesignSpecifications``.

Fully supported here: text, YAML, JSON, CSV tables, SPICE netlists.
Schematic images: clean :class:`SchematicImageParser` protocol with a mock
implementation (a real VLM-backed parser is a later stage; see docs/DECISIONS.md D6).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentic_raptor.utils.exceptions import SpecificationError

# ---------------------------------------------------------------------------
# Extracted-field record
# ---------------------------------------------------------------------------


@dataclass
class ExtractedField:
    """One specification field extracted from one modality."""

    name: str            # canonical DesignSpecifications field name
    value: Any
    source: str          # "text" | "structured" | "table" | "netlist" | "image"
    detail: str = ""     # e.g. matched text or file key
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Canonical field aliases and unit handling
# ---------------------------------------------------------------------------

#: alias (lowercase, non-alphanumeric stripped) → canonical field name
_FIELD_ALIASES: dict[str, str] = {
    "circuitclass": "circuit_class",
    "circuit": "circuit_class",
    "circuittype": "circuit_class",
    "type": "circuit_class",
    "technology": "technology",
    "tech": "technology",
    "process": "technology",
    "node": "technology",
    "supplyvoltage": "supply_voltage",
    "supply": "supply_voltage",
    "vdd": "supply_voltage",
    "vddv": "supply_voltage",
    "temperature": "temperature_c",
    "temperaturec": "temperature_c",
    "temp": "temperature_c",
    "gain": "target_gain_db",
    "gaindb": "target_gain_db",
    "targetgaindb": "target_gain_db",
    "dcgain": "target_gain_db",
    "gbw": "target_gbw_hz",
    "gbwhz": "target_gbw_hz",
    "targetgbwhz": "target_gbw_hz",
    "gainbandwidth": "target_gbw_hz",
    "unitygainbandwidth": "target_gbw_hz",
    "ugbw": "target_gbw_hz",
    "phasemargin": "minimum_phase_margin_deg",
    "phasemargindeg": "minimum_phase_margin_deg",
    "minimumphasemargindeg": "minimum_phase_margin_deg",
    "pm": "minimum_phase_margin_deg",
    "power": "maximum_power_w",
    "powerbudget": "maximum_power_w",
    "maximumpowerw": "maximum_power_w",
    "maxpower": "maximum_power_w",
    "area": "maximum_area_um2",
    "maximumareaum2": "maximum_area_um2",
    "slewrate": "minimum_slew_rate_v_per_s",
    "minimumslewratevpers": "minimum_slew_rate_v_per_s",
    "sr": "minimum_slew_rate_v_per_s",
    "outputswing": "minimum_output_swing_v",
    "minimumoutputswingv": "minimum_output_swing_v",
    "loadcapacitance": "load_capacitance_f",
    "loadcap": "load_capacitance_f",
    "loadcapacitancef": "load_capacitance_f",
    "cl": "load_capacitance_f",
    "cload": "load_capacitance_f",
    "commonmodeinput": "common_mode_input_v",
    "commonmodeinputv": "common_mode_input_v",
    "vcm": "common_mode_input_v",
}

_STRING_FIELDS = {"circuit_class", "technology"}

_UNIT_SCALE: dict[str, float] = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "m": 1e-3, "k": 1e3,
    "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}


def canonical_field(raw_name: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", raw_name.strip().lower())
    if key in _FIELD_ALIASES:
        return _FIELD_ALIASES[key]
    # Already-canonical names pass through.
    canonical = {
        "circuit_class", "technology", "supply_voltage", "temperature_c",
        "target_gain_db", "target_gbw_hz", "minimum_phase_margin_deg",
        "maximum_power_w", "maximum_area_um2", "minimum_slew_rate_v_per_s",
        "minimum_output_swing_v", "load_capacitance_f", "common_mode_input_v",
    }
    name = raw_name.strip().lower()
    return name if name in canonical else None


def parse_engineering_value(raw: str) -> float | None:
    """Parse '100 MHz', '1.8V', '500uW', '2p', '80 dB' → float in SI units."""
    text = str(raw).strip().lower().replace(",", "")
    match = re.match(
        r"^([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(meg|[tgkmunpfµ])?\s*"
        r"(hz|v|w|f|db|deg|°|a|s|v/s|v/us|um2|um\^2)?$",
        text,
    )
    if not match:
        return None
    value = float(match.group(1))
    prefix, unit = match.group(2), match.group(3)
    if prefix:
        # Ambiguity: 'm' before Hz conventionally means Mega in EDA specs (100mhz).
        if prefix == "m" and unit == "hz":
            value *= 1e6
        else:
            value *= _UNIT_SCALE[prefix]
    if unit == "v/us":
        value *= 1e6
    return value


# ---------------------------------------------------------------------------
# Text parser
# ---------------------------------------------------------------------------

_CIRCUIT_CLASS_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("two-stage ota", "two_stage_ota"),
    ("two stage ota", "two_stage_ota"),
    ("folded cascode", "folded_cascode_ota"),
    ("ota", "ota"),
    ("operational transconductance", "ota"),
    ("op-amp", "opamp"),
    ("opamp", "opamp"),
    ("operational amplifier", "opamp"),
    ("comparator", "comparator"),
    ("ldo", "ldo"),
    ("bandgap", "bandgap"),
    ("lna", "lna"),
    ("amplifier", "amplifier"),
)

_TECH_PATTERN = re.compile(
    r"\b((?:tsmc|smic|gf|umc|ibm|st)?\s*\d{1,3}\s?nm|\d+(?:\.\d+)?\s?um|sky130|gpdk045|ptm\w*)\b",
    re.IGNORECASE,
)

#: (field, regex) — value in group 'val', optional unit merged in group.
_TEXT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("target_gain_db", re.compile(r"gain\s*(?:of|is|:|=|≥|>=|at least)?\s*(?P<val>\d+(?:\.\d+)?)\s*db", re.I)),
    ("target_gbw_hz", re.compile(r"(?:gbw|gain[- ]?bandwidth|unity[- ]gain bandwidth|ugbw)\s*(?:of|is|:|=|≥|>=|at least)?\s*(?P<val>\d+(?:\.\d+)?\s*[kmg]?(?:meg)?\s*hz)", re.I)),
    ("minimum_phase_margin_deg", re.compile(r"phase\s*margin\s*(?:of|is|:|=|≥|>=|at least)?\s*(?P<val>\d+(?:\.\d+)?)\s*(?:deg|°|degrees)?", re.I)),
    ("supply_voltage", re.compile(r"(?:supply|vdd)\s*(?:voltage)?\s*(?:of|is|:|=)?\s*(?P<val>\d+(?:\.\d+)?)\s*v\b", re.I)),
    ("maximum_power_w", re.compile(r"power\s*(?:budget|consumption|dissipation)?\s*(?:of|is|:|=|≤|<=|under|below|at most)?\s*(?P<val>\d+(?:\.\d+)?\s*[munp]?\s*w)", re.I)),
    ("load_capacitance_f", re.compile(r"(?:load|cl|cload)\s*(?:capacitance|cap)?\s*(?:of|is|:|=)?\s*(?P<val>\d+(?:\.\d+)?\s*[munpf]?\s*f)\b", re.I)),
    ("minimum_slew_rate_v_per_s", re.compile(r"slew\s*rate\s*(?:of|is|:|=|≥|>=|at least)?\s*(?P<val>\d+(?:\.\d+)?)\s*v\s*/\s*(?P<per>us|µs|s)", re.I)),
    ("temperature_c", re.compile(r"(?:at|temperature)\s*(?:of|is|:|=)?\s*(?P<val>-?\d+(?:\.\d+)?)\s*(?:°c|c\b|celsius)", re.I)),
)


def parse_text(text: str) -> list[ExtractedField]:
    fields: list[ExtractedField] = []
    lowered = text.lower()
    for keyword, cls in _CIRCUIT_CLASS_KEYWORDS:
        if keyword in lowered:
            fields.append(ExtractedField("circuit_class", cls, "text", detail=keyword))
            break
    tech = _TECH_PATTERN.search(text)
    if tech:
        fields.append(
            ExtractedField("technology", re.sub(r"\s+", "", tech.group(1)).lower(), "text", detail=tech.group(0))
        )
    for name, pattern in _TEXT_RULES:
        m = pattern.search(text)
        if not m:
            continue
        raw = m.group("val")
        if name == "minimum_slew_rate_v_per_s":
            value = float(raw) * (1e6 if m.group("per").lower() in ("us", "µs") else 1.0)
        elif name in ("target_gain_db", "minimum_phase_margin_deg", "supply_voltage", "temperature_c"):
            value = float(raw)
        else:
            parsed = parse_engineering_value(raw)
            if parsed is None:
                continue
            value = parsed
        fields.append(ExtractedField(name, value, "text", detail=m.group(0).strip()))
    return fields


# ---------------------------------------------------------------------------
# Structured (YAML / JSON) parser
# ---------------------------------------------------------------------------


def parse_structured(path: str | Path) -> list[ExtractedField]:
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(raw_text)
    else:
        import yaml

        data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise SpecificationError(f"structured specification {p} must contain a mapping")
    # Allow one level of nesting under a 'specifications'/'specs' key.
    inner = data.get("specifications") or data.get("specs")
    if isinstance(inner, dict):
        data = {**data, **inner}
    return _fields_from_mapping(data, source="structured")


def _fields_from_mapping(data: dict[str, Any], source: str) -> list[ExtractedField]:
    out: list[ExtractedField] = []
    for key, raw_value in data.items():
        name = canonical_field(str(key))
        if name is None:
            continue
        if name in _STRING_FIELDS:
            out.append(ExtractedField(name, str(raw_value), source, detail=str(key)))
            continue
        if isinstance(raw_value, (int, float)):
            out.append(ExtractedField(name, float(raw_value), source, detail=str(key)))
        else:
            parsed = parse_engineering_value(str(raw_value))
            if parsed is not None:
                out.append(ExtractedField(name, parsed, source, detail=f"{key}={raw_value}"))
    return out


# ---------------------------------------------------------------------------
# CSV table parser
# ---------------------------------------------------------------------------


def parse_table(path: str | Path) -> list[ExtractedField]:
    """CSV with columns (parameter, value[, unit]) — header optional."""
    p = Path(path)
    out: list[ExtractedField] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(cell.strip() for cell in r)]
    if not rows:
        return out
    start = 1 if canonical_field(rows[0][0]) is None and rows[0][0].strip().lower() in ("parameter", "param", "spec", "name") else 0
    for row in rows[start:]:
        if len(row) < 2:
            continue
        name = canonical_field(row[0])
        if name is None:
            continue
        raw_value = row[1].strip()
        unit = row[2].strip() if len(row) > 2 else ""
        if name in _STRING_FIELDS:
            out.append(ExtractedField(name, raw_value, "table", detail=",".join(row)))
            continue
        parsed = parse_engineering_value(f"{raw_value} {unit}".strip()) if unit else parse_engineering_value(raw_value)
        if parsed is None:
            try:
                parsed = float(raw_value)
            except ValueError:
                continue
        out.append(ExtractedField(name, parsed, "table", detail=",".join(row)))
    return out


# ---------------------------------------------------------------------------
# SPICE netlist parser
# ---------------------------------------------------------------------------


def parse_netlist(path: str | Path) -> list[ExtractedField]:
    """Extract what a netlist reliably encodes: supply voltage, load cap, class hints."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[ExtractedField] = []
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("*"):
            for keyword, cls in _CIRCUIT_CLASS_KEYWORDS:
                if keyword in low:
                    out.append(ExtractedField("circuit_class", cls, "netlist", detail=stripped, confidence=0.8))
                    break
            continue
        tokens = low.split()
        if not tokens:
            continue
        # Vxxx node1 node2 [dc] value — supply if named vdd/vsupply or drives vdd net.
        if tokens[0].startswith("v") and len(tokens) >= 4:
            name, nodes = tokens[0], tokens[1:3]
            value_token = tokens[4] if len(tokens) >= 5 and tokens[3] == "dc" else tokens[3]
            value = parse_engineering_value(value_token) or _safe_float(value_token)
            if value is not None and ("vdd" in name or any("vdd" in n for n in nodes)):
                out.append(ExtractedField("supply_voltage", float(value), "netlist", detail=stripped))
        # Cxxx node1 node2 value — load cap if named cl/cload or touches out net.
        if tokens[0].startswith("c") and len(tokens) >= 4:
            value = parse_engineering_value(tokens[3]) or _safe_float(tokens[3])
            is_load = tokens[0] in ("cl", "cload") or any(n in ("out", "vout") for n in tokens[1:3])
            if value is not None and is_load:
                out.append(ExtractedField("load_capacitance_f", float(value), "netlist", detail=stripped))
    return out


def _safe_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Schematic image parsing (interface + mock)
# ---------------------------------------------------------------------------


class SchematicImageParser(Protocol):
    """Interface for future VLM-backed schematic understanding."""

    def parse(self, image_path: str) -> list[ExtractedField]: ...


@dataclass
class MockSchematicImageParser:
    """Deterministic stand-in used when no multimodal API is configured.

    Returns the configured fields (default: none) tagged with source="image".
    """

    fields: list[ExtractedField] = field(default_factory=list)

    def parse(self, image_path: str) -> list[ExtractedField]:
        if not Path(image_path).is_file():
            raise SpecificationError(f"schematic image not found: {image_path}")
        return [
            ExtractedField(f.name, f.value, "image", detail=f.detail or Path(image_path).name, confidence=f.confidence)
            for f in self.fields
        ]
