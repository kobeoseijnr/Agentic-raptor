"""Metric normalization, constraint margins, and real ngspice output parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_raptor.core.specifications import DesignSpecifications

#: canonical metric keys produced by every simulator adapter
CANONICAL_METRICS: tuple[str, ...] = (
    "gain_db",
    "gbw_hz",
    "phase_margin_deg",
    "power_w",
    "slew_rate_v_per_s",
    "output_swing_v",
    "area_um2",
)

_METRIC_ALIASES: dict[str, str] = {
    "gain": "gain_db",
    "dc_gain_db": "gain_db",
    "av_db": "gain_db",
    "gbw": "gbw_hz",
    "ugbw_hz": "gbw_hz",
    "unity_gain_bandwidth_hz": "gbw_hz",
    "pm": "phase_margin_deg",
    "phase_margin": "phase_margin_deg",
    "power": "power_w",
    "pdiss_w": "power_w",
    "slew_rate": "slew_rate_v_per_s",
    "sr_v_per_s": "slew_rate_v_per_s",
    "swing_v": "output_swing_v",
    "output_swing": "output_swing_v",
    "area": "area_um2",
}


def normalize_metrics(raw: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in raw.items():
        canonical = _METRIC_ALIASES.get(key.strip().lower(), key.strip().lower())
        if canonical in CANONICAL_METRICS:
            out[canonical] = float(value)
    return out


def compute_constraint_margins(
    metrics: dict[str, float], spec: DesignSpecifications
) -> dict[str, float]:
    """Signed relative margins; ≥ 0 means the constraint is met.

    minimum-type: (value − target) / |target|;  maximum-type: (target − value) / |target|.
    Only constraints present in the spec produce entries.
    """
    margins: dict[str, float] = {}

    def minimum(name: str, metric_key: str, target: float | None) -> None:
        if target is not None and metric_key in metrics and target != 0:
            margins[name] = (metrics[metric_key] - target) / abs(target)

    def maximum(name: str, metric_key: str, target: float | None) -> None:
        if target is not None and metric_key in metrics and target != 0:
            margins[name] = (target - metrics[metric_key]) / abs(target)

    minimum("gain_db", "gain_db", spec.target_gain_db)
    minimum("gbw_hz", "gbw_hz", spec.target_gbw_hz)
    minimum("phase_margin_deg", "phase_margin_deg", spec.minimum_phase_margin_deg)
    minimum("slew_rate_v_per_s", "slew_rate_v_per_s", spec.minimum_slew_rate_v_per_s)
    minimum("output_swing_v", "output_swing_v", spec.minimum_output_swing_v)
    maximum("power_w", "power_w", spec.maximum_power_w)
    maximum("area_um2", "area_um2", spec.maximum_area_um2)
    return margins


# ---------------------------------------------------------------------------
# Real ngspice stdout parsing
# ---------------------------------------------------------------------------

#: Ordered failure signatures scanned against combined stdout/stderr (lowercased).
#: First match wins; codes are stable identifiers for the coordinator.
FAILURE_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("no convergence", "convergence_failure"),
    ("convergence failed", "convergence_failure"),
    ("gmin stepping failed", "convergence_failure"),
    ("source stepping failed", "convergence_failure"),
    ("singular matrix", "singular_matrix"),
    ("timestep too small", "timestep_failure"),
    ("time step too small", "timestep_failure"),
    ("could not find model", "missing_model"),
    ("unable to find definition of model", "missing_model"),
    ("model .* not found", "missing_model"),
    ("unknown device", "malformed_netlist"),
    ("syntax error", "malformed_netlist"),
    ("error on line", "malformed_netlist"),
    ("mismatch of subckt", "malformed_netlist"),
    ("overflow", "numerical_overflow"),
    ("out of range", "numerical_overflow"),
)

_MEAS_LINE = re.compile(r"^\s*(\w+)\s*=\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
_PRINT_LINE = re.compile(r"^\s*([\w#()\.]+)\s*=\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
_MEAS_FAILED = re.compile(r"^\s*(\w+)\s*=\s*failed", re.IGNORECASE)


@dataclass
class NgspiceOutput:
    """Structured view of one ngspice batch run's stdout/stderr."""

    values: dict[str, float] = field(default_factory=dict)  # meas + printed scalars
    failed_measurements: list[str] = field(default_factory=list)
    failure_type: str | None = None
    failure_message: str | None = None


def parse_ngspice_stdout(stdout: str, stderr: str = "") -> NgspiceOutput:
    """Extract ``name = value`` lines (meas + print) and classify failures."""
    out = NgspiceOutput()
    for line in stdout.splitlines():
        failed = _MEAS_FAILED.match(line)
        if failed:
            out.failed_measurements.append(failed.group(1).lower())
            continue
        match = _MEAS_LINE.match(line) or _PRINT_LINE.match(line)
        if match:
            name = match.group(1).lower().replace("(", "_").replace(")", "").replace("#", "_")
            try:
                out.values[name] = float(match.group(2))
            except ValueError:  # pragma: no cover - regex guarantees float
                continue
    combined = (stdout + "\n" + stderr).lower()
    for signature, code in FAILURE_SIGNATURES:
        found = re.search(signature, combined) if any(ch in signature for ch in ".*[") else (signature in combined)
        if found:
            out.failure_type = code
            snippet = [ln for ln in (stdout + "\n" + stderr).splitlines() if signature.split(" ")[0] in ln.lower()]
            out.failure_message = snippet[0].strip() if snippet else code
            break
    return out


def metrics_from_ngspice(
    parsed: NgspiceOutput,
    supply_voltage: float,
    output_node: str,
) -> dict[str, float]:
    """Map parsed ngspice scalars to canonical metric keys.

    Phase-margin convention: the testbench sets ``set units=degrees``, so
    ``phase_at_ugf`` arrives in degrees and PM = 180 + phase_at_ugf. Values
    with magnitude < 2π are assumed to be radians (older ngspice ignoring the
    units setting) and converted.
    """
    import math

    metrics: dict[str, float] = {}
    v = parsed.values
    if "dc_gain_db" in v:
        metrics["gain_db"] = v["dc_gain_db"]
    if "ugf_hz" in v:
        metrics["gbw_hz"] = v["ugf_hz"]
        metrics["unity_gain_hz"] = v["ugf_hz"]
    if "phase_at_ugf" in v:
        phase = v["phase_at_ugf"]
        if abs(phase) < 2 * math.pi:  # radians fallback
            phase = math.degrees(phase)
        metrics["phase_margin_deg"] = 180.0 + phase
    ivdd = v.get("vvdd_branch")
    if ivdd is not None:
        metrics["power_w"] = abs(ivdd) * supply_voltage
    out_key = f"v_{output_node.lower()}"
    if out_key in v:
        metrics["output_dc_v"] = v[out_key]
    if "t_rise_lo" in v and "t_rise_hi" in v and v["t_rise_hi"] > v["t_rise_lo"]:
        # Slew from the transient testbench's two crossing times (0.6·step apart).
        dt = v["t_rise_hi"] - v["t_rise_lo"]
        metrics["slew_rate_v_per_s"] = v.get("slew_dv", 0.24) / dt if dt > 0 else 0.0
    return metrics
