"""Cross-field engineering sanity checks on a fused specification.

``DesignSpecifications.validate()`` already enforces hard physical ranges;
this validator adds *engineering plausibility* warnings that should not block
the pipeline but must be surfaced to the coordinator and logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_raptor.core.specifications import DesignSpecifications


@dataclass
class SpecificationCheck:
    code: str
    message: str
    severity: str  # "error" | "warning"


@dataclass
class SpecificationReview:
    is_plausible: bool
    checks: list[SpecificationCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)


def review_specification(spec: DesignSpecifications) -> SpecificationReview:
    checks: list[SpecificationCheck] = []

    def warn(code: str, message: str) -> None:
        checks.append(SpecificationCheck(code, message, "warning"))

    # gm ≈ 2π·GBW·CL; a first-order power sanity bound with modest gm/Id.
    if spec.target_gbw_hz and spec.load_capacitance_f and spec.maximum_power_w:
        gm_needed = 2.0 * 3.14159265 * spec.target_gbw_hz * spec.load_capacitance_f
        id_estimate = gm_needed / 20.0  # gm/Id ≈ 20 S/A (weak-moderate inversion)
        power_floor = id_estimate * spec.supply_voltage
        if power_floor > spec.maximum_power_w:
            warn(
                "POWER_BUDGET_TIGHT",
                f"first-order estimate needs ≥{power_floor:.2e} W for GBW/CL target but budget "
                f"is {spec.maximum_power_w:.2e} W",
            )

    if spec.target_gain_db is not None and spec.target_gain_db > 100.0:
        warn("VERY_HIGH_GAIN", f"gain target {spec.target_gain_db} dB likely needs multi-stage/cascode topology")

    if spec.minimum_output_swing_v is not None and spec.minimum_output_swing_v > 0.9 * spec.supply_voltage:
        warn(
            "SWING_NEAR_RAILS",
            f"output swing {spec.minimum_output_swing_v} V is >90% of supply {spec.supply_voltage} V",
        )

    if spec.supply_voltage < 0.6:
        warn("VERY_LOW_SUPPLY", f"supply {spec.supply_voltage} V is aggressive for analog design")

    if spec.minimum_slew_rate_v_per_s and spec.load_capacitance_f and spec.maximum_power_w:
        i_slew = spec.minimum_slew_rate_v_per_s * spec.load_capacitance_f
        if i_slew * spec.supply_voltage > spec.maximum_power_w:
            warn(
                "SLEW_POWER_CONFLICT",
                f"slew target needs ≥{i_slew:.2e} A into the load; exceeds power budget",
            )

    is_plausible = not any(c.severity == "error" for c in checks)
    return SpecificationReview(is_plausible=is_plausible, checks=checks)
