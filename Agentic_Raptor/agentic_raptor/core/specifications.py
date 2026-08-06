"""Design specification model with physical-range validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.utils.exceptions import SpecificationError

_ABS_ZERO_C = -273.15


@dataclass
class DesignSpecifications:
    """Target specifications for one analog design task.

    Optional targets are ``None`` when unconstrained. ``validate()`` raises
    :class:`SpecificationError` on physically meaningless values.
    """

    circuit_class: str
    technology: str
    supply_voltage: float
    temperature_c: float = 27.0
    target_gain_db: float | None = None
    target_gbw_hz: float | None = None
    minimum_phase_margin_deg: float | None = None
    maximum_power_w: float | None = None
    maximum_area_um2: float | None = None
    minimum_slew_rate_v_per_s: float | None = None
    minimum_output_swing_v: float | None = None
    load_capacitance_f: float | None = None
    common_mode_input_v: float | None = None
    additional_constraints: dict[str, float | str | bool] = field(default_factory=dict)
    #: field name → provenance ("text", "structured", "table", "netlist", "image", "default")
    source_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        def _fail(msg: str) -> None:
            raise SpecificationError(msg)

        if not self.circuit_class.strip():
            _fail("circuit_class must be non-empty")
        if not self.technology.strip():
            _fail("technology must be non-empty")
        if not 0.0 < self.supply_voltage <= 100.0:
            _fail(f"supply_voltage {self.supply_voltage} V outside (0, 100]")
        if not _ABS_ZERO_C < self.temperature_c <= 500.0:
            _fail(f"temperature_c {self.temperature_c} outside ({_ABS_ZERO_C}, 500]")
        if self.target_gbw_hz is not None and self.target_gbw_hz <= 0:
            _fail("target_gbw_hz must be positive")
        if self.minimum_phase_margin_deg is not None and not 0.0 <= self.minimum_phase_margin_deg <= 180.0:
            _fail("minimum_phase_margin_deg outside [0, 180]")
        if self.maximum_power_w is not None and self.maximum_power_w <= 0:
            _fail("maximum_power_w must be positive")
        if self.maximum_area_um2 is not None and self.maximum_area_um2 <= 0:
            _fail("maximum_area_um2 must be positive")
        if self.minimum_slew_rate_v_per_s is not None and self.minimum_slew_rate_v_per_s <= 0:
            _fail("minimum_slew_rate_v_per_s must be positive")
        if self.minimum_output_swing_v is not None and not 0.0 < self.minimum_output_swing_v <= self.supply_voltage:
            _fail("minimum_output_swing_v must be in (0, supply_voltage]")
        if self.load_capacitance_f is not None and self.load_capacitance_f < 0:
            _fail("load_capacitance_f must be non-negative")
        if self.common_mode_input_v is not None and not 0.0 <= self.common_mode_input_v <= self.supply_voltage:
            _fail("common_mode_input_v must be in [0, supply_voltage]")

    # -- feature view -------------------------------------------------------
    #: Ordered numeric fields used for embeddings / retrieval similarity.
    NUMERIC_FIELDS: tuple[str, ...] = (
        "supply_voltage",
        "temperature_c",
        "target_gain_db",
        "target_gbw_hz",
        "minimum_phase_margin_deg",
        "maximum_power_w",
        "maximum_area_um2",
        "minimum_slew_rate_v_per_s",
        "minimum_output_swing_v",
        "load_capacitance_f",
        "common_mode_input_v",
    )

    def feature_vector(self) -> list[float]:
        """Fixed-length numeric embedding; ``None`` targets map to 0.0."""
        out: list[float] = []
        for name in self.NUMERIC_FIELDS:
            value = getattr(self, name)
            out.append(0.0 if value is None else float(value))
        return out

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesignSpecifications:
        return cls(**data)
