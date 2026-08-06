"""Measurement library: mathematically validated metric extraction.

All frequency-domain metrics work directly on the complex transfer function
(freq[], complex H[]) exported from the simulator (`wrdata`), never on
pre-collapsed scalars. Phase is continuously unwrapped (np.unwrap on radians —
no naive ±360 fixes). Every metric returns MetricResult(value, status,
confidence, failure_reason, metadata); only status=="verified" values may be
reported. Ambiguity → value None, never an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

STATUSES = ("verified", "estimated", "ambiguous", "unsupported", "measurement_failed")

FAILURE_CLASSES_31 = (
    "phase_wrap_failure", "multiple_crossings", "no_unity_crossing", "crossing_ambiguous",
    "invalid_transfer_function", "nonfinite_transfer_function", "phase_noise",
    "measurement_instability",
)


@dataclass
class MetricResult:
    metric: str
    value: float | None
    status: str
    confidence: float
    method: str
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "value": self.value, "status": self.status,
                "confidence": self.confidence, "method": self.method,
                "failure_reason": self.failure_reason, "notes": self.metadata.get("notes", ""),
                **{k: v for k, v in self.metadata.items() if k != "notes"}}


def _validate_tf(freq: np.ndarray, h: np.ndarray) -> str | None:
    if len(freq) < 8 or len(freq) != len(h):
        return "invalid_transfer_function"
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(h.real)) or not np.all(np.isfinite(h.imag)):
        return "nonfinite_transfer_function"
    if np.all(np.abs(h) == 0):
        return "invalid_transfer_function"
    return None


def find_unity_crossings(freq: np.ndarray, h: np.ndarray) -> list[dict[str, Any]]:
    """All |H|=1 crossings with log-frequency linear interpolation."""
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-300))
    crossings = []
    for i in range(len(freq) - 1):
        a, b = mag_db[i], mag_db[i + 1]
        if a == 0.0:
            crossings.append({"crossing_frequency": float(freq[i]), "crossing_index": i,
                              "crossing_type": "exact", "t": 0.0})
        elif (a > 0) != (b > 0):
            t = a / (a - b)
            lf = np.log10(freq[i]) + t * (np.log10(freq[i + 1]) - np.log10(freq[i]))
            crossings.append({"crossing_frequency": float(10 ** lf), "crossing_index": i,
                              "crossing_type": "down" if a > 0 else "up", "t": float(t)})
    return crossings


def measure_gain(freq: np.ndarray, h: np.ndarray) -> MetricResult:
    bad = _validate_tf(freq, h)
    if bad:
        return MetricResult("dc_gain_db", None, "measurement_failed", 0.0, "low_freq_mag", bad)
    value = float(20 * np.log10(abs(h[0]))) if abs(h[0]) > 0 else None
    if value is None:
        return MetricResult("dc_gain_db", None, "measurement_failed", 0.0, "low_freq_mag",
                            "invalid_transfer_function")
    return MetricResult("dc_gain_db", value, "verified", 0.99, "low_freq_mag",
                        metadata={"notes": f"|H| at {freq[0]:.3g} Hz"})


def measure_ugbw(freq: np.ndarray, h: np.ndarray) -> MetricResult:
    bad = _validate_tf(freq, h)
    if bad:
        return MetricResult("ugbw_hz", None, "measurement_failed", 0.0, "unity_gain_interpolation", bad)
    crossings = [c for c in find_unity_crossings(freq, h) if c["crossing_type"] in ("down", "exact")]
    all_cross = find_unity_crossings(freq, h)
    if not all_cross:
        return MetricResult("ugbw_hz", None, "unsupported", 0.0, "unity_gain_interpolation",
                            "no_unity_crossing")
    if not crossings:
        return MetricResult("ugbw_hz", None, "ambiguous", 0.0, "unity_gain_interpolation",
                            "crossing_ambiguous", {"crossings": len(all_cross)})
    # Valid crossover rule (documented): the LAST downward crossing (final |H|=1
    # before magnitude stays below unity). Multiple downward crossings → ambiguous.
    down = [c for c in all_cross if c["crossing_type"] in ("down", "exact")]
    if len(down) > 1:
        return MetricResult("ugbw_hz", None, "ambiguous", 0.0, "unity_gain_interpolation",
                            "multiple_crossings", {"crossings": len(down)})
    c = down[0]
    conf = 0.99 if abs(c["t"] - 0.5) <= 0.5 else 0.7
    return MetricResult("ugbw_hz", c["crossing_frequency"], "verified", conf,
                        "unity_gain_interpolation",
                        metadata={"crossing_index": c["crossing_index"],
                                  "crossing_type": c["crossing_type"],
                                  "crossing_confidence": conf, "notes": "single valid crossing"})


def measure_bandwidth(freq: np.ndarray, h: np.ndarray) -> MetricResult:
    bad = _validate_tf(freq, h)
    if bad:
        return MetricResult("f3db_hz", None, "measurement_failed", 0.0, "minus3db_interpolation", bad)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-300))
    target = mag_db[0] - 3.0
    below = np.where(mag_db < target)[0]
    if len(below) == 0 or below[0] == 0:
        return MetricResult("f3db_hz", None, "unsupported", 0.0, "minus3db_interpolation",
                            "no_unity_crossing", {"notes": "-3dB point outside sweep"})
    i = below[0] - 1
    t = (mag_db[i] - target) / (mag_db[i] - mag_db[i + 1])
    lf = np.log10(freq[i]) + t * (np.log10(freq[i + 1]) - np.log10(freq[i]))
    return MetricResult("f3db_hz", float(10 ** lf), "verified", 0.95, "minus3db_interpolation")


def measure_phase_margin(freq: np.ndarray, h: np.ndarray) -> MetricResult:
    bad = _validate_tf(freq, h)
    if bad:
        return MetricResult("phase_margin_deg", None, "measurement_failed", 0.0,
                            "unity_gain_interpolation", bad)
    ug = measure_ugbw(freq, h)
    if ug.status != "verified":
        return MetricResult("phase_margin_deg", None,
                            "ambiguous" if ug.failure_reason in ("multiple_crossings", "crossing_ambiguous")
                            else "unsupported", 0.0, "continuous_unwrap+interpolation",
                            ug.failure_reason, {"measurement_status": "ambiguous_phase_margin"
                                                if ug.failure_reason != "no_unity_crossing" else "no_crossing"})
    phase = np.unwrap(np.angle(h))  # continuous unwrap in radians — no ±360 hacks
    # phase-noise check: local jumps beyond π between adjacent points post-unwrap
    dphi = np.abs(np.diff(phase))
    if np.any(dphi > np.pi):
        return MetricResult("phase_margin_deg", None, "ambiguous", 0.0,
                            "continuous_unwrap+interpolation", "phase_wrap_failure",
                            {"max_step_rad": float(dphi.max())})
    i = ug.metadata["crossing_index"]
    lf = np.log10(freq)
    lfx = np.log10(ug.value)
    t = (lfx - lf[i]) / (lf[i + 1] - lf[i]) if lf[i + 1] != lf[i] else 0.0
    phi_x = phase[i] + t * (phase[i + 1] - phase[i])
    # Reference DC phase: excess phase relative to low-frequency phase.
    #
    # BRANCH-CUT GUARD. An INVERTING amplifier sits exactly on the +-180 deg
    # branch cut of np.angle, whose principal value is (-pi, pi]. Rounding in
    # the imaginary part alone decides whether the DC sample returns +179.9
    # or -175.4 for the same circuit, and np.unwrap then propagates that
    # arbitrary choice as a ~360 deg offset through every later sample.
    #
    # Measured example (3-stage, nested Miller): phase[0] = -175.42 instead
    # of +179.87 gave delta = +92.65 deg and pm = +272.65, which the
    # >= 180 guard below discarded as "measurement_instability". The true
    # value was about -87 deg -- a real, reportable instability was silently
    # turned into "unmeasurable".
    #
    # A minimum-phase amplifier cannot GAIN phase as frequency rises, so a
    # positive accumulated delta is proof of the artefact rather than of a
    # measurement. Remove whole turns until the physics holds.
    delta = phi_x - phase[0]
    turns = 0
    while delta > 0 and turns < 4:
        delta -= 2 * np.pi
        turns += 1
    pm = 180.0 + np.degrees(delta)
    # normalise once into (-180, 540) sanity window without wrapping tricks
    if not np.isfinite(pm):
        return MetricResult("phase_margin_deg", None, "measurement_failed", 0.0,
                            "continuous_unwrap+interpolation", "nonfinite_transfer_function")
    if pm <= 0:
        return MetricResult("phase_margin_deg", float(pm), "verified", 0.9,
                            "continuous_unwrap+interpolation", None,
                            {"notes": "non-positive PM: unstable system (measured, not estimated)"})
    if pm >= 180:
        return MetricResult("phase_margin_deg", None, "ambiguous", 0.0,
                            "continuous_unwrap+interpolation", "measurement_instability",
                            {"raw_pm_deg": float(pm),
                             "measurement_status": "ambiguous_phase_margin"})
    return MetricResult("phase_margin_deg", float(pm), "verified",
                        min(0.99, ug.confidence), "continuous_unwrap+interpolation",
                        metadata={"crossing_frequency": ug.value, "notes": "single valid crossing"})


def measure_gain_margin(freq: np.ndarray, h: np.ndarray) -> MetricResult:
    bad = _validate_tf(freq, h)
    if bad:
        return MetricResult("gain_margin_db", None, "measurement_failed", 0.0, "phase180_interpolation", bad)
    phase = np.unwrap(np.angle(h))
    rel = np.degrees(phase - phase[0])
    idx = np.where(np.diff(np.sign(rel + 180.0)) != 0)[0]
    if len(idx) == 0:
        return MetricResult("gain_margin_db", None, "unsupported", 0.0, "phase180_interpolation",
                            "no_unity_crossing", {"notes": "phase never reaches -180deg in sweep"})
    i = int(idx[0])
    t = (rel[i] + 180.0) / (rel[i] - rel[i + 1]) if rel[i] != rel[i + 1] else 0.0
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-300))
    gm = -(mag_db[i] + t * (mag_db[i + 1] - mag_db[i]))
    return MetricResult("gain_margin_db", float(gm), "verified", 0.9, "phase180_interpolation")


def measure_all(freq: np.ndarray, h: np.ndarray) -> dict[str, MetricResult]:
    return {m.metric: m for m in (measure_gain(freq, h), measure_ugbw(freq, h),
                                  measure_bandwidth(freq, h), measure_phase_margin(freq, h),
                                  measure_gain_margin(freq, h))}


def load_wrdata_complex(path) -> tuple[np.ndarray, np.ndarray]:
    """ngspice `wrdata` output for a complex vector: freq re freq im columns."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("unexpected wrdata format")
    freq = data[:, 0]
    # 3 cols: freq re im · ≥4 cols: freq re freq im (ngspice repeats the scale)
    h = data[:, 1] + 1j * (data[:, 3] if data.shape[1] >= 4 else data[:, 2])
    return freq, h
