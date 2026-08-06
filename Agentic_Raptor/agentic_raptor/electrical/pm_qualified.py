"""Reference-qualified PM extension (knob #1).

Wraps the production extractor. When production says 'ambiguous', applies an
extension rule (single clean descending crossing, in-bounds, inverting DC
phase, no later smaller margin) — but extended verdicts are permitted ONLY if
the module's own golden-reference gate passes: stable ref verified positive,
unstable ref negative, and the MISWIRED ref NOT accepted by the extension.
If the gate fails, the wrapper falls back to production behaviour exactly.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from agentic_raptor.electrical import measurements as M

_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = {
    "stable": _ROOT / "artifacts/variant_verification/vv_2m00/run/acdata.txt",
    "unstable": _ROOT / "artifacts/variant_verification/vv_3m00/run/acdata.txt",
    "miswired": _ROOT / "artifacts/nested_miller/c4_1_s1.0_b1.0/run/acdata.txt",
}
_qualified: bool | None = None
_gate_report: dict = {}


def _extension(f, h) -> dict:
    """Candidate rule for lead-compensated margins production can't classify."""
    mag = 20 * np.log10(np.abs(h) + 1e-30)
    ph = np.unwrap(np.angle(h)) * 180 / np.pi
    cross = []
    for i in range(len(f) - 1):
        if mag[i] * mag[i + 1] < 0:
            t = mag[i] / (mag[i] - mag[i + 1])
            cross.append({"f": f[i] * (f[i + 1] / f[i]) ** t,
                          "ph": ph[i] + t * (ph[i + 1] - ph[i]),
                          "slope": (mag[i + 1] - mag[i]) / np.log10(f[i + 1] / f[i])})
    desc = [c for c in cross if c["slope"] < 0]
    rej = []
    if len(cross) != 1 or not desc:
        rej.append("not_single_descending_crossing")
    if abs(abs(ph[0]) - 180) > 20:          # require inverting DC phase branch
        rej.append("dc_phase_not_inverting_branch")
    if desc and f[-1] < 5 * desc[0]["f"]:
        rej.append("crossing_near_sweep_edge")
    pm = (desc[0]["ph"] + 180.0) if desc else None
    if pm is not None and not (0 < pm < 180):
        rej.append("pm_outside_physical_branch")
    return {"pm": None if rej else round(float(pm), 1),
            "rejections": rej, "crossings": len(cross)}


def qualify() -> dict:
    """Golden-reference gate for the extension. Cached."""
    global _qualified, _gate_report
    if _qualified is not None:
        return _gate_report
    rep = {}
    ok = True
    for name, p in GOLDEN.items():
        if not p.is_file():
            rep[name] = "missing"
            ok = False
            continue
        f, h = M.load_wrdata_complex(p)
        prod = M.measure_phase_margin(f, h)
        ext = _extension(f, h)
        rep[name] = {"production": (prod.value, prod.status), "extension": ext}
    if ok:
        s, u, m = rep["stable"], rep["unstable"], rep["miswired"]
        ok = (s["production"][1] == "verified" and (s["production"][0] or 0) > 0
              and u["production"][1] == "verified" and (u["production"][0] or 0) < 0
              and m["extension"]["pm"] is None)   # extension must NOT pass miswired
    rep["measurement_system_qualified"] = ok
    _qualified, _gate_report = ok, rep
    return rep


def measure_pm_qualified(f, h) -> dict:
    prod = M.measure_phase_margin(f, h)
    if prod.status == "verified":
        return {"verified_pm_deg": prod.value, "pm_status": "verified",
                "source": "production", "confidence": prod.confidence}
    gate = qualify()
    if not gate["measurement_system_qualified"]:
        return {"verified_pm_deg": None, "pm_status": "ambiguous",
                "source": "extension_disabled_gate_failed", "gate": gate}
    ext = _extension(f, h)
    if ext["pm"] is not None:
        return {"verified_pm_deg": ext["pm"], "pm_status": "verified_extended",
                "source": "reference_qualified_extension",
                "extension_conditions_passed": True}
    return {"verified_pm_deg": None, "pm_status": "ambiguous",
            "source": "extension", "rejections": ext["rejections"]}
