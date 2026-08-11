"""Part 1: the single primary figure of merit.

    FoM = (UGBW * C_LOAD) / IDD                      (SI units)
        = UGBW_MHz * C_LOAD_pF / IDD_mA               (paper-facing units)

Higher is better. This is a QUALITY metric, never a feasibility replacement:
a design that fails mandatory specification constraints must never be
preferred over a feasible one on FoM grounds alone -- callers store
``complete_pass``/``exact_spec_pass`` and ``fom`` separately and never
collapse them into one score (Part 1 policy; enforced by callers, not here).

IDD must be the TOTAL measured supply current from ngspice
(``AuthoritativeSpiceOutcome.idd_a`` / ``measure()``'s ``idd_a``), never the
MB-SAC ``ibx`` sizing knob (an optimizer variable, not a measurement). This
module never reads sizing vectors, so that substitution cannot happen here --
callers are responsible for passing the right value in.
"""

from __future__ import annotations

from typing import Any

FOM_VERSION = "FOM_V1_UGBW_CL_OVER_IDD"
FOM_FORMULA = "UGBW_MHz * C_LOAD_pF / IDD_mA"
FOM_UNITS = "MHz*pF/mA"

HZ_PER_MHZ = 1e6
F_PER_PF = 1e-12
A_PER_MA = 1e-3


def hz_to_mhz(hz: float) -> float:
    return hz / HZ_PER_MHZ


def f_to_pf(farads: float) -> float:
    return farads / F_PER_PF


def a_to_ma(amps: float) -> float:
    return amps / A_PER_MA


def compute_fom(ugbw_hz: float | None, c_load_f: float | None,
                idd_a: float | None) -> dict[str, Any]:
    """Compute FOM_V1_UGBW_CL_OVER_IDD from authoritative measured inputs.

    Returns ``fom_value=None`` (never a fabricated number, never a ZeroDivisionError)
    when any input is missing or IDD is non-positive -- a zero/negative IDD is
    not physically valid current draw and cannot be divided into.
    """
    valid = (ugbw_hz is not None and c_load_f is not None
             and idd_a is not None and idd_a > 0)
    ugbw_mhz = hz_to_mhz(ugbw_hz) if ugbw_hz is not None else None
    cload_pf = f_to_pf(c_load_f) if c_load_f is not None else None
    idd_ma = a_to_ma(idd_a) if idd_a is not None else None
    fom_value = (round(ugbw_mhz * cload_pf / idd_ma, 6) if valid else None)
    return {
        "fom_value": fom_value,
        "fom_formula": FOM_FORMULA,
        "fom_version": FOM_VERSION,
        "fom_units": FOM_UNITS,
        "ugbw_used": ugbw_hz,
        "cload_used": c_load_f,
        "idd_used": idd_a,
        "ugbw_mhz": ugbw_mhz,
        "cload_pf": cload_pf,
        "idd_ma": idd_ma,
        "valid": valid,
        "invalid_reason": (None if valid else
                           "missing_ugbw" if ugbw_hz is None else
                           "missing_cload" if c_load_f is None else
                           "missing_idd" if idd_a is None else
                           "idd_not_positive"),
    }
