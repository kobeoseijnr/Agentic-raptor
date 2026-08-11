"""Parts 3-7: post-optimization PVT (process/voltage/temperature) robustness
evaluation.

Runs strictly AFTER a nominal design is already sized and authoritatively
verified (Stages 1-10 of the main pipeline). Every corner is a FRESH,
independent, real ngspice call through the exact same qualification path the
nominal pipeline uses (``agentic_raptor.electrical.qualify_family`` via
``agentic_raptor.mb_sac.spec_sizing.measure``) -- no PVT grid is embedded
inside SAC sizing steps, and no PVT result is ever synthesized or reused from
a prior (sizing or nominal-verification) call.

Only the process corners this repository's PDK actually ships are
selectable -- see ``available_process_corners()``. Requesting an unavailable
corner name is a hard, explicit ``ConfigurationError``, never a silent
substitution.

Leakage: this module has no import of and no call into
``agentic_raptor.selfimprove_v2`` (RAG/SFT/SAC-replay/PUCT/DPO harvesting).
PVT results produced here must be treated by callers as evaluation-only and
never passed into ``harvest_run``/``record_pair`` -- enforced by callers
simply never doing so (see run_raptor_v2.py), not by anything in this file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from agentic_raptor.electrical import (NOMINAL_SUPPLY_V,
                                       NOMINAL_TEMPERATURE_C,
                                       _PDK_CORNER_DIR)
from agentic_raptor.utils.exceptions import ConfigurationError

#: Process corners this PDK ships model files for (verified present on disk
#: at import time via `available_process_corners`, not assumed). Do not add
#: names here that lack a `<name>.spice` file under `_PDK_CORNER_DIR` --
#: that would silently fabricate an unsupported corner.
KNOWN_PROCESS_CORNERS = ("tt", "ff", "ss", "sf", "fs")


def available_process_corners() -> dict[str, Path]:
    """Process corners actually present in this repo's configured PDK.

    Only reports corners whose `<name>.spice` model file exists on disk --
    never assumes availability. If FF/SS/SF/FS are missing from the PDK
    checkout, they simply will not appear here, and requesting them in a
    PvtConfig raises ConfigurationError rather than pretending they exist.
    """
    return {name: _PDK_CORNER_DIR / f"{name}.spice" for name in KNOWN_PROCESS_CORNERS
            if (_PDK_CORNER_DIR / f"{name}.spice").is_file()}


@dataclass
class PvtConfig:
    """Declarative PVT sweep configuration.

    ``supply_voltages``/``temperatures_c`` are ABSOLUTE values (volts / deg C),
    not deltas from nominal -- e.g. [1.62, 1.8, 1.98] for a +-10% VDD sweep.
    Defaults to nominal-only (tt / 1.8 V / 27 C), i.e. a single corner, so an
    enabled-but-unconfigured PvtConfig never silently explodes into a large
    sweep.
    """
    enabled: bool = False
    process_corners: tuple[str, ...] = ("tt",)
    supply_voltages: tuple[float, ...] = (NOMINAL_SUPPLY_V,)
    temperatures_c: tuple[float, ...] = (NOMINAL_TEMPERATURE_C,)
    #: a corner "counts" toward robust_complete_pass only if it's in this set;
    #: None means ALL configured corners are required (strict paper protocol)
    required_corner_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        self.process_corners = tuple(self.process_corners)
        self.supply_voltages = tuple(float(v) for v in self.supply_voltages)
        self.temperatures_c = tuple(float(t) for t in self.temperatures_c)
        if self.required_corner_ids is not None:
            self.required_corner_ids = tuple(self.required_corner_ids)
        if not self.enabled:
            return
        if not self.process_corners:
            raise ConfigurationError("pvt.process_corners is empty")
        if not self.supply_voltages:
            raise ConfigurationError("pvt.supply_voltages is empty")
        if not self.temperatures_c:
            raise ConfigurationError("pvt.temperatures_c is empty")
        avail = available_process_corners()
        unsupported = [c for c in self.process_corners if c not in avail]
        if unsupported:
            raise ConfigurationError(
                f"pvt.process_corners {unsupported} are not available in "
                f"the configured PDK (found on disk: {sorted(avail)}). This "
                "PDK checkout does not ship model files for them -- fix the "
                "configuration rather than simulating a corner that does "
                "not exist.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PvtConfig":
        known = {"enabled", "process_corners", "supply_voltages",
                 "temperatures_c", "required_corner_ids"}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"pvt config: unknown keys {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PvtConfig":
        import yaml
        p = Path(path)
        if not p.is_file():
            raise ConfigurationError(f"pvt config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if "pvt" in data:
            data = data["pvt"]
        if not isinstance(data, dict):
            raise ConfigurationError(f"{p}: pvt config must be a mapping")
        return cls.from_dict(data)


@dataclass
class PvtCorner:
    pvt_corner_id: str
    process_corner: str
    vdd: float
    temperature_c: float


def generate_corners(cfg: PvtConfig) -> list[PvtCorner]:
    """Cartesian product of process x voltage x temperature. Deterministic
    ordering (process, then voltage, then temperature) so corner ids are
    stable run to run for the same config."""
    corners = []
    for proc, vdd, temp in product(cfg.process_corners, cfg.supply_voltages,
                                    cfg.temperatures_c):
        cid = f"{proc}_{vdd:g}V_{temp:g}C"
        corners.append(PvtCorner(pvt_corner_id=cid, process_corner=proc,
                                 vdd=vdd, temperature_c=temp))
    return corners


def run_pvt_sweep(topology_id: str, graph, spec: dict, exe: str,
                  out_dir: Path, cfg: PvtConfig, *, label: str,
                  c_load_f: float | None = None) -> list[dict[str, Any]]:
    """Run one FRESH real ngspice call per configured corner. Never reuses a
    nominal sizing or verification call. Returns a list of per-corner record
    dicts (Part 5 schema)."""
    from agentic_raptor.electrical import effective_c_load
    from agentic_raptor.mb_sac.spec_sizing import margin_vector, measure, postsizing_outcome
    from agentic_raptor.topology_rl.stage3e2 import new_costs

    avail = available_process_corners()
    # Same resolution rule as every other real measurement (Stage 1.5): PVT
    # is allowed to vary process/voltage/temperature but must not silently
    # vary the load too -- c_load_f is normally the caller's already-resolved
    # run_cl (from run_raptor_v2.py), so this only re-derives from `spec` if
    # a caller genuinely passed nothing at all.
    cload = effective_c_load(spec, override=c_load_f)
    records = []
    for corner in generate_corners(cfg):
        tag = (f"pvt_{spec['spec_id']}_{label}_{corner.pvt_corner_id}_"
               f"{int(time.time() * 1000)}")
        call_id = f"pvt:{spec['spec_id']}:{label}:{corner.pvt_corner_id}:{int(time.time() * 1000)}"
        costs = new_costs()
        meas = measure(topology_id, graph, exe, out_dir, tag, costs,
                       pdk_file=avail[corner.process_corner],
                       supply_voltage=corner.vdd,
                       temperature_c=corner.temperature_c,
                       c_load_f=cload)
        mv = margin_vector(meas, spec)
        oc = postsizing_outcome(meas, spec)
        reasons = (oc["exact_failure_reason"].split("; ")
                  if oc["exact_failure_reason"] else [])
        records.append({
            "pvt_corner_id": corner.pvt_corner_id,
            "process_corner": corner.process_corner,
            "vdd": corner.vdd,
            "temperature_c": corner.temperature_c,
            "gain_db": meas.get("gain_db"),
            "pm_deg": meas.get("pm_deg"),
            "ugbw_hz": meas.get("ugbw_hz"),
            "idd_a": meas.get("idd_a"),
            "power_w": meas.get("power_w"),
            "gain_margin": mv.get("gain_margin_db"),
            "pm_margin": mv.get("pm_margin_deg"),
            "ugbw_margin": mv.get("ugbw_log_margin"),
            "power_margin": mv.get("power_margin"),
            "complete_pass": bool(oc["exact_spec_pass"]),
            "failure_reasons": reasons,
            "spice_call_id": call_id,
            "real_spice_calls": costs.get("real_spice_calls", 0),
        })
    return records


def aggregate_pvt(records: list[dict[str, Any]],
                  required_corner_ids: tuple[str, ...] | None = None
                  ) -> dict[str, Any]:
    """Parts 6-7: PVT Pass %, robust_complete_pass, worst-case diagnostics.

    Deliberately simple: no combined FoM x PVT score, no averaging of FoM
    across corners (Part 8). `robust_complete_pass` is strict -- ALL required
    corners (every configured corner, unless `required_corner_ids` narrows
    it) must pass every mandatory specification.
    """
    total = len(records)
    if total == 0:
        return {"total_pvt_corners": 0, "passed_pvt_corners": 0,
                "failed_pvt_corners": 0, "pvt_pass_percent": None,
                "robust_complete_pass": False,
                "worst_gain_db": None, "worst_pm_deg": None,
                "worst_ugbw_hz": None, "max_idd_a": None, "max_power_w": None,
                "worst_spec_margin": None, "failing_corner_count": 0}
    passed = [r for r in records if r["complete_pass"]]
    failed = [r for r in records if not r["complete_pass"]]
    required = records if required_corner_ids is None else [
        r for r in records if r["pvt_corner_id"] in required_corner_ids]
    robust = bool(required) and all(r["complete_pass"] for r in required)

    def _min(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return min(vals) if vals else None

    def _max(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return max(vals) if vals else None

    margins = [m for r in records
              for m in (r.get("gain_margin"), r.get("pm_margin"), r.get("ugbw_margin"))
              if m is not None]
    return {
        "total_pvt_corners": total,
        "passed_pvt_corners": len(passed),
        "failed_pvt_corners": len(failed),
        "pvt_pass_percent": round(100.0 * len(passed) / total, 4),
        "robust_complete_pass": robust,
        "worst_gain_db": _min("gain_db"),
        "worst_pm_deg": _min("pm_deg"),
        "worst_ugbw_hz": _min("ugbw_hz"),
        "max_idd_a": _max("idd_a"),
        "max_power_w": _max("power_w"),
        "worst_spec_margin": min(margins) if margins else None,
        "failing_corner_count": len(failed),
    }
