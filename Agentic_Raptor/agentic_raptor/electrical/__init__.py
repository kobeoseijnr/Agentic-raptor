"""Stage 3B: electrical qualification of device-level corpus families.

Backend-neutral records + an ngspice backend (reuses the Stage 2 stdout parser
and failure taxonomy). Adapted testbenches derive from the source AnalogGym
TB_Amplifier_ACDC.cir (supply 1.8 V, VCM ratio 0.25, CLOAD 500 pF, open-loop
Lfb/Cin ADM instance); transformations are path normalisation + analysis
wrapper only — no sizing/bias/topology changes. All metrics are measured or
null with explicit failure reasons; nothing is fabricated.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.spice.ngspice_simulator import discover_ngspice, ngspice_version
from agentic_raptor.spice.result_parser import parse_ngspice_stdout

VALIDATION_STATES = (
    "unvalidated", "source_netlist_present", "netlist_parse_valid", "dependencies_resolved",
    "testbench_ready", "simulation_started", "simulation_converged", "dc_operating_point_valid",
    "ac_analysis_valid", "transient_analysis_valid", "electrically_functional",
    "spec_conditioned_pass", "pvt_validated", "validation_failed", "mapping_required",
    "unsupported_simulator_syntax", "missing_dependency",
)

_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_AMP = _ROOT.parent / "RAPTOR_Legacy" / "AnalogGym" / "AnalogGym" / "Amplifier"
_PDK_CORNER_DIR = (_ROOT.parent / "RAPTOR_Legacy" / "AnalogGym" / "RGNN_RL" / "mosfet_model"
                    / "sky130_pdk" / "sky130_pdk" / "libs.tech" / "ngspice" / "corners")
_PDK_TT = _PDK_CORNER_DIR / "tt.spice"

# The single testbench every real ngspice call in v2 uses (nominal path AND
# PVT sweep alike): one supply rail (V1), a load resolved per-call, fixed
# nominal voltage/temperature unless a PVT corner overrides them.
# NOMINAL_CLOAD_F is ONLY the fallback used when no spec (or an explicit
# override) supplies a load -- see effective_c_load() below, which is the
# actual authoritative decision every real caller goes through as of the
# Stage 1.5 repair (2026-08-09). Before that repair, every caller left
# c_load_f=None and this constant was silently applied regardless of what a
# spec's `cl=` target said; that is what effective_c_load() now fixes.
# FoM/PVT must still read the value that was ACTUALLY simulated (never the
# unapplied spec target) -- that discipline doesn't change, only which value
# actually reaches the simulator now does.
NOMINAL_CLOAD_F = 500e-12
NOMINAL_SUPPLY_V = 1.8
NOMINAL_TEMPERATURE_C = 27.0


def effective_c_load(spec: dict | None, override: float | None = None) -> float:
    """THE single authoritative decision of what load a real simulation uses.

    effective = override if explicitly given
              else spec["load_capacitance_pf"] * 1e-12 if the spec states one
              else NOMINAL_CLOAD_F (a spec that never states a load, e.g. a
              synthetic/legacy caller with no `cl=` field, still needs SOME
              real number to simulate against).

    Every real measure()/qualify_*/PVT call must resolve its load through
    this function -- not by leaving c_load_f=None and letting a lower layer
    silently default to NOMINAL_CLOAD_F, which is how the spec's requested
    cl came to be measured but never applied (Stage 1.5 repair, 2026-08-09:
    the 81-run pilot's nominal.c_load_f was 500pF on every row regardless of
    the spec's stated 100pF/200pF/500pF cl -- see spec_registry.py).
    """
    if override is not None:
        return float(override)
    if spec and spec.get("load_capacitance_pf") is not None:
        return float(spec["load_capacitance_pf"]) * 1e-12
    return NOMINAL_CLOAD_F


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass
class ElectricalValidationRecord:
    topology_id: str
    graph_hash: str | None
    source: str
    simulator: str
    simulator_version: str | None
    run_id: str
    timestamp: float
    netlist_hash: str | None
    testbench_hash: str | None
    validation_stage: str
    structural_validation_status: str
    electrical_validation_status: str
    simulation_status: str
    spec_validation_status: str = "not_evaluated"
    pvt_validation_status: str = "not_evaluated"
    analyses_requested: list[str] = field(default_factory=list)
    analyses_completed: list[str] = field(default_factory=list)
    extracted_metrics: dict[str, float | None] = field(default_factory=dict)
    failure_class: str | None = None
    failure_message: str | None = None
    raw_output_paths: list[str] = field(default_factory=list)
    environment_id: str = ""
    split: str = "unknown"
    parent_run_id: str | None = None


def capture_environment(out_dir: Path) -> str:
    exe = discover_ngspice()
    env = {
        "environment_id": f"env-{uuid.uuid4().hex[:10]}",
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "ngspice_exe": exe,
        "ngspice_version": ngspice_version(exe) if exe else None,
        "pdk_tt_spice": str(_PDK_TT),
        "pdk_present": _PDK_TT.is_file(),
        "pdk_hash": _h(_PDK_TT.read_text(encoding="utf-8", errors="replace")[:200000]) if _PDK_TT.is_file() else None,
        "git_commit": None,  # repository is not under git (documented)
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    return env["environment_id"]


def audit_family(entry) -> dict[str, Any]:
    netlist = entry.path / "netlist.sp"
    text = netlist.read_text(encoding="utf-8", errors="replace") if netlist.is_file() else ""
    subckt = text.lower().split(".subckt", 1)[1].split()[0] if ".subckt" in text.lower() else None
    params_file = _LEGACY_AMP / "design_variables" / entry.metadata.get("name", "")
    return {
        "topology_id": entry.topology_id, "name": entry.metadata.get("name"),
        "graph_hash": entry.metadata.get("graph_hash"),
        "netlist_kind": "reusable_amplifier_subcircuit" if subckt else "unknown",
        "subckt_name": subckt,
        "device_count": text.lower().count("\nxm") + (1 if text.lower().startswith("xm") else 0),
        "models": ["sky130_fd_pr__nfet_01v8", "sky130_fd_pr__pfet_01v8"],
        "input_pair_polarity": input_pair_polarity(text),
        "input_vcm_ratio": input_vcm_ratio(input_pair_polarity(text)),
        "design_variables_present": params_file.is_file(),
        "pdk_present": _PDK_TT.is_file(),
        "source_testbench": str(_LEGACY_AMP / "amp_spice_testbench" / "TB_Amplifier_ACDC.cir"),
        "simulator_syntax": "ngspice/sky130",
        "immediately_runnable": bool(subckt and params_file.is_file() and _PDK_TT.is_file()),
        "blocking_reason": None if (subckt and params_file.is_file() and _PDK_TT.is_file())
        else ("missing_design_variables" if not params_file.is_file()
              else "missing_pdk" if not _PDK_TT.is_file() else "no_subckt"),
    }


def input_pair_polarity(netlist_text: str) -> str:
    """'nmos' | 'pmos' | 'unknown' -- the device type of the input pair.

    Found by locating the MOS whose GATE net is vinp/vinn. Device line layout
    is `xNAME drain gate source bulk model ...` in both the legacy AnalogGym
    netlists and the ones mapping/ builds.
    """
    for line in netlist_text.splitlines():
        f = line.split()
        if len(f) < 6 or not f[0].lower().startswith("xm"):
            continue
        if f[2].lower() in ("vinp", "vinn"):
            if "nfet" in f[5].lower():
                return "nmos"
            if "pfet" in f[5].lower():
                return "pmos"
    return "unknown"


def input_vcm_ratio(polarity: str) -> float:
    """Input common mode, as a fraction of the supply, per input-pair type.

    The legacy AnalogGym testbench hard-codes 0.25 (0.45 V at 1.8 V). Every
    legacy amplifier has a PMOS input pair, for which that is a correct bias.
    mapping/ builds NMOS-input amplifiers, and 0.45 V is BELOW the sky130 nfet
    threshold (vth = 0.564 V), so the pair never leaves subthreshold: measured
    at 0.25 the branch current is 121 nA and gm = 3.0 uS, versus 9.5 uA and
    191 uS at 0.5 -- a 63x transconductance loss that caps UGBW roughly 30x
    below target on every NMOS-input family.

    PMOS keeps 0.25, so legacy circuits are unaffected.
    """
    return 0.5 if polarity == "nmos" else 0.25


def build_testbench(entry, audit: dict[str, Any], *,
                    pdk_file: Path | None = None,
                    supply_voltage: float | None = None,
                    temperature_c: float | None = None,
                    c_load_f: float | None = None) -> str:
    """Adapted ADM open-loop TB from the source testbench (transform log in header).

    ``pdk_file``/``supply_voltage``/``temperature_c``/``c_load_f`` default to
    the nominal values (tt corner, 1.8 V, 27 C, 500 pF) so every existing
    caller is byte-identical unless it opts into a PVT corner explicitly.
    """
    pdk_file = pdk_file or _PDK_TT
    vdd = NOMINAL_SUPPLY_V if supply_voltage is None else supply_voltage
    temp = NOMINAL_TEMPERATURE_C if temperature_c is None else temperature_c
    cload = NOMINAL_CLOAD_F if c_load_f is None else c_load_f
    name = audit["subckt_name"]
    netlist_abs = (entry.path / "netlist.sp").resolve()
    polarity = input_pair_polarity(
        netlist_abs.read_text(encoding="utf-8", errors="replace")
        if netlist_abs.is_file() else "")
    vcm_ratio = input_vcm_ratio(polarity)
    params_abs = (_LEGACY_AMP / "design_variables" / entry.metadata.get("name", "")).resolve()
    return f"""* Stage3B adapted TB for {entry.topology_id} ({name})
* transforms: absolute include paths; combined-corner PDK path; ADM-only instance;
* op+ac wrapper with measurements. No size/topology changes.
* input pair = {polarity} -> VCM_ratio {vcm_ratio} (legacy PMOS default 0.25 unchanged;
* NMOS raised because 0.25 = 0.45 V sits below the sky130 nfet vth of 0.564 V)
.include {netlist_abs}
.include {params_abs}
.param mc_mm_switch=0
.param mc_pr_switch=0
.include {pdk_file.resolve()}
.PARAM supply_voltage = {vdd:.6g}
.PARAM VCM_ratio = {vcm_ratio}
.PARAM PARAM_CLOAD = {cload:.6g}
.TEMP {temp:.6g}
V1 vdd 0 'supply_voltage'
V2 vss 0 0
Vindc opin 0 'supply_voltage*VCM_ratio'
Vin signal_in 0 dc 'supply_voltage*VCM_ratio' ac 1
Lfb opout opout_dc 1T
Cin opout_dc signal_in 1T
Xop1 vss vdd opout_dc opin opout {name}
Cload1 opout 0 'PARAM_CLOAD'
.control
set units=degrees
op
print v(opout) v1#branch
ac dec 20 0.1 1e9
meas ac dcgain_db find vdb(opout) at=0.1
wrdata acdata.txt v(opout)
quit 0
.endc
.end
"""


def qualify_family(entry, audit: dict[str, Any], run_dir: Path, exe: str,
                   env_id: str, split: str, timeout_s: float = 180.0, *,
                   pdk_file: Path | None = None,
                   supply_voltage: float | None = None,
                   temperature_c: float | None = None,
                   c_load_f: float | None = None) -> ElectricalValidationRecord:
    """``pdk_file``/``supply_voltage``/``temperature_c``/``c_load_f`` default
    to nominal (tt, 1.8 V, 27 C, 500 pF) -- passing them is how a PVT corner
    sweep reuses this exact, real-ngspice qualification path instead of a
    second implementation."""
    vdd = NOMINAL_SUPPLY_V if supply_voltage is None else supply_voltage
    rec = ElectricalValidationRecord(
        topology_id=entry.topology_id, graph_hash=entry.metadata.get("graph_hash"),
        source=entry.source, simulator="ngspice", simulator_version=ngspice_version(exe) if exe else None,
        run_id=f"run-{uuid.uuid4().hex[:10]}", timestamp=time.time(),
        netlist_hash=None, testbench_hash=None,
        validation_stage="source_netlist_present",
        structural_validation_status="structurally_loaded",
        electrical_validation_status="unvalidated",
        simulation_status="not_started", environment_id=env_id, split=split,
        analyses_requested=["op", "ac"],
    )
    if not audit["immediately_runnable"]:
        rec.validation_stage = "missing_dependency"
        rec.electrical_validation_status = "missing_dependency"
        rec.failure_class = audit["blocking_reason"]
        return rec
    tb = build_testbench(entry, audit, pdk_file=pdk_file,
                         supply_voltage=vdd, temperature_c=temperature_c,
                         c_load_f=c_load_f)
    rec.netlist_hash = _h((entry.path / "netlist.sp").read_text(encoding="utf-8", errors="replace"))
    rec.testbench_hash = _h(tb)
    rec.validation_stage = "testbench_ready"
    run_dir.mkdir(parents=True, exist_ok=True)
    cir = run_dir / "tb.cir"
    cir.write_text(tb, encoding="utf-8")
    rec.simulation_status = "started"
    try:
        proc = subprocess.run([exe, "-b", str(cir)], capture_output=True, text=True,
                              timeout=timeout_s, cwd=str(run_dir), check=False)
        stdout, stderr = proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = ""
        timed_out = True
    log = run_dir / "ngspice.log"
    log.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
    rec.raw_output_paths = [str(cir), str(log)]
    parsed = parse_ngspice_stdout(stdout, stderr)
    if timed_out:
        rec.failure_class, rec.simulation_status = "timeout", "timeout"
        rec.electrical_validation_status = "validation_failed"
        return rec
    v = parsed.values
    vout = v.get("v_opout")
    ivdd = v.get("v1_branch")
    # Hardened frequency-domain metrics from the complex transfer function.
    from agentic_raptor.electrical.measurements import load_wrdata_complex, measure_all

    metric_reports: dict[str, Any] = {}
    acfile = run_dir / "acdata.txt"
    if acfile.is_file():
        try:
            freq, hcplx = load_wrdata_complex(acfile)
            metric_reports = {k: m.to_dict() for k, m in measure_all(freq, hcplx).items()}
        except (ValueError, OSError) as exc:
            metric_reports = {"error": {"metric": "all", "status": "measurement_failed",
                                        "failure_reason": f"invalid_transfer_function: {exc}"}}
    def _verified(name: str) -> float | None:
        m = metric_reports.get(name)
        return m["value"] if m and m.get("status") == "verified" else None

    idd_a = abs(ivdd) if ivdd is not None else None
    rec.extracted_metrics = {
        "dc_gain_db": _verified("dc_gain_db") if metric_reports else v.get("dcgain_db"),
        "ugbw_hz": _verified("ugbw_hz"),
        "phase_margin_deg": _verified("phase_margin_deg"),  # verified-only; else null
        "gain_margin_db": _verified("gain_margin_db"),
        "f3db_hz": _verified("f3db_hz"),
        # idd_a: the TOTAL measured supply current (|V1 branch current| from
        # the op-point) -- V1 is the circuit's one supply rail (V2 is a 0 V
        # ground reference, not a second current path), so this is already
        # the aggregate, not an approximation from a design/optimizer knob
        # such as Ibias. quiescent_power_w is derived from the SAME idd_a.
        "idd_a": idd_a,
        "quiescent_power_w": idd_a * vdd if idd_a is not None else None,
        "output_dc_v": vout,
    }
    rec.__dict__["metric_reports"] = metric_reports  # full status/confidence detail
    dc_ok = vout is not None and 0.02 < vout < (vdd - 0.02) and ivdd is not None
    ac_ok = rec.extracted_metrics["dc_gain_db"] is not None
    rec.simulation_status = "converged" if (dc_ok or ac_ok) else "failed"
    if not (dc_ok or ac_ok):
        rec.failure_class = parsed.failure_type or (
            "output_railed" if vout is not None else "operating_point_failure")
        rec.failure_message = parsed.failure_message
        rec.electrical_validation_status = "validation_failed"
        return rec
    rec.validation_stage = "dc_operating_point_valid" if dc_ok else "simulation_converged"
    if ac_ok:
        rec.analyses_completed = ["op", "ac"] if dc_ok else ["ac"]
        rec.validation_stage = "ac_analysis_valid"
        gain = rec.extracted_metrics["dc_gain_db"]
        if dc_ok and gain is not None and gain > 0:
            rec.electrical_validation_status = "electrically_functional"
        else:
            rec.electrical_validation_status = "validation_failed"
            rec.failure_class = "invalid_ac_transfer" if gain is None else "gain_below_unity"
    else:
        rec.analyses_completed = ["op"]
        rec.electrical_validation_status = "validation_failed"
        rec.failure_class = parsed.failure_type or "invalid_ac_transfer"
    return rec


def run_stage3b(data_root: Path | None = None, timeout_s: float = 180.0) -> dict[str, Any]:
    root = data_root or _ROOT / "datasets" / "simulation_memory"
    registry = TopologyRegistry(_ROOT / "datasets" / "topology_library")
    split_map: dict[str, str] = {}
    split_file = _ROOT / "datasets" / "topology_splits" / "family_split.json"
    if split_file.is_file():
        s = json.loads(split_file.read_text())
        for k in ("train", "validation", "test"):
            for t in s.get(k, []):
                split_map[t] = k
    families = [t for t in registry.filter_by_source("analoggym")
                if registry.get_topology(t).has_netlist]
    env_id = capture_environment(root)
    exe = discover_ngspice()
    runs, audits = [], []
    for tid in families:
        entry = registry.get_topology(tid)
        audit = audit_family(entry)
        audits.append(audit)
        run_dir = _ROOT / "artifacts" / "stage3b" / "runs" / tid
        rec = qualify_family(entry, audit, run_dir, exe, env_id, split_map.get(tid, "unknown"), timeout_s) \
            if exe else ElectricalValidationRecord(
                topology_id=tid, graph_hash=audit["graph_hash"], source="analoggym",
                simulator="ngspice", simulator_version=None, run_id=f"run-{uuid.uuid4().hex[:10]}",
                timestamp=time.time(), netlist_hash=None, testbench_hash=None,
                validation_stage="missing_dependency", structural_validation_status="structurally_loaded",
                electrical_validation_status="missing_dependency", simulation_status="blocked",
                failure_class="simulator_missing", environment_id=env_id,
                split=split_map.get(tid, "unknown"))
        runs.append(rec)
        # per-topology electrical_validation.json (references, atomic-ish)
        ev = entry.path / "electrical_validation.json"
        tmp = ev.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(rec), indent=1, default=str), encoding="utf-8")
        tmp.replace(ev)
    # level-4 memory
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runs.jsonl").open("w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(asdict(r), default=str) + "\n")
    with (root / "failures.jsonl").open("w", encoding="utf-8") as f:
        for r in runs:
            if r.failure_class:
                f.write(json.dumps({"run_id": r.run_id, "topology_id": r.topology_id,
                                    "failure_class": r.failure_class, "stage": r.validation_stage,
                                    "message": r.failure_message, "retryable": r.failure_class in ("timeout",),
                                    "repairable": False,
                                    "recommended_next_action": "diagnose logs; no automatic circuit changes"}) + "\n")
    with (root / "topology_summaries.jsonl").open("w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps({
                "topology_id": r.topology_id, "latest_run_id": r.run_id,
                "electrical_validation_status": r.electrical_validation_status,
                "successful_analyses": r.analyses_completed,
                "verified_metrics": {k: v for k, v in r.extracted_metrics.items() if v is not None},
                "failure_class": r.failure_class, "simulator": "ngspice",
                "environment_id": r.environment_id, "split": r.split}) + "\n")
    functional = [r for r in runs if r.electrical_validation_status == "electrically_functional"]
    by_class: dict[str, int] = {}
    for r in runs:
        if r.failure_class:
            by_class[r.failure_class] = by_class.get(r.failure_class, 0) + 1
    summary = {"families_audited": len(families), "attempted": len(runs),
               "electrically_functional": len(functional),
               "failures_by_class": by_class,
               "splits": {r.topology_id: r.split for r in runs}}
    (root / "index_metadata.json").write_text(json.dumps(
        {"level4_records": len(runs), **summary}, indent=1, default=str), encoding="utf-8")
    (root / "build_report.md").write_text(
        "# Level-4 simulation memory\n" + json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return {"summary": summary, "audits": audits, "runs": [asdict(r) for r in runs]}


if __name__ == "__main__":
    out = run_stage3b()
    print(json.dumps(out["summary"], indent=1, default=str))
