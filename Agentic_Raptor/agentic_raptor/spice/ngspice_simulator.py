"""Real ngspice backend implementing the Stage 1 ``SpiceSimulator`` interface.

Features: executable discovery, configurable binary, temporary working
directories with deterministic file names, subprocess execution with timeout,
stdout/stderr capture, return-code validation, raw-output preservation,
cleanup policy, and structured error reporting.

If ngspice is unavailable, ``simulate`` returns a failed ``SimulationResult``
with ``error_type="simulator_unavailable"`` — it never silently falls back to
the mock (fallback is an explicit coordinator/config decision).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from agentic_raptor.core.candidate import CircuitCandidate
from agentic_raptor.spice.interface import SimulationResult
from agentic_raptor.spice.model_library import ModelLibrary, load_model_library
from agentic_raptor.spice.netlist_builder import build_circuit
from agentic_raptor.spice.result_parser import (
    compute_constraint_margins,
    metrics_from_ngspice,
    parse_ngspice_stdout,
)
from agentic_raptor.spice.testbench_builder import (
    OperatingConditions,
    build_op_ac_testbench,
    build_tran_testbench,
)
from agentic_raptor.utils.logging import get_logger

logger = get_logger("agentic_raptor.spice.ngspice")

#: Discovery order. On Windows ngspice_con is the console binary; the GUI
#: ngspice.exe can open a window even in batch mode, so the console build wins.
_CANDIDATE_NAMES = ("ngspice_con", "ngspice")
_CANDIDATE_PATHS = (
    r"C:\Spice64\bin\ngspice_con.exe",
    r"C:\Spice64\bin\ngspice.exe",
    r"C:\Program Files\Spice64\bin\ngspice_con.exe",
    "/usr/local/bin/ngspice",
    "/usr/bin/ngspice",
)


def discover_ngspice(configured: str | None = None) -> str | None:
    """Resolve the ngspice executable: config > PATH (console first) > known paths."""
    if configured:
        return configured if Path(configured).is_file() else None
    for name in _CANDIDATE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    for path in _CANDIDATE_PATHS:
        if Path(path).is_file():
            return path
    return None


def ngspice_version(exe: str, timeout_s: float = 15.0) -> str | None:
    try:
        proc = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=timeout_s, check=False
        )
        for line in proc.stdout.splitlines():
            if "ngspice" in line.lower():
                return line.strip()
        return (proc.stdout.strip().splitlines() or [None])[0]
    except (OSError, subprocess.TimeoutExpired):
        return None


@dataclass
class NgspiceRun:
    returncode: int | None
    stdout: str
    stderr: str
    runtime_s: float
    netlist_path: str
    log_path: str
    timed_out: bool = False


class NgspiceSimulator:
    """Real SPICE evaluation of typed candidates."""

    def __init__(
        self,
        ngspice_exe: str | None = None,
        model_library: ModelLibrary | None = None,
        model_library_path: str | None = None,
        technology_label: str = "generic_1u_level1",
        workdir_root: str | None = None,
        keep_workdirs: bool = False,
        seed: int = 0,
    ) -> None:
        self.exe = discover_ngspice(ngspice_exe)
        self.model_library = model_library or load_model_library(model_library_path, technology_label)
        self.workdir_root = Path(workdir_root) if workdir_root else Path(tempfile.gettempdir()) / "agentic_raptor_spice"
        self.keep_workdirs = keep_workdirs
        self.seed = seed
        self._run_counter = 0

    def is_available(self) -> bool:
        return self.exe is not None

    def config_fingerprint(self) -> dict[str, str]:
        """Participates in the cache key."""
        return {
            "backend": "ngspice",
            "exe": self.exe or "unavailable",
            "version": ngspice_version(self.exe) or "unknown" if self.exe else "unavailable",
            "model_library": self.model_library.label,
        }

    # ------------------------------------------------------------------
    def simulate(
        self,
        candidate: CircuitCandidate,
        analyses: list[str],
        timeout_s: float = 60.0,
        corner: str = "typical",
        voltage_scale: float = 1.0,
        temperature_c: float | None = None,
    ) -> SimulationResult:
        started = time.monotonic()
        if self.exe is None:
            return SimulationResult(
                success=False,
                error_type="simulator_unavailable",
                error_message=(
                    "ngspice executable not found; set spice.ngspice_exe or install ngspice "
                    "(checked PATH for ngspice_con/ngspice and common install locations)"
                ),
                corner=corner,
                seed=self.seed,
            )
        spec = candidate.specifications
        sizing = candidate.sizing_state or candidate.topology.sizing_state()
        try:
            built = build_circuit(candidate.topology, sizing, self.model_library, candidate.candidate_id)
            oc = OperatingConditions.from_spec(spec, voltage_scale, temperature_c)
        except Exception as exc:  # SimulationError from builder
            return SimulationResult(
                success=False,
                error_type="malformed_netlist",
                error_message=str(exc),
                runtime_s=time.monotonic() - started,
                corner=corner,
                seed=self.seed,
            )

        metrics: dict[str, float] = {}
        raw_paths: list[str] = []
        failure_type: str | None = None
        failure_message: str | None = None

        wants_op_ac = any(a in analyses for a in ("op", "ac"))
        wants_tran = "tran" in analyses

        if wants_op_ac:
            netlist = build_op_ac_testbench(built, candidate.topology, spec, oc)
            run = self._execute(netlist, candidate.candidate_id, corner, "op_ac", timeout_s)
            raw_paths.append(run.log_path)
            parsed = parse_ngspice_stdout(run.stdout, run.stderr)
            metrics.update(metrics_from_ngspice(parsed, oc.vdd, built.output_nodes[0]))
            required = {"gain_db", "gbw_hz", "phase_margin_deg"}
            if run.timed_out:
                failure_type, failure_message = "timeout", f"ngspice exceeded {timeout_s}s"
            elif run.returncode not in (0, None):
                failure_type = parsed.failure_type or "nonzero_return_code"
                failure_message = parsed.failure_message or f"ngspice exit code {run.returncode}"
            elif not required.issubset(metrics):
                # Failure signatures classify only runs that actually failed to
                # measure — transient notes (e.g. recovered gmin stepping) are
                # not fatal when every required measurement is present.
                failure_type = parsed.failure_type or "missing_measurement"
                failure_message = parsed.failure_message or (
                    f"measurements missing: {sorted(required - set(metrics))} "
                    f"(failed: {parsed.failed_measurements})"
                )

        if wants_tran and failure_type is None:
            try:
                netlist = build_tran_testbench(built, candidate.topology, spec, oc)
            except Exception as exc:
                failure_type, failure_message = "malformed_netlist", str(exc)
            else:
                run = self._execute(netlist, candidate.candidate_id, corner, "tran", timeout_s)
                raw_paths.append(run.log_path)
                parsed = parse_ngspice_stdout(run.stdout, run.stderr)
                tran_metrics = {
                    k: v for k, v in metrics_from_ngspice(parsed, oc.vdd, built.output_nodes[0]).items()
                    if k == "slew_rate_v_per_s"
                }
                if run.timed_out:
                    failure_type, failure_message = "timeout", f"ngspice tran exceeded {timeout_s}s"
                elif not tran_metrics:
                    failure_type = parsed.failure_type or "missing_measurement"
                    failure_message = parsed.failure_message or "slew measurement missing"
                else:
                    metrics.update(tran_metrics)

        success = failure_type is None and bool(metrics)
        margins = compute_constraint_margins(metrics, spec) if success else {}
        return SimulationResult(
            success=success,
            metrics=metrics,
            constraint_margins=margins,
            raw_output_path=";".join(raw_paths) if raw_paths else None,
            runtime_s=time.monotonic() - started,
            error_type=failure_type,
            error_message=failure_message,
            corner=corner,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    def _execute(self, netlist: str, candidate_id: str, corner: str, tag: str, timeout_s: float) -> NgspiceRun:
        self._run_counter += 1
        run_dir = self.workdir_root / f"{candidate_id}_{corner}_{tag}_{self._run_counter:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        netlist_path = run_dir / "circuit.cir"
        log_path = run_dir / "ngspice.log"
        netlist_path.write_text(netlist, encoding="utf-8")

        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                [self.exe, "-b", str(netlist_path)],  # type: ignore[list-item]
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(run_dir),
                check=False,
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            returncode, timed_out = None, True
        except OSError as exc:
            stdout, stderr, returncode = "", f"failed to launch ngspice: {exc}", -1

        runtime = time.monotonic() - started
        log_path.write_text(
            f"$ {self.exe} -b {netlist_path}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
            f"--- returncode: {returncode} timed_out: {timed_out} runtime_s: {runtime:.3f} ---\n",
            encoding="utf-8",
        )
        if not self.keep_workdirs and not timed_out and returncode == 0:
            # Keep only the log (raw-output preservation) — remove bulky intermediates.
            for child in run_dir.iterdir():
                if child.name not in ("ngspice.log", "circuit.cir"):
                    child.unlink(missing_ok=True)
        return NgspiceRun(returncode, stdout, stderr, runtime, str(netlist_path), str(log_path), timed_out)
