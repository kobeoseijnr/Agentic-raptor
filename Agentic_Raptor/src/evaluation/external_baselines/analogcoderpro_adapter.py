"""AnalogCoder-Pro adapter (STAGE 5).

Invokes the UNMODIFIED baseline (external_baselines/AnalogCoderPro/run.py)
via subprocess in its own working directory, then parses its native output
tree (<model>/p<task>/<it>/...) into ExternalTopologyResult rows.

Adaptations (documented, environment-level only):
  * PYTHONUTF8=1 -- their file writes crash on Windows cp1252 otherwise;
  * default PySpice mode (their --ngspice mode is broken as shipped:
    prompt_template_ngspice.md missing at the frozen commit).

The baseline's own retry/diagnosis loop counts as its own repair and is
reported via llm_calls/retries; nothing external repairs its outputs.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from .schema import ExternalTopologyResult, count_devices, netlist_topology_hash

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "external_baselines" / "AnalogCoderPro"
STAGING = ROOT / "external_baselines" / "_staging" / "acp_specrun"
PYTHON = os.environ.get("AGR_EVAL_PYTHON",
                        r"C:\Users\kobeo\AppData\Local\Python\pythoncore-3.14-64\python.exe")
SPEC_TASK_ID = 30            # free non-optimize id (<=50) in their scheme
TSV_HEADER = "Id\tLevel\tCircuit\tInput\tOutput\tType\tSubmodule Name\tTestbench\tNormal"


def spec_to_circuit_text(spec_entry: dict) -> str:
    """Frozen AG spec -> their Circuit description field (mirrors their own
    quantitative phrasing style, e.g. task 54). Input adaptation only."""
    p = spec_entry.get("parsed_spec") or {}
    return ("a multi-stage CMOS operational amplifier (single-ended output) "
            f"achieving DC gain >= {p.get('gain_target_db')} dB, unity-gain "
            f"bandwidth >= {p.get('ugbw_target_hz'):.0f} Hz, and phase margin "
            f">= {p.get('phase_margin_target_deg')} degrees while driving a "
            f"{p.get('load_capacitance_pf')} pF load capacitor "
            "(preventing positive feedback)")


def run_frozen_spec(spec_entry: dict, n_samples: int, model: str = "gpt-5-mini",
                    timeout_s: int = 2400) -> list[ExternalTopologyResult]:
    """Spec-aligned Track-A run (option A): write a single-row problem_set.tsv
    into the STAGING copy (their checkout stays pristine; staging run.py is
    hash-verified byte-identical), run their unmodified pipeline, parse the
    exported *_netlist.sp per sample."""
    from .schema import common_netlist_graph_check, topology_iso_key
    spec_id = spec_entry.get("context_id", "?")
    row = "\t".join([str(SPEC_TASK_ID), "Hard", spec_to_circuit_text(spec_entry),
                     "Vinp, Vinn", "Vout", "Opamp", "SpecOpamp", "NA", "NA"])
    (STAGING / "problem_set.tsv").write_text(TSV_HEADER + "\n" + row + "\n",
                                             encoding="utf-8")
    out_root = STAGING / model / f"p{SPEC_TASK_ID}"
    if out_root.exists():
        import shutil
        shutil.rmtree(out_root)
    env = dict(os.environ, PYTHONUTF8="1")
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, "run.py", "--task_id", str(SPEC_TASK_ID),
         "--num_per_task", str(n_samples), "--model", model],
        cwd=STAGING, env=env, capture_output=True, text=True, timeout=timeout_s)
    wall = time.time() - t0
    rows: list[ExternalTopologyResult] = []
    if not out_root.exists():
        return [ExternalTopologyResult(
            baseline="analogcoderpro_specaligned", spec_id=spec_id, seed=0,
            valid_syntax=False,
            notes=f"no output; rc={proc.returncode}; stderr={proc.stderr[-250:]}")]
    for it_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        it = it_dir.name
        sp = sorted(it_dir.glob("*_netlist.sp"))
        success = list(it_dir.glob("*_success.py"))
        tokens = retries = None
        tok_files = (sorted(it_dir.glob("token_summary*.txt"))
                     or sorted(it_dir.glob("token_info*.txt")))
        texts = [f.read_text(encoding="utf-8", errors="replace")
                 for f in tok_files]
        for t in texts:
            m = re.search(r"Total tokens:\s*(\d+)", t)
            if m:
                tokens = (tokens or 0) + int(m.group(1))
            m = re.search(r"Total retries:\s*(\d+)", t)
            if m:
                retries = max(retries or 0, int(m.group(1)))
        net = sp[0].read_text(encoding="utf-8", errors="replace") if sp else ""
        # RAW PRESERVATION (fix 2026-08-28): the staging out_root is wiped per
        # spec, so archive every sample netlist + full raw dir listing into
        # artifacts BEFORE the next spec destroys it.
        arch_dir = ROOT / "artifacts" / "external_baselines" / "raw" / "acp"
        arch_dir.mkdir(parents=True, exist_ok=True)
        arch = arch_dir / f"{spec_id}_s{it}.sp"
        if net:
            arch.write_text(net, encoding="utf-8")
        g_ok, g_why = common_netlist_graph_check(net) if net else (False, "no netlist")
        rows.append(ExternalTopologyResult(
            baseline="analogcoderpro_specaligned", spec_id=spec_id,
            seed=int(it) if it.isdigit() else 0,
            raw_output_path=str(it_dir),
            netlist_path=str(arch) if net else "",
            valid_syntax=bool(sp),
            valid_graph=g_ok,
            simulatable=bool(success),
            topology_hash=topology_iso_key(net) if net else "",
            num_devices=count_devices(net) if net else None,
            generation_runtime_s=round(wall / max(1, n_samples), 2),
            llm_calls=(1 + (retries or 0)),
            llm_tokens=tokens,
            notes=f"spec-aligned (option A); graph_check={g_why}; "
                  "functional=their own Opamp check"))
    return rows


def run_task(task_id: int, n_samples: int, model: str = "gpt-5-mini",
             timeout_s: int = 1800) -> list[ExternalTopologyResult]:
    """Run one native problem for n_samples and parse every sample produced."""
    t0 = time.time()
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.run(
        [PYTHON, "run.py", "--task_id", str(task_id),
         "--num_per_task", str(n_samples), "--model", model],
        cwd=BASELINE, env=env, capture_output=True, text=True, timeout=timeout_s)
    wall = time.time() - t0
    out_dir = BASELINE / model / f"p{task_id}"
    rows: list[ExternalTopologyResult] = []
    if not out_dir.exists():
        rows.append(ExternalTopologyResult(
            baseline="analogcoderpro", spec_id=f"native_p{task_id}", seed=0,
            valid_syntax=False,
            notes=f"no output dir; rc={proc.returncode}; "
                  f"stderr_tail={proc.stderr[-300:]}"))
        return rows
    for it_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        it = it_dir.name
        success = list(it_dir.glob("*_success.py"))
        netlists = list(it_dir.glob("*_netlist_gen.py")) or success or \
            sorted(it_dir.glob(f"p{task_id}_{it}_*.py"))
        tokens = None
        tok_f = it_dir / "token_summary_final.txt"
        if tok_f.exists():
            m = re.search(r"Total tokens:\s*(\d+)", tok_f.read_text(encoding="utf-8",
                                                                    errors="replace"))
            tokens = int(m.group(1)) if m else None
        retries = None
        if tok_f.exists():
            m = re.search(r"Total retries:\s*(\d+)",
                          tok_f.read_text(encoding="utf-8", errors="replace"))
            retries = int(m.group(1)) if m else None
        code_text = (netlists[0].read_text(encoding="utf-8", errors="replace")
                     if netlists else "")
        rows.append(ExternalTopologyResult(
            baseline="analogcoderpro",
            spec_id=f"native_p{task_id}", seed=int(it) if it.isdigit() else 0,
            raw_output_path=str(it_dir),
            netlist_path=str(netlists[0]) if netlists else "",
            valid_syntax=bool(netlists),
            simulatable=bool(success),      # their functional check passed
            topology_hash=netlist_topology_hash(code_text) if code_text else "",
            num_devices=count_devices(code_text) if code_text else None,
            generation_runtime_s=round(wall / max(1, n_samples), 2),
            llm_calls=(1 + (retries or 0)),
            llm_tokens=tokens,
            spice_calls=None,   # PySpice sims are internal to their check loop
            notes=f"native problem {task_id}; their retry loop = their own repair"))
    return rows
