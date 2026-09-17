"""PANDA adapter (STAGE 5).

Scope in THIS environment (per repository_manifest): PANDA's Spectre sizing /
Virtuoso layout chain is NOT REPRODUCIBLE here; the separable TOPOLOGY track
runs natively:
  * local template generator + native validator -- offline, exercised now;
  * AnalogXpert LLM generation -- runnable with the available OpenAI key,
    wired in the full phase (adapter records which path produced a result).

The runner executes in a child process with cwd = the PANDA checkout; the
RAPTOR runtime never imports PANDA code.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .schema import ExternalTopologyResult, count_devices, netlist_topology_hash

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "external_baselines" / "PANDA"
RUNNER = Path(__file__).with_name("_panda_runner.py")
PYTHON = os.environ.get("AGR_EVAL_PYTHON",
                        r"C:\Users\kobeo\AppData\Local\Python\pythoncore-3.14-64\python.exe")


def spec_to_panda_request(spec_entry: dict) -> dict:
    """Frozen AG spec -> PANDA design_spec (their demo JSON shape).
    Pure format conversion; nothing about their algorithm changes."""
    p = spec_entry.get("parsed_spec") or {}
    prompt = spec_entry.get("prompt", "")
    return {
        "design_intent": f"Design an amplifier meeting: {prompt.splitlines()[0][:160]}",
        "design_spec": {
            "circuit_type": "ota",
            "target_gain_db": p.get("gain_target_db"),
            "target_ugbw_mhz": (p.get("ugbw_target_hz") or 0) / 1e6 or None,
            "phase_margin_deg": p.get("phase_margin_target_deg"),
            "load_cap": f"{p.get('load_capacitance_pf', 100)}p",
        }}


def run_spec(spec_entry: dict, seed: int = 0,
             timeout_s: int = 300) -> ExternalTopologyResult:
    req = spec_to_panda_request(spec_entry)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(req, f)
        req_path = f.name
    proc = subprocess.run([PYTHON, str(RUNNER), req_path], cwd=BASELINE,
                          capture_output=True, text=True, timeout=timeout_s,
                          env=dict(os.environ, PYTHONUTF8="1"))
    spec_id = spec_entry.get("context_id", "?")
    if proc.returncode != 0:
        return ExternalTopologyResult(
            baseline="panda", spec_id=spec_id, seed=seed, valid_syntax=False,
            notes=f"runner failed rc={proc.returncode}: {proc.stderr[-300:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if not out.get("generated"):
        return ExternalTopologyResult(
            baseline="panda", spec_id=spec_id, seed=seed, valid_syntax=False,
            generation_runtime_s=out.get("wall_s"),
            notes="template path produced nothing; LLM generation path "
                  "required (full phase). " + out.get("note", ""))
    net = out["netlist"]
    raw_dir = ROOT / "artifacts" / "external_baselines" / "raw" / "panda"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"{spec_id}_s{seed}.sp"
    raw.write_text(net, encoding="utf-8")
    return ExternalTopologyResult(
        baseline="panda", spec_id=spec_id, seed=seed,
        raw_output_path=str(raw), netlist_path=str(raw),
        valid_syntax=True, valid_graph=bool(out.get("validator_ok")),
        topology_hash=netlist_topology_hash(net),
        num_devices=count_devices(net),
        generation_runtime_s=out.get("wall_s"),
        llm_calls=0,
        notes=f"generator={out.get('generator')} "
              f"({out.get('topology_summary')}); NATIVE validator verdict; "
              "Spectre sizing NOT REPRODUCIBLE here (topology-only track)")


LLM_RUNNER = Path(__file__).with_name("_panda_llm_runner.py")
QUERY_TEMPLATE = """User Query1:

Stage Numbers: choose as needed to meet the electrical targets
Compensation: choose as needed
FeadBack: {{Type: None, FB Network: None}}
InputSignal1: Differential-Ended
OutputSignal1: Single-Ended
Input Type1: choose as needed
Electrical targets: DC gain >= {gain} dB, unity-gain bandwidth >= {ugbw:.0f} Hz, phase margin >= {pm} deg, load capacitor = {cl} pF
"""


def run_spec_llm(spec_entry: dict, seed: int = 0, model: str = "gpt-5-mini",
                 timeout_s: int = 900) -> ExternalTopologyResult:
    """PANDA's REAL generative mode: their Analog_designer.single_run verbatim
    (their prompt suite + self-refine loop = their own repair). Input
    adaptation only: our numeric spec rendered into their query schema with
    structural fields left to the model (no steering). base_url pinned to
    the user's endpoint inside the runner (never their proxy default)."""
    import time as _t
    from .schema import common_netlist_graph_check, topology_iso_key
    p = spec_entry.get("parsed_spec") or {}
    spec_id = spec_entry.get("context_id", "?")
    query = QUERY_TEMPLATE.format(gain=p.get("gain_target_db"),
                                  ugbw=p.get("ugbw_target_hz") or 0,
                                  pm=p.get("phase_margin_target_deg"),
                                  cl=p.get("load_capacitance_pf"))
    raw_dir = ROOT / "artifacts" / "external_baselines" / "raw" / "panda_llm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    qf = raw_dir / f"{spec_id}_s{seed}_query.txt"
    qf.write_text(query, encoding="utf-8")
    logf = raw_dir / f"{spec_id}_s{seed}_their_log.txt"
    proc = subprocess.run(
        [PYTHON, str(LLM_RUNNER), str(qf), str(logf), model],
        cwd=BASELINE / "topology_gen", capture_output=True, text=True,
        timeout=timeout_s, env=dict(os.environ, PYTHONUTF8="1"))
    out_lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not out_lines:
        return ExternalTopologyResult(
            baseline="panda_llm", spec_id=spec_id, seed=seed,
            valid_syntax=False,
            notes=f"runner rc={proc.returncode}: {proc.stderr[-250:]}")
    out = json.loads(out_lines[-1])
    if out.get("error") or not out.get("netlist"):
        return ExternalTopologyResult(
            baseline="panda_llm", spec_id=spec_id, seed=seed,
            valid_syntax=False, generation_runtime_s=out.get("wall_s"),
            raw_output_path=str(logf),
            notes=f"no netlist: {out.get('error','empty output')}"[:200])
    net = out["netlist"]
    nf = raw_dir / f"{spec_id}_s{seed}.sp"
    nf.write_text(net, encoding="utf-8")
    g_ok, g_why = common_netlist_graph_check(net)
    return ExternalTopologyResult(
        baseline="panda_llm", spec_id=spec_id, seed=seed,
        raw_output_path=str(logf), netlist_path=str(nf),
        valid_syntax=True, valid_graph=g_ok,
        simulatable=bool(out.get("their_check_pass")),
        topology_hash=topology_iso_key(net),
        num_devices=count_devices(net),
        generation_runtime_s=out.get("wall_s"),
        llm_calls=out.get("rounds"),
        notes=f"their self-refine rounds={out.get('rounds')}; their check="
              f"{out.get('their_check_pass')}; graph_check={g_why}")
