"""STAGE 6 v1 -- COMMON JUDGE over Track-A topologies (2026-08-28).

One shared measurement protocol applied to every generator's output:
  deck = the generated netlist AS-IS (its own device models, disclosed)
       + the SPEC's load capacitor (overriding any generator-chosen CL)
       + one shared AC testbench (ac source on Vinp, Vinn at bias)
       + one shared analysis (.op + .ac dec 10 1 10GHz)
  metrics = DC-gain (low-freq |vout|), UGBW (unity crossing), PM
  judge   = the spec's gain/UGBW/PM targets  ->  FinalPass (as-generated)
  corners = VT sweep (vdd +/-10% x 0/70C) on every passing design

v1 scope honesty:
  * "as-generated" feasibility -- no tuning yet (standardized sizing = v2);
  * ACP: fully measurable (self-contained level-1 netlists);
  * AG: judge results read from the frozen campaign traces (same ngspice
    judge family, sky130); PVT pending (campaign was nominal-only);
  * PANDA: NOT electrically measurable as-generated -- its cell-library
    topologies omit bias-voltage values by design (their flow assigns
    biases during Spectre sizing). Deferred to Stage-6 v2 with the tuner;
    recorded per-row as 'requires_sizing_stage'.
Every real ngspice invocation is counted. Output: stage6_results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "external_baselines" / "stage6_results.jsonl"
TRACK_A = ROOT / "artifacts" / "external_baselines" / "track_a_full.jsonl"
TRACES = ROOT / "artifacts" / "publication_v2" / "raptor_v2_runs"
NGSPICE = r"C:/Users/kobeo/OneDrive/Desktop/Spice64/bin/ngspice_con.exe"
CORNERS = {"LL": (0.9, 0), "LH": (0.9, 70), "HL": (1.1, 0), "HH": (1.1, 70)}


def _specs():
    d = json.loads((ROOT / "data/external_baseline_eval/specs_validation.json"
                    ).read_text(encoding="utf-8"))
    return {s["context_id"]: s for s in d["specs"]}


def _spice_count():
    if not OUT.exists():
        return 0
    return sum(json.loads(l).get("spice_calls") or 0
               for l in OUT.read_text(encoding="utf-8").splitlines())


def measure_acp_netlist(net_text: str, cl_pf: float, vdd_scale: float = 1.0,
                        temp_c: float | None = None) -> dict | None:
    """Shared judge for a self-contained flat netlist: impose spec CL, add
    AC testbench + control, run ngspice, extract gain/UGBW/PM."""
    lines = []
    for ln in net_text.splitlines():
        s = ln.strip()
        if re.match(r"^C\w*\s+Vout\s+0\s", s, re.I):
            continue                       # drop generator-chosen load
        if vdd_scale != 1.0:
            m = re.match(r"^(V\w+\s+\S+\s+\S+\s+)([0-9.eE+-]+)\s*$", s)
            if m and "vdd" in s.lower():
                s = f"{m.group(1)}{float(m.group(2)) * vdd_scale}"
        lines.append(s)
    body = "\n".join(lines)
    # ac stimulus: ride the existing Vinp source (replace with dc+ac)
    body = re.sub(r"(?im)^(Vinp\s+\S+\s+\S+\s+)([0-9.eE+-]+)\s*$",
                  lambda m: f"{m.group(1)}{m.group(2)} ac 1", body, count=1)
    deck = ["* stage6 shared judge", body,
            f"CLOAD_S6 Vout 0 {cl_pf}p"]
    if temp_c is not None:
        deck.append(f".temp {temp_c}")
    deck += [".control", "op", "ac dec 20 1 10G",
             "wrdata s6_out.csv vdb(Vout) cph(Vout)", "quit", ".endc", ".end"]
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "s6.cir"
        f.write_text("\n".join(deck), encoding="utf-8")
        try:
            subprocess.run([NGSPICE, "-b", str(f)], cwd=td,
                           capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return None
        outf = Path(td) / "s6_out.csv"
        if not outf.exists():
            return None
        rows = []
        for ln in outf.read_text().splitlines():
            p = ln.split()
            if len(p) >= 4:
                try:
                    rows.append((float(p[0]), float(p[1]), float(p[3])))
                except ValueError:
                    pass
    if not rows:
        return None
    gain_db = rows[0][1]
    ugbw = None
    pm = None
    for i in range(1, len(rows)):
        if rows[i - 1][1] >= 0 > rows[i][1]:
            f0, g0 = rows[i - 1][0], rows[i - 1][1]
            f1, g1 = rows[i][0], rows[i][1]
            t = g0 / (g0 - g1)
            ugbw = f0 * (f1 / f0) ** t     # log-interp crossing
            ph = rows[i - 1][2] + t * (rows[i][2] - rows[i - 1][2])
            pm = 180.0 + math.degrees(ph) if abs(ph) < 7 else 180.0 + ph
            pm = ((pm + 180) % 360) - 180
            pm = abs(pm)
            break
    return {"gain_db": gain_db, "ugbw_hz": ugbw, "pm_deg": pm}


def judge(meas: dict | None, p: dict) -> tuple[bool, dict]:
    if not meas or meas.get("ugbw_hz") is None:
        return False, {"why": "no unity crossing / sim failed"}
    ok = (meas["gain_db"] >= p["gain_target_db"]
          and meas["ugbw_hz"] >= p["ugbw_target_hz"]
          and (meas["pm_deg"] or 0) >= p["phase_margin_target_deg"])
    return ok, {}


def part_acp() -> None:
    specs = _specs()
    rows = [json.loads(l) for l in TRACK_A.read_text(encoding="utf-8").splitlines()]
    acp = [r for r in rows if r["baseline"] == "analogcoderpro_specaligned"
           and r["valid_graph"] and r.get("netlist_path")]
    with OUT.open("a", encoding="utf-8") as f:
        for i, r in enumerate(acp):
            spec = specs.get(r["spec_id"])
            if spec is None:
                continue
            p = spec["parsed_spec"]
            npth = Path(r["netlist_path"])
            if not npth.exists():
                f.write(json.dumps({
                    "baseline": "analogcoderpro", "spec_id": r["spec_id"],
                    "seed": r["seed"], "stage": "netlist_lost",
                    "final_pass": None, "spice_calls": 0,
                    "notes": "raw netlist wiped by pre-fix staging cleanup; "
                             "regenerated in the archived rerun"}) + "\n")
                continue
            net = npth.read_text(encoding="utf-8", errors="replace")
            t0 = time.time()
            calls = 0
            meas = measure_acp_netlist(net, p["load_capacitance_pf"])
            calls += 1
            ok, _ = judge(meas, p)
            corners_pass = None
            if ok:
                corners_pass = 0
                for cname, (vs, tc) in CORNERS.items():
                    m2 = measure_acp_netlist(net, p["load_capacitance_pf"],
                                             vdd_scale=vs, temp_c=tc)
                    calls += 1
                    o2, _ = judge(m2, p)
                    corners_pass += int(o2)
            fom = None
            if meas and meas.get("ugbw_hz"):
                fom = ((meas["gain_db"] - p["gain_target_db"]) / max(p["gain_target_db"], 1)
                       + (meas["ugbw_hz"] - p["ugbw_target_hz"]) / max(p["ugbw_target_hz"], 1)
                       + ((meas["pm_deg"] or 0) - p["phase_margin_target_deg"])
                       / max(p["phase_margin_target_deg"], 1))
            f.write(json.dumps({
                "baseline": "analogcoderpro", "spec_id": r["spec_id"],
                "seed": r["seed"], "stage": "as_generated_common_judge",
                "meas": meas, "final_pass": ok, "fom_rel": fom,
                "vt_corners_pass": corners_pass, "spice_calls": calls,
                "runtime_s": round(time.time() - t0, 2)}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(acp)}] ...", flush=True)
    print(f"acp: judged {len(acp)} netlists; total stage6 spice={_spice_count()}")


def part_ag() -> None:
    """AG rows from the frozen campaign traces: the SAME judge family
    (qualify_device_graph ngspice testbench, sky130) already measured every
    run's returned design; extract nominal pass/FoM per (spec, seed)."""
    import glob as _g
    results_files = sorted((ROOT / "artifacts/publication_v3/ablation_v3"
                            ).glob("results_TIER3_merged.jsonl"))
    rows = [json.loads(l) for l in results_files[0].read_text(
        encoding="utf-8").splitlines()] if results_files else []
    a0 = [r for r in rows if r.get("ablation_id") == "A0"]
    with OUT.open("a", encoding="utf-8") as f:
        for r in a0:
            nom = r.get("nominal") or {}
            f.write(json.dumps({
                "baseline": "agentic_raptor", "spec_id": f"heldout_{r['spec_index']}",
                "seed": r.get("pipeline_seed"), "stage": "native_pipeline_judged",
                "meas": {"gain_db": nom.get("gain_db"), "ugbw_hz": nom.get("ugbw_hz"),
                         "pm_deg": nom.get("pm_deg")},
                "final_pass": bool(nom.get("complete_pass")),
                "fom_rel": (r.get("fom") or {}).get("fom_value"),
                "vt_corners_pass": None, "spice_calls": 0,
                "notes": "frozen Tier-3 A0 measurement (sky130 judge); "
                         "0 NEW spice; PVT pending"}) + "\n")
    print(f"ag: extracted {len(a0)} judged rows (0 new spice)")


def part_panda() -> None:
    rows = [json.loads(l) for l in TRACK_A.read_text(encoding="utf-8").splitlines()]
    pd = [r for r in rows if r["baseline"] in ("panda", "panda_llm")
          and r.get("valid_graph")]
    with OUT.open("a", encoding="utf-8") as f:
        for r in pd:
            f.write(json.dumps({
                "baseline": r["baseline"], "spec_id": r["spec_id"],
                "seed": r["seed"], "stage": "requires_sizing_stage",
                "final_pass": None, "spice_calls": 0,
                "notes": "PANDA topologies omit bias values by design (their "
                         "flow biases during sizing); electrical verdict "
                         "deferred to Stage-6 v2 tuner"}) + "\n")
    print(f"panda: {len(pd)} rows marked requires_sizing_stage (documented)")


def part_summary() -> None:
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    by = defaultdict(list)
    for r in rows:
        by[r["baseline"]].append(r)
    print(f"\n=== STAGE-6 v1 COMMON-JUDGE SUMMARY ({len(rows)} rows) ===")
    for b, rs in sorted(by.items()):
        n = len(rs)
        judged = [r for r in rs if r.get("final_pass") is not None]
        ok = sum(1 for r in judged if r["final_pass"])
        spice = sum(r.get("spice_calls") or 0 for r in rs)
        specs_any = defaultdict(bool)
        for r in judged:
            specs_any[r["spec_id"]] |= bool(r["final_pass"])
        anyp = sum(specs_any.values())
        foms = sorted(r["fom_rel"] for r in judged
                      if r.get("fom_rel") is not None and r["final_pass"])
        med = foms[len(foms)//2] if foms else None
        vt = [r["vt_corners_pass"] for r in judged
              if r.get("vt_corners_pass") is not None]
        vt4 = sum(1 for v in vt if v == 4)
        print(f"{b:22s} rows={n:4d} judged={len(judged):4d} pass={ok:4d} "
              f"specs_any_pass={anyp:3d}/{len(specs_any) or '-'} "
              f"fom_med={med if med is None else round(med,2)} "
              f"vt_4of4={vt4}/{len(vt) if vt else '-'} spice={spice}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["acp", "ag", "panda", "summary"])
    a = ap.parse_args()
    {"acp": part_acp, "ag": part_ag, "panda": part_panda,
     "summary": part_summary}[a.part]()
