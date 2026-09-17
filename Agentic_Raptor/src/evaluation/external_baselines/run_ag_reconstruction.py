"""AG winner reconstruction: native PVT cell + symmetric common-tuner FoM arm.

Every Tier-3 trace records the winning branch's canonical family + full
7-knob sizing_vector. The winning DESIGN is rebuilt through the pipeline's
own deterministic machinery (map_family family template -> apply_knobs), then
GATED: the rebuilt design's nominal measurement must match the trace's
recorded nominal (|gain delta| <= 2 dB, UGBW ratio in [0.5, 2]) before any
number is reported. Winners that fail the gate (e.g. tier-2 cascode/class-AB
structural blocks not encoded in the family string, or AlphaZero-edited
graphs) are EXCLUDED AND COUNTED -- never silently substituted.

Per verified design:
  native arm    : 4 VT corners on the rebuilt design  -> the AG PVT cell
  symmetric arm : 13-call common tune (same protocol as baselines' per-
                  candidate budget) + corners on the tuned winner -> the
                  symmetric FoM entry
Output: artifacts/topology_baselines/ag_reconstruction.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation.external_baselines.run_stage6_v2_tuner import (  # noqa: E402
    CORNERS, run_deck, score)
from src.evaluation.external_baselines.run_topology_baselines import (  # noqa: E402
    specs29, subckt_builder, tune_budget)

ART = ROOT / "artifacts" / "topology_baselines"
OUT = ART / "ag_reconstruction.jsonl"
MERGED = ROOT / "artifacts/publication_v3/ablation_v3/results_TIER3_merged.jsonl"
TRACES = ROOT / "artifacts/publication_v2/raptor_v2_runs"


def build_design(family: str, sizing_vector: dict):
    from agentic_raptor.mapping import map_family, emit_netlist
    from agentic_raptor.mb_sac.spec_sizing import KNOB_NAMES, apply_knobs
    n = int(family[0])
    comp = family.split("_", 1)[1]
    blocks = {"none": [], "miller": ["C"], "rc": ["RC_series"]}[comp]
    g, status = map_family(
        SimpleNamespace(topology_id="agrecon"),
        {"topology_id": "agrecon", "gain_stages": n,
         "functional_blocks": blocks, "unresolved_blocks": [],
         "mapping_readiness": "mapping_ready", "graph_hash": None,
         "cascode_input": False, "class_ab_output": False})
    if g is None:
        return None, status
    knobs = [sizing_vector.get(k, 1.0) for k in KNOB_NAMES]
    g2 = apply_knobs(g, knobs)
    return emit_netlist(g2, "extckt"), "ok"


def main(arm: str, limit: int | None):
    sp = specs29()
    merged = [json.loads(l) for l in MERGED.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in merged if r["ablation_id"] == arm]
    done = set()
    if OUT.exists():
        done = {(json.loads(l)["arm"], json.loads(l)["seed"],
                 json.loads(l)["spec_index"])
                for l in OUT.read_text(encoding="utf-8").splitlines()
                if l.strip()}
    n_run = 0
    with OUT.open("a", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: (x["pipeline_seed"],
                                             x["spec_index"])):
            i, seed = r["spec_index"], r["pipeline_seed"]
            if (arm, seed, i) in done:
                continue
            if limit and n_run >= limit:
                break
            n_run += 1
            p = sp[i]
            cl = p["load_capacitance_pf"]
            nom = r.get("nominal") or {}
            fam, sel = r.get("selected_family"), r.get("selected_hash")
            # sizing_vector: from the trace's stage6 branch matching the
            # selected hash
            sv = None
            ms = sorted(TRACES.glob(
                f"ABLv3HELDOUT29_{arm}_s{seed}_heldout_{i:03d}_*.json"))
            if ms:
                tr = json.loads(ms[-1].read_text(encoding="utf-8"))
                for br in (tr.get("stage6_sizing") or {}).values():
                    if isinstance(br, dict) and br.get("topology_hash") == sel:
                        sv = br.get("sizing_vector")
            row = {"arm": arm, "seed": seed, "spec_index": i, "family": fam,
                   "selected_hash": sel}
            if not (fam and sv and nom.get("gain_db") is not None):
                row.update({"status": "missing_inputs",
                            "have": {"family": bool(fam), "sizing": bool(sv),
                                     "nominal": nom.get("gain_db") is not None}})
                f.write(json.dumps(row) + "\n")
                f.flush()
                continue
            body, st = build_design(fam, sv)
            if body is None:
                row.update({"status": "realize_failed", "why": st})
                f.write(json.dumps(row) + "\n")
                f.flush()
                continue
            build = subckt_builder(cl)
            t0 = time.time()
            m0 = run_deck(build(body), "opout")
            calls = 1
            gd = (abs((m0 or {}).get("gain_db", -999)
                      - nom["gain_db"]) if m0 else 999)
            ur = ((m0.get("ugbw_hz") or 0) / nom["ugbw_hz"]
                  if m0 and m0.get("ugbw_hz") and nom.get("ugbw_hz")
                  else 0)
            gate_ok = gd <= 2.0 and 0.5 <= ur <= 2.0
            row.update({"gate_gain_delta_db": round(gd, 2),
                        "gate_ugbw_ratio": round(ur, 3),
                        "gate_ok": gate_ok,
                        "rebuilt_meas": m0, "trace_nominal": {
                            k: nom.get(k) for k in ("gain_db", "ugbw_hz",
                                                    "pm_deg")}})
            if not gate_ok:
                row["status"] = "gate_failed_excluded"
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"[{arm} s{seed} i{i}] GATE FAIL "
                      f"dGain={gd:.1f}dB ugbw_ratio={ur:.2f}", flush=True)
                continue
            ok0, fom0 = score(m0, p)
            # native arm: VT corners on the rebuilt design
            nat = 0
            for vs, tc in CORNERS.values():
                mc = run_deck(build(body, vdd_scale=vs, temp_c=tc), "opout")
                calls += 1
                okc, _ = score(mc, p)
                nat += int(okc)
            # symmetric arm: same 13-call tune as baseline candidates
            t = tune_budget(build, body,
                            ["w", "cap", "res", "isrc", "mmul"], p, "opout",
                            9000 + seed * 100 + i, 13)
            calls += t["calls"]
            sym_c = None
            if t["pass"]:
                sym_c = 0
                for vs, tc in CORNERS.values():
                    mc = run_deck(build(t["body"], vdd_scale=vs, temp_c=tc),
                                  "opout")
                    calls += 1
                    okc, _ = score(mc, p)
                    sym_c += int(okc)
            row.update({"status": "ok",
                        "native_pass": ok0, "native_fom": round(fom0, 4),
                        "native_vt_pass": nat,
                        "sym_pass": t["pass"], "sym_fom": t["fom"],
                        "sym_vt_pass": sym_c,
                        "recon_spice_calls": calls,
                        "runtime_s": round(time.time() - t0, 1)})
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{arm} s{seed} i{i}] {fam} gate OK (d={gd:.1f}dB) "
                  f"native pass={ok0} vt={nat}/4 | sym pass={t['pass']} "
                  f"fom={t['fom']} vt={sym_c}", flush=True)
    print("done:", n_run, "designs processed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["AG_FULL", "A0"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.arm, a.limit)
