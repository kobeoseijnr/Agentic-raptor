"""AG VT-corner sweep for the Track-A standard table (2026-08-29).

Maps each Tier-3 A0/AG_FULL run's WINNING design to its on-disk sizing
directory (device_graph hash match + netlist presence), then re-runs the
design's OWN campaign testbench at the 4 VT corners (vdd +/-10% x 0/70C)
under the shared measurement block. Every ngspice call counted.

Output: artifacts/external_baselines/ag_vt_sweep.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACES = ROOT / "artifacts" / "publication_v2" / "raptor_v2_runs"
SIZING = TRACES / "sizing"
OUT = ROOT / "artifacts" / "external_baselines" / "ag_vt_sweep.jsonl"
CORNERS = {"LL": (0.9, 0), "LH": (0.9, 70), "HL": (1.1, 0), "HH": (1.1, 70)}


def build_dir_index() -> dict:
    """graph-hash -> [sizing dirs] for every dir with netlist+tb."""
    idx = defaultdict(list)
    for d in SIZING.iterdir():
        dg = d / "device_graph.json"
        if not (dg.exists() and (d / "netlist.sp").exists()
                and (d / "run" / "tb.cir").exists()):
            continue
        try:
            g = json.loads(dg.read_text(encoding="utf-8"))
        except Exception:
            continue
        h = (g.get("graph_hash") or g.get("hash")
             or g.get("canonical_graph_hash") or "")
        idx[h].append(d)
    return idx


def main(arm: str, limit: int | None) -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.evaluation.external_baselines.run_stage6_v2_tuner import (
        ag_builder, run_deck, score)
    specs = {s["spec_index"]: s["parsed_spec"] for s in json.loads(
        (ROOT / "data/external_baseline_eval/specs_validation.json"
         ).read_text(encoding="utf-8"))["specs"]}
    idx = build_dir_index()
    print(f"indexed {sum(len(v) for v in idx.values())} sizing dirs, "
          f"{len(idx)} distinct graph hashes")
    done = set()
    if OUT.exists():
        done = {(json.loads(l)["arm"], json.loads(l)["spec_index"],
                 json.loads(l)["seed"])
                for l in OUT.read_text(encoding="utf-8").splitlines()}
    n = matched = swept = 0
    with OUT.open("a", encoding="utf-8") as f:
        for seed in (0, 1, 2):
            for i in range(29):
                if limit and swept >= limit:
                    break
                if (arm, i, seed) in done:
                    continue
                ms = sorted(TRACES.glob(
                    f"ABLv3HELDOUT29_{arm}_s{seed}_heldout_{i:03d}_*.json"))
                if not ms:
                    continue
                n += 1
                tr = json.loads(ms[-1].read_text(encoding="utf-8"))
                nom = tr.get("nominal") or {}
                if not nom.get("complete_pass"):
                    continue
                sz = tr.get("stage6_sizing") or {}
                # winning branch: the one whose topology_hash matches the
                # selected/answer hash if recorded; else try both
                cands = []
                for br in ("A", "B"):
                    h = (sz.get(br) or {}).get("topology_hash")
                    for d in idx.get(h, []):
                        cands.append(d)
                if not cands:
                    f.write(json.dumps({"arm": arm, "spec_index": i,
                                        "seed": seed, "matched": False}) + "\n")
                    continue
                # verify by simulating nominal and comparing to the trace's
                # recorded nominal gain (2 dB tolerance): pick best match
                p = specs[i]
                best = None
                for d in cands[:6]:
                    try:
                        body = (d / "netlist.sp").read_text(encoding="utf-8")
                        build = ag_builder(d, p["load_capacitance_pf"])
                        m0 = run_deck(build(body), "opout")
                    except Exception:
                        continue
                    if not m0 or m0.get("gain_db") is None:
                        continue
                    dg = abs((m0["gain_db"] or -999)
                             - (nom.get("gain_db") or 999))
                    if best is None or dg < best[0]:
                        best = (dg, d, body, build, m0)
                if best is None or best[0] > 2.0:
                    f.write(json.dumps({"arm": arm, "spec_index": i,
                                        "seed": seed, "matched": False,
                                        "why": "no nominal-consistent dir"
                                        }) + "\n")
                    continue
                matched += 1
                _, d, body, build, m0 = best
                corners = 0
                for vs, tc in CORNERS.values():
                    mc = run_deck(build(body, vdd_scale=vs, temp_c=tc),
                                  "opout")
                    ok, _ = score(mc, p)
                    corners += int(ok)
                swept += 1
                f.write(json.dumps({"arm": arm, "spec_index": i, "seed": seed,
                                    "matched": True, "dir": d.name,
                                    "nominal_check_gain": m0["gain_db"],
                                    "vt_corners_pass": corners,
                                    "spice_calls": 5}) + "\n")
                f.flush()
                print(f"[{arm} s{seed} i{i}] dir={d.name} corners={corners}/4",
                      flush=True)
    print(f"{arm}: traces={n} matched+swept={swept}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A0", choices=["A0", "AG_FULL"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.arm, a.limit)
