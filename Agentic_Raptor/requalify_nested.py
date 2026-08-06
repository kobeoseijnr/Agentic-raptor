"""Evidence-based PM verification + requalification + bandwidth optimisation."""
import copy
import itertools
import json
import numpy as np
from pathlib import Path
from agentic_raptor.electrical.measurements import load_wrdata_complex

NM = Path("artifacts/nested_miller")
GAIN_T, PM_T, UGBW_T = 73.42, 50.0, 1e6


def verified_pm(acpath: Path) -> dict:
    """Evidence-based verifier: 8 conditions, supports >90deg lead margins."""
    f, h = load_wrdata_complex(acpath)
    mag = 20 * np.log10(np.abs(h) + 1e-30)
    ph = np.unwrap(np.angle(h)) * 180 / np.pi
    cross, reasons, rej = [], [], []
    for i in range(len(f) - 1):
        if mag[i] * mag[i + 1] < 0:
            t = mag[i] / (mag[i] - mag[i + 1])
            fc = f[i] * (f[i + 1] / f[i]) ** t
            pc = ph[i] + t * (ph[i + 1] - ph[i])
            sl = (mag[i + 1] - mag[i]) / np.log10(f[i + 1] / f[i])
            cross.append({"f": float(fc), "ph": float(pc), "slope": float(sl),
                          "pm": float(pc + 180.0)})   # convention: PM = ph - (-180)
    out = {"crossing_count": len(cross),
           "all_crossing_frequencies_hz": [round(c["f"]) for c in cross],
           "all_crossing_slopes_db_per_dec": [round(c["slope"], 1) for c in cross],
           "phase_at_each_crossing_deg": [round(c["ph"], 1) for c in cross],
           "estimated_pm_candidates_deg": [round(c["pm"], 1) for c in cross]}
    desc = [c for c in cross if c["slope"] < 0]
    if not cross:
        rej.append("no_unity_crossing")
    elif not desc:
        rej.append("ascending_crossing_only")
    else:
        sel = desc[0]
        reasons.append("descending_crossing_selected")
        if f[-1] < 5 * sel["f"]:
            rej.append("out_of_frequency_range")
        else:
            reasons.append("crossing_within_sweep_bounds")
        if not (np.isfinite(sel["pm"]) and -180 < sel["pm"] < 180):
            rej.append("pm_outside_physical_branch")
        later_worse = [c for c in desc[1:] if c["pm"] < sel["pm"]]
        if later_worse:
            rej.append("later_crossing_with_smaller_margin")
        else:
            reasons.append("no_later_smaller_margin")
        if abs(ph[0] + 180) > 30 and abs(ph[0]) > 30:
            rej.append("dc_phase_unexpected")
        else:
            reasons.append("dc_phase_consistent_with_inverting_loop")
        out["selected_crossing_hz"] = round(sel["f"])
        if not rej:
            out["verified_pm_deg"] = round(sel["pm"], 1)
    out["pm_status"] = "verified" if "verified_pm_deg" in out else \
        (rej[0] if rej else "no_unity_crossing")
    out["verification_reasons"] = reasons
    out["rejection_reasons"] = rej
    return out


print("=== TASK 4: REGRESSION ON REFERENCES ===")
refs = {
    "stable_2stage(vv_2m00,expect ~+6)": "artifacts/variant_verification/vv_2m00/run/acdata.txt",
    "unstable_3stage(vv_3m00,expect ~-6)": "artifacts/variant_verification/vv_3m00/run/acdata.txt",
    "miswired_nested(sweep1)": str(NM / "c4_1_s1.0_b1.0/run/acdata.txt"),
    "corrected_nested(sweep2)": str(NM / "c1_0.15_s1.5_b2.0/run/acdata.txt"),
}
for name, p in refs.items():
    q = Path(p)
    r = verified_pm(q) if q.is_file() else {"pm_status": "missing"}
    print(name, "->", r.get("pm_status"), r.get("verified_pm_deg"),
          r.get("estimated_pm_candidates_deg"))

print("=== TASK 5: REQUALIFY CORRECTED SWEEP (stored traces) ===")
rows = [json.loads(x) for x in (NM / "sweep.jsonl").read_text().splitlines()]
requal = []
for r in rows:
    tag = f"c{r['cc1_pf']:g}_{r['cc2_pf']:g}_s{r['s23']}_b{r['ib']}"
    ac = NM / tag / "run" / "acdata.txt"
    if not ac.is_file():
        continue
    v = verified_pm(ac)
    requal.append({**r, "tag": tag, "pm_status": v["pm_status"],
                   "verified_pm_deg": v.get("verified_pm_deg")})
ver = [r for r in requal if r["verified_pm_deg"] is not None
       and r["verified_pm_deg"] > 0]
ge50 = [r for r in ver if r["verified_pm_deg"] >= PM_T]
best = max(ver, key=lambda r: (r["verified_pm_deg"] >= PM_T, r["ugbw_hz"] or 0),
           default=None)
print(f"requalified: {len(requal)} | verified PM>0: {len(ver)} | PM>=50: {len(ge50)}")
print("best:", json.dumps(best))
(NM / "requalified.jsonl").write_text("\n".join(json.dumps(r) for r in requal),
                                      encoding="utf-8")

al_path = Path("artifacts/variant_verification/ALLOWLIST.json")
al = json.loads(al_path.read_text())
promoted = bool(ge50)
al.setdefault("templates", {})["three_stage_nested_miller"] = {
    "promoted": promoted, "template_hash": "2eafaa8d95d87c81c9849d641e2ca4cb",
    "topology_verified": bool(ver), "verified_regions": ver[:10],
    "note": "sweep1 (hash 440f...) preserved as invalid-realisation campaign"}
al_path.write_text(json.dumps(al, indent=1), encoding="utf-8")
if ver:
    with Path("datasets/simulation_memory/self_improvement_runs.jsonl").open(
            "a", encoding="utf-8") as f:
        for r in ver:
            f.write(json.dumps({"level": "L4", "source": "nested_requalified",
                                "stages": 3, "stability": "verified_stable",
                                "pm": r["verified_pm_deg"],
                                "gain": r["gain_db"]}) + "\n")
print("topology_verified:", bool(ver), "| allow_list_promoted:", promoted)

if ver:
    print("=== TASK 6: BANDWIDTH OPTIMISATION (joint, real SPICE) ===")
    from agentic_raptor.mapping import map_family
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import (apply_edit,
                                                          qualify_device_graph)

    class _S:
        topology_id = "three_stage_nested_miller"
    base, _ = map_family(_S(), {"topology_id": _S.topology_id, "gain_stages": 3,
                                "functional_blocks": ["C"], "unresolved_blocks": [],
                                "mapping_readiness": "mapping_ready",
                                "graph_hash": None})
    nested, _a = apply_edit(base, "REPLACE_SIMPLE_MILLER_WITH_NESTED")
    exe = discover_ngspice()
    costs = new_costs()
    opt_rows = []
    for cc1, ib, rzx in itertools.product((0.1e-12, 0.25e-12, 0.5e-12),
                                          (2.0, 4.0), (0.5, 1.0)):
        g = copy.deepcopy(nested)
        for d in g.devices:
            if d.device_id == "CC1":
                d.sizing["value"] = cc1
            elif d.device_id == "CC2":
                d.sizing["value"] = cc1 * 0.15
            elif d.device_id == "RZ1":
                d.sizing["value"] = 700.0 * rzx
            elif d.kind in ("nmos", "pmos") and d.group and d.group.startswith("cs"):
                d.sizing["w"] = max(0.42, min(100.0, d.sizing["w"] * 1.5))
            elif d.kind == "isrc":
                d.sizing["value"] = d.sizing["value"] * ib
        tag = f"bw_c{cc1*1e12:g}_i{ib}_r{rzx}"
        q = qualify_device_graph(_S.topology_id, g, NM, exe, tag, costs)
        m = q.get("metrics") or {}
        ac = NM / tag / "run" / "acdata.txt"
        v = verified_pm(ac) if ac.is_file() else {"pm_status": "no_ac"}
        row = {"cc1_pf": cc1 * 1e12, "ib": ib, "rz_x": rzx,
               "gain_db": m.get("dc_gain_db"), "ugbw_hz": m.get("ugbw_hz"),
               "power_w": m.get("quiescent_power_w"),
               "verified_pm_deg": v.get("verified_pm_deg"),
               "pm_status": v["pm_status"]}
        row["frozen_spec_met"] = bool(
            row["verified_pm_deg"] and row["verified_pm_deg"] >= PM_T
            and (row["gain_db"] or 0) >= GAIN_T and (row["ugbw_hz"] or 0) >= UGBW_T)
        opt_rows.append(row)
        print(row)
    (NM / "bandwidth_opt.jsonl").write_text(
        "\n".join(json.dumps(r) for r in opt_rows), encoding="utf-8")
    passing = [r for r in opt_rows if r["frozen_spec_met"]]
    bb = max([r for r in opt_rows if r["verified_pm_deg"]
              and r["verified_pm_deg"] >= PM_T] or opt_rows,
             key=lambda r: r["ugbw_hz"] or 0, default=None)
    print(json.dumps({"frozen_spec_passes": len(passing),
                      "best_bandwidth_candidate": bb,
                      "spice_calls": costs["real_spice_calls"]}, indent=1))
