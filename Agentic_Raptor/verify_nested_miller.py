"""three_stage_nested_miller: systematic REAL-ngspice verification sweep.

Sweeps Cc1 x Cc2 x stage-2/3 sizing x bias (36 combos), stores every
measurement, promotes to the allow-list ONLY combos with convergent, stable,
positive-PM results, updates RAG/L4 + topology metadata, then evaluates the
frozen spec (gain>=73.42dB, PM>=50deg, UGBW>=1MHz, 200pF).

Run:  python verify_nested_miller.py
"""
import copy
import itertools
import json
from pathlib import Path

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.electrical import discover_ngspice
from agentic_raptor.mapping import map_family
from agentic_raptor.mb_sac.stage3d2 import V3
from agentic_raptor.topology_rl.stage3e2 import new_costs
from agentic_raptor.topology_rl.stage3e2_edits import (
    apply_edit, device_graph_hash, qualify_device_graph)

OUT = Path("artifacts/nested_miller").resolve()
OUT.mkdir(parents=True, exist_ok=True)
MEM = Path("datasets/simulation_memory")
exe = discover_ngspice()


class _S:
    topology_id = "three_stage_nested_miller"


base, _ = map_family(_S(), {"topology_id": _S.topology_id, "gain_stages": 3,
                            "functional_blocks": ["C"], "unresolved_blocks": [],
                            "mapping_readiness": "mapping_ready", "graph_hash": None})
nested, audit = apply_edit(base, "REPLACE_SIMPLE_MILLER_WITH_NESTED")
print("template hash:", audit["child_hash"], "| dims:", audit["action_dim_after"])

costs = new_costs()
rows = []
GAIN_T, PM_T, UGBW_T = 73.42, 50.0, 1e6
for cc1, ratio, s23, ib in itertools.product(
        (0.25e-12, 0.5e-12, 1e-12, 2e-12),   # retuned: UGBW must rise ~20x
        (0.15, 0.3), (1.0, 1.5), (1.0, 2.0)):
    cc2 = cc1 * ratio
    g = copy.deepcopy(nested)
    for d in g.devices:
        if d.device_id == "CC1":
            d.sizing["value"] = cc1
        elif d.device_id == "CC2":
            d.sizing["value"] = cc2
        elif d.kind in ("nmos", "pmos") and d.group and d.group.startswith("cs"):
            d.sizing["w"] = max(0.42, min(100.0, d.sizing["w"] * s23))
        elif d.kind == "isrc":
            d.sizing["value"] = d.sizing["value"] * ib
    tag = f"c{cc1*1e12:g}_{cc2*1e12:g}_s{s23}_b{ib}"
    q = qualify_device_graph(_S.topology_id, g, OUT, exe, tag, costs)
    m = q.get("metrics") or {}
    row = {"cc1_pf": cc1 * 1e12, "cc2_pf": cc2 * 1e12, "s23": s23, "ib": ib,
           "gain_db": m.get("dc_gain_db"), "pm_deg": m.get("phase_margin_deg"),
           "ugbw_hz": m.get("ugbw_hz"), "power_w": m.get("quiescent_power_w"),
           "converged": bool(m), "stable": q.get("stability") == "verified_stable",
           "spec_met": bool(m.get("phase_margin_deg") and m.get("dc_gain_db")
                            and m["phase_margin_deg"] >= PM_T
                            and m["dc_gain_db"] >= GAIN_T
                            and (m.get("ugbw_hz") or 0) >= UGBW_T)}
    rows.append(row)
    print(row)

(OUT / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                 encoding="utf-8")
stable = [r for r in rows if r["stable"]]
passing = [r for r in rows if r["spec_met"]]
best = (sorted(passing or stable,
               key=lambda r: (r["pm_deg"] or -999, r["gain_db"] or -999),
               reverse=True) or [None])[0]

# allow-list promotion: ONLY on verified stability
verdict = {"template": "three_stage_nested_miller",
           "template_hash": audit["child_hash"],
           "combos_tested": len(rows), "converged": sum(r["converged"] for r in rows),
           "stable": len(stable), "frozen_spec_pass": len(passing),
           "best": best, "real_spice_calls": costs["real_spice_calls"],
           "promoted": bool(stable)}
al_path = Path("artifacts/variant_verification/ALLOWLIST.json")
al = json.loads(al_path.read_text())
al.setdefault("templates", {})["three_stage_nested_miller"] = {
    "promoted": bool(stable), "stable_regions": stable[:10],
    "failed_regions": [r for r in rows if not r["stable"]][:5]}
al_path.write_text(json.dumps(al, indent=1), encoding="utf-8")
if stable:   # RAG/L4 + metadata only on verified success
    with (MEM / "self_improvement_runs.jsonl").open("a", encoding="utf-8") as f:
        for r in stable:
            f.write(json.dumps({"level": "L4", "source": "nested_miller_sweep",
                                "stages": 3, "stability": "verified_stable",
                                "pm": r["pm_deg"], "gain": r["gain_db"]}) + "\n")
    (OUT / "topology_metadata.json").write_text(json.dumps(
        {"family": "three_stage_nested_miller", "hash": audit["child_hash"],
         "sizing_variables": ["W1", "W2", "W3", "Ib", "Cc1", "Cc2", "Rz1"],
         "stable_regions": stable, "provenance": "verified sweep"}, indent=1),
        encoding="utf-8")
(OUT / "VERDICT.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in verdict.items() if k != "best"}, indent=1))
print("BEST:", json.dumps(best))
