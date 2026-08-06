"""One-time electrical verification sweep of every corpus variant combination.

Realises each (stages, comp, buffer, fb) combo the corpus can teach, measures
it on REAL ngspice, and writes a verified allow-list. The proposal validator
then refuses any combo that failed — circuits we can't build are never taught,
proposed, or simulated again until their template is fixed.

Run:  python verify_variants.py     (~16 SPICE calls, a few minutes)
"""
import itertools
import json
from pathlib import Path

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.electrical import discover_ngspice
from agentic_raptor.mapping import map_family
from agentic_raptor.mb_sac.stage3d2 import V3
from agentic_raptor.topology_rl.stage3e2 import new_costs
from agentic_raptor.topology_rl.stage3e2_edits import (EditRejected, apply_edit,
                                                       qualify_device_graph)

OUT = Path("artifacts/variant_verification").resolve()
exe = discover_ngspice()
rows = []
for stages, comp, buf, fb in itertools.product(
        (1, 2, 3), ("none", "miller", "rc"), (False, True), (False, True)):
    if stages == 1 and comp != "none":
        continue

    class _S:
        topology_id = f"vv_{stages}{comp[0]}{int(buf)}{int(fb)}"
    g, _ = map_family(_S(), {
        "topology_id": _S.topology_id, "gain_stages": stages,
        "functional_blocks": ["C"] if comp != "none" else [],
        "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
        "graph_hash": None})
    try:
        if buf:
            g, _a = apply_edit(g, "ADD_SUPPORTED_OUTPUT_STAGE")
        if comp == "rc":
            g, _a = apply_edit(g, "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE")
        if fb:
            g, _a = apply_edit(g, "CONNECT_VERIFIED_FEEDBACK_PATH")
    except EditRejected as exc:
        rows.append({"stages": stages, "comp": comp, "buffer": buf, "fb": fb,
                     "verdict": f"edit_rejected:{exc}", "buildable": False})
        continue
    costs = new_costs()
    q = qualify_device_graph(_S.topology_id, g, OUT, exe, _S.topology_id, costs)
    m = q.get("metrics") or {}
    gain, pm = m.get("dc_gain_db"), m.get("phase_margin_deg")
    ok = q.get("electrical") == "electrically_functional" and (gain or -99) > 0
    rows.append({"stages": stages, "comp": comp, "buffer": buf, "fb": fb,
                 "gain_db": round(gain, 1) if gain is not None else None,
                 "pm_deg": round(pm, 1) if pm is not None else None,
                 "stability": q.get("stability"), "buildable": bool(ok)})
    print(rows[-1])

allow = [{k: r[k] for k in ("stages", "comp", "buffer", "fb")}
         for r in rows if r["buildable"]]
result = {"combos_tested": len(rows),
          "buildable": len(allow), "allow_list": allow, "all": rows}
(OUT / "ALLOWLIST.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
print(f"\nverified buildable: {len(allow)}/{len(rows)} -> {OUT / 'ALLOWLIST.json'}")
