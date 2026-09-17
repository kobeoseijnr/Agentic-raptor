"""Step-1 report: HELDOUT29 winners under the 4-corner V/T protocol.

Consumes artifacts/publication_v3/heldout29_pvt4/heldout29_pvt4_runs.jsonl
(written by tools/pvt4_heldout29.py --sweep) and answers the questions a DATE
reviewer asks: how many designs survive all four corners, which corner is the
killer, which constraint fails there, and how that differs by topology family.

Read-only. Writes HELDOUT29_PVT4_REPORT.md + pvt4_by_corner.csv next to the jsonl.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

D = Path(r"C:\Users\kobeo\OneDrive\Desktop\raptor1\Agentic_Raptor\artifacts\publication_v3\heldout29_pvt4")
rows = [json.loads(l) for l in (D / "heldout29_pvt4_runs.jsonl").open(encoding="utf-8") if l.strip()]
n = len(rows)


def P(s: str) -> None:
    print(str(s).encode("ascii", "replace").decode())


robust = sum(1 for r in rows if r.get("robust_complete_pass"))
pcts = [r.get("pvt_pass_percent") or 0.0 for r in rows]

# per-corner: how many designs fail at each corner, and why
by_corner_fail: Counter = Counter()
by_corner_reason: dict[str, Counter] = defaultdict(Counter)
for r in rows:
    for c in r["corners"]:
        if not c.get("complete_pass"):
            by_corner_fail[c["pvt_corner_id"]] += 1
            for reason in (c.get("failure_reasons") or ["unspecified"]):
                by_corner_reason[c["pvt_corner_id"]][reason.split(" ")[0] if reason else "unspecified"] += 1

# how many corners each design fails
fail_count_dist = Counter(r.get("failed_pvt_corners", 0) for r in rows)

# per family
fam: dict[str, dict] = defaultdict(lambda: {"n": 0, "robust": 0, "pcts": []})
for r in rows:
    f = fam[str(r.get("family"))]
    f["n"] += 1
    f["robust"] += int(bool(r.get("robust_complete_pass")))
    f["pcts"].append(r.get("pvt_pass_percent") or 0.0)

# worst-case margins across corners (what the paper's PVT column should carry)
worst_pm = [min((c.get("pm_deg") or 0) for c in r["corners"]) for r in rows]
worst_gain = [min((c.get("gain_db") or 0) for c in r["corners"]) for r in rows]

md = []
md.append("# HELDOUT29 winners under 4-corner V/T PVT\n")
md.append("Protocol: process tt x VDD {1.62, 1.98} V x T {0, 70} C = 4 corners, per metric_definitions.md.")
md.append("Same 74 verified R2 winners; sized graphs rebuilt via the pipeline's own _realise + apply_knobs path")
md.append("(validation: worst relative diff 0.000e+00 vs recorded nominal on 5 winners).\n")
md.append("| | single-corner (R2 as run) | **4-corner (this sweep)** |")
md.append("|---|---:|---:|")
md.append(f"| robust (all corners pass) | 74/74 = 100.0% | **{robust}/{n} = {100*robust/n:.1f}%** |")
md.append(f"| mean PVT pass % | 100.0 | **{st.mean(pcts):.1f}** |")
md.append(f"| corners per design | 1 | 4 |\n")

md.append("## Which corner kills designs\n")
md.append("| corner | designs failing | dominant failure |")
md.append("|---|---:|---|")
for cid in ("tt_1.62V_0C", "tt_1.62V_70C", "tt_1.98V_0C", "tt_1.98V_70C"):
    top = by_corner_reason[cid].most_common(1)
    md.append(f"| {cid} | {by_corner_fail.get(cid, 0)}/{n} | {top[0][0] + ' (' + str(top[0][1]) + ')' if top else '-'} |")
md.append("")

md.append("## Corners failed per design\n")
md.append("| corners failed | designs |")
md.append("|---:|---:|")
for k in sorted(fail_count_dist):
    md.append(f"| {k} | {fail_count_dist[k]} |")
md.append("")

md.append("## By topology family\n")
md.append("| family | n | robust | robust % | mean PVT % |")
md.append("|---|---:|---:|---:|---:|")
for f, v in sorted(fam.items(), key=lambda kv: -kv[1]["n"]):
    md.append(f"| {f} | {v['n']} | {v['robust']} | {100*v['robust']/v['n']:.1f}% | {st.mean(v['pcts']):.1f} |")
md.append("")

md.append("## Worst-case across corners (medians over the 74)\n")
md.append(f"- worst-corner phase margin: median {st.median(worst_pm):.1f} deg")
md.append(f"- worst-corner gain: median {st.median(worst_gain):.1f} dB\n")

md.append("## What this means for the paper\n")
md.append(f"The R2 campaign's '100% PVT' was a single-corner check. Under the declared 4-corner protocol, "
          f"{robust}/{n} ({100*robust/n:.1f}%) of the same designs are robust. Report the 4-corner number; "
          f"the single-corner figure must not appear as 'PVT'.")

(D / "HELDOUT29_PVT4_REPORT.md").write_text("\n".join(md), encoding="utf-8")
with (D / "pvt4_by_corner.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["stem", "seed", "family", "corner", "gain_db", "pm_deg", "ugbw_hz", "idd_a", "complete_pass", "failure_reasons"])
    for r in rows:
        for c in r["corners"]:
            w.writerow([r["stem"], r["seed"], r.get("family"), c["pvt_corner_id"], c.get("gain_db"), c.get("pm_deg"),
                        c.get("ugbw_hz"), c.get("idd_a"), c.get("complete_pass"), "; ".join(c.get("failure_reasons") or [])])

P("\n".join(md))
P(f"\nwrote {D / 'HELDOUT29_PVT4_REPORT.md'}")
