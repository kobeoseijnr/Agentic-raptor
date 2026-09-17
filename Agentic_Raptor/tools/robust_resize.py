"""Robust selection experiment on the 74 HELDOUT29 R2 winners (2026-09-07).

WHY
Under the 4-corner protocol only 36/74 R2 winners are robust (tools/pvt4_*).
The winners carry a median nominal PM margin of 2.1 deg (robust ones: 10.4).
Diagnostic on one winner (3s_rc, 61-call history): 37 of 61 sized points
PASS the spec, one with 48.7 deg PM margin at 84 uA -- yet the pipeline
delivered the last tail point with 2.9 deg margin at 490 uA, because the
agentic arm's final pick is select_by="fom_mguard": the FoM argmax
(UGBW*CL/Idd) among passing points with only a 6 dB gain guard. FoM is blind
to phase margin. The runner-up design is no alternative either: the
Supervisor banked its budget into the winner (3 calls) and it fails every
corner (heldout29_pvt4_backup_runs.jsonl: 2/74 robust).

WHAT THIS MEASURES
One control sizing run per design (production sac_size, R2 settings:
select_by=fom_mguard, margin_tail=12, same seed, SPICE budget matched to the
calls R2 spent on that winner), then FOUR selection rules applied to the SAME
history, each pick verified against the ORIGINAL spec and swept at the same
4 corners. Rules differ ONLY in which already-simulated point is delivered:

  r2_fom_mguard   R2's rule (self-check: must equal the sizer's own pick)
  pm_cushion_fom  FoM argmax among passing points that ALSO hold the 10 deg
                  PM cushion the reward already encodes (+ 6 dB gain guard)
  worst_margin    maximise the smallest cushion-normalised margin
                  min(pm/10deg, gain/6dB, log10(ugbw)/0.5) among passing
                  points -- the legacy-RAPTOR `worst_margin` analogue
  pm_margin       max PM margin among passing points (upper bound; shows the
                  FoM price of robustness)

Zero extra sizing SPICE for any rule; 4 corner calls per distinct pick,
counted. Nothing here modifies the pipeline, checkpoints, corpora or the
sealed blindtest; writes only under artifacts/publication_v3/heldout29_pvt4/
resize/. Resumable per (stem, seed). --report -> ROBUST_RESIZE_REPORT.md.

COST ~ (R2 calls + 4 x distinct picks) per design ~ 1.5 min -> 74 designs
~ 1.9 h. --limit N to split.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(r"C:\Users\kobeo\OneDrive\Desktop\raptor1\Agentic_Raptor")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from agentic_raptor.electrical import discover_ngspice                              # noqa: E402
from agentic_raptor.electrical.fom import compute_fom                                # noqa: E402
from agentic_raptor.electrical.pvt_eval import PvtConfig, aggregate_pvt, run_pvt_sweep  # noqa: E402
from agentic_raptor.mb_sac.spec_sizing import (KNOB_NAMES, PM_CUSHION_DEG, apply_knobs,  # noqa: E402
                                               margin_vector, postsizing_outcome, sac_size)
from agentic_raptor.topology_rl.stage3e2 import new_costs                            # noqa: E402

# reuse the step-1 harness's winner enumeration + graph rebuild (validated: rel diff 0.0)
_spec = importlib.util.spec_from_file_location("pvt4", ROOT / "tools/pvt4_heldout29.py")
_h = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, [sys.argv[0]]
_spec.loader.exec_module(_h)
sys.argv = _argv

OUTD = ROOT / "artifacts/publication_v3/heldout29_pvt4/resize"
RUNS = OUTD / "ROBUST_RESIZE.jsonl"
REPORT = OUTD / "ROBUST_RESIZE_REPORT.md"
MARGIN_TAIL = 12                       # R2 agentic arm: kwargs["margin_tail"] = 12
GAIN_GUARD_DB = 6.0                    # fom_mguard's guard
UGBW_CUSHION_LOG = 0.5                 # spec_reward's ugbw cushion
RULES = ("r2_fom_mguard", "pm_cushion_fom", "worst_margin", "pm_margin")
CFG4 = PvtConfig(enabled=True, process_corners=("tt",), supply_voltages=(1.62, 1.98),
                 temperatures_c=(0.0, 70.0))


def P(s: str) -> None:
    print(str(s).encode("ascii", "replace").decode(), flush=True)


def r2_calls(w) -> int:
    d = json.load(open(w["file"], encoding="utf-8"))
    return int(d["stage6_sizing"][w["label"]]["spice_calls"])


def fom_of(r: dict, cl: float) -> float | None:
    return compute_fom(r.get("ugbw_hz"), r.get("c_load_f") or cl, r.get("idd_a")).get("fom_value")


def select(rule: str, results: list[dict], spec: dict, cl: float) -> dict | None:
    passing = [r for r in results if postsizing_outcome(r, spec)["exact_spec_pass"]]
    if not passing:
        return None
    mv = {id(r): (r.get("margin_vector") or margin_vector(r, spec)) for r in passing}

    def gm(r): return mv[id(r)].get("gain_margin_db") or 0.0
    def pmm(r): return mv[id(r)].get("pm_margin_deg") or 0.0
    def um(r): return mv[id(r)].get("ugbw_log_margin") or 0.0

    if rule == "r2_fom_mguard":
        pool = [r for r in passing if gm(r) >= GAIN_GUARD_DB] or passing
        sc = [(fom_of(r, cl), r) for r in pool]
        sc = [(f, r) for f, r in sc if f is not None]
        return max(sc, key=lambda t: t[0])[1] if sc else max(passing, key=lambda r: r["reward"])
    if rule == "pm_cushion_fom":
        pool = [r for r in passing if gm(r) >= GAIN_GUARD_DB and pmm(r) >= PM_CUSHION_DEG]
        if not pool:                        # cushion unreachable: fall back to R2's rule
            return select("r2_fom_mguard", results, spec, cl)
        sc = [(fom_of(r, cl), r) for r in pool]
        sc = [(f, r) for f, r in sc if f is not None]
        return max(sc, key=lambda t: t[0])[1] if sc else pool[0]
    if rule == "worst_margin":
        return max(passing, key=lambda r: min(pmm(r) / PM_CUSHION_DEG, gm(r) / GAIN_GUARD_DB,
                                              um(r) / UGBW_CUSHION_LOG))
    if rule == "pm_margin":
        return max(passing, key=pmm)
    raise ValueError(rule)


PER_BRANCH_BUDGET = 32      # ablation_v3 ExperimentBudget.max_optimization_spice_calls
RUNS_BACKUP = OUTD / "ROBUST_RESIZE_BACKUP.jsonl"


def run(limit: int, rules: tuple[str, ...], backup: bool = False,
        budget_override: int | None = None, stems: set | None = None) -> None:
    """backup=True: size each run's RUNNER-UP topology (stage8 backup_design)
    under the same control settings at the per-branch budget (32) -- R2 gave
    it ~3 calls after the Supervisor banked its budget into the winner -- and
    apply the same rules + corners. Feeds the --policy report."""
    global RUNS
    if backup:
        RUNS = (OUTD / f"ROBUST_RESIZE_BACKUP_b{budget_override}.jsonl") if budget_override else RUNS_BACKUP
    exe = discover_ngspice()
    store = _h.load_store()
    ws = list(_h.winners("backup" if backup else "selected"))
    if stems:
        ws = [w for w in ws if f"{w['stem']}:{w['seed']}" in stems]
    if limit:
        ws = ws[:limit]
    OUTD.mkdir(parents=True, exist_ok=True)
    scratch = OUTD / "_scratch"
    scratch.mkdir(exist_ok=True)
    done = set()
    if RUNS.exists():
        for l in RUNS.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["stem"], r["seed"]))
    todo = [w for w in ws if (w["stem"], w["seed"]) not in done]
    P(f"[resize] winners={len(ws)} rules={list(rules)} done={len(done)} todo={len(todo)}")
    t0 = time.time()
    with RUNS.open("a", encoding="utf-8") as fh:
        for i, w in enumerate(todo, 1):
            rec = {"stem": w["stem"], "seed": w["seed"], "label": w["label"],
                   "family": w["family"], "hash": w["hash"], "spec": w["spec"]}
            try:
                budget = budget_override or (PER_BRANCH_BUDGET if backup else r2_calls(w))
                cl = w["c_load_f"]
                g0 = _h._realise(json.loads(store[w["hash"]]))
                tid = f"rs{'B' if backup else ''}{budget_override or ''}_{w['stem'][:24]}_s{w['seed']}"
                r = sac_size(tid, g0, w["spec"], exe, scratch, new_costs(), budget=budget,
                             seed=w["seed"], persist=False, early_stop_on_pass=False,
                             select_by="fom_mguard", margin_tail_calls=MARGIN_TAIL,
                             use_surrogate=True, use_ranker=True, c_load_f=cl)
                results = r["results"]
                n_pass = sum(postsizing_outcome(x, w["spec"])["exact_spec_pass"] for x in results)
                picks = {rule: select(rule, results, w["spec"], cl) for rule in rules}
                r2_ok = (picks.get("r2_fom_mguard") is not None
                         and picks["r2_fom_mguard"]["step"] == r["best"]["step"])
                swept: dict[int, dict] = {}
                out_rules = {}
                pvt_calls = 0
                for rule, pk in picks.items():
                    if pk is None:
                        out_rules[rule] = None
                        continue
                    stp = int(pk["step"])
                    if stp not in swept:
                        g = apply_knobs(g0, [float(pk["knobs"][k]) for k in KNOB_NAMES])
                        corners = run_pvt_sweep(w["hash"], g, w["spec"], exe, scratch, CFG4,
                                                label=f"{w['label']}_{rule}", c_load_f=cl)
                        pvt_calls += len(corners)
                        agg = aggregate_pvt(corners, CFG4.required_corner_ids)
                        swept[stp] = {"corners": corners, "robust": bool(agg.get("robust_complete_pass")),
                                      "passed_corners": agg.get("passed_pvt_corners"),
                                      "worst_pm": min((c.get("pm_deg") or 0) for c in corners)}
                    mvv = pk.get("margin_vector") or margin_vector(pk, w["spec"])
                    out_rules[rule] = {"step": stp,
                                       "nominal": {k: pk.get(k) for k in ("gain_db", "pm_deg", "ugbw_hz", "idd_a", "power_w")},
                                       "pm_margin": mvv.get("pm_margin_deg"), "gain_margin": mvv.get("gain_margin_db"),
                                       "fom": fom_of(pk, cl), "sizing_vector": pk["knobs"],
                                       **swept[stp]}
                rec.update(status="ok", budget=budget, sizing_calls=len(results), n_passing_points=n_pass,
                           r2_rule_matches_sizer=r2_ok, sizer_best_step=r["best"]["step"],
                           pvt_calls=pvt_calls, rules=out_rules)
                summ = " ".join(f"{k[:6]}={'R' if v and v['robust'] else '-'}{((v or {}).get('pm_margin') or 0):4.1f}"
                                for k, v in out_rules.items())
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[:300],
                           trace=traceback.format_exc()[-800:])
                summ = rec["error"][:80]
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            P(f"  [{len(done)+i:3d}/{len(done)+len(todo)}] {w['stem']:30} s{w['seed']} {str(w['family']):10} "
              f"pass_pts={rec.get('n_passing_points')!s:>3}/{rec.get('sizing_calls')!s:<3} r2ok={rec.get('r2_rule_matches_sizer')!s:5} "
              f"{summ} ({(time.time()-t0)/60:.1f} min)")


def report() -> None:
    rows = [json.loads(l) for l in RUNS.open(encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r["status"] == "ok"]
    ref = {(r["stem"], r["seed"]): r for r in
           (json.loads(l) for l in (OUTD.parent / "heldout29_pvt4_runs.jsonl").open(encoding="utf-8") if l.strip())}
    n0 = len(ref); rob0 = sum(1 for r in ref.values() if r.get("robust_complete_pass"))
    md = ["# Robust selection over the same sizing history (74 HELDOUT29 winners, 4-corner protocol)\n",
          f"One control sizing run per design (R2 settings, budget matched to R2's calls), {len(ok)} designs; "
          f"self-check R2-rule == sizer pick on {sum(r['r2_rule_matches_sizer'] for r in ok)}/{len(ok)}. "
          f"Median passing points per history: {st.median(r['n_passing_points'] for r in ok):.0f}.\n",
          "| rule | delivered a passing design | **robust (4 corners)** | median nominal PM margin | median FoM | median I_dd (uA) |",
          "|---|---:|---:|---:|---:|---:|",
          f"| R2 campaign as run (reference) | {n0}/{n0} | **{rob0}/{n0} = {100*rob0/n0:.1f}%** | 2.1 fail / 10.4 robust | - | - |"]
    for rule in RULES:
        have = [r["rules"].get(rule) for r in ok if r["rules"].get(rule)]
        if not have:
            continue
        rob = sum(p["robust"] for p in have)
        md.append(f"| {rule} | {len(have)}/{len(ok)} | **{rob}/{len(ok)} = {100*rob/len(ok):.1f}%** | "
                  f"{st.median(p['pm_margin'] for p in have):.1f} | "
                  f"{st.median(p['fom'] for p in have if p['fom'] is not None):.1f} | "
                  f"{st.median((p['nominal']['idd_a'] or 0)*1e6 for p in have):.1f} |")
    md.append("\n## Per family (robust / n)\n")
    fams = sorted({str(r["family"]) for r in ok})
    md.append("| family | R2 | " + " | ".join(RULES) + " |")
    md.append("|---|---:|" + "---:|" * len(RULES))
    for f in fams:
        rr = [r for r in ref.values() if str(r.get("family")) == f]
        cells = [f"{sum(1 for r in rr if r.get('robust_complete_pass'))}/{len(rr)}"]
        rs = [r for r in ok if str(r["family"]) == f]
        for rule in RULES:
            cells.append(f"{sum(1 for r in rs if (r['rules'].get(rule) or {}).get('robust'))}/{len(rs)}")
        md.append(f"| {f} | " + " | ".join(cells) + " |")
    md += ["\n## Notes",
           "- Every rule picks from the SAME already-simulated history: zero extra sizing SPICE. Corner calls: 4 per distinct pick.",
           "- Verdicts are always against the original spec; the pass criteria and the corner protocol are unchanged.",
           f"- errors: {len(rows) - len(ok)}"]
    REPORT.write_text("\n".join(md), encoding="utf-8")
    P("\n".join(md)); P(f"\nwrote {REPORT}")


def policy(rule: str = "worst_margin", backup_budget: int | None = None) -> None:
    """Corner-aware delivery: size BOTH candidates fully, pick each by `rule`,
    corner-check both (8 calls), deliver the robust one (winner preferred; if
    neither is robust deliver the winner's pick). SPICE per run: A 61 + B 32 +
    8 corners vs R2's 61 + 3 + 0 (A's R2 calls already include B's banked
    budget; A is re-sized at its own R2 call count for comparability)."""
    NL = chr(10)
    A = {(r["stem"], r["seed"]): r for r in (json.loads(l) for l in RUNS.open(encoding="utf-8") if l.strip())
         if r["status"] == "ok"}
    B = {(r["stem"], r["seed"]): r for r in (json.loads(l) for l in RUNS_BACKUP.open(encoding="utf-8") if l.strip())
         if r["status"] == "ok"} if RUNS_BACKUP.exists() else {}
    n_b61 = 0
    if backup_budget:       # overlay runner-ups re-sized at a larger budget (subset of runs)
        fb = OUTD / f"ROBUST_RESIZE_BACKUP_b{backup_budget}.jsonl"
        if fb.is_file():
            for r in (json.loads(l) for l in fb.open(encoding="utf-8") if l.strip()):
                if r["status"] == "ok":
                    B[(r["stem"], r["seed"])] = r
                    n_b61 += 1
    keys = sorted(A)
    a_rob = b_rob = either = 0
    fam_rows = {}
    delivered_fams = {}
    rows_md = []
    for k in keys:
        pa = (A[k]["rules"].get(rule) or {})
        pb = ((B.get(k) or {}).get("rules") or {}).get(rule) or {}
        ra, rb = bool(pa.get("robust")), bool(pb.get("robust"))
        a_rob += ra; b_rob += rb; either += (ra or rb)
        fa, fb = str(A[k]["family"]), str((B.get(k) or {}).get("family"))
        f = fam_rows.setdefault(fa, [0, 0, 0])
        f[0] += 1; f[1] += ra; f[2] += (ra or rb)
        deliver = fa if ra or not rb else fb
        delivered_fams[deliver] = delivered_fams.get(deliver, 0) + 1
        if not ra:
            rows_md.append(f"| {k[0]} | s{k[1]} | {fa} | {'-' if not B.get(k) else fb} | "
                           f"{'robust' if rb else ('not sized' if not B.get(k) else 'not robust')} |")
    n = len(keys)
    md = [f"# Corner-aware delivery policy (rule = {rule}) on {n} HELDOUT29 runs" + NL,
          "| policy | robust (4 corners) | SPICE per run |", "|---|---:|---:|",
          "| R2 as run (FoM argmax, winner only) | 36/74 = 48.6% | 64 (61 A + 3 B) |",
          f"| winner only, {rule} selection | {a_rob}/{n} = {100*a_rob/n:.1f}% | 61 + 4 corners |",
          f"| runner-up only, {rule} selection (budget 32) | {b_rob}/{n} = {100*b_rob/n:.1f}% | 32 + 4 corners |",
          (f"| **corner-check both, deliver the robust one** | **{either}/{n} = {100*either/n:.1f}%** | "
           + (f"A 61 + B 32 ({n - n_b61} runs) / B {backup_budget} ({n_b61} runs) + 8 corners; "
              f"uniform policy = 61 + {backup_budget} + 8 = {69 + backup_budget} per run (lower bound on robustness)"
              if backup_budget else "61 + 32 + 8 corners") + " |"),
          NL + "## Per winner family: n / winner robust / either robust" + NL,
          "| family | n | winner | either |", "|---|---:|---:|---:|"]
    for f, (c, a, e) in sorted(fam_rows.items()):
        md.append(f"| {f} | {c} | {a} | {e} |")
    md += [NL + "## Families delivered under the policy" + NL, "| family | delivered |", "|---|---:|"]
    md += [f"| {f} | {c} |" for f, c in sorted(delivered_fams.items(), key=lambda t: -t[1])]
    md += [NL + "## Runs where the winner is not robust" + NL,
           "| spec | seed | winner family | runner-up family | runner-up |", "|---|---|---|---|---|"] + rows_md
    md += [NL + "## Notes", "- Verdicts against the original spec; corner protocol unchanged; every SPICE call counted.",
           f"- runner-up rows available: {len(B)}/{n}"]
    out = OUTD / f"CORNER_AWARE_POLICY_{rule}{'_b' + str(backup_budget) if backup_budget else ''}.md"
    out.write_text(NL.join(md), encoding="utf-8")
    P(NL.join(md)); P(NL + f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--backup", action="store_true", help="with --run: size the runner-up topologies instead")
    ap.add_argument("--budget", type=int, default=None, help="with --run --backup: override the runner-up budget")
    ap.add_argument("--stems", default=None, help="comma list of stem:seed to restrict --run to")
    ap.add_argument("--backup-budget", type=int, default=None, help="with --policy: overlay ROBUST_RESIZE_BACKUP_b<N>.jsonl")
    ap.add_argument("--policy", default=None, metavar="RULE",
                    help="corner-aware delivery report combining winner + runner-up picks by RULE")
    a = ap.parse_args()
    if a.run:
        run(a.limit, tuple(x for x in a.rules.split(",") if x), backup=a.backup, budget_override=a.budget,
            stems=set(a.stems.split(",")) if a.stems else None)
    if a.report:
        report()
    if a.policy:
        policy(a.policy, a.backup_budget)
    if not (a.run or a.report or a.policy):
        P("nothing to do: --run [--backup --limit N --rules a,b] | --report | --policy RULE")


if __name__ == "__main__":
    main()
