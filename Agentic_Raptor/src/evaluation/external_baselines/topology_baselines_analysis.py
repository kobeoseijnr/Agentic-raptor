"""Final analysis for the CktGen/AnalogGenie/Agentic-RAPTOR topology comparison.

One rule for every method, computed from on-disk data:
  pass  = gain_db >= target AND ugbw_hz >= target AND pm_deg >= target (3-target
          shared-judge rule; ngspice-measured nominals only)
  fom   = uniform relative margin (run_stage6_v2_tuner.score), passes only
Baselines come from artifacts/topology_baselines/results.jsonl (common 65-call
protocol). RAPTOR rows come from the frozen Tier-3 merged campaign
(native end-to-end pipeline, 65-call budget, same specs/judge) -- the Stage-20
"End-to-End System Comparison" arm. AG P@5 cells are frozen Track-A
measurements (pre-cleanup provenance, footnoted).
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.evaluation.external_baselines.run_stage6_v2_tuner import score  # noqa: E402

ART = ROOT / "artifacts" / "topology_baselines"
MERGED = ROOT / "artifacts/publication_v3/ablation_v3/results_TIER3_merged.jsonl"
R2 = ROOT / "artifacts/publication_v3/ablation_v3/results_20260830_201205.jsonl"  # AG_FULL R2: repairs + fom_mguard delivery (canonical)


def specs29():
    d = json.loads((ROOT / "data/external_baseline_eval/specs_validation.json"
                    ).read_text(encoding="utf-8"))
    return {s["spec_index"]: s["parsed_spec"] for s in d["specs"]}


def load_all():
    sp = specs29()
    per = []   # per (method, seed, spec): dict
    _res = ART / "results_bounded.jsonl"   # 2026-08-31: bounds-parity tuner + idd
    for l in _res.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        per.append({
            "method": r["method"], "seed": r["seed"], "spec": r["spec_index"],
            "final_pass": bool(r["final_pass"]), "p5": bool(r["pass_at_5"]),
            "calls": r["optimization_calls"], "pvt_calls": r["pvt_calls"],
            "vt": r["vt_corners_pass"],
            "runtime": r["generation_time_s"] + r["eval_time_s"],
            "gen_s": r["generation_time_s"], "eval_s": r["eval_time_s"],
            "fom": r["final_fom"] if r["final_pass"] else None,
            "fom_power": (r.get("final_fom_power")
                          if r["final_pass"] else None),
            "fom_spec": (((sp[r["spec_index"]]["ugbw_target_hz"] / 1e6)
                          * sp[r["spec_index"]]["load_capacitance_pf"]
                          / (r["final_idd_a"] * 1e3))
                         if r["final_pass"] and r.get("final_idd_a")
                         else None),
            "valid_cands": sum(1 for c in r["candidates"] if c.get("valid")),
            "n_cands": r["n_candidates"],
        })
    # RAPTOR: AG_FULL from the REPAIRED campaign (HELDOUT29R2,
    # 2026-08-30: diverse top-2 + capability gate + two-phase margin tail,
    # PVT measured natively); A0 stays the frozen pre-repair reference.
    by_ix = {}
    src_rows = (
        [r for l in R2.read_text(encoding="utf-8").splitlines() if l.strip()
         if (r := json.loads(l))["ablation_id"] == "AG_FULL"]      # repaired campaign ONLY
        + [r for l in MERGED.read_text(encoding="utf-8").splitlines() if l.strip()
           if (r := json.loads(l))["ablation_id"] == "A0"])        # frozen reference ONLY
    for r in src_rows:
        i, seed = r["spec_index"], r["pipeline_seed"]
        p = sp[i]
        n = r["nominal"] or {}
        meas = {"gain_db": n.get("gain_db"), "ugbw_hz": n.get("ugbw_hz"),
                "pm_deg": n.get("pm_deg")}
        ok, fom = (score(meas, p) if n.get("ugbw_hz") is not None
                   else (False, None))
        m = "ag_full" if r["ablation_id"] == "AG_FULL" else "ag_a0"
        pvpct = ((r.get("pvt") or {}).get("pvt_pass_percent"))
        per.append({
            "method": m, "seed": seed, "spec": i,
            "final_pass": bool(ok), "p5": None,
            "calls": (r["spice"] or {}).get("total_calls"),
            "pvt_calls": (r.get("pvt") or {}).get("total_corners") or 0,
            "vt": (round(pvpct / 25.0) if pvpct is not None else None),
            "runtime": r.get("runtime_s"), "gen_s": None, "eval_s": None,
            "fom": (round(fom, 4) if ok else None),
            "fom_power": ((r.get("fom") or {}).get("fom_value")
                          if ok else None),
            "fom_spec": (((p["ugbw_target_hz"] / 1e6)
                          * p["load_capacitance_pf"] / (n["idd_a"] * 1e3))
                         if ok and n.get("idd_a") else None),
            "valid_cands": None, "n_cands": None,
        })
    return per


def agg(rows):
    n = len(rows)
    fp = [r["final_pass"] for r in rows]
    foms = [r["fom"] for r in rows if r["fom"] is not None]
    rts = [r["runtime"] for r in rows if r["runtime"] is not None]
    calls = [r["calls"] for r in rows if r["calls"] is not None]
    vts = [r["vt"] for r in rows if r["vt"] is not None]
    p5s = [r["p5"] for r in rows if r["p5"] is not None]
    def ms(xs):
        if not xs:
            return None, None, None
        m = statistics.mean(xs)
        s = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return m, s, statistics.median(xs)
    fom_m, fom_s, fom_med = ms(foms)
    fpw = [r.get("fom_power") for r in rows if r.get("fom_power") is not None]
    _, _, fom_pw_med = ms(fpw)
    fsp = [r.get("fom_spec") for r in rows if r.get("fom_spec") is not None]
    _, _, fom_sp_med = ms(fsp)
    rt_m, rt_s, _ = ms(rts)
    fp_rate = sum(fp) / n
    ci = 1.96 * math.sqrt(max(fp_rate * (1 - fp_rate), 1e-12) / n)
    return {"n": n, "final_pass": fp_rate, "fp_ci95": ci,
            "p5": (sum(p5s) / len(p5s)) if p5s else None,
            "calls": statistics.mean(calls) if calls else None,
            "runtime_mean": rt_m, "runtime_std": rt_s,
            "fom_mean": fom_m, "fom_std": fom_s, "fom_median": fom_med,
            "fom_power_median": fom_pw_med,
            "fom_spec_median": fom_sp_med,
            "n_feasible": len(foms),
            "pvt_pct": (100 * sum(vts) / (4 * len(vts))) if vts else None}


def paired_tests(per):
    from scipy import stats as st
    key = lambda r: (r["seed"], r["spec"])
    by = {}
    for r in per:
        by.setdefault(r["method"], {})[key(r)] = r
    lines = ["# Statistical tests (paired on common (seed, spec) keys)", ""]
    for base in ("cktgen", "analogtobi"):
        common = sorted(set(by["ag_full"]) & set(by.get(base, {})))
        a = [by["ag_full"][k] for k in common]
        b = [by[base][k] for k in common]
        n = len(common)
        # FinalPass: exact McNemar (binomial on discordant pairs)
        b01 = sum(1 for x, y in zip(a, b) if x["final_pass"] and not y["final_pass"])
        b10 = sum(1 for x, y in zip(a, b) if y["final_pass"] and not x["final_pass"])
        disc = b01 + b10
        p_mcn = (st.binomtest(min(b01, b10), disc, 0.5).pvalue if disc else 1.0)
        # FoM: Wilcoxon on pairs where both feasible
        pairs = [(x["fom"], y["fom"]) for x, y in zip(a, b)
                 if x["fom"] is not None and y["fom"] is not None]
        if len(pairs) >= 6 and any(p[0] != p[1] for p in pairs):
            w = st.wilcoxon([p[0] for p in pairs], [p[1] for p in pairs])
            fom_line = (f"Wilcoxon W={w.statistic:.0f}, p={w.pvalue:.2e} "
                        f"(n={len(pairs)} both-feasible pairs)")
        else:
            fom_line = f"insufficient both-feasible pairs (n={len(pairs)})"
        # Runtime: Wilcoxon
        rp = [(x["runtime"], y["runtime"]) for x, y in zip(a, b)
              if x["runtime"] and y["runtime"]]
        wr = st.wilcoxon([p[0] for p in rp], [p[1] for p in rp])
        d = sum(x["final_pass"] - y["final_pass"] for x, y in zip(a, b)) / n
        lines += [f"## RAPTOR (AG_FULL) vs {base}",
                  f"- common keys: {n}",
                  f"- FinalPass delta: {d:+.3f} "
                  f"(AG-better discordant {b01}, {base}-better {b10}); "
                  f"exact McNemar p={p_mcn:.2e}",
                  f"- FoM: {fom_line}",
                  f"- Runtime: Wilcoxon W={wr.statistic:.0f}, "
                  f"p={wr.pvalue:.2e} (n={len(rp)})", ""]
    (ART / "statistical_tests.md").write_text("\n".join(lines), encoding="utf-8")
    print("statistical_tests.md written")


LABEL = {"cktgen": "CktGen (pretrained)",
         "analogtobi": "AnalogToBi (pretrained)",
         "analoggenie": "AnalogGenie (pretrained, retired row)",
         "ag_full": "RAPTOR (AG-Full, end-to-end)",
         "ag_a0": "-- A0 no-agent pipeline (reference)"}
P5_FROZEN = {"ag_full": "0.529†", "ag_a0": "0.862†"}


def tables(per):
    methods = ["cktgen", "analogtobi", "ag_full", "ag_a0"]
    A = {m: agg([r for r in per if r["method"] == m]) for m in methods}
    # merge reconstruction cells (native PVT + symmetric FoM), if present
    rec_p = ART / "ag_reconstruction_summary.json"
    REC = (json.loads(rec_p.read_text(encoding="utf-8"))
           if rec_p.exists() else {})
    for m, arm in (("ag_a0", "A0"),):
        if arm in REC:
            A[m]["pvt_pct"] = REC[arm]["native_pvt_pct"]
            A[m]["fom_median"] = REC[arm]["sym_fom_median"]
            A[m]["rec_coverage"] = REC[arm]["coverage"]
            A[m]["sym_pvt"] = REC[arm]["sym_pvt_pct"]
    def f(x, d=3):
        return "n/a" if x is None else f"{x:.{d}f}"
    md = ["# Primary comparison table — HELDOUT29, seeds 0-2, shared sky130 "
          "ngspice judge, 65-call budget", "",
          "| Method | FinalPass ↑ | P@5 ↑ | Calls ↓ | Calls+Train ↓ | "
          "Runtime (s) ↓ | Runtime+Train (s) ↓ | FoM ↑ | PVT (%) ↑ |",
          "|---|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lcccccccc}", r"\toprule",
           r"Method & FinalPass$\uparrow$ & P@5$\uparrow$ & Calls$\downarrow$ & "
           r"Calls+Train$\downarrow$ & Runtime (s)$\downarrow$ & "
           r"Runtime+Train (s)$\downarrow$ & FoM$\uparrow$ & PVT (\%)$\uparrow$\\",
           r"\midrule"]
    # Runtime+Train = campaign grand total: online x 87 runs + one-time training
    N_RUNS = 87
    TRAIN_H = {"cktgen": ("~40 h**", 40.0), "analogtobi": ("22.8 h††", 22.8),
               "ag_full": ("529 s§", 529 / 3600), "ag_a0": ("529 s§", 529 / 3600)}
    for m in methods:
        a = A[m]
        p5 = P5_FROZEN.get(m, f(a["p5"]))
        if m == "cktgen":
            ct = f"{a['calls']:.0f} + 0**"
        elif m == "analogtobi":
            ct = f"{a['calls']:.0f} + 0††"
        elif m == "analoggenie":
            ct = f"{a['calls']:.0f} + N/A‡"
        else:
            ct = f"{a['calls']:.0f} + 0§"
        online_h = a["runtime_mean"] * N_RUNS / 3600
        if m in TRAIN_H:
            mark, th = TRAIN_H[m]
            suffix = mark.lstrip("~0123456789. hs")   # trailing footnote marker
            rt_t = f"~{online_h + th:.1f} h{suffix}"
        else:
            rt_t = f"{online_h:.1f} h + N/A‡"
        pvt = f(a["pvt_pct"], 1) if a["pvt_pct"] is not None else "n/a"
        if a.get("rec_coverage"):
            pvt += f"¶ ({a['rec_coverage']})"
        pw = a.get("fom_power_median")
        fs = a.get("fom_spec_median")
        fom_cell = ((f"{fs:,.1f}" if fs is not None else "n/a")
                    + f" ({a['fom_median']:.0f}ᵐ/"
                    + (f"{pw:,.0f}ᵖ)" if pw is not None else "n/aᵖ)"))
        if a.get("rec_coverage"):
            fom_cell += "¶"
        row = [LABEL[m], f(a["final_pass"]), p5, f"{a['calls']:.0f}", ct,
               f"{a['runtime_mean']:.0f}", rt_t, fom_cell, pvt]
        md.append("| " + " | ".join(row) + " |")
        tex.append(" & ".join(str(x) for x in row) + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    md += ["",
           "† frozen Track-A P@5 measurement (native 5-proposal protocol, "
           "pre-cleanup provenance).",
           "‡ official pretrained checkpoint; historical training SPICE/"
           "runtime not reconstructible (CktGen: OCB behavioral labels; "
           "AnalogGenie: human-designed corpus, no training SPICE). "
           "offline_training_data: CktGen 10k OCB circuits, AnalogGenie 3,350 "
           "circuits.",
           "§ AG SFT proposer fine-tune: MEASURED 528.7 s wall (1,500 steps, "
           "917 records, sft_adapter_tier2_sft.json), zero SPICE in the "
           "training loop.",
           "†† AnalogToBi training cost AUTHOR-REPORTED (cited, not our "
           "measurement): 22.8 h on an NVIDIA RTX 5880 (arXiv:2603.08720, "
           "100k iters, batch 64, best val at step 97,500); training uses "
           "zero SPICE (human-designed corpus of 1,588 circuits). "
           "offline_training_data: 1,588 circuits (+renaming augmentation).",
           "** CktGen training cost EXTRAPOLATED AT MEASURED RATE (not a "
           "measured total): official 600-epoch recipe probed locally at "
           "238.2 s/epoch -> ~39.7 h; zero SPICE in training loop.",
           "Runtime+Train = CAMPAIGN GRAND TOTAL: online runtime x 87 runs "
           "(29 specs x 3 seeds) + one-time benchmark-specific training; the "
           "per-spec online cost stays in the Runtime column.",
           "¶ AG cells from the winner-reconstruction study "
           "(ag_reconstruction.jsonl): each campaign winner rebuilt from its "
           "recorded family + 7-knob sizing vector, gated on matching the "
           "recorded nominal (|dGain|<=2 dB, UGBW ratio 0.5-2); coverage "
           "shown as verified/total, excluded winners are edited graphs the "
           "family string cannot encode. PVT = VT corners on the rebuilt "
           "native design. FoM = the SYMMETRIC arm (same 13-call "
           "margin-maximizing tune the baseline candidates received), making "
           "the FoM column protocol-matched across all rows; AG's "
           "native-objective FoM (power-efficiency units) is reported in the "
           "Tier-3 table, not here.",
           "FoM cell = SPEC-REFERENCED FoM (margin-FoMᵐ / power-FoMᵖ in "
           "parentheses, full disclosure). Spec-referenced FoM = "
           "UGBW_target x CL / IDD (MHz*pF/mA): the benchmark FoM with "
           "the TARGET bandwidth substituted for the achieved one -- in "
           "a spec-driven benchmark FinalPass rewards meeting targets "
           "and PVT rewards robustness, so FoM measures the CURRENT "
           "COST of delivering the required spec; bandwidth overshoot "
           "nobody requested earns no credit. Identical measurements, "
           "identical formula, all methods. margin-FoMᵐ = uniform "
           "relative-margin score (shared judge); power-FoMᵖ = "
           "UGBW*CL/IDD (MHz*pF/mA) from the measured DC supply current "
           "of the SAME returned design -- both medians over passing "
           "specs. The specs carry no current limit; the two FoMs "
           "expose the overshoot-vs-efficiency trade explicitly. "
           "Margin-FoM detail: median over "
           "passing specs; means in aggregate_results.csv. Baselines: common "
           "65-call tuner over the method's first-5 candidates (13 each); AG: "
           "its own end-to-end pipeline under the same 65-call envelope. PVT "
           "corners (vdd±10% × 0/70°C) on the returned design, "
           "counted separately."]
    (ART / "comparison_table.md").write_text("\n".join(md), encoding="utf-8")
    (ART / "comparison_table.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(md))
    # aggregate csv
    with open(ART / "aggregate_results.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n", "final_pass", "fp_ci95", "p5", "calls",
                    "runtime_mean", "runtime_std", "fom_mean", "fom_std",
                    "fom_median", "n_feasible", "pvt_pct"])
        for m in methods:
            a = A[m]
            w.writerow([m] + [a[k] for k in
                              ("n", "final_pass", "fp_ci95", "p5", "calls",
                               "runtime_mean", "runtime_std", "fom_mean",
                               "fom_std", "fom_median", "n_feasible",
                               "pvt_pct")])


def per_spec_csv(per):
    with open(ART / "per_spec_results.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "spec_id", "final_pass", "pass_at_5",
                    "online_spice_calls", "training_spice_calls",
                    "calls_plus_train", "online_runtime_s",
                    "training_runtime_s", "runtime_plus_train_s", "final_fom",
                    "pvt_pass_pct", "valid_topology", "topology_rank"])
        for r in sorted(per, key=lambda x: (x["method"], x["seed"], x["spec"])):
            pvt = (100 * r["vt"] / 4) if r["vt"] is not None else ""
            w.writerow([r["method"], r["seed"], r["spec"],
                        int(r["final_pass"]),
                        ("" if r["p5"] is None else int(r["p5"])),
                        r["calls"], "NA", "NA", r["runtime"], "NA", "NA",
                        ("" if r["fom"] is None else r["fom"]), pvt,
                        ("" if r["valid_cands"] is None else r["valid_cands"]),
                        ""])
    print("per_spec_results.csv written")


def topology_table(per):
    rows = [json.loads(l) for l in (ART / "results.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    lines = ["# Topology-specific secondary table", "",
             "| Method | Valid topology (%) | Unique (%) | Novel (%) | "
             "Top-5 feasible (%) | FinalPass | Best FoM | Gen time (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for m in ("cktgen", "analogtobi"):
        rs = [r for r in rows if r["method"] == m]
        cands = [c for r in rs for c in r["candidates"]]
        valid = 100 * sum(1 for c in cands if c.get("valid")) / len(cands)
        # uniqueness on realized structures per method (proxy: candidate FoM
        # trace identity is not usable; use per-spec distinct passing ranks)
        feas = 100 * sum(1 for c in cands if c.get("pass")) / len(cands)
        fp = sum(r["final_pass"] for r in rs) / len(rs)
        best = max((r["final_fom"] for r in rs
                    if r["final_fom"] is not None), default=None)
        gen = statistics.mean(r["generation_time_s"] for r in rs)
        uniq = "see note"
        lines.append(f"| {LABEL[m]} | {valid:.1f} | {uniq} | n/a* | "
                     f"{feas:.1f} | {fp:.3f} | "
                     f"{best if best is not None else 'n/a'} | {gen:.1f} |")
    lines += ["",
              "\\* Novelty vs each method's own training corpus is not "
              "measurable without corpus-wide isomorphism baselines; not "
              "claimed. Uniqueness note: CktGen candidates collapse onto the "
              "benchmark's cascade family space at realization (distinct "
              "abstract graphs, small realized-family alphabet); AnalogGenie "
              "candidates are structurally distinct transistor graphs "
              "(unconditional pool, 5 per seed)."]
    (ART / "topology_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("topology_table.md written")


def breakdowns(per):
    with open(ART / "runtime_breakdown.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "spec_id", "generation_s", "eval_s",
                    "total_s"])
        for r in per:
            if r["method"].startswith("ag"):
                continue
            w.writerow([r["method"], r["seed"], r["spec"], r["gen_s"],
                        r["eval_s"], r["runtime"]])
    with open(ART / "spice_call_breakdown.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", "spec_id", "optimization_calls",
                    "pvt_calls", "training_calls"])
        for r in per:
            w.writerow([r["method"], r["seed"], r["spec"], r["calls"],
                        r["pvt_calls"], 0 if r["method"].startswith("ag")
                        else "NA"])
    print("breakdown CSVs written")


if __name__ == "__main__":
    per = load_all()
    tables(per)
    per_spec_csv(per)
    paired_tests(per)
    topology_table(per)
    breakdowns(per)
