"""Extends run_family_spec_gate.py to a SECOND structural axis: output_buffer
and local_feedback, which build_diverse_corpus.py currently never targets
(target_for() hard-codes buf=False, fb=False for every family).

These are NOT cosmetic labels -- stage3e4.py's real-candidate qualification
path applies ADD_SUPPORTED_OUTPUT_STAGE / CONNECT_VERIFIED_FEEDBACK_PATH
edits to the device graph when they're set, so a (family, buf, fb) triple can
be a genuinely different circuit from its (family, False, False) baseline,
not just a different label on the same netlist. Right now the SFT proposer is
never taught these variants exist -- corpus_diverse.json's 5 distinct
canonical graphs are exactly the 5 realizable families at buf=fb=False, and
nothing else.

This gate answers, per (family, buf, fb) combo not already covered by the
base gate: is there ANY corpus spec (in that family's difficulty band) this
combo can meet? Real SPICE, same best-of-N-repeats existence-claim logic as
run_family_spec_gate.py (sac_size is not seed-reproducible), so a pass here
is evidence, not a lucky draw.

Writes artifacts/publication_v2/family_spec_gate/SUMMARY_bufsfb.json --
separate file, does NOT touch SUMMARY.json (the base 5-family gate that
corpus_diverse.json's realizable_families() already reads and that the
diversity-repair comparison depends on staying intact).

COST WARNING: this is 5 families x 3 new combos (buf,fb) x --per-family specs
x --repeats sizing calls, each a real SPICE optimization loop (~5-7 min each
going by the base gate's logged per-probe seconds). At the base gate's
defaults (per-family=3, repeats=3) that's up to 135 real sizing runs, i.e.
plausibly several hours. Start with smaller --per-family/--repeats for a
first pass; widen only once the first pass shows which combos are even
worth probing more carefully.

Run:  python run_family_spec_gate_bufsfb.py --per-family 2 --repeats 2
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v2/family_spec_gate"
FAMILIES = ["2s_none", "2s_miller", "2s_rc", "3s_miller", "3s_rc"]
#: (False, False) is the base gate's own axis -- already probed, already
#: what corpus_diverse.json uses. Only the NEW combos need real SPICE time.
NEW_COMBOS = [(True, False), (False, True), (True, True)]


def realise_variant(fam: str, buf: bool, fb: bool):
    """(family, buf, fb) -> device graph, or None if the edit is structurally
    illegal for this family (EditRejected), which is itself a real result --
    not every family need support every combo."""
    from agentic_raptor.topology_rl.stage3e2_edits import (EditRejected,
                                                            apply_edit)
    from run_puct_ablation import realise_class
    g = realise_class(fam)
    try:
        if buf:
            g, _a = apply_edit(g, "ADD_SUPPORTED_OUTPUT_STAGE")
        if fb:
            g, _a = apply_edit(g, "CONNECT_VERIFIED_FEEDBACK_PATH")
    except EditRejected as exc:
        return None, str(exc)
    return g, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--per-family", type=int, default=2,
                    help="specs probed per (family,buf,fb), easiest first "
                         "(base gate default is 3; start lower here given "
                         "3x more combos to cover)")
    ap.add_argument("--repeats", type=int, default=2,
                    help="sizing attempts per probe, best-of-N (base gate "
                         "default is 3; sac_size is not seed-reproducible)")
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()
    families = [f for f in FAMILIES if f in args.families.split(",")]

    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mb_sac.spec_sizing import sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from run_family_spec_gate import corpus_specs_by_family, difficulty

    by = corpus_specs_by_family()
    exe = discover_ngspice()
    runs = (OUT / "runs_bufsfb").resolve()
    runs.mkdir(parents=True, exist_ok=True)
    report = {}
    for fam in families:
        specs = by.get(fam, [])
        if not specs:
            for buf, fb in NEW_COMBOS:
                report[f"{fam}_buf{int(buf)}_fb{int(fb)}"] = {"status": "NO_CORPUS_SPECS"}
            continue
        idxs = sorted({0, len(specs) // 4, len(specs) // 2})[:args.per_family]
        for buf, fb in NEW_COMBOS:
            key = f"{fam}_buf{int(buf)}_fb{int(fb)}"
            g, rejected = realise_variant(fam, buf, fb)
            if g is None:
                report[key] = {"status": "EDIT_REJECTED", "reason": rejected}
                print(f"{key:20} EDIT_REJECTED: {rejected}", flush=True)
                continue
            results = []
            for i in idxs:
                spec = specs[i]
                t0 = time.time()
                trials = []
                for t in range(args.repeats):
                    sz = sac_size(f"fsgb_{key}_{i}", g, spec, exe, runs,
                                  new_costs(), budget=args.budget,
                                  seed=7 + t, persist=False)
                    trials.append(sz)
                best_sz = min(trials, key=lambda s: (
                    not s["outcome"]["exact_spec_pass"],
                    s["outcome"]["normalized_distance_to_feasibility"]
                    if s["outcome"]["normalized_distance_to_feasibility"]
                    is not None else float("inf")))
                o, b = best_sz["outcome"], best_sz["best"]
                n_pass = sum(1 for s in trials if s["outcome"]["exact_spec_pass"])
                results.append({
                    "rank": i, "percentile": round(100 * i / len(specs)),
                    "context_id": spec.get("context_id"),
                    "gain_target": spec["gain_target_db"],
                    "pm_target": spec["phase_margin_target_deg"],
                    "ugbw_target": spec["ugbw_target_hz"],
                    "load_pf": spec["load_capacitance_pf"],
                    "achieved_gain": b["gain_db"], "achieved_pm": b["pm_deg"],
                    "achieved_ugbw": b["ugbw_hz"],
                    "exact_pass": o["exact_spec_pass"],
                    "distance": o["normalized_distance_to_feasibility"],
                    "reason": o.get("exact_failure_reason"),
                    "repeats": args.repeats, "passes": n_pass,
                    "seconds": round(time.time() - t0, 1)})
                print(f"{key:20} p{results[-1]['percentile']:>3} "
                     f"gain>={spec['gain_target_db']:6.1f} "
                     f"pm>={spec['phase_margin_target_deg']:4.1f} "
                     f"ugbw>={spec['ugbw_target_hz']:.0e} | "
                     f"pass={o['exact_spec_pass']!s:5} ({n_pass}/{args.repeats}) "
                     f"{o.get('exact_failure_reason') or ''}", flush=True)
            passing = [r for r in results if r["exact_pass"]]
            report[key] = {
                "family": fam, "buf": buf, "fb": fb,
                "corpus_specs": len(specs), "probed": len(results),
                "passing": len(passing),
                "realizable_for_some_spec": bool(passing),
                "hardest_percentile_passed": (max(r["percentile"] for r in passing)
                                              if passing else None),
                "results": results}

    realizable = [k for k, r in report.items() if r.get("realizable_for_some_spec")]
    doc = {"budget": args.budget, "per_family": args.per_family,
          "repeats": args.repeats, "combos": report,
          "realizable_combos": realizable,
          "realizable_count": len(realizable),
          "note": ("NEW structural axis beyond the base 5-family gate -- "
                   "buf=False,fb=False is that gate, unchanged, at "
                   "family_spec_gate/SUMMARY.json"),
          "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    from agentic_raptor.publication.artifact_provenance import stamp
    doc = stamp(doc, model_type="family_spec_gate_bufsfb")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY_bufsfb.json").write_text(
        json.dumps(doc, indent=1, default=str), encoding="utf-8")
    print(f"\nrealizable (family,buf,fb) combos: {realizable}")
    print(f"count: {len(realizable)}/{len(families) * len(NEW_COMBOS)}")
    print(f"written -> {(OUT / 'SUMMARY_bufsfb.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
