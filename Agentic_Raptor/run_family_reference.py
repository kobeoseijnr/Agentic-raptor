"""Fix 6: known-feasible reference circuit per structure family.

The point is attribution. If `3s_miller` cannot reach spec even when a human
hands it good device sizes, then a poor optimiser score for that family says
nothing about the optimiser -- the physical realisation or the testbench is
at fault. Blaming SAC for it would be a measurement error, not a finding.

For each of the five corpus families this searches a small, deliberately
coarse grid for ONE sizing that meets its own reference specification on real
ngspice, then records:

  * the netlist actually simulated;
  * the operating point, stability, gain, PM, UGBW;
  * the knob values with explicit units;
  * a repeat run proving reproducibility.

Families with no feasible reference are marked UNRESOLVED and must be
excluded from the optimiser comparison until the implementation is repaired.

Run:  python run_family_reference.py [--budget 40]
"""
import argparse
import json
import time
from pathlib import Path

from agentic_raptor.publication import PUB
from run_puct_ablation import CORPUS_CLASSES, realise_class

ROOT = PUB.parent.parent
OUT = PUB.parent / "publication_v2" / "family_reference"


def family_of(stages: int, comp: str) -> str:
    comp = {"miller_cap": "miller", "rc_nulling": "rc"}.get(comp, comp)
    return f"{stages}s_{comp}"

def corpus_reference_specs() -> dict:
    """Reference spec per family, DERIVED FROM THE CORPUS.

    Hand-picked references were wrong in a way that quietly invalidated the
    gate. The corpus pairs UNCOMPENSATED families with LIGHT loads and
    compensated ones with heavy loads -- physically sensible, since a light
    load needs less compensation:

        2s_none    mean load  74 pF        3s_none   mean load  72 pF
        2s_miller  mean load 612 pF        3s_rc     mean load 494 pF

    The earlier hand-written references used a flat 200 pF for everything,
    testing 3s_none at 2.8x the load the system ever asks of it and then
    recording the resulting instability as "family unrealisable". A gate must
    test a family against the conditions it is actually used under, or it
    measures the test author rather than the circuit.

    Per-family MEDIAN of gain / PM / UGBW / load over the corpus records that
    use that family. Falls back to the previous constants if the corpus is
    unavailable.
    """
    import statistics as st

    from agentic_raptor.llm_dpo import integrity as ig
    path = ROOT / "artifacts/stage3e4/corpus.json"
    if not path.is_file():
        return dict(_FALLBACK_SPECS)
    corpus = json.loads(path.read_text(encoding="utf-8"))
    by = {}
    for r in corpus["records"]:
        s = ig.parse_spec(r.get("prompt") or "")
        if not s:
            continue
        fam = family_of(r["stages"], r["comp"])
        by.setdefault(fam, []).append(s)
    out = {}
    for fam, specs in by.items():
        out[fam] = {
            "gain_target_db": round(st.median(
                x["gain_target_db"] for x in specs), 2),
            "phase_margin_target_deg": round(st.median(
                x["phase_margin_target_deg"] for x in specs), 1),
            "load_capacitance_pf": round(st.median(
                x["load_capacitance_pf"] for x in specs), 1),
            "ugbw_target_hz": float(st.median(
                x["ugbw_target_hz"] for x in specs)),
            "source": "corpus_median", "n_specs": len(specs)}
    for fam, spec in _FALLBACK_SPECS.items():
        out.setdefault(fam, dict(spec, source="fallback_constant"))
    return out


#: only used when the corpus is missing
_FALLBACK_SPECS = {
    "2s_none": {"gain_target_db": 40.0, "phase_margin_target_deg": 45.0,
                "load_capacitance_pf": 200.0, "ugbw_target_hz": 1e4},
    "2s_miller": {"gain_target_db": 45.0, "phase_margin_target_deg": 50.0,
                  "load_capacitance_pf": 200.0, "ugbw_target_hz": 1e4},
    # 2s_rc replaced 3s_none in the target space (uncompensated 3-stage is
    # unstable by construction); same difficulty band as 2s_miller, which it
    # strictly dominates by nulling the RHP zero
    "2s_rc": {"gain_target_db": 45.0, "phase_margin_target_deg": 55.0,
              "load_capacitance_pf": 200.0, "ugbw_target_hz": 1e4},
    "3s_miller": {"gain_target_db": 65.0, "phase_margin_target_deg": 50.0,
                  "load_capacitance_pf": 200.0, "ugbw_target_hz": 1e4},
    "3s_rc": {"gain_target_db": 65.0, "phase_margin_target_deg": 55.0,
              "load_capacitance_pf": 200.0, "ugbw_target_hz": 1e4},
}
REFERENCE_SPECS = corpus_reference_specs()

#: units are part of the contract: a silent um-vs-m or pF-vs-F error would
#: look exactly like an optimiser failure
KNOB_UNITS = {"s1_w": "relative multiplier on stage-1 device width",
              "s2_w": "relative multiplier on stage-2/output device width",
              "s1_l": "relative multiplier on stage-1 device length",
              "s2_l": "relative multiplier on stage-2 device length",
              "cap_x": "relative multiplier on compensation/load capacitor",
              "ib_x": "relative multiplier on bias current source"}


def main():
    ap = argparse.ArgumentParser()
    # 40 was too few once compensation reached the pole-splitting region:
    # the search lands on a stable-but-slow point and has to trade capacitance
    # back for bandwidth. Measured on 3s_miller -- budget 40: UGBW 6745 Hz
    # (fail); budget 80: UGBW 11001 Hz (PASS). A reference test that fails for
    # want of search budget says nothing about realisability.
    ap.add_argument("--budget", type=int, default=80,
                    help="real-SPICE calls per family")
    ap.add_argument("--families", default=",".join(CORPUS_CLASSES))
    args = ap.parse_args()
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mb_sac.spec_sizing import (KNOB_NAMES,
                                                   postsizing_outcome,
                                                   sac_size)
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    exe = discover_ngspice()
    runs = (OUT / "runs").resolve()
    runs.mkdir(parents=True, exist_ok=True)
    results = {}
    for fam in args.families.split(","):
        spec = REFERENCE_SPECS[fam]
        t0 = time.time()
        g = realise_class(fam)
        sz = sac_size(f"ref_{fam}", g, spec, exe, runs, new_costs(),
                      budget=args.budget, seed=7, persist=False)
        best, out = sz["best"], sz["outcome"]
        feasible = bool(out["exact_spec_pass"])
        rec = {
            "family": fam, "reference_spec": spec,
            "feasible": feasible,
            "status": "OK" if feasible else "UNRESOLVED",
            "best_knobs": best.get("knobs"),
            "knob_units": KNOB_UNITS,
            "achieved": {"gain_db": best.get("gain_db"),
                         "pm_deg": best.get("pm_deg"),
                         "ugbw_hz": best.get("ugbw_hz"),
                         "power_w": best.get("power_w")},
            "operating_point_valid": out.get("operating_point_valid"),
            "spice_converged": out.get("spice_converged"),
            "stability_status": out.get("stability_status"),
            "constraints": out.get("constraints"),
            "exact_failure_reason": out.get("exact_failure_reason"),
            "worst_failing_constraint": out.get("worst_failing_constraint"),
            "distance": out.get("normalized_distance_to_feasibility"),
            "spice_calls": sz["spice_calls"],
            "seconds": round(time.time() - t0, 1)}
        # reproducibility: same knobs, same seed, must give the same verdict
        if best.get("knobs"):
            sz2 = sac_size(f"rep_{fam}", realise_class(fam), spec, exe, runs,
                           new_costs(), budget=args.budget, seed=7,
                           persist=False)
            o2 = sz2["outcome"]
            rec["reproducible"] = (
                bool(o2["exact_spec_pass"]) == feasible)
            rec["repeat_distance"] = o2.get(
                "normalized_distance_to_feasibility")
        results[fam] = rec
        print(f"{fam:11} {rec['status']:10} gain="
              f"{(rec['achieved']['gain_db'] or 0):7.2f} dB  pm="
              f"{(rec['achieved']['pm_deg'] or 0):7.2f} deg  "
              f"dist={rec['distance']}  "
              f"reason={rec['exact_failure_reason']}")
    OUT.mkdir(parents=True, exist_ok=True)
    unresolved = [f for f, r in results.items() if not r["feasible"]]
    doc = {"budget": args.budget, "families": results,
           "resolved": [f for f, r in results.items() if r["feasible"]],
           "unresolved": unresolved,
           "note": ("families listed as unresolved have no known-feasible "
                    "reference under the current PDK/testbench and MUST be "
                    "excluded from the optimiser comparison -- a poor score "
                    "there measures the realisation, not the optimiser"),
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "SUMMARY.json").write_text(json.dumps(doc, indent=1, default=str),
                                      encoding="utf-8")
    print(f"\nresolved  : {doc['resolved']}")
    print(f"unresolved: {unresolved}")


if __name__ == "__main__":
    main()
