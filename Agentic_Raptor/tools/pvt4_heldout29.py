"""4-corner V/T PVT re-sweep of the HELDOUT29 R2 winners (step 1 of the DATE plan).

WHY
The R2 campaign ran PVT with the launcher defaults (--pvt-corners tt
--pvt-voltages 1.8 --pvt-temps-c 27): ONE corner. Every "100% PVT" cell rests on
total_pvt_corners == 1. The benchmark protocol (metric_definitions.md) is four
V/T corners: VDD +/-10% x {0, 70} C at process tt. This re-sweeps the same 74
verified winners under that protocol without re-running the pipeline.

HOW THE SIZED GRAPH IS REBUILT (byte-identical to the pipeline's own path)
run_raptor_v2.py:447:  g = c["device_graph"] if ... else _realise(c["obj"])
bandit_selector.py:128 sets device_graph=None, so every winner went through
_realise(proposal). Then run_raptor_v2.py:537: apply_knobs(g, knob_vector).
    hash (stage6.topology_hash) -> corpus response -> json.loads -> _realise
    -> apply_knobs(g, [sizing_vector[k] for k in KNOB_NAMES]) -> sized graph
    -> run_pvt_sweep(...) -> aggregate_pvt(...)

VALIDATION FIRST
--validate re-runs the single nominal corner (tt, 1.8 V, 27 C) on N winners and
diffs gain/pm/ugbw against the values the pipeline itself recorded in
pvt.corners[0]. If the rebuild does not reproduce those, STOP: the 4-corner
numbers would be garbage. Only sweep after validation passes.

GUARDRAILS
Read-only on every run file and on the sealed blindtest. Writes ONLY to
artifacts/publication_v3/heldout29_pvt4/. Never touches production checkpoints.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\kobeo\OneDrive\Desktop\raptor1\Agentic_Raptor")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from agentic_raptor.electrical import discover_ngspice                      # noqa: E402
from agentic_raptor.electrical.pvt_eval import (PvtConfig, aggregate_pvt,    # noqa: E402
                                                 run_pvt_sweep, generate_corners)
from agentic_raptor.mb_sac.spec_sizing import KNOB_NAMES, apply_knobs        # noqa: E402
from run_puct_ablation import _realise                                        # noqa: E402

RUN_GLOB = "artifacts/publication_v2/raptor_v2_runs/ABLv3HELDOUT29R2_AG_FULL_s*_heldout_*.json"   # overridden by --tag
CORPORA = ("artifacts/stage3e4/corpus.json",
           "artifacts/publication_v3/tier2/corpus_tier2_extension.json",
           "artifacts/publication_v3/tier2/corpus_tier2_mixed.json")
OUT = ROOT / "artifacts/publication_v3/heldout29_pvt4"


def P(s: str) -> None:
    print(str(s).encode("ascii", "replace").decode(), flush=True)


def load_store() -> dict[str, str]:
    store: dict[str, str] = {}
    for p in CORPORA:
        for r in json.load(open(p, encoding="utf-8"))["records"]:
            for k in ("variant_hash", "canonical_graph_hash"):
                if r.get(k):
                    store.setdefault(r[k], r["response"])
    return store


def _nominal_pass(meas: dict | None, targets: dict) -> bool:
    """Same three constraints stage 9 applies (gain, pm, ugbw >= target)."""
    if not meas:
        return False
    return all(meas.get(k) is not None and float(meas[k]) >= float(targets[k])
               for k in ("gain_db", "pm_deg", "ugbw_hz"))


def winners(which: str = "selected"):
    """which="selected": the 74 R2 winners (the design the ranker chose, exact
    nominal pass). which="backup": for those SAME 74 runs, the OTHER sized
    design (stage8 backup_design) -- stage 9 measured it at nominal too, so
    its nominal pass is derived from stage9.measured vs targets."""
    for f in sorted(glob.glob(RUN_GLOB)):
        d = json.load(open(f, encoding="utf-8"))
        s9 = d["stage9_verification"]
        if not s9.get("selected_exact_pass"):
            continue
        sel = d["stage8_ranker"]["selected_design"]
        # robust-delivery campaigns record the DELIVERED design (may be the runner-up)
        sel = (s9.get("robust_delivery") or {}).get("delivered") or sel
        if which == "backup":
            sel = d["stage8_ranker"].get("backup_design")
            if not sel or sel not in d["stage6_sizing"]:
                continue
            bpass = _nominal_pass((s9.get("measured") or {}).get(sel), s9["targets"])
        s6 = d["stage6_sizing"][sel]
        yield {
            "file": f,
            "stem": f.split("_heldout_", 1)[1].rsplit(".json", 1)[0],
            "seed": int(f.split("_s", 1)[1].split("_", 1)[0]),
            "label": sel,
            "hash": s6["topology_hash"],
            "sizing_vector": s6["sizing_vector"],
            "spec": d["stage1_spec"]["spec"],
            "c_load_f": (d.get("nominal") or {}).get("simulated_c_load_f")
                        or d["stage1_spec"].get("effective_c_load_f"),
            "recorded_corner0": ((d.get("pvt") or {}).get("corners") or [None])[0],
            "family": next((r.get("canonical_family") for r in d["stage5_alphazero"]["ranking"]
                            if r.get("canonical_graph_hash") == s6["topology_hash"]), None),
            "backup_nominal_pass": (bpass if which == "backup" else True),
        }


def rebuild(w, store):
    obj = json.loads(store[w["hash"]])
    g = _realise(obj)
    if g is None:
        raise RuntimeError(f"_realise returned None for {w['hash']}")
    knobs = [float(w["sizing_vector"][k]) for k in KNOB_NAMES]
    return apply_knobs(g, knobs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=0,
                    help="re-run the single nominal corner on N winners and diff vs recorded")
    ap.add_argument("--sweep", action="store_true", help="run the 4-corner protocol on all winners")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backup", action="store_true",
                    help="sweep the runner-up (stage8 backup_design) of the same 74 runs instead")
    ap.add_argument("--tag", default="ABLv3HELDOUT29R2_AG_FULL", help="campaign tag prefix of the run files")
    ap.add_argument("--corners", default="tt", help="comma list of process corners, e.g. ss,tt,ff")
    a = ap.parse_args()
    global RUN_GLOB, OUT
    RUN_GLOB = f"artifacts/publication_v2/raptor_v2_runs/{a.tag}_s*_heldout_*.json"
    pcs = tuple(x.strip() for x in a.corners.split(",") if x.strip())
    if a.tag != "ABLv3HELDOUT29R2_AG_FULL" or pcs != ("tt",):
        OUT = ROOT / f"artifacts/publication_v3/heldout29_pvt4/{a.tag}__{'_'.join(pcs)}"

    from agentic_raptor.electrical import pvt_eval as PE
    nom_v = getattr(PE, "NOMINAL_SUPPLY_V", None)
    nom_t = getattr(PE, "NOMINAL_TEMPERATURE_C", None)
    P(f"[cfg] NOMINAL_SUPPLY_V={nom_v}  NOMINAL_TEMPERATURE_C={nom_t}")
    assert abs(float(nom_v) - 1.8) < 1e-9, f"nominal supply is {nom_v}, corners assume 1.8 V"

    exe = discover_ngspice()
    store = load_store()
    ws = list(winners("backup" if a.backup else "selected"))
    if a.limit:
        ws = ws[: a.limit]
    P(f"[cfg] ngspice={exe}")
    P(f"[cfg] winners={len(ws)}  store_hashes={len(store)}")
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = OUT / "_ngspice_scratch"
    scratch.mkdir(exist_ok=True)

    if a.validate:
        cfg = PvtConfig(enabled=True, process_corners=("tt",), supply_voltages=(1.8,),
                        temperatures_c=(27.0,))
        P(f"\n=== VALIDATE: nominal corner on {min(a.validate, len(ws))} winners vs recorded ===")
        P(f"{'winner':34}{'metric':>8}{'recorded':>14}{'rebuilt':>14}{'rel diff':>10}")
        worst = 0.0
        for w in ws[: a.validate]:
            rec = w["recorded_corner0"]
            g = rebuild(w, store)
            rows = run_pvt_sweep(w["hash"], g, w["spec"], exe, scratch, cfg,
                                 label=w["label"], c_load_f=w["c_load_f"])
            got = rows[0]
            for m in ("gain_db", "pm_deg", "ugbw_hz"):
                r0, r1 = rec.get(m), got.get(m)
                if r0 is None or r1 is None:
                    P(f"{w['stem'] + '/s' + str(w['seed']):34}{m:>8}{str(r0):>14}{str(r1):>14}{'n/a':>10}")
                    continue
                rel = abs(r1 - r0) / max(abs(r0), 1e-12)
                worst = max(worst, rel)
                P(f"{w['stem'] + '/s' + str(w['seed']):34}{m:>8}{r0:>14.4g}{r1:>14.4g}{rel:>10.2e}")
        verdict = "PASS -- rebuild reproduces the pipeline's nominal measurement" if worst < 1e-3 \
            else f"FAIL -- worst rel diff {worst:.3e}; do NOT trust a 4-corner sweep from this rebuild"
        P(f"\n[validate] worst relative diff = {worst:.3e}  ->  {verdict}")
        (OUT / "validation.json").write_text(json.dumps({"n": min(a.validate, len(ws)),
                                                          "worst_rel_diff": worst,
                                                          "verdict": verdict}, indent=1), encoding="utf-8")
        return

    if a.sweep:
        cfg = PvtConfig(enabled=True, process_corners=pcs, supply_voltages=(1.62, 1.98),
                        temperatures_c=(0.0, 70.0))
        ids = [c.pvt_corner_id for c in generate_corners(cfg)]
        P(f"\n=== SWEEP: {len(ids)} corners {ids} on {len(ws)} winners ===")
        outp = OUT / ("heldout29_pvt4_backup_runs.jsonl" if a.backup else "heldout29_pvt4_runs.jsonl")
        # RESUME: the jsonl is written one winner at a time, so a relaunch after
        # a timeout skips what is already on disk instead of re-simulating it.
        # (stem, seed) uniquely identifies a run -- one selected design per run --
        # so it is the resume key; it also tolerates rows written by the earlier
        # harness version that carried no "label" field.
        done: set[tuple[str, int]] = set()
        if outp.exists():
            for l in outp.open(encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    done.add((r["stem"], int(r["seed"])))
        todo = [w for w in ws if (w["stem"], w["seed"]) not in done]
        P(f"[resume] already on disk: {len(done)}   remaining: {len(todo)}")
        t0 = time.time()
        with outp.open("a", encoding="utf-8") as fh:
            for i, w in enumerate(todo, 1):
                g = rebuild(w, store)
                if a.backup:
                    # stage 9 only simulated the runner-up when the winner
                    # FAILED, so for these 74 runs its nominal result is not
                    # on record: measure it (1 call) before the corners.
                    nom_cfg = PvtConfig(enabled=True, process_corners=("tt",),
                                        supply_voltages=(1.8,), temperatures_c=(27.0,))
                    nom = run_pvt_sweep(w["hash"], g, w["spec"], exe, scratch, nom_cfg,
                                        label=w["label"], c_load_f=w["c_load_f"])[0]
                    w["backup_nominal_pass"] = bool(nom.get("complete_pass"))
                    w["backup_nominal"] = {k: nom.get(k) for k in
                                           ("gain_db", "pm_deg", "ugbw_hz", "idd_a", "failure_reasons")}
                rows = run_pvt_sweep(w["hash"], g, w["spec"], exe, scratch, cfg,
                                     label=w["label"], c_load_f=w["c_load_f"])
                agg = aggregate_pvt(rows, cfg.required_corner_ids)
                rec = {"stem": w["stem"], "seed": w["seed"], "label": w["label"],
                       "family": w["family"], "hash": w["hash"], "corners": rows,
                       "backup_nominal_pass": w.get("backup_nominal_pass", True),
                       "backup_nominal": w.get("backup_nominal"),
                       **{k: v for k, v in agg.items()}}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                rc = bool(agg.get("robust_complete_pass"))
                P(f"  [{len(done)+i:2d}/{len(ws)}] {w['stem']:30} s{w['seed']} {str(w['family']):10} "
                  f"pass={agg.get('passed_pvt_corners')}/{agg.get('total_pvt_corners')} "
                  f"robust={rc}  ({(time.time()-t0)/60:.1f} min)")
        # SUMMARY from the FULL jsonl, not the loop -- correct after any resume.
        allr = [json.loads(l) for l in outp.open(encoding="utf-8") if l.strip()]
        n = len(allr)
        robust = sum(1 for r in allr if r.get("robust_complete_pass"))
        pcts = [r.get("pvt_pass_percent") or 0.0 for r in allr]
        per_fam: dict[str, list[int]] = {}
        for r in allr:
            per_fam.setdefault(str(r.get("family")), []).append(int(bool(r.get("robust_complete_pass"))))
        summ = {"protocol": f"{'/'.join(pcs)} x VDD{{1.62,1.98}} x T{{0,70}}C ({len(ids)} corners)",
                "n_winners": n, "robust_complete_pass": robust,
                "robust_pct": round(100 * robust / n, 1) if n else None,
                "mean_pvt_pass_percent": round(st.mean(pcts), 1) if pcts else None,
                "per_family_robust": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_fam.items())},
                "single_corner_R2_reference": "74/74 = 100.0% at total_pvt_corners=1"}
        (OUT / ("heldout29_pvt4_backup_summary.json" if a.backup else "heldout29_pvt4_summary.json")).write_text(json.dumps(summ, indent=1), encoding="utf-8")
        P("\n=== 4-CORNER RESULT ===")
        for k, v in summ.items():
            P(f"  {k}: {v}")
        return

    P("nothing to do: pass --validate N or --sweep")


if __name__ == "__main__":
    main()
