"""Counterfactual value targets: measure SPEC-INCOMPATIBLE structure classes.

Diagnosis this repairs: of the accumulated post-sizing value targets, 354/355
pair a structure family with a spec of its OWN stage tier -- the value head
never observes a wrong-tier design's outcome. Because 2-stage specs are the
easier ones, the head learns "2-stage looks good" unconditionally and keeps a
2-stage proposal offered against a 3-stage spec (measured: 0/11 rescues).

This module sizes each TRAIN-split spec against the incompatible tier's
classes with real ngspice and records the measured outcome through the same
value_target_from_outcome() the pipeline uses, so the head finally sees what
a mismatch costs.

Leakage discipline: TRAIN-split specs only -- heldout/blindtest never enter
value training. Output goes to its own file and nothing reads it unless
value_refresh is invoked with include_counterfactual=True, so campaigns and
the F-battery keep the exact value net they were measured with.

Run:  python build_counterfactual_values.py [--specs 60] [--per-spec 2]
"""
import argparse
import json
import time
from pathlib import Path

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.publication import ROOT
from run_puct_ablation import compatible_classes, realise_class

CF_FILE = ROOT / "datasets/simulation_memory/az_counterfactual_targets.jsonl"
OUT = ROOT / "artifacts/publication/counterfactual_values"
ALL_CLASSES = ["2s_none", "2s_miller", "2s_rc", "3s_miller", "3s_rc"]


def incompatible_classes(spec, limit: int) -> list:
    """Classes from a DIFFERENT stage tier than the spec requires."""
    ok = set(compatible_classes(spec))
    tier = next(iter(ok))[0]
    return [c for c in ALL_CLASSES if c[0] != tier][:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=int, default=60,
                    help="train-split specs to cover (0 = all)")
    ap.add_argument("--per-spec", type=int, default=2,
                    help="incompatible classes measured per spec")
    ap.add_argument("--budget", type=int, default=12)
    args = ap.parse_args()
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mb_sac.spec_sizing import sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import device_graph_hash

    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    train = [r for r in corpus["records"] if r["split"] == "train"]
    if args.specs:
        train = train[:args.specs]
    exe = discover_ngspice()
    out = (OUT / "runs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if CF_FILE.is_file():
        done = {(json.loads(x)["context_key"],
                 json.loads(x)["topology_family"])
                for x in CF_FILE.read_text(encoding="utf-8").splitlines()
                if x.strip()}
    print(f"train specs: {len(train)}, already measured: {len(done)}")
    written = 0
    for i, r in enumerate(train):
        spec = ig.parse_spec(r["prompt"])
        if not spec:
            continue
        key = f"{i:03d}_{r['context_id']}"
        for cls in incompatible_classes(spec, args.per_spec):
            if (key, cls) in done:
                continue
            t0 = time.time()
            costs = new_costs()
            g = realise_class(cls)
            sz = sac_size(f"cf_{i:03d}_{cls}", g, spec, exe, out, costs,
                          budget=args.budget, seed=17, persist=False)
            o = sz["outcome"]
            rec = {"graph_hash": device_graph_hash(g),
                   "topology_family": cls, "spec": spec,
                   "context_id": r["context_id"], "context_key": key,
                   **sz["value"],
                   "value_source": "REAL_POST_SIZING_SPICE",
                   "counterfactual": True,
                   "spec_compatible": False,
                   "sizing_budget": sz["budget"],
                   "spice_calls": sz["spice_calls"],
                   "uncertainty": 0.0 if (sz["best"]["stability"] and
                                          str(sz["best"]["stability"])
                                          .startswith("verified")) else 1.0,
                   "campaign_id": "counterfactual_probe", "generation": 0}
            CF_FILE.parent.mkdir(parents=True, exist_ok=True)
            with CF_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            written += 1
            print(f"[{i+1}/{len(train)}] {key} {cls} "
                  f"value={rec['value_target']} dist="
                  f"{o['normalized_distance_to_feasibility']} "
                  f"({round(time.time()-t0,1)}s)")
    rows = [json.loads(x) for x in
            CF_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_fam = {}
    for x in rows:
        by_fam.setdefault(x["topology_family"], []).append(x["value_target"])
    summary = {"written_this_run": written, "total": len(rows),
               "mean_value_by_family": {
                   k: round(sum(v) / len(v), 4) for k, v in sorted(
                       by_fam.items())},
               "file": str(CF_FILE)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
