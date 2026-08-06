"""Publication-v2 smoke battery (run BEFORE the full rerun).

5 tasks (T006 T076 T021 T100 T042), equal 16-call budgets:
  C2 grid | C3 TPE | C6 old spec-SAC | C9 repaired hybrid

Also: v2 freeze manifest, A/B/C benchmark splits, RAG snapshot checks and
memory-usage logging verification. Gate criteria printed at the end —
proceed to the full v2 rerun only on PASS.

Run:  python run_smoke_v2.py
"""
import json
import time
from pathlib import Path

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.llm_dpo import rag
from agentic_raptor.publication import FREEZE, PUB, ROOT

V2 = ROOT / "artifacts/publication_v2"
SMOKE_TASKS = ["T006", "T076", "T021", "T100", "T042"]


def v2_freeze():
    from agentic_raptor.publication.freeze import build_manifest
    from agentic_raptor.mb_sac.hybrid_sizing import (ACTION_SPACE_VERSION,
                                                     SCHEMA as HS)
    man = build_manifest(with_tests=False)
    man["v2"] = {"action_space_version": ACTION_SPACE_VERSION,
                 "hybrid_schema": HS,
                 "reward": "feasibility_first_v2",
                 "rag": "structured_filtered_v2",
                 "sft_gate": "parent-vs-SFT-vs-DPO strongest accepted",
                 "negative_controls_preserved": [
                     "poisoned DPO queue (campaign archives)",
                     "dpo_no_gate / dpo_no_integrity arms",
                     "stale value net backup (.pre_postsizing_*)",
                     "legacy action space + reward (C8/RW0 arms)",
                     "brittle RAG path (L3 record)"]}
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / "freeze_manifest.json").write_text(json.dumps(man, indent=1),
                                             encoding="utf-8")
    return man


def v2_splits():
    audit = json.loads((FREEZE / "feasibility_audit.json").read_text())
    bench = json.loads((FREEZE / "benchmark.json").read_text())
    cls_of = {}
    for r in audit["results"]:
        for t in r["tasks"]:
            cls_of[t] = r["classification"]
    splits = {"A_feasible": [], "B_optimizer_challenge": [], "C_stress": []}
    for t in bench["tasks"]:
        c = cls_of.get(t["task_id"], "unknown")
        if c == "known_feasible":
            splits["A_feasible"].append(t["task_id"])
        elif c in ("optimizer_limited",):
            splits["B_optimizer_challenge"].append(t["task_id"])
        else:
            splits["C_stress"].append(t["task_id"])
    doc = {**{k: sorted(v) for k, v in splits.items()},
           "counts": {k: len(v) for k, v in splits.items()},
           "policy": "A: headline electrical results; B: optimizer research; "
                     "C: reported separately, never as ordinary failures",
           "source": "feasibility_audit.json",
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (V2 / "benchmark_splits.json").write_text(json.dumps(doc, indent=1),
                                              encoding="utf-8")
    return doc


def rag_snapshots():
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    fails = []
    for r in [x for x in corpus["records"] if x["split"] == "train"][:20]:
        spec = ig.parse_spec(r["prompt"])
        p2 = rag.augment_prompt(r["prompt"], spec, r["stages"])
        chk = rag.snapshot_check(p2)
        if not all(chk.values()):
            fails.append({"context": r["context_id"], "checks": chk})
    held = [x for x in corpus["records"] if x["split"] == "heldout"]
    leak = [e for e in (rag.retrieve(ig.parse_spec(held[0]["prompt"]),
                                     held[0]["stages"]) or [])
            if e.get("context_id") in {h["context_id"] for h in held}]
    return {"prompt_snapshot_failures": fails,
            "evidence_from_validation_contexts": leak}


def smoke_sizing():
    from agentic_raptor.publication.sizing_baselines import (_mk_graph,
                                                             _sample_loop,
                                                             _grid, _tpe,
                                                             _summ)
    from agentic_raptor.mb_sac.spec_sizing import sac_size
    from agentic_raptor.mb_sac.hybrid_sizing import hybrid_size
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    bench = json.loads((FREEZE / "benchmark.json").read_text())
    tasks = [t for t in bench["tasks"] if t["task_id"] in SMOKE_TASKS]
    exe = discover_ngspice()
    out = (V2 / "smoke_runs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tasks:
        tid, g = _mk_graph(t)
        spec = dict(t["spec"])
        comp = t["target_class"].split("_", 1)[1]
        for m in ("C2", "C3", "C6_old", "C9_hybrid"):
            t0 = time.time()
            costs = new_costs()
            if m == "C2":
                res = _sample_loop(tid, g, spec, exe, out, costs, 16, 11,
                                   _grid)
            elif m == "C3":
                res = _sample_loop(tid, g, spec, exe, out, costs, 16, 11,
                                   _tpe)
            elif m == "C6_old":
                sz = sac_size(tid, g, spec, exe, out, costs, budget=16,
                              seed=11, persist=False)
                res = sz["results"]
            else:
                sz = hybrid_size(tid, g, spec, exe, out, costs, budget=16,
                                 seed=11, family=t["target_class"],
                                 comp=comp)
                res = sz["results"]
                rows_mem = sz["memory"]
            rows.append({"task_id": t["task_id"],
                         "class": t["target_class"], "method": m,
                         **_summ(res, spec, t0)})
            print(t["task_id"], m, "exact", rows[-1]["exact_pass"],
                  "dist", rows[-1]["distance"])
    return rows


def main():
    print("=== v2 freeze + splits ===")
    v2_freeze()
    sp = v2_splits()
    print("splits:", sp["counts"])
    print("=== RAG snapshot checks ===")
    rs = rag_snapshots()
    print("prompt failures:", len(rs["prompt_snapshot_failures"]),
          "| validation-context evidence leaks:",
          len(rs["evidence_from_validation_contexts"]))
    print("=== smoke sizing (5 tasks x 4 methods x 16 calls) ===")
    rows = smoke_sizing()
    agg = {}
    for r in rows:
        agg.setdefault(r["method"], []).append(r)
    summary = {m: {"exact": sum(x["exact_pass"] for x in rs_),
                   "mean_dist": round(sum((1.0 if x["distance"] is None else x["distance"])
                                          for x in rs_) / len(rs_), 4)}
               for m, rs_ in sorted(agg.items())}
    old, new = summary.get("C6_old", {}), summary.get("C9_hybrid", {})
    gate = {"rag_snapshots_clean": not rs["prompt_snapshot_failures"]
            and not rs["evidence_from_validation_contexts"],
            "hybrid_not_worse_than_old_sac":
                (new.get("exact", 0), -new.get("mean_dist", 9))
                >= (old.get("exact", 0), -old.get("mean_dist", 9)),
            "aggregate_improvement":
                new.get("exact", 0) > old.get("exact", 0)
                or new.get("mean_dist", 9) < old.get("mean_dist", 9)}
    doc = {"summary": summary, "rows": rows, "rag_checks": rs,
           "gate": gate, "PASS": all(gate.values()),
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (V2 / "SMOKE.json").write_text(json.dumps(doc, indent=1, default=str),
                                   encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("gate:", json.dumps(gate, indent=1))
    print("SMOKE PASS:" if doc["PASS"] else "SMOKE FAIL:", doc["PASS"])


if __name__ == "__main__":
    main()
