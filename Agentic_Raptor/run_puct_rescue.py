"""PUCT rescue test: can the repaired search correct a WEAK proposer's
mistakes? Complements run_puct_ablation.py (L8 proposer, "does no harm")
without touching its results — everything here writes to
artifacts/publication/puct_rescue/.

Proposer = L2 (initial SFT, first training only). Measured on the same
held-out specs: ~31% stage accuracy, so most proposals pick the wrong
design class — real mistakes for the search to catch. L2 only emits valid
JSON when the RAG/KNOWN evidence lines are stripped from the prompt
(qwen ablation L2 vs L3), so proposals are generated on stripped prompts.

Arms (same semantics as P0/P8 in run_puct_ablation.py):
  R0  L2 proposal executed directly (no search)   — shows the damage
  R8  repaired PUCT on the L2 proposal            — shows the rescue

Run:  python run_puct_rescue.py [--tasks 29] [--budget 12]
      (phase 1 needs GPU once; re-runs reuse cached proposals)
"""
import argparse
import json
import re
import time

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.publication import PUB, ROOT
from run_puct_ablation import (compatible_classes, realise_class,
                               run_puct_fixed, _realise)

OUT = PUB / "puct_rescue"
_RAG_RE = re.compile(r"^### (RAG|KNOWN) [^\n]*\n", re.M)


def get_weak_proposals(n_tasks: int) -> dict:
    """Phase 1 (GPU, cached): L2 proposals on RAG-stripped held-out prompts."""
    cache = OUT / "proposals.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    import torch
    from run_qwen_ablation import _load, resolve_arms
    from agentic_raptor.llm_dpo.stage3e4 import generate
    adapter = resolve_arms()["L2"]["adapter"]
    assert adapter, "initial SFT checkpoint missing"
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    held = [r for r in corpus["records"] if r["split"] == "heldout"][:n_tasks]
    tok, model = _load(adapter)
    props, no_valid = {}, []
    for i, r in enumerate(held):
        # context_id is NOT unique per record (two distinct specs can share
        # one label, differing only e.g. in ugbw_target_hz) -- keying by it
        # alone silently collapses distinct held-out tasks into one another
        key = f"{i:03d}_{r['context_id']}"
        prompt = _RAG_RE.sub("", r["prompt"])
        c = None
        for s in range(6):        # L2 needs ~3 attempts on average
            cand = generate(model, tok, prompt, sample_seed=s)
            if cand["valid"]:
                c = cand
                break
        if not c:
            # KEEP these. A spec the proposer cannot answer at all is the
            # purest test of whether search stands on its own -- dropping
            # them silently removed 9 of 29 tasks, 4 of them the hard ones.
            no_valid.append(key)
            props[key] = {"prompt": prompt, "obj": None, "graph_hash": None,
                          "context_id": r["context_id"], "llm_failed": True,
                          "spec": ig.parse_spec(r["prompt"])}
            continue
        props[key] = {
            "prompt": prompt, "obj": c["obj"], "graph_hash": c["graph_hash"],
            "context_id": r["context_id"], "llm_failed": False,
            "spec": ig.parse_spec(r["prompt"])}
    del model
    torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(props, indent=1), encoding="utf-8")
    (OUT / "no_valid_proposal.json").write_text(
        json.dumps(no_valid, indent=1), encoding="utf-8")
    return props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=29)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--arms", default="R0,R8")
    args = ap.parse_args()
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mb_sac.spec_sizing import sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import device_graph_hash
    props = get_weak_proposals(args.tasks)
    print(f"weak proposals cached: {len(props)}")
    exe = discover_ngspice()
    out = OUT.resolve() / "runs"
    out.mkdir(parents=True, exist_ok=True)
    rows_f = OUT / "puct_rescue_rows.jsonl"
    done = set()
    if rows_f.is_file():
        done = {(json.loads(x)["context_id"], json.loads(x)["arm"])
                for x in rows_f.read_text().splitlines() if x.strip()}
    for task_key, p in props.items():
        spec = p["spec"]
        ctx_id = p.get("context_id", task_key)
        ok_classes = compatible_classes(spec)
        llm_failed = p.get("llm_failed") or p.get("obj") is None
        # with no proposal there is no proposal class; the search starts from
        # the simplest compatible structure and may move anywhere in the pool
        default_cls = f"{ok_classes[0][0]}s_none"
        prop_cls = (default_cls if llm_failed else
                    f"{len(p['obj']['stages'])}s_"
                    + ig.compensation_class(p["obj"]))
        prop_ok = (None if llm_failed else prop_cls in ok_classes)
        # RX arms cover the no-proposal specs: RX0 = fixed default structure
        # (no LLM, no search), RX8 = search picks unaided
        arms = (["RX0", "RX8"] if llm_failed
                else [a for a in args.arms.split(",") if a in ("R0", "R8")])
        for arm in arms:
            if (task_key, arm) in done:
                continue
            t0 = time.time()
            costs = new_costs()
            g = realise_class(prop_cls) if llm_failed else _realise(p["obj"])
            action = "DEFAULT_STRUCTURE" if llm_failed else "EXECUTE_PROPOSAL"
            sel, visits = None, {}
            if arm in ("R8", "RX8"):
                sel, visits, pc = run_puct_fixed(
                    p.get("obj"), spec, ctx_id,
                    root_cls=prop_cls if llm_failed else None)
                if sel and sel.startswith("a_sel_") and sel[6:] != pc:
                    g = realise_class(sel[6:])
                    action = f"SWITCH_CLASS:{sel[6:]}"
                else:
                    action = "KEEP"
            sz = sac_size(f"pr_{task_key[:3]}_{ctx_id[-6:]}", g, spec, exe,
                          out, costs, budget=args.budget, seed=17,
                          persist=False)
            o = sz["outcome"]
            switched_to = (action.split(":", 1)[1]
                           if action.startswith("SWITCH_CLASS") else None)
            row = {"context_id": task_key, "spec_label": ctx_id,
                   "arm": arm, "action": action,
                   "llm_failed": bool(llm_failed),
                   "puct_selected": sel, "visits": visits,
                   "proposal_class": prop_cls,
                   "compatible_classes": ok_classes,
                   "proposal_correct": prop_ok,
                   "rescued": (prop_ok is False
                               and switched_to in ok_classes),
                   "bad_overturn": (prop_ok is True
                                    and switched_to is not None
                                    and switched_to not in ok_classes),
                   "executed_graph_hash": device_graph_hash(g),
                   "search_spice": costs["real_spice_calls"],
                   "sizing_spice": sz["spice_calls"],
                   "exact_pass": o["exact_spec_pass"],
                   "constraints_passed": o["hard_constraints_passed"],
                   "distance": o["normalized_distance_to_feasibility"],
                   "best_gain_db": sz["best"]["gain_db"],
                   "best_pm_deg": sz["best"]["pm_deg"],
                   "runtime_s": round(time.time() - t0, 1)}
            with rows_f.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(task_key, arm, action,
                  "prop_ok", prop_ok, "exact", o["exact_spec_pass"],
                  "dist", o["normalized_distance_to_feasibility"])
    rows = [json.loads(x) for x in rows_f.read_text().splitlines()
            if x.strip()]
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    def _mean_dist(rs):
        return (round(sum(r["distance"] or 1 for r in rs) / len(rs), 4)
                if rs else None)
    summary = {arm: {"tasks": len(rs),
                     "exact_pass": sum(r["exact_pass"] for r in rs),
                     "mean_distance": _mean_dist(rs),
                     "mean_distance_wrong_proposals": _mean_dist(
                         [r for r in rs if not r["proposal_correct"]]),
                     "mean_distance_right_proposals": _mean_dist(
                         [r for r in rs if r["proposal_correct"]])}
               for arm, rs in sorted(by_arm.items())}
    # the question the extension exists to answer: on specs the proposer
    # could not answer AT ALL, does search beat a fixed default structure?
    rx0 = {r["context_id"]: r for r in by_arm.get("RX0", [])}
    rx8 = by_arm.get("RX8", [])
    if rx8:
        b = w = t = 0
        for r in rx8:
            o = rx0.get(r["context_id"])
            if not o:
                continue
            d8, d0 = r["distance"], o["distance"]
            if d8 is None or d0 is None or abs(d8 - d0) < 1e-6:
                t += 1
            elif d8 < d0:
                b += 1
            else:
                w += 1
        summary["llm_produced_nothing"] = {
            "tasks": len(rx8),
            "search_closer": b, "search_further": w, "tied": t,
            "search_exact_pass": sum(r["exact_pass"] for r in rx8),
            "default_exact_pass": sum(o["exact_pass"]
                                      for o in rx0.values()),
            "search_changed_structure": sum(
                1 for r in rx8 if r["action"].startswith("SWITCH_CLASS")),
            "mean_distance_search": _mean_dist(rx8),
            "mean_distance_default": _mean_dist(list(rx0.values()))}
    r8 = by_arm.get("R8", [])
    wrong = [r for r in r8 if r["proposal_correct"] is False]
    right = [r for r in r8 if r["proposal_correct"] is True]
    summary["rescue"] = {
        "wrong_proposals": len(wrong),
        "rescued": sum(r["rescued"] for r in wrong),
        "rescue_rate": (round(sum(r["rescued"] for r in wrong)
                              / len(wrong), 3) if wrong else None),
        "right_proposals": len(right),
        "bad_overturns": sum(r["bad_overturn"] for r in right)}
    # The tier-based rescue counters above only see STAGE-COUNT changes, and
    # measured runs showed that is the wrong axis: every exact spec pass came
    # from switching compensation WITHIN the compatible tier (2s_miller ->
    # 2s_none), which scores as neither a rescue nor a bad overturn. The
    # paired comparison below asks the question that actually matters -- on
    # the same task, did searching beat executing the proposal?
    base = {r["context_id"]: r for r in by_arm.get("R0", [])}
    better = worse = tie = 0
    gained = lost = 0
    for r in r8:
        b = base.get(r["context_id"])
        if not b:
            continue
        gained += int(r["exact_pass"] and not b["exact_pass"])
        lost += int(b["exact_pass"] and not r["exact_pass"])
        d8, d0 = r["distance"], b["distance"]
        if d8 is None or d0 is None or abs(d8 - d0) < 1e-6:
            tie += 1
        elif d8 < d0:
            better += 1
        else:
            worse += 1
    summary["search_vs_no_search"] = {
        "paired_tasks": better + worse + tie,
        "search_closer": better, "search_further": worse, "tied": tie,
        "exact_passes_gained_by_searching": gained,
        "exact_passes_lost_by_searching": lost,
        "switched_within_compatible_tier": sum(
            1 for r in r8 if r["action"].startswith("SWITCH_CLASS")
            and r["action"].split(":", 1)[1] in r["compatible_classes"]),
        "net_exact_pass": (sum(r["exact_pass"] for r in r8)
                           - sum(b["exact_pass"] for b in base.values()))}
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
