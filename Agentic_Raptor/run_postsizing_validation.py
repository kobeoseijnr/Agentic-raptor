"""Task 9: small post-sizing validation BEFORE another full campaign.

For a small balanced subset of train specs:
  proposal structure -> realisation -> SAC sizing under a fixed budget,
  once with the LEGACY raw-metric reward and once with the CORRECTED
  target-saturating reward (same seed, same budget), all on real ngspice.

Pass criterion: the corrected reward must not sacrifice gain for excess PM —
its best designs must dominate legacy on gain-margin while keeping PM >= the
target (or get closer to joint feasibility).

Structures come from the corpus records' target structures by default (no
GPU needed). Pass --with-model to have the generation-2 model propose them
instead (GPU; requires AGENTIC_RAPTOR_ADAPTER pointing at the checkpoint).

Run:  python run_postsizing_validation.py [--with-model] [--budget 16]
"""
import argparse
import json
import time
from pathlib import Path

from agentic_raptor.electrical import discover_ngspice
from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.mapping import map_family
from agentic_raptor.mb_sac.spec_sizing import (legacy_reward, sac_size,
                                               spec_reward)
from agentic_raptor.topology_rl.stage3e2 import new_costs

O4 = Path("artifacts/stage3e4").resolve()
OUT = Path("artifacts/postsizing_validation").resolve()
OUT.mkdir(parents=True, exist_ok=True)


def pick_specs(corpus, n=4):
    """Small balanced subset: different structure classes, loads, gains."""
    by_class = {}
    for r in corpus["records"]:
        if r["split"] == "train":
            by_class.setdefault((r["stages"], r["comp"]), r)
    return list(by_class.values())[:n]


def realise(rec, args):
    """Structure to size: corpus target structure (default) or a fresh
    generation-2 model proposal (--with-model)."""
    from agentic_raptor.llm_dpo.stage3e4 import variant_text
    obj = json.loads(variant_text(rec["stages"], rec["comp"], False, False))
    source = "corpus_target_structure"
    if args.with_model:
        import os
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from agentic_raptor.llm_dpo import MODEL_ID
        from agentic_raptor.llm_dpo.stage3e4 import generate
        adapter = os.environ["AGENTIC_RAPTOR_ADAPTER"]
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        tok.pad_token = tok.eos_token
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, device_map="auto"), adapter)
        c = generate(model, tok, rec["prompt"], sample_seed=0)
        del model
        torch.cuda.empty_cache()
        if c["valid"]:
            obj, source = c["obj"], f"gen2_model({adapter})"

    class _S:
        topology_id = f"psval_{rec['context_id'][-8:]}"
    g, _ = map_family(_S(), {
        "topology_id": _S.topology_id, "gain_stages": len(obj["stages"]),
        "functional_blocks": ["C"] if obj.get("compensation") else [],
        "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
        "graph_hash": None})
    return _S.topology_id, g, obj, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true")
    ap.add_argument("--budget", type=int, default=16)
    args = ap.parse_args()
    corpus = json.loads((O4 / "corpus.json").read_text())
    exe = discover_ngspice()
    report = {"budget_per_run": args.budget, "specs": [],
              "timestamp": time.time()}
    for rec in pick_specs(corpus):
        spec = ig.parse_spec(rec["prompt"])
        tid, g, obj, source = realise(rec, args)
        row = {"context_id": rec["context_id"],
               "class": f"{rec['stages']}s_{rec['comp']}",
               "structure_source": source,
               "spec": {k: spec[k] for k in ("gain_target_db",
                                             "phase_margin_target_deg",
                                             "load_capacitance_pf",
                                             "ugbw_target_hz")}}
        for label, fn in (("legacy_reward", legacy_reward),
                          ("corrected_reward", spec_reward)):
            # persist=False: the A/B must compare rewards from identical cold
            # starts — warm-started memory would contaminate the comparison
            sz = sac_size(tid, g, spec, exe, OUT, new_costs(),
                          budget=args.budget, seed=7, reward_fn=fn,
                          persist=False)
            b, o = sz["best"], sz["outcome"]
            row[label] = {"best_gain_db": b["gain_db"],
                          "best_pm_deg": b["pm_deg"],
                          "stable": b["stable"],
                          "gain_margin_db": o["margin_vector"]["gain_margin_db"],
                          "pm_margin_deg": o["margin_vector"]["pm_margin_deg"],
                          "outcome_tier": o["outcome_tier"],
                          "exact_spec_pass": o["exact_spec_pass"],
                          "hard_constraints_passed":
                              o["hard_constraints_passed"],
                          "distance": o["normalized_distance_to_feasibility"],
                          "spice_calls": sz["spice_calls"],
                          "best_knobs": b["knobs"]}
        # nominal (no sizing) for reference
        from agentic_raptor.mb_sac.spec_sizing import measure, postsizing_outcome
        nom = measure(tid, g, exe, OUT, "nominal", new_costs())
        no = postsizing_outcome(nom, spec)
        row["nominal"] = {"gain_db": nom["gain_db"], "pm_deg": nom["pm_deg"],
                          "stable": nom["stable"],
                          "outcome_tier": no["outcome_tier"],
                          "distance": no["normalized_distance_to_feasibility"]}
        report["specs"].append(row)
        print(json.dumps(row, indent=1, default=str))
    # Task 9 pass criteria (v2 — maps directly to the requirement text):
    #  1. NEVER excess-PM at gain's expense: no corrected best may sit more
    #     than the cushion above the PM target while gain is below target.
    #  2. Nominal anchor: corrected best is never worse than no sizing.
    #  3. Aggregate: corrected mean distance-to-feasibility <= legacy's
    #     (single-run per-spec comparisons sit inside search noise).
    from agentic_raptor.mb_sac.spec_sizing import PM_CUSHION_DEG
    no_pm_chasing, anchored, dists_c, dists_l = [], [], [], []
    for row in report["specs"]:
        c = row["corrected_reward"]
        no_pm_chasing.append(not (
            (c["pm_margin_deg"] or 0) > PM_CUSHION_DEG
            and (c["gain_margin_db"] or 0) < 0))
        anchored.append(c["distance"] is None
                        or row["nominal"]["distance"] is None
                        or c["distance"] <= row["nominal"]["distance"] + 1e-9)
        dists_c.append(c["distance"])
        dists_l.append(row["legacy_reward"]["distance"])
    mean = lambda xs: sum(x for x in xs if x is not None) / max(
        1, sum(1 for x in xs if x is not None))
    report["criteria_version"] = 2
    report["criteria"] = {
        "no_pm_excess_at_gain_expense": no_pm_chasing,
        "never_worse_than_nominal": anchored,
        "mean_distance_corrected": round(mean(dists_c), 4),
        "mean_distance_legacy": round(mean(dists_l), 4)}
    report["corrected_reward_acceptable"] = (
        all(no_pm_chasing) and all(anchored)
        and mean(dists_c) <= mean(dists_l) + 1e-9)
    report["per_spec_verdicts"] = [a and b for a, b in
                                   zip(no_pm_chasing, anchored)]
    (OUT / "REPORT.json").write_text(json.dumps(report, indent=1,
                                                default=str),
                                     encoding="utf-8")
    print("\ncorrected_reward_acceptable:", report["corrected_reward_acceptable"])
    if report["corrected_reward_acceptable"]:
        print("proceed with:  python run_self_improvement.py 3")
    else:
        print("DO NOT run the full campaign — inspect", OUT / "REPORT.json")


if __name__ == "__main__":
    main()
