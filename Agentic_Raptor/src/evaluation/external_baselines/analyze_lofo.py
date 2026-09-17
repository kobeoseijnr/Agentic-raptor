"""LOFO post-campaign analysis: does the proposer generate a family it never
saw? Run AFTER the LOFO campaign completes. READ-ONLY over the LOFO results +
traces and the frozen R2 baseline. Writes LOFO_REPORT.md + lofo_metrics.json.

Usage:
  python -m src.evaluation.external_baselines.analyze_lofo --family 3s_rc \
      --results artifacts/publication_v3/ablation_v3/results_<LOFO_run>.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R2 = ROOT / "artifacts/publication_v3/ablation_v3/results_20260830_201205.jsonl"


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def fam_of_hash_map(trace_glob: str) -> dict:
    """Map canonical_graph_hash -> family from stage5 rankings across traces."""
    h2f = {}
    for fp in glob.glob(trace_glob):
        tr = json.loads(Path(fp).read_text(encoding="utf-8"))
        for e in (tr.get("stage5_alphazero") or {}).get("ranking") or []:
            if e.get("canonical_graph_hash") and e.get("canonical_family"):
                h2f[e["canonical_graph_hash"]] = e["canonical_family"]
    return h2f


def main(family: str, results: str, tag: str):
    out = ROOT / "artifacts/publication_v3/lofo" / family
    out.mkdir(parents=True, exist_ok=True)
    res = jl(Path(results))
    lofo = [r for r in res if r["ablation_id"] == "AG_FULL"]
    # baseline R2 (family present)
    r2 = [r for r in jl(R2)]
    r2_win_fam = {(r["pipeline_seed"], r["spec_index"]): r["selected_family"]
                  for r in r2 if (r.get("nominal") or {}).get("complete_pass")}
    focus = sorted({i for (s, i), f in r2_win_fam.items() if f == family})

    # proposal-pool family emergence (from traces of the LOFO run)
    tg = str(ROOT / "artifacts/publication_v2/raptor_v2_runs"
             / f"*{tag}*AG_FULL*.json")
    h2f = fam_of_hash_map(tg)
    emergence = {"specs_with_family_in_pool": set(), "total_specs": set(),
                 "runs_with_family_in_pool": 0, "total_runs": 0,
                 "family_proposals": 0, "total_proposals": 0}
    for fp in glob.glob(tg):
        tr = json.loads(Path(fp).read_text(encoding="utf-8"))
        s3 = tr.get("stage3_propose") or {}
        hashes = s3.get("proposal_hashes") or []
        if not hashes:
            continue
        si = tr.get("stage1_spec", {}).get("spec_index")
        fams = [h2f.get(h) for h in hashes]
        emergence["total_runs"] += 1
        emergence["total_specs"].add(si)
        emergence["total_proposals"] += len(hashes)
        n_fam = sum(1 for f in fams if f == family)
        emergence["family_proposals"] += n_fam
        if n_fam:
            emergence["runs_with_family_in_pool"] += 1
            emergence["specs_with_family_in_pool"].add(si)

    # win/pass metrics
    lofo_pass = {(r["pipeline_seed"], r["spec_index"]):
                 bool((r.get("nominal") or {}).get("complete_pass"))
                 for r in lofo}
    lofo_winfam = {(r["pipeline_seed"], r["spec_index"]): r.get("selected_family")
                   for r in lofo if (r.get("nominal") or {}).get("complete_pass")}
    n = len(lofo_pass)
    n_pass = sum(lofo_pass.values())
    family_wins = sum(1 for f in lofo_winfam.values() if f == family)

    def passrate(specs):
        ks = [(s, i) for (s, i) in lofo_pass if i in specs]
        return (sum(lofo_pass[k] for k in ks), len(ks))

    focus_pass = passrate(set(focus))
    rest = set(i for (_, i) in lofo_pass) - set(focus)
    rest_pass = passrate(rest)

    r2_focus = [(r["pipeline_seed"], r["spec_index"]) for r in r2
                if r["spec_index"] in focus]
    r2_focus_pass = sum(1 for r in r2 if r["spec_index"] in focus
                        and (r.get("nominal") or {}).get("complete_pass"))

    emerged = len(emergence["specs_with_family_in_pool"])
    metrics = {
        "family": family, "n_runs": n, "results_file": results,
        "focus_specs_family_dependent": focus,
        "A_family_emergence": {
            "specs_with_family_in_pool": sorted(emergence["specs_with_family_in_pool"]),
            "n_specs_emerged": emerged,
            "n_specs_total": len(emergence["total_specs"]),
            "runs_with_family_in_pool": emergence["runs_with_family_in_pool"],
            "total_runs": emergence["total_runs"],
            "family_proposals": emergence["family_proposals"],
            "total_proposals": emergence["total_proposals"],
            "emergence_rate_runs": round(emergence["runs_with_family_in_pool"]
                                         / max(1, emergence["total_runs"]), 3),
            "interpretation": "STRONG generalization if > 0: the proposer "
                              "generated a family absent from its SFT corpus, "
                              "RAG memory, and with the critic rule disabled."},
        "B_family_win_rate": {
            "wins_in_held_out_family": family_wins,
            "of_total_wins": n_pass,
            "note": "if the family emerges AND wins, the strongest evidence"},
        "C_finalpass": {
            "overall_lofo": f"{n_pass}/{n} = {n_pass/n:.3f}",
            "focus_specs_lofo": f"{focus_pass[0]}/{focus_pass[1]}",
            "focus_specs_r2_baseline": f"{r2_focus_pass}/{len(r2_focus)}",
            "rest_specs_lofo": f"{rest_pass[0]}/{rest_pass[1]}",
            "note": "focus = specs whose full-corpus winner was the held-out "
                    "family; the pass gap there is the cost of removing it."},
        "D_alternative_solutions": dict(Counter(
            lofo_winfam[k] for k in lofo_winfam if k[1] in set(focus))),
    }
    (out / "lofo_metrics.json").write_text(json.dumps(metrics, indent=1),
                                           encoding="utf-8")

    verdict = ("STRONG: family emerged from a proposer that never saw it"
               if emerged > 0 else
               "BOUNDARY: family did not emerge -- SFT knowledge, not "
               "compositional generation, carries this family")
    report = f"""# LOFO Generalization Result -- held-out family: {family}

The family **{family}** was removed from the SFT corpus (208 records), the RAG
memory (76 records), and the critic deep-comp rule was disabled
(AGR_LOFO_DISABLE_DEEPCOMP=1); the bandit selector and DPO ranker were frozen
(downstream -- cannot introduce an unproposed family); the realization library
was unchanged (scope: this tests learned PROPOSER discovery, not realization).
The adapter was retrained on the {917-208}-record filtered corpus and the
held-out campaign re-run on the 29 heldout specs x 3 seeds.

## Verdict: {verdict}

## A. Family emergence (the key metric)
- Proposal pools containing {family}: {emergence['runs_with_family_in_pool']}
  /{emergence['total_runs']} runs, on {emerged}/{len(emergence['total_specs'])}
  distinct specs.
- {family} proposals: {emergence['family_proposals']} of
  {emergence['total_proposals']} total.
- Emergence rate (runs): {metrics['A_family_emergence']['emergence_rate_runs']}.

## B. Held-out-family win rate
- Winners in {family}: {family_wins} of {n_pass} passing runs.

## C. FinalPass
- Overall (LOFO): {metrics['C_finalpass']['overall_lofo']}
- Family-dependent focus specs {focus}: LOFO
  {metrics['C_finalpass']['focus_specs_lofo']} vs full-corpus R2 baseline
  {metrics['C_finalpass']['focus_specs_r2_baseline']}.
- Remaining specs (graceful degradation): {metrics['C_finalpass']['rest_specs_lofo']}.

## D. What the system used instead on the focus specs
{json.dumps(metrics['D_alternative_solutions'], indent=1)}

## Claim guidance
- If emergence > 0 and the family wins: "RAPTOR generalizes to a
  held-out topology FAMILY: with {family} removed from all learned components,
  the proposer still discovers and successfully applies it on
  {focus_pass[0]}/{focus_pass[1]} of the family-dependent held-out specs."
- If emergence = 0: report the honest boundary -- the family's coverage came
  from SFT/retrieval memory, not compositional generation; the system degrades
  to the next-best family it CAN generate ({metrics['D_alternative_solutions']}),
  which is itself a useful graceful-degradation result.
- Either way: this is a genuine family-level test that the instance-level
  heldout evaluation cannot provide.
"""
    (out / "LOFO_REPORT.md").write_text(report, encoding="utf-8")
    print(f"verdict: {verdict}")
    print(f"emergence: {emergence['runs_with_family_in_pool']}/"
          f"{emergence['total_runs']} runs, {emerged} specs")
    print(f"family wins: {family_wins}/{n_pass}")
    print(f"focus FinalPass: LOFO {focus_pass[0]}/{focus_pass[1]} vs "
          f"R2 {r2_focus_pass}/{len(r2_focus)}")
    print(f"outputs: {(out / 'LOFO_REPORT.md')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="3s_rc")
    ap.add_argument("--results", required=True,
                    help="the LOFO campaign results_<...>.jsonl")
    ap.add_argument("--tag", default="LOFO_3s_rc",
                    help="trace-file tag substring (default LOFO_3s_rc)")
    a = ap.parse_args()
    main(a.family, a.results, a.tag)
