"""TRACK-A PILOT (chain validation, small budgets -- approved 2026-08-28).

Purpose: validate the measurement chain (baseline -> adapter ->
ExternalTopologyResult -> metrics), NOT to produce comparable numbers yet.
The three parts intentionally run on different spec sets at pilot stage;
spec alignment across generators is the full-phase design decision.
Output: artifacts/external_baselines/pilot_track_a.jsonl (+ summary print).

Parts:
  panda    5 frozen validation specs -> PANDA native template+validator path
  ag       AG topology generation EXTRACTED from the frozen Tier-3 campaign
           traces on the same validation specs (A0 arm, seeds 0-2, first 5
           spec indices) -- zero new compute, provenance recorded
  acp      AnalogCoder-Pro native problems 1-5 x 5 samples (LLM budget
           ~25 calls, approved)
  summary  aggregate whatever rows exist

Usage: python -m src.evaluation.external_baselines.run_track_a_pilot --part panda
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "external_baselines" / "pilot_track_a.jsonl"
TRACES = ROOT / "artifacts" / "publication_v2" / "raptor_v2_runs"


def part_panda() -> None:
    from .panda_adapter import run_spec
    from .schema import write_results
    specs = json.loads((ROOT / "data/external_baseline_eval/specs_validation.json"
                        ).read_text(encoding="utf-8"))["specs"][:5]
    rows = [run_spec(s, seed=0) for s in specs]
    write_results(rows, OUT)
    for r in rows:
        print(f"panda {r.spec_id}: syntax={r.valid_syntax} graph={r.valid_graph} "
              f"| {r.notes[:70]}")


def part_ag() -> None:
    from .schema import ExternalTopologyResult, write_results
    rows = []
    for seed in (0, 1, 2):
        for idx in range(5):
            matches = sorted(TRACES.glob(
                f"ABLv3HELDOUT29_A0_s{seed}_heldout_{idx:03d}_*.json"))
            if not matches:
                continue
            tr = json.loads(matches[-1].read_text(encoding="utf-8"))
            prop = tr.get("stage3_propose") or {}
            hashes = prop.get("proposal_hashes") or []
            for k, h in enumerate(hashes):
                rows.append(ExternalTopologyResult(
                    baseline="agentic_raptor_generator",
                    spec_id=f"heldout_{idx}", seed=seed,
                    raw_output_path=str(matches[-1]),
                    valid_syntax=True, valid_graph=True,   # validator-passed by construction
                    topology_hash=h,
                    llm_calls=(prop.get("attempts") if k == 0 else 0),
                    notes=("EXTRACTED from frozen Tier-3 campaign trace "
                           f"(A0 arm; distinct={prop.get('distinct')}, "
                           f"families={prop.get('distinct_family_count')}); "
                           "no new compute")))
    write_results(rows, OUT)
    print(f"ag: extracted {len(rows)} proposal rows from frozen traces")


def part_acp() -> None:
    from .analogcoderpro_adapter import run_task
    from .schema import write_results
    for task in (1, 2, 3, 4, 5):
        rows = run_task(task, n_samples=5)
        write_results(rows, OUT)
        ok = sum(1 for r in rows if r.simulatable)
        print(f"acp task {task}: {len(rows)} samples, functional {ok}", flush=True)


def part_summary() -> None:
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["baseline"]].append(r)
    print(f"\n=== TRACK-A PILOT SUMMARY ({len(rows)} rows) ===")
    print("NOTE: pilot parts run on DIFFERENT spec sets -- chain validation "
          "only, NOT cross-comparable.")
    for b, rs in sorted(by.items()):
        n = len(rs)
        vs = sum(1 for r in rs if r["valid_syntax"])
        vg = sum(1 for r in rs if r["valid_graph"])
        fn = sum(1 for r in rs if r["simulatable"])
        uniq = len({r["topology_hash"] for r in rs if r["topology_hash"]})
        lc = sum(r["llm_calls"] or 0 for r in rs)
        print(f"{b:26s} n={n:3d} valid_syntax={vs:3d} valid_graph={vg:3d} "
              f"functional={fn:3d} unique_topologies={uniq:3d} llm_calls={lc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["panda", "ag", "acp", "summary"])
    a = ap.parse_args()
    {"panda": part_panda, "ag": part_ag, "acp": part_acp,
     "summary": part_summary}[a.part]()
