"""TRACK-A FULL RUN, option A (approved 2026-08-28): all 29 frozen validation
specs, spec-aligned across generators. Output: artifacts/external_baselines/
track_a_full.jsonl. Test set (blindtest) stays sealed.

Parts:
  acp     AnalogCoder-Pro spec-aligned, 29 specs x --samples (LLM budget)
  panda   PANDA template path, 29 specs (offline; LLM path wired separately)
  ag      AG generation extracted from the frozen Tier-3 traces: A0 arm,
          seeds 0-2, ALL 29 heldout indices (zero new compute, provenance
          recorded per row)
  summary aggregate with iso-deduplicated diversity
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "external_baselines" / "track_a_full.jsonl"
TRACES = ROOT / "artifacts" / "publication_v2" / "raptor_v2_runs"


def _specs():
    return json.loads((ROOT / "data/external_baseline_eval/specs_validation.json"
                       ).read_text(encoding="utf-8"))["specs"]


def part_acp(samples: int) -> None:
    from .analogcoderpro_adapter import run_frozen_spec
    from .schema import write_results
    done = set()
    if OUT.exists():
        for l in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r["baseline"] == "analogcoderpro_specaligned":
                done.add(r["spec_id"])
    for i, spec in enumerate(_specs()):
        sid = spec["context_id"]
        if sid in done:
            print(f"[{i+1}/29] {sid}: already done, skipped", flush=True)
            continue
        rows = run_frozen_spec(spec, n_samples=samples)
        write_results(rows, OUT)
        ok = sum(1 for r in rows if r.simulatable)
        gg = sum(1 for r in rows if r.valid_graph)
        print(f"[{i+1}/29] {sid}: {len(rows)} samples, graph {gg}, "
              f"functional {ok}", flush=True)


def part_panda() -> None:
    from .panda_adapter import run_spec
    from .schema import write_results
    for i, spec in enumerate(_specs()):
        r = run_spec(spec, seed=0)
        write_results([r], OUT)
        print(f"[{i+1}/29] panda {r.spec_id}: graph={r.valid_graph}", flush=True)


def part_ag() -> None:
    from .schema import ExternalTopologyResult, write_results
    rows = []
    for seed in (0, 1, 2):
        for idx in range(29):
            matches = sorted(TRACES.glob(
                f"ABLv3HELDOUT29_A0_s{seed}_heldout_{idx:03d}_*.json"))
            if not matches:
                continue
            tr = json.loads(matches[-1].read_text(encoding="utf-8"))
            prop = tr.get("stage3_propose") or {}
            for k, h in enumerate(prop.get("proposal_hashes") or []):
                rows.append(ExternalTopologyResult(
                    baseline="agentic_raptor_generator",
                    spec_id=f"heldout_{idx}", seed=seed,
                    raw_output_path=str(matches[-1]),
                    valid_syntax=True, valid_graph=True,
                    topology_hash=h,
                    llm_calls=(prop.get("attempts") if k == 0 else 0),
                    notes=(f"frozen Tier-3 trace (A0); distinct="
                           f"{prop.get('distinct')}, families="
                           f"{prop.get('distinct_family_count')}")))
    write_results(rows, OUT)
    print(f"ag: {len(rows)} proposal rows extracted")


def part_pandallm() -> None:
    from .panda_adapter import run_spec_llm
    from .schema import write_results
    done = set()
    if OUT.exists():
        for l in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r["baseline"] == "panda_llm":
                done.add((r["spec_id"], r["seed"]))
    for i, spec in enumerate(_specs()):
        sid = spec["context_id"]
        if (sid, 0) in done:
            print(f"[{i+1}/29] {sid}: already done, skipped", flush=True)
            continue
        r = run_spec_llm(spec, seed=0)
        write_results([r], OUT)
        print(f"[{i+1}/29] panda_llm {sid}: syntax={r.valid_syntax} "
              f"graph={r.valid_graph} their_check={r.simulatable} "
              f"rounds={r.llm_calls} ({r.generation_runtime_s}s)", flush=True)


def part_summary() -> None:
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["baseline"]].append(r)
    print(f"\n=== TRACK-A FULL (option A) SUMMARY -- {len(rows)} rows ===")
    for b, rs in sorted(by.items()):
        n = len(rs)
        vs = sum(1 for r in rs if r["valid_syntax"])
        vg = sum(1 for r in rs if r["valid_graph"])
        fn = sum(1 for r in rs if r["simulatable"])
        uniq = len({r["topology_hash"] for r in rs if r["topology_hash"]})
        toks = sum(r.get("llm_tokens") or 0 for r in rs)
        print(f"{b:30s} n={n:4d} syntax={vs:4d} graph={vg:4d} "
              f"functional(native)={fn:4d} unique_iso={uniq:3d} tokens={toks:,}")
    print("NOTE: 'functional' = each method's native check at this stage; the "
          "uniform after-common-sizing verdict is Stage 6.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["acp", "panda", "ag", "pandallm", "summary"])
    ap.add_argument("--samples", type=int, default=5)
    a = ap.parse_args()
    if a.part == "acp":
        part_acp(a.samples)
    else:
        {"panda": part_panda, "ag": part_ag, "pandallm": part_pandallm,
     "summary": part_summary}[a.part]()
