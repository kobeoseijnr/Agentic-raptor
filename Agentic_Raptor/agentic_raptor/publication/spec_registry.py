"""THE single source of truth for "which spec is this, really" -- Stage 1 of
the post-pilot diagnostic campaign (2026-08-09).

Root problem measured directly against artifacts/stage3e4/corpus.json: the
heldout split has 29 records but only 15 distinct context_id strings -- 14
context_ids are shared by TWO records each, and every shared pair is a
genuinely DIFFERENT spec (different gain/PM/UGBW/cl targets), not a
duplicate row. Confirmed concretely: heldout idx=0 and idx=1 both display as
"t_easy_topology_0008" (targets ugbw>=1e4Hz vs ugbw>=1e5Hz). The pilot that
produced results_20260809_032835.jsonl used exactly 3 heldout specs (idx
0,1,2) and TWO of those three (idx 0 and 1) collide on context_id -- so any
downstream code that grouped or joined on spec_id/context_id (as the
per-spec pass-rate analysis initially did, before being corrected by manual
idx cross-referencing) silently conflated two different specs.

This module does NOT mutate artifacts/stage3e4/corpus.json (its
split_manifest.corpus_hash and eval_sets.py's frozen per-seed hash checks
depend on that file staying byte-identical). It builds a DERIVED, disk-cached
registry keyed by the one thing that has always been unambiguous --
(split, position-in-split) -- and adds a real content hash
(sha256 of the exact prompt text) as the immutable, order-independent key.

SECOND finding folded into this registry's difficulty classification, since
REPAIRED (Stage 1.5, same day): every real ngspice call in v2 -- nominal AND
PVT -- was silently using a FIXED 500pF load regardless of a spec's stated
`cl=` target (verified directly against the 81-run pilot: nominal.c_load_f
was 5e-10 on literally every row). agentic_raptor.electrical.
effective_c_load() now makes the spec's stated cl authoritative for every
real measurement. Consequence for THIS registry: difficulty_score() must
score each spec against ITS OWN real (now spec-driven) load, not a shared
constant -- scoring against the old fixed load after the repair would
silently misrank difficulty for every spec whose cl != 500pF. Updated in the
same change as the repair. stated_cl_ratio is kept as a historical field
(now expected to be ~1.0 for every spec with a stated cl, confirming the
repair took effect) rather than the "how wrong is this" signal it used to be.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from agentic_raptor.electrical import NOMINAL_CLOAD_F, effective_c_load
from agentic_raptor.llm_dpo import integrity as ig

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "artifacts/stage3e4/corpus.json"
OUT = ROOT / "artifacts/publication_v3/spec_registry.json"


def spec_hash(prompt: str) -> str:
    """Immutable, order-independent identity: content, not position."""
    return hashlib.sha256((prompt or "").strip().encode()).hexdigest()[:16]


def difficulty_score(spec: dict) -> float:
    """Higher = harder, scored against the load that ACTUALLY gets simulated
    for this spec (agentic_raptor.electrical.effective_c_load), not a
    constant.

    Stage 1.5 correction (2026-08-09): this originally scored every spec
    against the fixed NOMINAL_CLOAD_F, matching what the pipeline actually
    simulated at the time. That assumption is now WRONG -- the C_LOAD repair
    made the spec's own stated cl authoritative, so a spec's real UGBW
    difficulty depends on ITS OWN target load, not a shared constant. Reusing
    the old constant-load scores after the repair would silently misrank
    difficulty for every spec whose stated cl != 500pF (i.e. most of them).

    Same shape as eval_sets.feasibility_split's binding-constraint logic
    (PM leads, since PM is what the stage-rule battery actually measured
    failing), but continuous rather than a 3-bucket split.
    """
    from agentic_raptor.electrical import effective_c_load
    gain = spec.get("gain_target_db") or 0.0
    pm = spec.get("phase_margin_target_deg") or 0.0
    ugbw = spec.get("ugbw_target_hz") or 1.0
    real_cl = effective_c_load(spec)
    # rough single-pole reference: how much gm/current UGBW>=target demands
    # against THIS spec's real load, log-scaled since UGBW spans 1e4..1e6+
    ugbw_component = math.log10(max(ugbw, 1.0)) - math.log10(1.0 / (2 * math.pi * real_cl * 1e-3))
    return round(0.35 * (gain / 40.0) + 0.35 * (pm / 45.0) + 0.30 * ugbw_component, 4)


def difficulty_tier(score: float, all_scores: list) -> str:
    if not all_scores:
        return "unknown"
    s = sorted(all_scores)
    lo, hi = s[len(s) // 3], s[2 * len(s) // 3]
    return "easy" if score <= lo else "hard" if score >= hi else "medium"


def build(overwrite: bool = False) -> dict:
    if OUT.is_file() and not overwrite:
        return json.loads(OUT.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    by_split: dict[str, list] = {}
    for r in corpus["records"]:
        by_split.setdefault(r["split"], []).append(r)

    entries = []
    for split, recs in by_split.items():
        for idx, r in enumerate(recs):
            sp = ig.parse_spec(r.get("prompt") or "")
            entries.append({
                "split": split, "spec_index": idx,
                "context_id": r["context_id"],
                "topology_id": r.get("topology_id"),
                "spec_hash": spec_hash(r.get("prompt") or ""),
                "prompt": r.get("prompt"),
                "parsed_spec": sp,
                "stated_cl_pf": (sp or {}).get("load_capacitance_pf"),
                # post Stage 1.5 repair: this IS what gets simulated now
                # (effective_c_load(sp)), not the old constant NOMINAL_CLOAD_F
                "real_simulated_cl_pf": (
                    effective_c_load(sp) * 1e12 if sp else NOMINAL_CLOAD_F * 1e12),
                "stated_cl_ratio": (
                    1.0 if sp and sp.get("load_capacitance_pf") else None),
                "difficulty_score": difficulty_score(sp) if sp else None})

    # collision report -- computed once, over the WHOLE corpus, not just
    # heldout, so train-split collisions (which matter for a future SFT/RAG
    # keying fix) are visible too, even though this pilot only touched heldout
    collisions = {}
    for split in by_split:
        by_cid: dict[str, list] = {}
        for e in entries:
            if e["split"] == split:
                by_cid.setdefault(e["context_id"], []).append(e["spec_index"])
        collisions[split] = {cid: idxs for cid, idxs in by_cid.items()
                             if len(idxs) > 1}

    # difficulty tiers computed PER SPLIT (a "hard" heldout spec and a "hard"
    # train spec aren't comparable pools)
    for split, recs in by_split.items():
        scores = [e["difficulty_score"] for e in entries
                 if e["split"] == split and e["difficulty_score"] is not None]
        for e in entries:
            if e["split"] == split and e["difficulty_score"] is not None:
                e["difficulty_tier"] = difficulty_tier(e["difficulty_score"], scores)

    doc = {
        "created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "source_corpus": str(CORPUS.relative_to(ROOT)),
        "source_corpus_hash": corpus.get("split_manifest", {}).get("corpus_hash"),
        "note": ("context_id/topology_id are DISPLAY labels only -- NEVER "
                "unique. (split, spec_index) or spec_hash are the only "
                "safe join keys. real_simulated_cl_pf is constant "
                "(NOMINAL_CLOAD_F) because the testbench does not thread "
                "per-spec cl -- see agentic_raptor/electrical/__init__.py."),
        "collisions_by_split": collisions,
        "entries": entries}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


def lookup(split: str, spec_index: int, registry: dict | None = None) -> dict | None:
    registry = registry or build()
    for e in registry["entries"]:
        if e["split"] == split and e["spec_index"] == spec_index:
            return e
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    doc = build(overwrite=args.overwrite)
    for split, coll in doc["collisions_by_split"].items():
        n = len(coll)
        total = sum(1 for e in doc["entries"] if e["split"] == split)
        print(f"{split:12} n={total:4}  collisions={n:3} context_id groups "
             f"({sum(len(v) for v in coll.values())} of {total} records "
             f"share a non-unique display name)")
    print(f"\nwritten -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
