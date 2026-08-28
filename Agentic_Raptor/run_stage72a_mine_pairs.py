"""Stage 7.2A Part 5: mine cross-branch DPO pairs from existing
POST_CLOAD_FIX_V1 artifacts (sac_replay.jsonl), offline, no new SPICE.

Run:  python run_stage72a_mine_pairs.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agentic_raptor.publication.eval_sets import excluded_context_ids
from agentic_raptor.ranking import pair_mining as pm

ROOT = Path(__file__).resolve().parent
REPLAY_PATH = ROOT / "artifacts/publication_v2/live_streams_post_cload_v1/sac_replay.jsonl"
TRUSTED_PATH = ROOT / "datasets/ranker_preference_queue/trusted_pairs_post_cload_v1.jsonl"
OUT_DIR = ROOT / "artifacts/publication_v3/stage7_2a_pair_mining"
TMP_SURROGATE_DIR = OUT_DIR / "tmp_surrogates"
MAX_CANDIDATES_PER_BRANCH = 5


def load_spec_lookup() -> dict:
    rows = [json.loads(l) for l in TRUSTED_PATH.read_text(encoding="utf-8").splitlines()
           if l.strip()]
    lookup = {}
    for r in rows:
        h = r["spec"].get("spec_hash")
        if h:
            lookup[h] = r["spec"]
    return lookup


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec_by_hash = load_spec_lookup()
    print(f"spec lookup: {len(spec_by_hash)} distinct spec_hash", flush=True)

    runs = pm.load_runs_from_replay(REPLAY_PATH)
    paired = pm.pair_runs_by_spec_seed(runs)
    print(f"true per-branch runs: {len(runs)}; paired (A,B) runs: {len(paired)}", flush=True)

    excluded = excluded_context_ids()
    all_mined = []
    n_skipped_no_spec = 0
    n_skipped_protected = 0
    n_skipped_no_surrogate = 0
    t0 = time.time()
    for i, one in enumerate(paired):
        spec = spec_by_hash.get(one["spec_hash"])
        if spec is None:
            n_skipped_no_spec += 1
            continue
        if spec.get("spec_id") in excluded:
            n_skipped_protected += 1
            continue
        mined = pm.mine_cross_branch_pairs(one, spec, TMP_SURROGATE_DIR, excluded,
                                           MAX_CANDIDATES_PER_BRANCH)
        if not mined:
            n_skipped_no_surrogate += 1
        all_mined.extend(mined)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(paired)}] mined so far: {len(all_mined)} "
                 f"({round(time.time()-t0,1)}s elapsed)", flush=True)

    kept, n_dupes = pm.deduplicate_pairs(all_mined)
    print(f"\ntotal raw mined pairs: {len(all_mined)}", flush=True)
    print(f"canonical duplicates removed: {n_dupes}", flush=True)
    print(f"final deduplicated mined pairs: {len(kept)}", flush=True)
    print(f"runs skipped (no spec found): {n_skipped_no_spec}", flush=True)
    print(f"runs skipped (protected spec): {n_skipped_protected}", flush=True)
    print(f"runs skipped (insufficient data for a surrogate): {n_skipped_no_surrogate}", flush=True)

    out_path = OUT_DIR / "MINED_PAIRS.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, default=str) + "\n")
    print(f"\nwritten -> {out_path}", flush=True)

    both_feasible = sum(1 for p in kept
                        if p["outcome_a"]["exact_spec_pass"] and p["outcome_b"]["exact_spec_pass"])
    print(f"both-feasible mined pairs: {both_feasible}", flush=True)
    print(f"total wall time: {round(time.time()-t0,1)}s", flush=True)


if __name__ == "__main__":
    main()
