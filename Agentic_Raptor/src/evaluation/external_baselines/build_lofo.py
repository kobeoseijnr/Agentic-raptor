"""Leave-one-family-out (LOFO) generalization experiment builder.

Removes ONE topology family from every LEARNED component's training data, so a
held-out campaign can test whether the proposer generates a family it never
saw. READ-ONLY over frozen inputs; writes only new *_lofo_<family> artifacts.

Family knowledge audit (2026-09-06) -- where a family lives, and the treatment:
  1. SFT corpus  (corpus_tier2_extension.json, 917 recs) -> FILTER OUT the
     family's records; retrain the adapter on the remainder (user runs).
  2. RAG memory  (rag_memory_v2_post_cload_v1_clean.jsonl, 174) -> FILTER OUT;
     pass the filtered file via --rag-memory.
  3. Critic deep-comp rule -> DISABLE via AGR_LOFO_DISABLE_DEEPCOMP=1 (it names
     the preferred comp at depth>min, i.e. exactly 3s_rc on 2-stage-ceiling
     specs -- a scripted re-injection of the removed family).
  4. Bandit selector (BANDIT_TOP2_V2) + DPO ranker -> FROZEN, DISCLOSED as
     strictly DOWNSTREAM: they can only select/rank among candidates the
     proposer already generated; neither can introduce a family the proposer
     did not propose. (Refitting them from family-filtered history is possible
     but unnecessary for the causal question and would confound the result.)
  5. Mapping/realization library -> UNCHANGED (keeps the family's template).
     Scope note: LOFO tests whether the learned PROPOSER can DISCOVER/COMPOSE
     an unseen family, not whether the fixed realization layer can physically
     emit it. Stated as the experiment's scope in the report.

Usage:
  python -m src.evaluation.external_baselines.build_lofo --family 3s_rc
Outputs (under artifacts/publication_v3/lofo/<family>/):
  corpus_lofo_<family>.json, rag_lofo_<family>.jsonl, LOFO_MANIFEST.json,
  and the ready-to-run retrain + campaign commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SFT_CORPUS = ROOT / "artifacts/publication_v3/tier2/corpus_tier2_extension.json"
RAG_CLEAN = (ROOT / "artifacts/publication_v2/selfimprove"
             / "rag_memory_v2_post_cload_v1_clean.jsonl")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rec_family(r: dict) -> str:
    return f"{r.get('stages')}s_{r.get('comp')}"


def main(family: str, also_signature_variants: bool):
    out = ROOT / "artifacts/publication_v3/lofo" / family
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. SFT corpus ----
    corpus = json.loads(SFT_CORPUS.read_text(encoding="utf-8"))
    recs = corpus["records"]
    n0 = len(recs)
    before = Counter(rec_family(r) for r in recs)
    before_sig = Counter(r.get("topology_signature") for r in recs)

    def drop(r: dict) -> bool:
        if rec_family(r) == family:
            return True
        # signature variants (e.g. 3s_rc_ab) share the family's core; drop them
        # too when requested so no near-variant leaks the structure.
        if also_signature_variants and str(
                r.get("topology_signature", "")).startswith(family + "_"):
            return True
        return False

    kept = [r for r in recs if not drop(r)]
    dropped = n0 - len(kept)
    corpus_lofo = dict(corpus)
    corpus_lofo["records"] = kept
    corpus_lofo["lofo"] = {"held_out_family": family, "dropped": dropped,
                           "kept": len(kept), "source": SFT_CORPUS.name,
                           "drop_signature_variants": also_signature_variants}
    corpus_bytes = json.dumps(corpus_lofo, indent=1).encode()
    (out / f"corpus_lofo_{family}.json").write_bytes(corpus_bytes)

    # ---- 2. RAG memory ----
    rag_lines = [json.loads(l) for l in RAG_CLEAN.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    rag_before = Counter(r.get("family") for r in rag_lines)
    rag_kept = [r for r in rag_lines if r.get("family") != family]
    rag_dropped = len(rag_lines) - len(rag_kept)
    rag_text = "\n".join(json.dumps(r) for r in rag_kept) + "\n"
    (out / f"rag_lofo_{family}.jsonl").write_text(rag_text, encoding="utf-8")

    # ---- verify the family is truly gone ----
    assert not any(rec_family(r) == family for r in kept), "SFT leak"
    assert not any(r.get("family") == family for r in rag_kept), "RAG leak"

    adapter_out = ROOT / "artifacts/publication_v3/lofo" / family / f"sft_adapter_lofo_{family}"
    manifest = {
        "created": "2026-09-06",
        "experiment": "leave-one-family-out",
        "held_out_family": family,
        "drop_signature_variants": also_signature_variants,
        "components": {
            "sft_corpus": {
                "source": str(SFT_CORPUS.relative_to(ROOT)),
                "source_sha256": sha256_bytes(SFT_CORPUS.read_bytes()),
                "records_before": n0, "records_after": len(kept),
                "dropped": dropped,
                "family_counts_before": dict(before),
                "signature_counts_before": {k: v for k, v in before_sig.items()},
                "output": f"corpus_lofo_{family}.json",
                "output_sha256": sha256_bytes(corpus_bytes),
                "treatment": "filtered_out + retrain"},
            "rag_memory": {
                "source": str(RAG_CLEAN.relative_to(ROOT)),
                "source_sha256": sha256_bytes(RAG_CLEAN.read_bytes()),
                "records_before": len(rag_lines),
                "records_after": len(rag_kept), "dropped": rag_dropped,
                "family_counts_before": dict(rag_before),
                "output": f"rag_lofo_{family}.jsonl",
                "output_sha256": sha256_bytes(rag_text.encode()),
                "treatment": "filtered_out"},
            "critic_deep_comp_rule": {
                "treatment": "disabled via AGR_LOFO_DISABLE_DEEPCOMP=1",
                "reason": "names preferred comp at depth>min == held-out family "
                          "on 2-stage-ceiling specs; scripted re-injection"},
            "bandit_selector": {
                "treatment": "frozen, disclosed downstream",
                "reason": "selects among proposed candidates only; cannot "
                          "introduce an unproposed family"},
            "dpo_ranker": {
                "treatment": "frozen, disclosed downstream",
                "reason": "ranks sized candidates only; same argument"},
            "mapping_library": {
                "treatment": "unchanged",
                "scope_note": "LOFO tests learned PROPOSER discovery of the "
                              "family, not realization-layer capability"}},
        "primary_metrics": [
            "family_emergence_rate = fraction of held-out specs whose proposal "
            "pool contains the held-out family (from a proposer that never "
            "saw it)",
            "held_out_family_win_rate = fraction of winners in the held-out "
            "family",
            "FinalPass on the family-dependent specs vs the full-corpus system "
            "(the 17 heldout specs whose R2 winner was 3s_rc are the natural "
            "focus set)",
            "graceful-degradation: FinalPass/FoM on the remaining specs"],
        "retrain_command": [
            "python", "train_proposer_diverse.py",
            "--corpus", str((out / f'corpus_lofo_{family}.json')),
            "--out", str(adapter_out), "--steps", "1500"],
        "campaign_command": [
            "python", "run_ablation_v3.py", "--tag", f"LOFO_{family}",
            "--split", "heldout", "--specs", "29", "--seeds", "0,1,2",
            "--arms", "AG_FULL", "--adapter", str(adapter_out),
            "--rag-memory", str((out / f'rag_lofo_{family}.jsonl'))],
        "campaign_env": {"AGR_LOFO_DISABLE_DEEPCOMP": "1"},
    }
    (out / "LOFO_MANIFEST.json").write_text(json.dumps(manifest, indent=1),
                                            encoding="utf-8")

    print(f"=== LOFO builder: held-out family {family} ===")
    print(f"SFT corpus : {n0} -> {len(kept)} records (dropped {dropped})")
    print(f"   families before: {dict(before.most_common())}")
    print(f"RAG memory : {len(rag_lines)} -> {len(rag_kept)} (dropped {rag_dropped})")
    print(f"   families before: {dict(rag_before.most_common())}")
    print(f"critic deep-comp: disabled via AGR_LOFO_DISABLE_DEEPCOMP=1")
    print(f"bandit/DPO: frozen (downstream, cannot inject family)")
    print(f"outputs in: {out.relative_to(ROOT)}")
    print("\nRETRAIN (user, GPU ~ measured 529 s for tier2 at 1500 steps):")
    print("  " + " ".join(manifest["retrain_command"]))
    print("\nCAMPAIGN (user, ~4.25 h for 29 specs x 3 seeds):")
    print("  set AGR_LOFO_DISABLE_DEEPCOMP=1  (PowerShell: $env:AGR_LOFO_DISABLE_DEEPCOMP='1')")
    print("  " + " ".join(manifest["campaign_command"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="3s_rc",
                    help="topology family to hold out (default 3s_rc)")
    ap.add_argument("--keep-signature-variants", action="store_true",
                    help="keep near-variants like 3s_rc_ab (default: drop them)")
    a = ap.parse_args()
    main(a.family, not a.keep_signature_variants)
