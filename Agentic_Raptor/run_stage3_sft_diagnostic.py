"""Stage 3 diagnostic: why did A3 (base Qwen) produce zero usable topology
graphs, and what does the SFT adapter actually teach?

=== Item 1: SFT adapter audit (result, not a TODO -- concrete evidence) ===
corpus_diverse.json's RESPONSE (the actual supervised label) is PURELY
STRUCTURAL -- stages/compensation/buffer/feedback JSON, confirmed by direct
inspection, zero SPICE/gain/pm/ugbw fields anywhere in a training record.
Response labels come from a deterministic rule (soft_family_order() +
realizable_families()), never from the prompt text.

TWO real, non-label provenance dependencies found:
1. realizable_families() reads family_spec_gate/SUMMARY.json, built under
   PRE_CLOAD_FIX (probed loads 50-1000pF against the old fixed-500pF
   simulator). Already found 5/5 families realizable, so a rebuild would
   very likely reproduce the same target space, but this is not confirmed.
2. Training PROMPTS (not responses) contain a baked-in "### KNOWN ..."
   evidence line -- traced directly to stage3e4.py's L4 injection reading
   datasets/simulation_memory/self_improvement_runs.jsonl, the SAME file
   already flagged and archived elsewhere in this project for predating
   the VCM fix (phase margins measured at 1/79th of design current). This
   is real: the adapter's INPUT distribution contains stale evidence text.
   It does not corrupt the supervised LABEL (deterministic, independent of
   this text), so it does not compromise the serialization/family-choice
   capability SFT teaches -- but it is a genuine, confirmed staleness the
   adapter carries, not a hypothetical one.

VERDICT: PROVISIONALLY SAFE_TO_REUSE for this diagnostic (the question here
is format/structure learning vs base-model capability, which does not hinge
on either dependency above) -- NOT fully re-certified for final paper use
without a rebuilt family_spec_gate and a corpus rebuilt after that.

=== Item 8: old A3 config-bug audit (result) ===
Read agentic_raptor/publication/ablation_v3.py directly: A3_NO_SFT =
AblationConfig("A3", "NO_SFT", ..., use_sft=False) -- dataclass, every
OTHER field (use_rag, use_exclusion_conditioning, use_mcts, use_sac,
use_dpo, budget, ...) inherits A0_FULL's identical defaults. use_sft itself
is not even in to_run_pipeline_kwargs() -- it only ever controls WHICH
proposer gets loaded, via run_ablation_v3.py's _proposer_requirement()
grouping ("base" vs "sft") and run_qwen_ablation._load(adapter): `if
adapter: m = PeftModel.from_pretrained(m, adapter)` -- when adapter is
None, this changes NOTHING else (same MODEL_ID, same tokenizer, same
dtype/device logic, same generation ladder, same RAG, same exclusion
conditioning). No config/checkpoint/prompt/parser/tokenizer bug found.

Does NOT modify propose_diverse()/propose_diverse_excl() (the production
generation loop) -- those intentionally discard raw text for
failed/invalid attempts (the finding this diagnostic needs). This is a
STANDALONE instrumented loop, same sampling ladder/seeding/RAG as
production, that keeps every attempt's raw decoded text and classifies
every failure into the exact A-I taxonomy.

Run:  python run_stage3_sft_diagnostic.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v3/stage3_sft_diagnostic"
DEFAULT_ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"

ADAPTER_SAFETY_NOTE = (
    "PROVISIONALLY SAFE_TO_REUSE. Response labels are purely structural "
    "(zero SPICE fields, confirmed by direct corpus inspection). Two real "
    "non-label dependencies: (1) realizable_families() reads "
    "family_spec_gate/SUMMARY.json, still PRE_CLOAD_FIX -- 5/5 families "
    "already realizable, rebuild would very likely reproduce the same "
    "target space but this is unconfirmed; (2) training PROMPTS contain a "
    "'### KNOWN' line sourced from self_improvement_runs.jsonl, confirmed "
    "pre-VCM-fix and already flagged stale elsewhere in this project -- "
    "input-distribution staleness, not label corruption. NOT fully "
    "re-certified for final paper use.")

CONFIG_BUG_AUDIT_NOTE = (
    "NO CONFIG BUG FOUND. A3_NO_SFT differs from A0_FULL in EXACTLY ONE "
    "dataclass field (use_sft=False); every other field -- use_rag, "
    "use_exclusion_conditioning, use_mcts, use_sac, use_dpo, budget -- "
    "inherits A0's identical default. use_sft is not even read by "
    "to_run_pipeline_kwargs(); it only selects which proposer checkpoint "
    "run_ablation_v3.py loads. run_qwen_ablation._load(None) changes "
    "nothing except skipping the PeftModel.from_pretrained call -- same "
    "MODEL_ID, tokenizer, dtype/device, generation ladder, RAG, exclusion "
    "conditioning as the SFT path.")

# A-I failure taxonomy, exactly as specified.
TAXONOMY = {
    "A": "empty_refusal_irrelevant_text",
    "B": "yaml_serialization_failure",
    "C": "required_field_missing",
    "D": "parser_failure",
    "E": "schema_failure",
    "F": "graph_construction_failure",
    "G": "structural_electrical_validator_failure",
    "H": "canonical_duplicate",
    "I": "valid_graph",
}


def classify(raw_text: str, obj: dict | None, valid: bool,
            reasons: list, is_duplicate: bool) -> str:
    """Returns one TAXONOMY key (A-I). Never labels a serialization/schema
    failure as a circuit-quality problem -- those are structurally
    distinct branches below."""
    stripped = raw_text.strip()
    if not stripped or all(not c.isalnum() for c in stripped[:40]):
        return "A"      # empty / refusal / irrelevant text
    if "{" not in stripped:
        return "A"      # no attempt at the expected format at all
    if obj is None:
        return "B"      # found '{' but JSON/serialization didn't parse
    if "stages" not in obj or not obj.get("stages"):
        return "C"      # parsed JSON, but the required entity is absent
    if not valid:
        if any(r.startswith("malformed_") for r in reasons):
            return "E"   # schema failure: wrong types where dicts expected
        if any("unsupported" in r or "block" in r for r in reasons):
            return "F"   # graph construction: block/role the mapper can't realise
        return "G"       # parsed + schema-shaped, but structurally/electrically invalid
    if is_duplicate:
        return "H"
    return "I"


def instrumented_ladder(model, tok, prompt: str, target_k: int, ladder, seed0: int,
                        *, checkpoint_id: str, sft_loaded: bool,
                        rag_context_ids: list, spec_hash: str) -> dict:
    import torch

    from agentic_raptor.llm_dpo import parse_proposal_text, proposal_dict_valid
    from agentic_raptor.llm_dpo.stage3e4 import variant_hash
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    seen: dict[str, dict] = {}
    attempts_log = []
    attempts = 0
    for temp, top_p, n in ladder:
        for _j in range(n):
            if len(seen) >= target_k:
                break
            attempts += 1
            torch.manual_seed(seed0 * 1000 + attempts)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=260, do_sample=True,
                                     temperature=temp, top_p=top_p,
                                     pad_token_id=tok.eos_token_id)
            raw = tok.decode(out[0, ids.shape[1]:])
            obj = parse_proposal_text(raw)
            valid, reasons = (proposal_dict_valid(obj) if obj is not None
                              else (False, []))
            h = variant_hash(obj) if (obj is not None and valid) else None
            is_dup = bool(h and h in seen)
            cls = classify(raw, obj, valid, reasons, is_dup)
            attempts_log.append({
                "attempt": attempts, "checkpoint_id": checkpoint_id,
                "sft_loaded": sft_loaded, "seed": seed0, "temperature": temp,
                "top_p": top_p, "spec_hash": spec_hash,
                "rag_context_ids": rag_context_ids,
                "raw_response": raw[:800],   # capped -- these can run long at high temp
                "parsed": obj is not None, "valid": bool(valid and not is_dup),
                "validator_reasons": reasons, "graph_hash": h,
                "taxonomy_code": cls, "taxonomy_label": TAXONOMY[cls]})
            if obj is not None and valid and not is_dup:
                seen[h] = {"obj": obj, "temperature": temp, "top_p": top_p}
        if len(seen) >= target_k:
            break
    from collections import Counter
    class_counts = Counter(a["taxonomy_label"] for a in attempts_log)
    n_parsed = sum(1 for a in attempts_log if a["parsed"])
    return {"target_k": target_k, "distinct": len(seen), "attempts": attempts,
           "candidates": [{"graph_hash": h, **v} for h, v in seen.items()],
           "attempts_log": attempts_log,
           "taxonomy_counts": dict(class_counts),
           "raw_completion_rate": sum(
               1 for a in attempts_log if a["taxonomy_code"] != "A") / max(1, attempts),
           "schema_success_rate": n_parsed / max(1, attempts),
           "parse_success_rate": n_parsed / max(1, attempts),
           "graph_construction_success_rate": sum(
               1 for a in attempts_log
               if a["taxonomy_code"] in ("G", "H", "I")) / max(1, attempts),
           "validator_success_rate": sum(
               1 for a in attempts_log
               if a["taxonomy_code"] in ("H", "I")) / max(1, attempts),
           "valid_rate": sum(1 for a in attempts_log if a["valid"]) / max(1, attempts)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=int, default=6)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--split", default="train",
                    help="train (default), matching Stage 2's fix -- "
                         "heldout has almost no unprotected specs left")
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--target-k", type=int, default=5)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=== ITEM 1: ADAPTER SAFETY AUDIT ===")
    print(ADAPTER_SAFETY_NOTE, "\n")
    print("=== ITEM 8: A3 CONFIG-BUG AUDIT ===")
    print(CONFIG_BUG_AUDIT_NOTE, "\n")

    from agentic_raptor.llm_dpo.stage3e4 import DIVERSITY_LADDER
    from agentic_raptor.publication.eval_sets import excluded_context_ids
    from agentic_raptor.publication.preflight import CLEAN_RAG_PATH
    from agentic_raptor.publication.spec_registry import build as build_registry
    from run_qwen_ablation import _load
    from run_raptor_v2 import rag_stage

    protected = excluded_context_ids()
    reg = build_registry()
    entries = [e for e in reg["entries"] if e["split"] == args.split
              and e["context_id"] not in protected and e["parsed_spec"]]
    by_tier: dict[str, list] = {}
    for e in entries:
        by_tier.setdefault(e["difficulty_tier"], []).append(e)
    per_tier = max(1, args.specs // 3)
    diag = []
    for tier in ("easy", "medium", "hard"):
        pool = sorted(by_tier.get(tier, []), key=lambda e: e["spec_index"])
        diag.extend(pool[:per_tier])
    if len(diag) < args.specs:
        used = {e["spec_hash"] for e in diag}
        spare = sorted((e for e in entries if e["spec_hash"] not in used),
                       key=lambda e: e["spec_index"])
        diag.extend(spare[:args.specs - len(diag)])
    diag = diag[:args.specs]

    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    pool_recs = [r for r in corpus["records"] if r["split"] == args.split]

    print(f"loading BASE model (no adapter)...")
    tok_base, model_base = _load(None)
    print(f"loading SFT model (adapter={args.adapter})...")
    tok_sft, model_sft = _load(args.adapter)
    print("both models loaded\n", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for e in diag:
        idx = e["spec_index"]
        rec = pool_recs[idx]
        spec = dict(e["parsed_spec"], spec_id=e["context_id"],
                   spec_hash=e["spec_hash"], topology_id=rec.get("topology_id"))
        # Item 2: the SAME frozen, corrected Stage-2 RAG implementation,
        # identically for BOTH conditions -- computed ONCE per spec so
        # base and SFT see byte-identical injected context.
        rag = rag_stage(spec, rec["prompt"], use_rag=True, memory_path=str(CLEAN_RAG_PATH))
        for seed in seeds:
            t0 = time.time()
            base_res = instrumented_ladder(
                model_base, tok_base, rag["prompt"], args.target_k, DIVERSITY_LADDER, seed,
                checkpoint_id="Qwen3-4B-base", sft_loaded=False,
                rag_context_ids=rag["retrieval_ids"], spec_hash=e["spec_hash"])
            sft_res = instrumented_ladder(
                model_sft, tok_sft, rag["prompt"], args.target_k, DIVERSITY_LADDER, seed,
                checkpoint_id=str(args.adapter), sft_loaded=True,
                rag_context_ids=rag["retrieval_ids"], spec_hash=e["spec_hash"])
            row = {"spec_index": idx, "spec_hash": e["spec_hash"],
                  "context_id": e["context_id"], "difficulty_tier": e["difficulty_tier"],
                  "seed": seed, "rag_context_ids": rag["retrieval_ids"],
                  "base": base_res, "sft": sft_res,
                  "elapsed_s": round(time.time() - t0, 1)}
            results.append(row)
            (OUT / "results.jsonl").open("a", encoding="utf-8").write(
                json.dumps(row, default=str) + "\n")
            print(f"idx={idx:3} tier={e['difficulty_tier']:6} seed={seed} "
                 f"BASE(distinct={base_res['distinct']},valid_rate={base_res['valid_rate']:.2f},"
                 f"taxonomy={base_res['taxonomy_counts']}) "
                 f"SFT(distinct={sft_res['distinct']},valid_rate={sft_res['valid_rate']:.2f}) "
                 f"{row['elapsed_s']:.0f}s", flush=True)

    print(f"\ndone: {len(results)} paired comparisons -> {OUT / 'results.jsonl'}")


if __name__ == "__main__":
    main()
