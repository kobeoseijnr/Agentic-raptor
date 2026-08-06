"""Case C, Step 8: does the retrained proposer actually propose five?

This is the ONLY test that settles Problem A. Everything before it -- the
realizability gate, the corpus rebuild, the retrain -- makes five topologies
POSSIBLE. This measures whether the model emits them.

Protocol is identical to the frozen baseline (baseline_diversity.json) so the
two numbers are comparable: DIVERSITY_LADDER, target_k=5, <=20 attempts,
max temperature 1.5, distinctness by canonical variant hash. Nothing is
enumerated or substituted -- every candidate is the model's own output, and
if the ladder exhausts below 5 that is reported as a failure, not padded.

Evaluated on the HELDOUT split of the original corpus. The repaired corpus
trained on `train` only, so these contexts were never seen. The blindtest
split is not touched.

Run (use the pythoncore interpreter -- conda base has a broken torch):
  python run_success_at_5.py --adapter artifacts/publication_v2/proposer_repair/sft_adapter_diverse
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v2/proposer_repair"
BASELINE = OUT / "baseline_diversity.json"
TARGET_K = 5


def family_of(obj) -> str:
    """Structure class of a proposal, for coverage reporting."""
    from agentic_raptor.llm_dpo import integrity as ig
    try:
        return f"{len(obj['stages'])}s_" + ig.compensation_class(obj)
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--contexts", type=int, default=0,
                    help="limit heldout contexts (0 = all); the baseline used 6")
    ap.add_argument("--target-k", type=int, default=TARGET_K)
    ap.add_argument("--out", default=str(OUT / "success_at_5.json"))
    ap.add_argument("--conditioning", choices=("temperature", "exclusion"),
                    default="exclusion",
                    help="'temperature' is the baseline protocol (widen "
                         "sampling only). 'exclusion' additionally asks for a "
                         "structure DIFFERENT from those already proposed -- "
                         "the conditioning the repaired corpus trains but "
                         "which no serve path previously used")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"not a peft adapter directory: {adapter}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from agentic_raptor.llm_dpo import MODEL_ID
    from agentic_raptor.llm_dpo.stage3e4 import (O4, propose_diverse,
                                                 propose_diverse_excl)
    from agentic_raptor.ranking import directory_sha256

    # The adapter records the base it was trained against. Trusting MODEL_ID
    # instead would let a shell without AGENTIC_RAPTOR_TOPOLOGY_LLM attach a
    # 3B-shaped LoRA to the 4B default -- and a diversity number measured on
    # a different base model is not comparable to the frozen baseline at all.
    cfg = json.loads((adapter / "adapter_config.json").read_text(
        encoding="utf-8"))
    base_id = cfg.get("base_model_name_or_path")
    if base_id and base_id != MODEL_ID:
        raise SystemExit(
            f"base-model mismatch\n"
            f"  adapter was trained on : {base_id}\n"
            f"  this shell resolves to : {MODEL_ID}\n"
            f"set AGENTIC_RAPTOR_TOPOLOGY_LLM={base_id}, or retrain against "
            f"{MODEL_ID} so the comparison holds the base model constant.")

    corpus = json.loads((O4 / "corpus.json").read_text(encoding="utf-8"))
    held = [r for r in corpus["records"] if r["split"] == "heldout"]
    if args.contexts:
        held = held[:args.contexts]

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    # mirror the training loader exactly: bf16 on GPU. Omitting dtype loads
    # fp32, which is ~2x the VRAM and OOMs a laptop GPU on a 3-4B model.
    cuda = torch.cuda.is_available()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if cuda else torch.float32,
        device_map="auto" if cuda else None)
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()

    print(f"adapter  : {adapter}")
    print(f"base     : {MODEL_ID}  ({'bf16/gpu' if cuda else 'fp32/cpu'})")
    print(f"contexts : {len(held)} (heldout)")
    print(f"target_k : {args.target_k}\n", flush=True)

    rows = []
    for i, r in enumerate(held):
        t0 = time.time()
        if args.conditioning == "exclusion":
            res = propose_diverse_excl(model, tok, r["prompt"],
                                       target_k=args.target_k, seed0=i + 1,
                                       family_fn=family_of)
        else:
            res = propose_diverse(model, tok, r["prompt"],
                                  target_k=args.target_k, seed0=i + 1)
        cands = res["candidates"]
        fams = sorted({family_of(c["obj"]) for c in cands})
        # context_id is NOT unique in the heldout split: 29 records carry only
        # 15 distinct ids while all 29 prompts differ. Keying on it alone
        # would look like the same context measured twice.
        rows.append({
            "eval_key": f"{i:02d}_{r['context_id']}",
            "context_id": r["context_id"],
            "prompt_sha8": __import__("hashlib").sha256(
                r["prompt"].encode()).hexdigest()[:8],
            "distinct": len(cands),
            "reached_target": len(cands) >= args.target_k,
            "attempts": res.get("attempts"),
            "max_temperature": max((c["temperature"] for c in cands),
                                   default=None),
            "classes": fams,
            "graph_hashes": sorted(c["graph_hash"] for c in cands),
            "seconds": round(time.time() - t0, 1)})
        print(f"[{i+1:>2}/{len(held)}] {r['context_id']:<28} "
              f"#{rows[-1]['prompt_sha8']} "
              f"distinct={len(cands)}/{args.target_k} "
              f"attempts={res.get('attempts')} "
              f"{'REACHED' if rows[-1]['reached_target'] else '       '} "
              f"{','.join(fams)}", flush=True)

    reached = [x for x in rows if x["reached_target"]]
    classes_ever = sorted({c for x in rows for c in x["classes"]})
    from build_diverse_corpus import ALL_FAMILIES
    # The model can still emit structures the repaired corpus excluded --
    # 3s_none is a valid GRAPH that the realizability gate showed is not a
    # realizable AMPLIFIER (uncompensated 3-stage, unstable by construction).
    # Counting it as diversity would overstate how much USABLE design space
    # the proposer covers, so it is reported separately rather than netted out.
    unrealizable = sorted(set(classes_ever) - set(ALL_FAMILIES))
    for x in rows:
        x["realizable_classes"] = [c for c in x["classes"] if c in ALL_FAMILIES]
        x["unrealizable_classes"] = [c for c in x["classes"]
                                     if c not in ALL_FAMILIES]
    doc = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "proposer": {"checkpoint_path": str(adapter),
                     "checkpoint_sha256": directory_sha256(str(adapter)),
                     "training_method": "SFT on corpus_diverse.json"},
        "protocol": {"target_k": args.target_k,
                     "ladder": "DIVERSITY_LADDER (4+4+6+6, T 0.8->1.5)",
                     "max_attempts_budget": 20,
                     "distinctness": "canonical variant hash",
                     "split": "heldout",
                     "conditioning": args.conditioning,
                     "identical_to_baseline": args.conditioning == "temperature"},
        "results": {
            "contexts": len(rows),
            "mean_distinct": round(sum(x["distinct"] for x in rows)
                                   / max(len(rows), 1), 3),
            "specs_reaching_target": len(reached),
            "share_reaching_target": round(len(reached) / max(len(rows), 1), 3),
            "mean_attempts": round(sum(x["attempts"] or 0 for x in rows)
                                   / max(len(rows), 1), 1),
            "classes_ever_produced": classes_ever,
            "classes_never_produced": sorted(set(ALL_FAMILIES) - set(classes_ever)),
            "unrealizable_classes_produced": unrealizable,
            "specs_emitting_unrealizable": sum(
                1 for x in rows if x["unrealizable_classes"]),
            "mean_realizable_distinct": round(
                sum(len(x["realizable_classes"]) for x in rows)
                / max(len(rows), 1), 3),
            "distinct_histogram": dict(sorted(
                Counter(x["distinct"] for x in rows).items()))},
        "per_spec": rows}

    if BASELINE.is_file():
        b = json.loads(BASELINE.read_text(encoding="utf-8"))["results"]
        doc["baseline_comparison"] = {
            "baseline_mean_distinct": b.get("baseline_mean_distinct"),
            "baseline_ladder_mean_distinct": b.get("ladder_mean_distinct"),
            "baseline_share_reaching_target": b.get("share_reaching_target"),
            "baseline_classes_ever_produced": b.get("classes_ever_produced"),
            "now_mean_distinct": doc["results"]["mean_distinct"],
            "now_share_reaching_target": doc["results"]["share_reaching_target"],
            "now_classes_ever_produced": classes_ever}

    passed = doc["results"]["share_reaching_target"] >= 1.0
    doc["verdict"] = f"PROPOSER DIVERSITY PASS: {passed}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")

    r = doc["results"]
    print(f"\nmean distinct        : {r['mean_distinct']}  "
          f"(baseline 1.0 fixed-T, 1.67 with ladder)")
    print(f"reached {args.target_k}/{args.target_k}          : "
          f"{r['specs_reaching_target']}/{r['contexts']} "
          f"({100*r['share_reaching_target']:.0f}%)   (baseline 0%)")
    print(f"classes produced     : {classes_ever}")
    print(f"classes never seen   : {r['classes_never_produced']}")
    if r["unrealizable_classes_produced"]:
        print(f"UNREALIZABLE emitted : {r['unrealizable_classes_produced']} "
              f"on {r['specs_emitting_unrealizable']}/{r['contexts']} specs "
              f"(counts toward distinct, but cannot be built)")
        print(f"mean REALIZABLE dist : {r['mean_realizable_distinct']}")
    print(f"\nwritten -> {args.out}")
    print(doc["verdict"])


if __name__ == "__main__":
    main()
