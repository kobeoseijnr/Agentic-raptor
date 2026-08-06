"""Qwen/SFT ablation L0-L8: raw proposal quality on the frozen validation
set, BEFORE PUCT or SAC. GPU required. Resume-safe: completed arms are
skipped; each arm's results are saved separately and aggregated at the end.

Arms (checkpoints resolve from real campaign artifacts; absent ones are
recorded as unavailable, never silently substituted):
  L0 base Qwen, RAG/KNOWN lines stripped
  L1 base Qwen, full prompt (RAG)
  L2 initial SFT (gen0_sft of the reference campaign), RAG stripped
  L3 initial SFT, full prompt
  L4 initial SFT + HISTORICAL POISONED DPO adapter (negative control)
  L5 initial SFT + repaired integrity DPO (first accepted DPO checkpoint)
  L6 latest self-earned SFT (final gen SFT checkpoint)
  L7 latest accepted DPO checkpoint (safety-gated)
  L8 final accepted checkpoint of the latest campaign (full learner)

Run:  python run_qwen_ablation.py [--arms L0,L1,...] [--attempts 4]
"""
import argparse
import json
import re
import time
from pathlib import Path

from agentic_raptor.llm_dpo import MODEL_ID
from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.llm_dpo.stage3e4 import generate
from agentic_raptor.publication import PUB, ROOT

OUT = PUB / "qwen_ablation"
_RAG_RE = re.compile(r"^### (RAG|KNOWN) [^\n]*\n", re.M)


def _campaigns():
    return sorted((ROOT / "artifacts/self_improvement").glob("camp_*"))


def _gens(camp):
    f = camp / "logs/generations.jsonl"
    if not f.is_file():
        return []
    return [json.loads(x) for x in
            f.read_text(encoding="utf-8").splitlines()]


def resolve_arms() -> dict:
    camps = [c for c in _campaigns() if _gens(c)]
    ref, latest = camps[0] if camps else None, camps[-1] if camps else None
    ref_g, lat_g = _gens(ref) if ref else [], _gens(latest) if latest else []
    init_sft = (ROOT / ref_g[0]["sft"]["output_sft_checkpoint"]
                if ref_g else None)
    poisoned = ROOT / "artifacts/stage3e4/dpo_adapter_qwen"
    first_dpo = next((ROOT / g["dpo"]["output_dpo_checkpoint"]
                      for c in camps for g in _gens(c)
                      if g.get("dpo_update_accepted") and "dpo" in g), None)
    last_sft = (ROOT / lat_g[-1]["sft"]["output_sft_checkpoint"]
                if lat_g else None)
    # most recent ACCEPTED DPO checkpoint across ALL campaigns (the latest
    # campaign may have skipped DPO entirely once the proposer converged)
    last_dpo = next((ROOT / g["dpo"]["output_dpo_checkpoint"]
                     for c in reversed(camps) for g in reversed(_gens(c))
                     if g.get("dpo_update_accepted") and "dpo" in g), None)
    accepted = (ROOT / lat_g[-1]["accepted_checkpoint"] if lat_g else None)

    def ck(p):
        return str(p) if p and Path(p).is_dir() and \
            (Path(p) / "adapter_config.json").is_file() else None
    return {
        "L0": {"adapter": None, "rag": False, "desc": "base, no RAG"},
        "L1": {"adapter": None, "rag": True, "desc": "base, RAG"},
        "L2": {"adapter": ck(init_sft), "rag": False,
               "desc": f"initial SFT ({ref.name if ref else '?'}/gen0)"},
        "L3": {"adapter": ck(init_sft), "rag": True, "desc": "initial SFT+RAG"},
        "L4": {"adapter": ck(poisoned), "rag": True,
               "desc": "historical poisoned DPO (negative control)"},
        "L5": {"adapter": ck(first_dpo), "rag": True,
               "desc": "first accepted integrity DPO"},
        "L6": {"adapter": ck(last_sft), "rag": True,
               "desc": f"self-earned SFT ({latest.name if latest else '?'})"},
        "L7": {"adapter": ck(last_dpo), "rag": True,
               "desc": "latest accepted DPO"},
        "L8": {"adapter": ck(accepted), "rag": True,
               "desc": "final accepted checkpoint (full learner)"},
    }


# ------------------------- RAG battery (RAG0-RAG6) ---------------------------
def _l4_lines(kind: str) -> list:
    l4p = ROOT / "datasets/simulation_memory/self_improvement_runs.jsonl"
    if not l4p.is_file():
        return []
    out = []
    for x in l4p.read_text(encoding="utf-8").splitlines()[-40:]:
        if not x.strip():
            continue
        e = json.loads(x)
        st = e.get("stability")
        if not st or not e.get("stages"):
            continue
        if kind == "success" and st != "verified_stable":
            continue
        if kind == "failure" and st != "verified_unstable":
            continue
        out.append(f"{e['stages']}stage {st}"
                   + (f" pm={round(e['pm'])}deg" if e.get("pm") is not None
                      else ""))
    return out[-4:]


def _registry_lines() -> list:
    sump = ROOT / "datasets/simulation_memory/topology_summaries.jsonl"
    out = []
    if sump.is_file():
        for x in sump.read_text(encoding="utf-8").splitlines()[:200]:
            e = json.loads(x)
            if e.get("stability_status") == "verified_stable":
                out.append(f"registry:{e['topology_id']} verified_stable")
            if len(out) >= 2:
                break
    return out


def rag_transform(arm: str, prompt: str) -> str:
    """Evaluation-time retrieval-content variants. RAG/KNOWN lines are
    stripped first, then the arm's retrieval content is injected."""
    base = _RAG_RE.sub("", prompt)
    rag_id = next((ln for ln in prompt.splitlines()
                   if ln.startswith("### RAG ")), None)

    def inject(lines):
        if not lines:
            return base
        block = "### KNOWN " + "; ".join(lines) + "\n"
        return base.replace("### BLOCKS", block + "### BLOCKS")
    if arm == "RAG0":
        return base
    if arm == "RAG1":
        return base.replace("### BLOCKS", (rag_id + "\n" if rag_id else "")
                            + "### BLOCKS")
    if arm == "RAG2":
        return inject(_registry_lines())
    if arm == "RAG3":
        return inject(_l4_lines("success"))
    if arm == "RAG4":
        return inject(_l4_lines("failure"))
    if arm == "RAG5":
        return inject(_l4_lines("success") + _l4_lines("failure"))
    # RAG6: full hierarchical (id line + registry + measured evidence)
    return base.replace("### BLOCKS", (rag_id + "\n" if rag_id else "")
                        + "### KNOWN "
                        + "; ".join(_registry_lines()
                                    + _l4_lines("success")
                                    + _l4_lines("failure"))
                        + "\n### BLOCKS")


def run_rag_battery(attempts: int = 4) -> dict:
    """RAG0-RAG6 on the FINAL accepted checkpoint, frozen validation set."""
    import torch
    arms = resolve_arms()
    adapter = arms["L8"]["adapter"]
    assert adapter, "final checkpoint missing"
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    held = [r for r in corpus["records"] if r["split"] == "heldout"]
    tok, model = _load(adapter)
    out = {}
    for arm in ("RAG0", "RAG1", "RAG2", "RAG3", "RAG4", "RAG5", "RAG6"):
        t0 = time.time()
        rows, uniq = [], set()
        for r in held:
            prompt = rag_transform(arm, r["prompt"])
            cands = [generate(model, tok, prompt, sample_seed=s)
                     for s in range(attempts)]
            best = next((c for c in cands if c["valid"]), None)
            if best:
                uniq.add(best["graph_hash"])
            rows.append({"valid1": bool(cands[0]["valid"]),
                         "match": bool(best and best["graph_hash"]
                                       == r["variant_hash"])})
        n = len(rows)
        out[arm] = {"first_attempt_validity":
                        round(sum(r["valid1"] for r in rows) / n, 3),
                    "structure_match":
                        round(sum(r["match"] for r in rows) / n, 3),
                    "unique_structures": len(uniq),
                    "inference_s": round(time.time() - t0, 1)}
        print(arm, json.dumps(out[arm]))
    del model
    torch.cuda.empty_cache()
    out["checkpoint"] = adapter
    (OUT / "RAG_BATTERY.json").write_text(json.dumps(out, indent=1),
                                          encoding="utf-8")
    return out


def _load(adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if torch.cuda.is_available()
        else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return tok, m


def eval_arm(name, cfg, held, attempts=4) -> dict:
    import torch
    if cfg["adapter"] is None and name not in ("L0", "L1"):
        return {"arm": name, "unavailable": "checkpoint missing",
                "desc": cfg["desc"]}
    tok, model = _load(cfg["adapter"])
    rows, uniq, resp = [], set(), {}
    t0 = time.time()
    for r in held:
        prompt = r["prompt"] if cfg["rag"] else _RAG_RE.sub("", r["prompt"])
        first_valid = None
        cands = []
        for s in range(attempts):
            c = generate(model, tok, prompt, sample_seed=s)
            cands.append(c)
            if c["valid"] and first_valid is None:
                first_valid = s
        best = next((c for c in cands if c["valid"]), None)
        row = {"context_id": r["context_id"],
               "first_attempt_valid": bool(cands[0]["valid"]),
               "attempts_to_valid": (first_valid + 1
                                     if first_valid is not None else None),
               "structure_match": bool(best and best["graph_hash"]
                                       == r["variant_hash"]),
               "topk_match": any(c["valid"] and c["graph_hash"]
                                 == r["variant_hash"] for c in cands),
               "stage_correct": bool(best and len(best["obj"]["stages"])
                                     == r["stages"]),
               "comp_correct": bool(best and ig.compensation_class(best["obj"])
                                    == {"miller_cap": "miller",
                                        "rc_nulling": "rc"}.get(r["comp"],
                                                                r["comp"]))}
        if best:
            uniq.add(best["graph_hash"])
            rh = ig.candidate_identity(best["obj"])["normalized_response_hash"]
            resp[rh] = resp.get(rh, 0) + 1
        rows.append(row)
    del model
    torch.cuda.empty_cache()
    n = len(rows)
    valid_n = sum(1 for r in rows if r["attempts_to_valid"])
    return {"arm": name, "desc": cfg["desc"], "adapter": cfg["adapter"],
            "rag": cfg["rag"], "contexts": n, "attempts_per_context": attempts,
            "first_attempt_validity":
                round(sum(r["first_attempt_valid"] for r in rows) / n, 3),
            "mean_attempts_to_valid": round(
                sum(r["attempts_to_valid"] or attempts + 1 for r in rows) / n,
                2),
            "structure_match": round(
                sum(r["structure_match"] for r in rows) / n, 3),
            "topk_structure_match": round(
                sum(r["topk_match"] for r in rows) / n, 3),
            "stage_accuracy": round(
                sum(r["stage_correct"] for r in rows) / n, 3),
            "comp_accuracy": round(
                sum(r["comp_correct"] for r in rows) / n, 3),
            "unique_structures": len(uniq),
            "top_response_share": round(
                max(resp.values()) / max(1, valid_n), 3) if resp else 0.0,
            "inference_s": round(time.time() - t0, 1), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--rag-battery", action="store_true")
    args = ap.parse_args()
    if args.rag_battery:
        OUT.mkdir(parents=True, exist_ok=True)
        run_rag_battery(args.attempts)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    arms = resolve_arms()
    wanted = args.arms.split(",") if args.arms else list(arms)
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    fe = json.loads((ROOT / "artifacts/stage3e4/frozen_exam.json").read_text())
    live = ig.build_exam_manifest(corpus)
    assert live["frozen_exam_hash"] == fe["frozen_exam_hash"], "exam drifted"
    held = [r for r in corpus["records"] if r["split"] == "heldout"]
    for name in wanted:
        f = OUT / f"{name}.json"
        if f.is_file():
            prev = json.loads(f.read_text())
            if "unavailable" not in prev:
                print(f"{name}: already done, skipping")
                continue
            print(f"{name}: was unavailable, retrying with current "
                  f"checkpoints")
        print(f"=== {name}: {arms[name]['desc']} ===")
        res = eval_arm(name, arms[name], held, args.attempts)
        f.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "rows"},
                         indent=1))
    agg = {n: {k: v for k, v in json.loads((OUT / f"{n}.json").read_text()
                                           ).items() if k != "rows"}
           for n in arms if (OUT / f"{n}.json").is_file()}
    (OUT / "SUMMARY.json").write_text(json.dumps(agg, indent=1),
                                      encoding="utf-8")
    print("\nsaved:", OUT / "SUMMARY.json")


if __name__ == "__main__":
    main()
