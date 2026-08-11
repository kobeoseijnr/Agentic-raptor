"""Stage 4 diagnostic: does exclusion conditioning create useful diversity,
or just force Qwen away from strong topologies?

Part A (cheap, no SPICE, no PUCT/DPO): a PER-ATTEMPT instrumented ladder --
not just propose_and_validate()'s final candidate set -- comparing
conditioning="exclusion" vs conditioning="temperature" (A4's actual
production values, verified against agentic_raptor/publication/ablation_v3.py
by audit_a4_config() below, printed every run). Every attempt (including
ones the production candidate set silently drops) is classified into
exactly one of:
  RUN_REHIT              -- a canonical graph already produced earlier in
                             THIS ladder run
  CORPUS_OR_MEMORY_REHIT  -- not a run-rehit, but already present in the
                             frozen structural corpus (corpus_diverse.json)
                             or the frozen corrected Stage-2 RAG memory
                             (agentic_raptor.publication.preflight.
                             CLEAN_RAG_PATH)
  NOVEL_CANONICAL_GRAPH   -- neither

Part B (real SPICE, small controlled subset): sizes the SAME candidates
from both pools under IDENTICAL conditions (same sizing method, budget,
seed policy) to check whether the diversity Part A measures actually
translates into different downstream feasibility/quality -- diversity that
never changes the sizing outcome is diversity in label only. Only run
AFTER reviewing Part A's output (Part A must be inspected first: an
implementation bug in the exclusion mechanism would otherwise waste real
SPICE budget measuring a mechanism that isn't doing what it claims to).

Run:  python run_stage4_diversity_diagnostic.py
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v3/stage4_diversity_diagnostic"
DEFAULT_ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"
BASE_CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"


# ---------------------------------------------------------------------------
# Section 2: A4 vs FULL configuration audit -- run BEFORE any generation.
# ---------------------------------------------------------------------------
def audit_a4_config() -> dict:
    """Confirms A4_NO_DIVERSITY differs from A0_FULL ONLY in
    use_exclusion_conditioning (-> conditioning="temperature" vs
    "exclusion"), by diffing their real to_run_pipeline_kwargs() and budget
    dataclasses field-by-field. Any other difference is reported as a
    discrepancy, not silently accepted."""
    from agentic_raptor.publication.ablation_v3 import A0_FULL, A4_NO_DIVERSITY
    kw0, kw4 = A0_FULL.to_run_pipeline_kwargs(), A4_NO_DIVERSITY.to_run_pipeline_kwargs()
    kwarg_diffs = {k: (kw0[k], kw4[k]) for k in kw0 if kw0[k] != kw4.get(k)}
    # both sides normalised through to_dict() (which itself asdict()s the
    # nested PvtConfig) -- comparing a dict to a raw PvtConfig object would
    # always be unequal even when every field matches, a false positive.
    bud0, bud4 = A0_FULL.budget.to_dict(), A4_NO_DIVERSITY.budget.to_dict()
    budget_diffs = {k: (v, bud4.get(k)) for k, v in bud0.items() if v != bud4.get(k)}
    expected = {"conditioning"}
    unexpected = (set(kwarg_diffs) - expected) | set(budget_diffs)
    return {
        "a0_kwargs": kw0, "a4_kwargs": kw4,
        "kwarg_diffs": kwarg_diffs, "budget_diffs": budget_diffs,
        "clean": not unexpected,
        "unexpected_diffs": sorted(unexpected),
        "note": ("A4 must differ from A0 ONLY in `conditioning` "
                "(exclusion vs temperature) -- both share target_k, "
                "the same DIVERSITY_LADDER temperature schedule, RAG, "
                "SFT adapter, model checkpoint, validator, and "
                "canonicalization by construction (propose_and_validate() "
                "dispatches on `conditioning` alone; everything else is a "
                "shared module-level constant or caller-supplied argument "
                "identical across both calls).")}


# ---------------------------------------------------------------------------
def graph_distance(obj_a: dict, obj_b: dict) -> int:
    """Every structural degree of freedom this schema can actually realise
    (agentic_raptor.llm_dpo.stage3e4.VARIANTS: stages(1-3) x
    compensation(none/miller/rc) x buffer(bool) x feedback(bool)) --
    confirmed exhaustive, not an incomplete proxy: there is no FIFTH
    independently-variable structural dimension anywhere else in the
    schema (current-mirror arrangement, gain-stage block choice, and
    connectivity are all determined BY these four for every realisable
    variant). canonicalize (variant_hash) already collapses device-order/
    serialization differences before this is ever called -- two objects
    reaching graph_distance already agree on everything canonicalization
    controls."""
    from agentic_raptor.llm_dpo.integrity import compensation_class
    sa, sb = len(obj_a.get("stages", [])), len(obj_b.get("stages", []))
    ca, cb = compensation_class(obj_a), compensation_class(obj_b)
    ba, bb = bool(obj_a.get("output_buffer")), bool(obj_b.get("output_buffer"))
    fa, fb = bool(obj_a.get("local_feedback")), bool(obj_b.get("local_feedback"))
    return int(sa != sb) + int(ca != cb) + int(ba != bb) + int(fa != fb)


def load_known_hashes(base_corpus: dict, rag_memory_path: Path) -> set:
    """The 'frozen known topology/RAG/corpus memory' Section 3B compares
    against: every canonical hash in the structural training corpus
    (corpus_diverse.json) union every canonical hash in the frozen
    corrected Stage-2 RAG memory (CLEAN_RAG_PATH). NOT self_improvement_
    runs.jsonl (archived, pre-fix) -- same exclusion this project's other
    SFT-facing code now applies everywhere."""
    known = {r.get("canonical_graph_hash") for r in base_corpus.get("records") or []
             if r.get("canonical_graph_hash")}
    if rag_memory_path.is_file():
        for line in rag_memory_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("variant"):
                known.add(e["variant"])
    return known


def family_pass_rates(rag_memory_path: Path) -> dict:
    """Historical measured pass rate per family from the SAME frozen RAG
    memory -- an already-existing, non-learned prior (Section 8: 'do not
    introduce a new learned metric'), used only to check whether exclusion
    systematically shifts generation toward historically weak families
    (Section 7)."""
    if not rag_memory_path.is_file():
        return {}
    n, passes = Counter(), Counter()
    for line in rag_memory_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        fam = e.get("family")
        if not fam:
            continue
        n[fam] += 1
        if e.get("exact_spec_pass") or e.get("postsizing"):
            passes[fam] += 1
    return {fam: round(passes[fam] / n[fam], 3) for fam in n}


# ---------------------------------------------------------------------------
# Section 3/5/11: per-attempt instrumented ladder (exclusion or
# temperature), classifying every attempt -- not just the final candidate
# set propose_and_validate() would report.
# ---------------------------------------------------------------------------
def instrumented_ladder_v2(model, tok, prompt: str, *, target_k: int, ladder,
                           seed0: int, conditioning: str,
                           known_hashes: set) -> dict:
    import torch

    from agentic_raptor.llm_dpo import integrity as ig
    from agentic_raptor.llm_dpo import parse_proposal_text, proposal_dict_valid
    from agentic_raptor.llm_dpo.stage3e4 import EXCLUDE_LINE, variant_hash
    from run_stage3_sft_diagnostic import TAXONOMY, classify

    seen: dict = {}          # canonical_hash -> accepted candidate, THIS run
    attempts_log = []
    attempts, rungs_used = 0, []
    for temp, top_p, n in ladder:
        used = False
        for _j in range(n):
            if len(seen) >= target_k:
                break
            used = True
            attempts += 1
            excluded_hashes_before = set(seen.keys())
            excluded_families_before = {c["family"] for c in seen.values()}
            if conditioning == "exclusion" and seen:
                ex = ", ".join(f"{c['family']}/{h[:12]}" for h, c in seen.items())
                cur = prompt.replace("### PROPOSAL", f"{EXCLUDE_LINE}{ex}\n### PROPOSAL")
            else:
                cur = prompt
            enc = tok(cur, return_tensors="pt").to(next(model.parameters()).device)
            ids = enc.input_ids
            torch.manual_seed(seed0 * 1000 + attempts)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=enc.attention_mask,
                                     max_new_tokens=260, do_sample=True,
                                     temperature=temp, top_p=top_p,
                                     pad_token_id=tok.eos_token_id)
            raw = tok.decode(out[0, ids.shape[1]:])
            obj = parse_proposal_text(raw)
            valid, reasons = (proposal_dict_valid(obj) if obj is not None
                              else (False, []))
            h = variant_hash(obj) if (obj is not None and valid) else None
            fam = None
            if obj is not None and valid:
                try:
                    fam = f"{len(obj['stages'])}s_" + ig.compensation_class(obj)
                except Exception:
                    fam = "unknown"
            run_rehit = bool(h and h in seen)
            corpus_or_memory_rehit = bool(h and not run_rehit and h in known_hashes)
            novel = bool(h and not run_rehit and not corpus_or_memory_rehit)
            cls = classify(raw, obj, valid, reasons, is_duplicate=run_rehit)
            violated_exclusion = bool(conditioning == "exclusion" and seen
                                      and h and h in excluded_hashes_before)
            switched_family = bool(fam and excluded_families_before
                                   and fam not in excluded_families_before)
            attempts_log.append({
                "attempt": attempts, "temperature": temp, "top_p": top_p,
                "prompt_word_count": len(cur.split()),
                "exclusion_applied": (conditioning == "exclusion" and bool(seen)),
                "excluded_hashes_before": sorted(excluded_hashes_before),
                "excluded_families_before": sorted(excluded_families_before),
                "raw_response": raw[:800], "parsed": obj is not None,
                "valid": bool(valid and not run_rehit),
                "validator_reasons": reasons, "graph_hash": h, "family": fam,
                "taxonomy_code": cls, "taxonomy_label": TAXONOMY[cls],
                "run_rehit": run_rehit,
                "corpus_or_memory_rehit": corpus_or_memory_rehit,
                "novel_canonical_graph": novel,
                "violated_exclusion": violated_exclusion,
                "switched_family": switched_family})
            if obj is not None and valid and not run_rehit:
                seen[h] = {"obj": obj, "family": fam, "temperature": temp,
                          "top_p": top_p, "canonical_graph_hash": h}
        if used:
            rungs_used.append({"temperature": temp, "top_p": top_p})
        if len(seen) >= target_k:
            break
    return {"target_k": target_k, "distinct": len(seen), "attempts": attempts,
           "rungs_used": rungs_used, "max_temperature":
               rungs_used[-1]["temperature"] if rungs_used else None,
           "candidates": [{"canonical_graph_hash": h, **v} for h, v in seen.items()],
           "attempts_log": attempts_log}


def _attempts_to_k(log: list, target_k: int) -> int | None:
    """The attempt number at which the cumulative count of DISTINCT valid
    canonical hashes first reaches target_k, or None if it never does."""
    seen = set()
    for a in log:
        if a["valid"] and a["graph_hash"]:
            seen.add(a["graph_hash"])
            if len(seen) >= target_k:
                return a["attempt"]
    return None


def measure_pool(ladder_out: dict, target_k: int) -> dict:
    """Section 4's full metric set, computed from the per-attempt log --
    not from the (already-deduplicated-by-construction) final candidate
    set alone, which is why canonical_duplicate_rate/RUN_REHIT rate were
    trivially 0 in the previous version of this script (propose_and_
    validate()'s own candidate list can never contain a duplicate; only
    the discarded attempts can)."""
    log = ladder_out["attempts_log"]
    attempts = ladder_out["attempts"]
    cands = ladder_out["candidates"]
    hashes = [c["canonical_graph_hash"] for c in cands]
    families = [c["family"] for c in cands]
    dists = [graph_distance(cands[i]["obj"], cands[j]["obj"])
            for i in range(len(cands)) for j in range(i + 1, len(cands))]
    n_parsed = sum(1 for a in log if a["parsed"])
    n_run_rehit = sum(1 for a in log if a["run_rehit"])
    n_corpus_rehit = sum(1 for a in log if a["corpus_or_memory_rehit"])
    n_novel = sum(1 for a in log if a["novel_canonical_graph"])
    attempts_to_k = _attempts_to_k(log, target_k)
    excl_applicable = [a for a in log if a["exclusion_applied"]]
    excl_compliant = [a for a in excl_applicable if not a["violated_exclusion"]]
    return {
        "attempts": attempts, "raw_successful_generations": len(cands),
        "schema_success_rate": n_parsed / max(1, attempts),
        "graph_construction_success_rate": sum(
            1 for a in log if a["taxonomy_code"] in ("G", "H", "I")) / max(1, attempts),
        "validator_success_rate": sum(
            1 for a in log if a["taxonomy_code"] in ("H", "I")) / max(1, attempts),
        "valid_at_k": len(cands) / target_k,
        "unique_at_k": len(set(hashes)) / max(1, len(hashes)),
        "run_rehit_count": n_run_rehit, "run_rehit_rate": n_run_rehit / max(1, attempts),
        "corpus_or_memory_rehit_count": n_corpus_rehit,
        "corpus_or_memory_rehit_rate": n_corpus_rehit / max(1, attempts),
        "novel_canonical_graph_count": n_novel,
        "novel_valid_yield": n_novel / max(1, attempts),
        "unique_yield": len(cands) / max(1, attempts),
        "attempts_to_k_unique_valid": attempts_to_k,
        "family_diversity_count": len(set(families)), "families": families,
        "family_distribution": dict(Counter(families)),
        "mean_pairwise_graph_distance": (sum(dists) / len(dists)) if dists else None,
        "canonical_duplicate_rate": n_run_rehit / max(1, n_run_rehit + len(cands)),
        "max_temperature_reached": ladder_out["max_temperature"],
        "exclusion_compliance_rate": (
            len(excl_compliant) / len(excl_applicable) if excl_applicable else None),
        "exclusion_applicable_attempts": len(excl_applicable),
        "family_switch_rate": (
            sum(1 for a in excl_applicable if a["switched_family"]) / len(excl_applicable)
            if excl_applicable else None),
        "taxonomy_counts": dict(Counter(a["taxonomy_label"] for a in log)),
        "max_prompt_word_count": max((a["prompt_word_count"] for a in log), default=None),
        "candidates": cands}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=int, default=6)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--split", default="train",
                    help="train (default) -- heldout has almost no "
                         "unprotected specs left (27/29 consumed by the "
                         "frozen evaluation set), confirmed in Stage 2")
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--target-k", type=int, default=5)
    ap.add_argument("--size-subset", type=int, default=0,
                    help="how many (spec,seed) pairs get Part B real sizing "
                         "-- kept at 0 by default; review Part A's output "
                         "first, then re-run with this set once a "
                         "meaningful diversity difference is confirmed "
                         "(Section 9)")
    ap.add_argument("--sizing-budget", type=int, default=16)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # ---- Section 2: A4 config audit, BEFORE any generation -------------
    audit = audit_a4_config()
    print("=== SECTION 2: A4 vs A0_FULL CONFIG AUDIT ===")
    print(json.dumps(audit, indent=1, default=str))
    if not audit["clean"]:
        print(f"\n!!! DISCREPANCY FOUND: {audit['unexpected_diffs']} !!!\n"
             "Stopping -- A4 is not a fair isolated ablation as configured. "
             "Fix agentic_raptor/publication/ablation_v3.py before "
             "proceeding with this diagnostic.")
        return
    print("A4 configuration verified fair: differs from A0_FULL only in "
         "`conditioning` (exclusion vs temperature).\n")

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
    print(f"diagnostic specs ({len(diag)}, unprotected, split={args.split!r}):")
    for e in diag:
        print(f"  idx={e['spec_index']:3} tier={e['difficulty_tier']:6} "
             f"hash={e['spec_hash']} {e['context_id']}")
    print()

    base_corpus = json.loads(BASE_CORPUS.read_text(encoding="utf-8"))
    known_hashes = load_known_hashes(base_corpus, CLEAN_RAG_PATH)
    fam_pass = family_pass_rates(CLEAN_RAG_PATH)
    print(f"known (corpus+RAG) canonical hashes: {len(known_hashes)}")
    print(f"historical family pass rates (from frozen RAG memory): "
         f"{fam_pass}\n")

    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    pool_recs = [r for r in corpus["records"] if r["split"] == args.split]

    tok, model = _load(str(args.adapter))
    print("model loaded\n", flush=True)

    from agentic_raptor.llm_dpo.stage3e4 import DIVERSITY_LADDER

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for e in diag:
        idx = e["spec_index"]
        rec = pool_recs[idx]
        spec = dict(e["parsed_spec"], spec_id=e["context_id"],
                   spec_hash=e["spec_hash"], topology_id=rec.get("topology_id"))
        rag = rag_stage(spec, rec["prompt"], use_rag=True, memory_path=str(CLEAN_RAG_PATH))
        for seed in seeds:
            t0 = time.time()
            excl = instrumented_ladder_v2(
                model, tok, rag["prompt"], target_k=args.target_k,
                ladder=DIVERSITY_LADDER, seed0=seed, conditioning="exclusion",
                known_hashes=known_hashes)
            temp = instrumented_ladder_v2(
                model, tok, rag["prompt"], target_k=args.target_k,
                ladder=DIVERSITY_LADDER, seed0=seed, conditioning="temperature",
                known_hashes=known_hashes)
            m_excl, m_temp = measure_pool(excl, args.target_k), measure_pool(temp, args.target_k)
            row = {"spec_index": idx, "spec_hash": e["spec_hash"],
                  "difficulty_tier": e["difficulty_tier"], "seed": seed,
                  "exclusion": m_excl, "temperature": m_temp,
                  "elapsed_s": round(time.time() - t0, 1)}
            results.append(row)
            (OUT / "results_partA.jsonl").open("a", encoding="utf-8").write(
                json.dumps(row, default=str) + "\n")
            # Section 5: every attempt's raw record (exclusion prompt/
            # context sent, resulting hash, violation/switch flags) --
            # measure_pool() only consumes this to compute aggregates and
            # never persisted it; kept in a separate file (not inlined
            # into results_partA.jsonl) since each entry carries an
            # 800-char raw_response and there are ~20-40 attempts per
            # pair per condition.
            for label, ladder_out in (("exclusion", excl), ("temperature", temp)):
                with (OUT / "results_partA_attempts_log.jsonl").open(
                        "a", encoding="utf-8") as f:
                    for a in ladder_out["attempts_log"]:
                        f.write(json.dumps({
                            "spec_index": idx, "spec_hash": e["spec_hash"],
                            "seed": seed, "conditioning": label, **a},
                            default=str) + "\n")
            print(f"idx={idx:3} tier={e['difficulty_tier']:6} seed={seed} "
                 f"EXCL(distinct={m_excl['raw_successful_generations']},"
                 f"fam={m_excl['family_diversity_count']},"
                 f"run_rehit={m_excl['run_rehit_rate']:.2f},"
                 f"novel_yield={m_excl['novel_valid_yield']:.2f},"
                 f"compliance={m_excl['exclusion_compliance_rate']}) "
                 f"TEMP(distinct={m_temp['raw_successful_generations']},"
                 f"fam={m_temp['family_diversity_count']},"
                 f"run_rehit={m_temp['run_rehit_rate']:.2f},"
                 f"novel_yield={m_temp['novel_valid_yield']:.2f}) "
                 f"{row['elapsed_s']:.0f}s", flush=True)

    print(f"\nPart A done: {len(results)} paired comparisons -> "
         f"{OUT / 'results_partA.jsonl'}")

    n = max(1, len(results))
    for label in ("exclusion", "temperature"):
        mean_rehit = sum(r[label]["run_rehit_rate"] for r in results) / n
        mean_novel_yield = sum(r[label]["novel_valid_yield"] for r in results) / n
        print(f"{label}: mean RUN_REHIT rate = {mean_rehit:.3f} "
             f"({mean_rehit*100:.1f}%), mean novel_valid_yield = "
             f"{mean_novel_yield:.3f}")

    # ---- Part B: real sizing on a small controlled subset ------------------
    if args.size_subset > 0:
        from agentic_raptor.electrical import discover_ngspice
        from agentic_raptor.mb_sac.spec_sizing import sac_size
        from agentic_raptor.topology_rl.stage3e2 import new_costs
        from run_puct_ablation import _realise
        exe = discover_ngspice()
        subset = results[:args.size_subset]
        sizing_out = []
        for row in subset:
            idx, seed = row["spec_index"], row["seed"]
            spec = dict([e for e in diag if e["spec_index"] == idx][0]["parsed_spec"],
                       spec_id=[e for e in diag if e["spec_index"] == idx][0]["context_id"])
            for pool_name in ("exclusion", "temperature"):
                cands = row[pool_name]["candidates"]
                for rank, c in enumerate(cands[:2]):
                    g = _realise(c["obj"])
                    tid = f"s4_{pool_name}_{idx}_{seed}_{rank}"
                    sz = sac_size(tid, g, spec, exe,
                                 OUT / "sizing", new_costs(),
                                 budget=args.sizing_budget, seed=17,
                                 persist=False)
                    sizing_out.append({
                        "spec_index": idx, "seed": seed, "pool": pool_name,
                        "rank": rank, "canonical_family": c["family"],
                        "exact_spec_pass": sz["outcome"]["exact_spec_pass"],
                        "distance": sz["outcome"]["normalized_distance_to_feasibility"],
                        "gain_db": sz["best"]["gain_db"], "pm_deg": sz["best"]["pm_deg"],
                        "ugbw_hz": sz["best"]["ugbw_hz"],
                        "c_load_f": sz["best"]["c_load_f"],
                        "spice_calls": sz["spice_calls"]})
                    print(f"  sized {pool_name} idx={idx} seed={seed} rank={rank}: "
                         f"pass={sz['outcome']['exact_spec_pass']} "
                         f"dist={sz['outcome']['normalized_distance_to_feasibility']}",
                         flush=True)
        (OUT / "results_partB_sizing.json").write_text(
            json.dumps(sizing_out, indent=1, default=str), encoding="utf-8")
        print(f"\nPart B done: {len(sizing_out)} real sizing runs -> "
             f"{OUT / 'results_partB_sizing.json'}")
    else:
        print("\nPart B skipped (--size-subset 0, the default) -- review "
             "Part A's output above, then re-run with --size-subset N once "
             "a meaningful diversity difference is confirmed.")


if __name__ == "__main__":
    main()
