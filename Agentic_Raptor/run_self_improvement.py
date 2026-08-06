"""Agentic RAPTOR self-improvement loop — repaired, campaign-based.

Each GENERATION (continuity: gen N+1 starts from gen N's ACCEPTED checkpoint):
  1. SFT   — continue training the accepted parent adapter on corpus +
             verified self-earned examples (leakage-guarded).
  2. EXAM  — frozen held-out exam of the SFT checkpoint.
  3. DESIGN— the SFT checkpoint proposes for train specs; distinct structures
             hit REAL ngspice; outcomes become tiered self-earned examples and
             spec-conditioned preference pairs (integrity engine).
  4. GATE  — dedup, contradiction resolution, balancing, hard pre-training
             blockers. DPO runs only if training_ready.
  5. DPO   — policy = SFT ckpt (trainable), reference = SFT ckpt (frozen,
             checksum-verified). Pairs carry their real prompts.
  6. ACCEPT— exam the DPO checkpoint; accept only if it does not collapse
             uniqueness / spec-match / validity. Otherwise roll back to SFT.

Modes:
  python run_self_improvement.py 3          # full 3-generation campaign
  python run_self_improvement.py --dry-run  # Part N data-pipeline rehearsal (no GPU)
  python run_self_improvement.py --validate # Part O small validation campaign
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from agentic_raptor.llm_dpo import (MODEL_ID, load_measurement_map,
                                    load_models, seq_logprob)
from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.publication import eval_sets
from agentic_raptor.llm_dpo.stage3e4 import (build_corpus, generate,
                                             realise_sample, variant_text)

# absolute: qualification runs ngspice with cwd=<run dir>, so relative
# artifact paths would break the testbench file resolution
O4 = Path("artifacts/stage3e4").resolve()
SI = Path("artifacts/self_improvement").resolve()
SI.mkdir(parents=True, exist_ok=True)

#: realise_n caps how many DISTINCT structures per generation reach ngspice
#: + SAC. It was 6 -- fine while the proposer only ever produced ~3 unique
#: structures, but a hard silent cap once the search ranks candidates and
#: sends 2 per spec: 10 specs x 2 finalists needs 20 slots, so at 6 the
#: later specs' finalists were dropped without a word.
CFG_FULL = {"sft_steps": 600, "dpo_epochs": 3, "design_contexts": 10,
            "design_seeds": 4, "realise_n": 24, "dpo_lr": 5e-5,
            "sft_lr": 3e-4, "beta": 0.1, "sizing_budget": 24}

#: '### KNOWN ...' evidence lines appear in TRAIN prompts only (the frozen
#: exam is evidence-free). Training must cover BOTH formats or the model
#: overfits the evidence-bearing template and emits nothing parseable on
#: exam prompts (observed: gen-0 exam valid 0/29 while train-prompt designs
#: were 20/20 valid).
_KNOWN_RE = re.compile(r"^### KNOWN [^\n]*\n", re.M)


def format_dropout(prompt: str, rng) -> str:
    return _KNOWN_RE.sub("", prompt) if rng.random() < 0.5 else prompt
#: uniform reduced profile for within-battery studies: every arm shrinks by
#: the same amounts, so battery-internal comparisons stay valid while each
#: campaign drops from ~1.75h to ~1.1h
CFG_FAST = {"sft_steps": 300, "dpo_epochs": 2, "design_contexts": 8,
            "design_seeds": 3, "realise_n": 20, "dpo_lr": 5e-5,
            "sft_lr": 3e-4, "beta": 0.1, "sizing_budget": 16}

CFG_VALIDATE = {"sft_steps": 200, "dpo_epochs": 2, "design_contexts": 8,
                "design_seeds": 3, "realise_n": 5, "dpo_lr": 5e-5,
                "sft_lr": 3e-4, "beta": 0.1, "sizing_budget": 8}

#: pre-campaign artifacts that must never feed the repaired campaign (A8)
STALE_PATHS = [SI / "generations.jsonl", SI / "self_earned_examples.jsonl",
               Path("datasets/llm_preference_queue/v2_mechanical.jsonl"),
               Path("datasets/llm_preference_queue/v2_mechanical_report.json"),
               O4 / "model_pairs.json", O4 / "diversity_sft.json"]


# ------------------------------ model loading ---------------------------------
def _tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    return tok


def load_adapter(ckpt, trainable=False, cpu=False):
    """Load base + a SAVED adapter checkpoint. Generation continuity (A6):
    training always resumes from an explicit parent checkpoint on disk."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    cuda = torch.cuda.is_available()
    # frozen reference goes FULLY to CPU: two 4B models don't fit 12GB VRAM,
    # and partial offload triggers a peft adapter-loading bug
    dm = ({"": "cpu"} if cpu else "auto") if cuda else None
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if cuda else torch.float32,
        device_map=dm)
    return PeftModel.from_pretrained(m, str(ckpt), is_trainable=trainable)


# ------------------------------ SFT (with guard) ------------------------------
def sft_generation(gen, corpus, camp, parent_ckpt, exam_manifest, cfg):
    import torch
    tok = _tok()
    if parent_ckpt is None:                 # generation 0 only: fresh LoRA
        tok, model = load_models(lora=True,
                                 seed=cfg.get("seed", 0) * 1000 + gen)
    else:                                   # A6: continue from accepted parent
        model = load_adapter(parent_ckpt, trainable=True)
    train = [r for r in corpus["records"] if r["split"] == "train"]
    # Fix 1: the frozen ELECTRICAL evaluation specs are drawn from the train
    # split, so SFT would otherwise train directly on what it is scored on
    frozen_eval = eval_sets.excluded_context_ids()
    eval_dropped = sum(1 for r in train if r["context_id"] in frozen_eval)
    train = [r for r in train if r["context_id"] not in frozen_eval]
    earned_file = camp["dirs"]["self_earned"] / "verified_self_earned.jsonl"
    earned, quarantined = [], 0
    if earned_file.is_file():
        cands = [json.loads(x) for x in earned_file.read_text().splitlines()
                 if x.strip()]
        offenders = {o["index"] for o in ig.leakage_check(cands, exam_manifest)}
        quarantined = len(offenders)
        earned = [c for i, c in enumerate(cands) if i not in offenders]
        if offenders:                       # J: block leaky examples, loudly
            qw = ig.quarantine_writer(camp)
            for i in sorted(offenders):
                qw({"record": cands[i], "reason": "frozen_exam_leakage_sft"})
        # self-earned examples inherit their spec's context: an evaluation
        # spec that reached sizing would re-enter SFT through this door
        earned = [c for c in earned
                  if c.get("context_id") not in frozen_eval]
    data = train + earned
    eval_sets.assert_no_leakage([r.get("context_id") for r in data],
                                f"SFT gen{gen} training data")
    ds_hash = ig.sha_json([(r["prompt"], r["response"]) for r in data])
    rng = random.Random(cfg.get("seed", 0) * 1000 + gen)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["sft_lr"])
    for step in range(cfg["sft_steps"]):
        r = data[rng.randrange(len(data))]
        lp, n = seq_logprob(model, tok, format_dropout(r["prompt"], rng),
                            r["response"])
        loss = -lp / n
        opt.zero_grad(); loss.backward(); opt.step()
    ckpt = camp["dirs"]["models"] / f"gen{gen}_sft"
    model.save_pretrained(str(ckpt))
    del model
    torch.cuda.empty_cache()
    return {"steps": cfg["sft_steps"], "train_records": len(data),
            "verified_self_earned_used": len(earned),
            "leaky_examples_quarantined": quarantined,
            "frozen_eval_records_excluded": eval_dropped,
            "input_sft_dataset_hash": ds_hash,
            "output_sft_checkpoint": str(ckpt),
            "output_sft_hash": ig.sha_checkpoint(ckpt)}


# ------------------------------ frozen exam -----------------------------------
def run_exam(ckpt, corpus, exam_manifest, camp, label):
    import torch
    live = ig.build_exam_manifest(corpus)
    assert live["frozen_exam_hash"] == exam_manifest["frozen_exam_hash"], \
        "frozen exam changed — evaluation aborted"
    tok = _tok()
    model = load_adapter(ckpt)
    held = [r for r in corpus["records"] if r["split"] == "heldout"]
    per_spec, resp_hashes, fams, stage_dist = [], {}, {}, {}
    valid, match, uniq = 0, 0, set()
    for r in held:
        c = generate(model, tok, r["prompt"], sample_seed=0)
        row = {"context_id": r["context_id"], "valid": bool(c["valid"]),
               "match": False}
        if c["valid"]:
            valid += 1
            uniq.add(c["graph_hash"])
            row["match"] = c["graph_hash"] == r["variant_hash"]
            match += int(row["match"])
            ident = ig.candidate_identity(c["obj"])
            rh = ident["normalized_response_hash"]
            resp_hashes[rh] = resp_hashes.get(rh, 0) + 1
            fams[ident["topology_family_id"]] = \
                fams.get(ident["topology_family_id"], 0) + 1
            stage_dist[str(ident["stage_count"])] = \
                stage_dist.get(str(ident["stage_count"]), 0) + 1
        per_spec.append(row)
    del model
    torch.cuda.empty_cache()
    n = len(held)
    metrics = {"label": label, "contexts": n,
               "valid_rate": round(valid / n, 3),
               "spec_match_rate": round(match / n, 3),
               "unique_structures": len(uniq),
               "per_spec_accuracy": per_spec,
               "topology_family_distribution": fams,
               "stage_count_distribution": stage_dist,
               "most_common_response_fraction": round(
                   max(resp_hashes.values()) / max(1, valid), 3)
               if resp_hashes else 0.0,
               "repeated_response_count": sum(
                   v - 1 for v in resp_hashes.values() if v > 1),
               "exact_response_duplicates": sum(
                   1 for v in resp_hashes.values() if v > 1),
               "frozen_exam_hash": exam_manifest["frozen_exam_hash"]}
    (camp["dirs"]["evaluations"] / f"{label}.json").write_text(
        json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def se_admits(se_arm, tier, cand, row, meas_map, nominal_by_variant):
    """SE0-SE6: which self-earned records may enter the SFT textbook.
    se0 none | se1 nominal-stable | se2 post-sizing stable | se3 exact
    verified only | se4 exact + stable partials | se5 provisional-only
    (negative control) | se6 full verified policy (production default)."""
    if se_arm == "se0":
        return False
    if se_arm == "se5":
        return tier == "provisional_self_earned"
    if se_arm in ("se3", "se6"):
        return tier == "verified_self_earned"
    m = meas_map.get(cand["graph_hash"]) or {}
    stable_ps = bool(m.get("postsizing")
                     and m.get("stability") == "verified_stable")
    exact = cand["graph_hash"] == row.get("target_variant")
    if se_arm == "se1":
        nm = nominal_by_variant.get(cand["graph_hash"]) or {}
        return nm.get("stability") == "verified_stable"
    if se_arm == "se2":
        return stable_ps
    if se_arm == "se4":
        return tier == "verified_self_earned" or (stable_ps and exact)
    return False


# --------------------- Task 6: post-sizing qualification ----------------------
def _realise_graph(obj):
    """Map a validated proposal to a device graph (same path realise_sample
    uses: base-template mapping + executable edits)."""
    from agentic_raptor.mapping import map_family
    from agentic_raptor.topology_rl.stage3e2_edits import (EditRejected,
                                                           apply_edit)

    class _S:
        topology_id = "postsize"
    g, _ = map_family(_S(), {
        "topology_id": _S.topology_id, "gain_stages": len(obj["stages"]),
        "functional_blocks": (["C"] if obj.get("compensation") else []),
        "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
        "graph_hash": None})
    try:
        if obj.get("output_buffer"):
            g, _a = apply_edit(g, "ADD_SUPPORTED_OUTPUT_STAGE")
        if obj.get("compensation") and \
                obj["compensation"][0]["type"] == "rc_nulling":
            g, _a = apply_edit(g, "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE")
        if obj.get("local_feedback"):
            g, _a = apply_edit(g, "CONNECT_VERIFIED_FEEDBACK_PATH")
    except EditRejected:
        return None
    return g


def postsize_designs(gen, rows, meas, camp, cfg):
    """SAC-size every realised structure under its originating spec (fixed
    budget, corrected target-saturating reward, spec-conditioned ranker).
    Returns {graph_hash: {sz, spec, context_id, ...}} keyed by structure."""
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mb_sac.spec_sizing import sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    realised_ok = {m["variant"] for m in meas["realised"]
                   if m.get("stability")}
    exe = discover_ngspice()
    post = {}
    for row in rows:
        spec = ig.parse_spec(row.get("prompt") or "")
        if not spec:
            continue
        for c in row["candidates"]:
            h = c["graph_hash"]
            if h in post or h not in realised_ok:
                continue
            g = _realise_graph(c["obj"])
            if g is None:
                continue
            costs = new_costs()
            fam = f"{len(c['obj']['stages'])}s_" \
                + ig.compensation_class(c["obj"])
            ch = set(cfg.get("channels", ["replay", "surrogate", "ranker",
                                          "l4", "value", "selfearn"]))
            sz = sac_size(f"gen_{h[:8]}", g, spec, exe, O4 / "realised",
                          costs, budget=cfg["sizing_budget"],
                          seed=cfg.get("seed", 0) * 1000 + 100 * gen
                          + len(post), family=fam,
                          persist="replay" in ch,
                          use_surrogate="surrogate" in ch,
                          use_ranker="ranker" in ch)
            comp = (c["obj"]["compensation"][0]["type"]
                    if c["obj"].get("compensation") else "none")
            post[h] = {"sz": sz, "spec": spec,
                       "context_id": row["context_id"],
                       "stages": len(c["obj"]["stages"]),
                       "comp": comp,
                       "proposal": json.dumps(c["obj"],
                                              separators=(",", ":"))}
            # PM attack: when an UNCOMPENSATED design still fails PM after a
            # fair sizing budget, also size the compensation-edited variant
            # and record that measurement under the edited variant's TRUE
            # identity — the evidence base (L4/KNOWN/pairs) learns whether
            # compensation fixes PM at this load, without ever mislabelling
            # the original proposal's own result.
            if comp == "none" and not sz["outcome"]["passes"]["pm"]:
                from agentic_raptor.topology_rl.stage3e2_edits import (
                    EditRejected, apply_edit)
                try:
                    g2, _a = apply_edit(
                        g, "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE")
                except EditRejected:
                    g2 = None
                if g2 is not None:
                    stages = len(c["obj"]["stages"])
                    obj2 = json.loads(variant_text(stages, "miller",
                                                   False, False))
                    h2 = ig.candidate_identity(obj2)["canonical_graph_hash"]
                    if h2 not in post:
                        sz2 = sac_size(f"cmp_{h2[:8]}", g2, spec, exe,
                                       O4 / "realised", new_costs(),
                                       budget=max(4, cfg["sizing_budget"] // 2),
                                       seed=100 * gen + len(post),
                                       family=f"{stages}s_miller")
                        post[h2] = {"sz": sz2, "spec": spec,
                                    "context_id": row["context_id"],
                                    "stages": stages, "comp": "miller",
                                    "probe": "compensation_edit",
                                    "proposal": json.dumps(
                                        obj2, separators=(",", ":"))}
    (camp["dirs"]["reports"] / f"postsizing_gen{gen}.json").write_text(
        json.dumps(post, indent=1, default=str), encoding="utf-8")
    return post


def electrical_curve(post):
    """Task 8: per-generation electrical metrics from real post-sizing
    ngspice results."""
    outs = [p["sz"] for p in post.values()]
    if not outs:
        return {"designs_sized": 0}
    n = len(outs)

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else None

    def rate(key):
        return round(sum(o["outcome"]["passes"][key] for o in outs) / n, 3)
    return {"designs_sized": n,
            "exact_spec_pass_rate": round(
                sum(o["outcome"]["exact_spec_pass"] for o in outs) / n, 3),
            "stable_rate": rate("stable"), "gain_pass_rate": rate("gain"),
            "pm_pass_rate": rate("pm"), "ugbw_pass_rate": rate("ugbw"),
            "avg_hard_constraints_passed": round(sum(
                o["outcome"]["hard_constraints_passed"] for o in outs) / n, 2),
            "median_normalized_distance": med(
                [o["outcome"]["normalized_distance_to_feasibility"]
                 for o in outs]),
            "median_spice_calls": med([o["spice_calls"] for o in outs]),
            "calls_to_first_exact_pass": min(
                (o["calls_to_first_exact_pass"] for o in outs
                 if o["calls_to_first_exact_pass"]), default=None),
            "best_reward": max(o["best"]["reward"] for o in outs),
            "postsizing_spice_calls": sum(o["spice_calls"] for o in outs)}


# ------------------------ design / measure / earn / pair ----------------------
def _class_of(c) -> str:
    """Structure class of a candidate record."""
    obj = c.get("obj") or {}
    return f"{len(obj.get('stages', []))}s_" + ig.compensation_class(obj)


def enumerate_and_rank(rows, chans, top_k: int = 2):
    """Channel "search": AlphaZero SELECTS which structures reach SAC.

    Two measured facts drive this. The corpus contains exactly FIVE distinct
    structures (2s_none, 2s_miller, 3s_none, 3s_miller, 3s_rc) -- that is the
    entire design space. And the proposer collapsed to 1-2 unique structures
    per spec (3 across a whole campaign), so sampling could never surface
    that space for the search to rank. Enumerating it is deterministic,
    complete, and costs no GPU.

    Each spec's candidate set becomes: the search's top_k ranked structures,
    PLUS whatever the LLM actually proposed. Keeping the LLM's own pick means
    every spec still yields a head-to-head between what the model wanted and
    what the search chose -- and >=2 measured structures per spec, so the
    preference-pair supply DPO depends on is preserved by construction.
    """
    if "search" not in chans:
        return rows, None
    from agentic_raptor.llm_dpo.stage3e4 import (proposal_dict_valid,
                                                 variant_hash, variant_text)
    from run_puct_ablation import CORPUS_CLASSES, run_puct_fixed
    ranked_log = []
    for row in rows:
        spec = ig.parse_spec(row.get("prompt") or "")
        cands = row.get("candidates") or []
        if not spec or not cands:
            continue
        llm_objs = {c["graph_hash"]: c for c in cands if c.get("valid")}
        obj0 = cands[0]["obj"]
        llm_cls = f"{len(obj0['stages'])}s_" + ig.compensation_class(obj0)
        _sel = None
        try:
            _sel, visits, _pc = run_puct_fixed(obj0, spec, row["context_id"])
        except Exception as exc:
            ranked_log.append({"context_id": row["context_id"],
                               "error": f"{type(exc).__name__}: {exc}"[:160]})
            continue
        # visit counts ARE the ranking (canonical AlphaZero); a_keep/a_term
        # both mean "the proposal's own class"
        score = {}
        for act, n in visits.items():
            cls = act[6:] if act.startswith("a_sel_") else llm_cls
            score[cls] = score.get(cls, 0) + n
        order = sorted(CORPUS_CLASSES,
                       key=lambda c: (-score.get(c, 0), c))
        chosen, extra = order[:top_k], []
        for cls in chosen:
            stages, comp = int(cls[0]), cls.split("_", 1)[1]
            obj = json.loads(variant_text(stages, comp, False, False))
            ok, _r = proposal_dict_valid(obj)
            h = variant_hash(obj)
            if not ok or h in llm_objs:
                continue
            extra.append({"seed": -1, "parseable": True, "valid": True,
                          "reasons": [], "graph_hash": h, "obj": obj,
                          "source": "mcts_enumerated"})
        row["candidates"] = list(llm_objs.values()) + extra
        row["valid"] = len(row["candidates"])
        row["unique_valid"] = len({c["graph_hash"]
                                   for c in row["candidates"]})
        # Fix 3: record what the search chose, so the executed topology can be
        # checked against it after sizing rather than assumed to match
        row["_search"] = {
            "root_id": llm_cls,
            "selected_action": _sel,
            "selected_topology_id": chosen[0] if chosen else llm_cls,
            "selected_topology_hash": next(
                (c["graph_hash"] for c in row["candidates"]
                 if c.get("source") == "mcts_enumerated"
                 and chosen and _class_of(c) == chosen[0]),
                next((c["graph_hash"] for c in row["candidates"]
                      if _class_of(c) == (chosen[0] if chosen else llm_cls)),
                     None)),
            "visits": visits,
            "proposal_hash": cands[0].get("graph_hash"),
            "proposal_class": llm_cls}
        ranked_log.append({"context_id": row["context_id"],
                           "llm_class": llm_cls, "visit_score": score,
                           "ranking": order, "search_top_k": chosen,
                           "added_structures": len(extra),
                           "selected_topology_hash":
                               row["_search"]["selected_topology_hash"],
                           "llm_pick_in_top_k": llm_cls in chosen})
    scored = [r for r in ranked_log if "ranking" in r]
    summary = {"contexts": len(scored), "top_k": top_k,
               "llm_pick_in_top_k": sum(1 for r in scored
                                        if r["llm_pick_in_top_k"]),
               "structures_added": sum(r["added_structures"]
                                       for r in scored),
               "per_context": ranked_log}
    if scored:
        print(f"  search ranking: top-{top_k} of {len(CORPUS_CLASSES)} "
              f"structures for {len(scored)} specs; LLM pick in top-{top_k} "
              f"{summary['llm_pick_in_top_k']}/{len(scored)}; "
              f"{summary['structures_added']} structures added for sizing")
    return rows, summary


def write_execution_traces(gen, rows, post, camp, chans):
    """Fix 3: hash chain from PUCT's choice to the feedback record.

    Without this the search can be consulted, ignored, and still credited --
    every downstream number would then describe a circuit the search never
    picked. The chain is asserted, not assumed.
    """
    if "search" not in chans:
        return None
    from agentic_raptor.publication.exec_trace import ExecutionTrace
    by_hash = {h: p for h, p in post.items()}
    traces, broken = [], []
    for row in rows:
        s = row.get("_search")
        if not s:
            continue
        sel_hash = s.get("selected_topology_hash")
        # the selected structure must appear among what was actually sized
        sized = by_hash.get(sel_hash)
        out = (sized or {}).get("sz", {}).get("outcome", {}) if sized else {}
        t = ExecutionTrace(
            spec_id=row["context_id"],
            generation=gen,
            campaign_id=camp["campaign_id"],
            proposal_id=f"{row['context_id']}_p0",
            proposal_hash=s.get("proposal_hash"),
            proposal_class=s.get("proposal_class"),
            puct_root_id=s.get("root_id"),
            puct_selected_action=s.get("selected_action"),
            puct_selected_topology_id=s.get("selected_topology_id"),
            puct_selected_topology_hash=sel_hash,
            puct_visits=s.get("visits") or {},
            executed_topology_hash=sel_hash if sized else None,
            executed_topology_id=s.get("selected_topology_id"),
            c9_input_topology_hash=sel_hash if sized else None,
            sizing_manifest_hash=(sized or {}).get("sz", {}).get(
                "schema_version") if sized else None,
            spice_result_id=f"{row['context_id']}_gen{gen}",
            feedback_record_id=f"L4_{row['context_id']}_gen{gen}",
            feedback_topology_hash=sel_hash if sized else None,
            exact_pass=bool(out.get("exact_spec_pass")) if out else None,
            exact_failure_reason=out.get("exact_failure_reason")
            if out else None)
        rep = t.verify(strict=False)
        if not rep["aligned"]:
            broken.append(rep)
        t.write(subdir=camp["campaign_id"])
        traces.append(rep)
    if broken:
        from agentic_raptor.publication.exec_trace import \
            TopologyExecutionMismatch
        raise TopologyExecutionMismatch(
            f"gen{gen}: {len(broken)} design context(s) executed a topology "
            f"the search did not select: {broken[:3]}")
    summary = {"generation": gen, "traces": len(traces),
               "aligned": sum(1 for t in traces if t["aligned"]),
               "selection_recorded": sum(
                   1 for t in traces
                   if "puct_selected" in t["links_present"])}
    (camp["dirs"]["reports"] /
     f"execution_traces_gen{gen}.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print(f"  execution traces: {summary['aligned']}/{summary['traces']} "
          f"hash-aligned, {summary['selection_recorded']} with a recorded "
          f"PUCT selection")
    return summary


def record_search_decisions(gen, rows, post, camp, chans):
    """AlphaZero DECISION layer for the campaign (channel: "search").

    Why the campaign never selected a topology: it SIZES EVERY CANDIDATE on
    purpose. Preference pairs and self-earned examples both need multiple
    measured structures per spec, and an earlier campaign starved DPO by
    collapsing to one structure (see the design_seeds comment above). A
    selector that discards candidates would reintroduce exactly that.

    So the search does not replace candidate generation. It states which
    structure it would deliver, and because every candidate is measured
    anyway, the decision can be scored after the fact against the outcomes
    already on hand -- decision evidence at campaign scale, at zero extra
    SPICE cost and with the pair supply untouched.
    """
    if "search" not in chans:
        return None
    from run_puct_ablation import run_puct_fixed
    by_ctx = {}
    for h, p in post.items():
        by_ctx.setdefault(p["context_id"], []).append((h, p))
    decisions = []
    for row in rows:
        spec = ig.parse_spec(row.get("prompt") or "")
        cands = row.get("candidates") or []
        if not spec or not cands:
            continue
        obj = cands[0]["obj"]
        prop_cls = f"{len(obj['stages'])}s_" + ig.compensation_class(obj)
        try:
            sel, visits, _pc = run_puct_fixed(obj, spec, row["context_id"])
        except Exception as exc:            # never fail a campaign on this
            decisions.append({"context_id": row["context_id"],
                              "error": f"{type(exc).__name__}: {exc}"[:160]})
            continue
        chosen = sel[6:] if (sel or "").startswith("a_sel_") else prop_cls
        # score the decision against structures this campaign already sized
        outcomes = {}
        for _h, p in by_ctx.get(row["context_id"], []):
            fam = f"{p['stages']}s_" + {"miller_cap": "miller",
                                        "rc_nulling": "rc"}.get(
                                            p["comp"], p["comp"])
            d = p["sz"]["outcome"]["normalized_distance_to_feasibility"]
            if d is not None and (fam not in outcomes or d < outcomes[fam]):
                outcomes[fam] = d
        best_fam = min(outcomes, key=outcomes.get) if outcomes else None
        decisions.append({
            "context_id": row["context_id"], "proposal_class": prop_cls,
            "search_selected": sel, "search_class": chosen,
            "changed_proposal": chosen != prop_cls,
            "visits": visits,
            "measured_distance_by_class": outcomes,
            "best_measured_class": best_fam,
            "search_picked_best_measured": (best_fam is not None
                                            and chosen == best_fam),
            "proposal_was_best_measured": (best_fam is not None
                                           and prop_cls == best_fam)})
    scored = [d for d in decisions if d.get("best_measured_class")]
    summary = {
        "generation": gen, "contexts": len(decisions),
        "scored_contexts": len(scored),
        "search_changed_proposal": sum(1 for d in decisions
                                       if d.get("changed_proposal")),
        "search_picked_best": sum(1 for d in scored
                                  if d["search_picked_best_measured"]),
        "proposal_picked_best": sum(1 for d in scored
                                    if d["proposal_was_best_measured"]),
        "decisions": decisions}
    (camp["dirs"]["reports"] /
     f"search_decisions_gen{gen}.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"  search decisions: changed {summary['search_changed_proposal']}"
          f"/{summary['contexts']}, picked best "
          f"{summary['search_picked_best']}/{summary['scored_contexts']} "
          f"(LLM alone {summary['proposal_picked_best']})")
    return summary


def design_and_measure(gen, corpus, camp, ckpt, exam_manifest, cfg, lineage):
    import torch
    tok = _tok()
    model = load_adapter(ckpt)
    train = [r for r in corpus["records"] if r["split"] == "train"]
    # Fix 1 -- single choke point. Excluding the frozen evaluation specs from
    # the DESIGN pool stops them reaching sizing at all, which in turn keeps
    # them out of L4/RAG memory, AlphaZero value targets, surrogate and
    # ranker training, self-earned examples and DPO pairs: every one of those
    # is built from what was designed and measured here.
    frozen_eval = eval_sets.excluded_context_ids()
    train = [r for r in train if r["context_id"] not in frozen_eval]
    # training contexts stay FRESH per generation (they are training data);
    # only the evaluation set is frozen -- see eval_sets module docstring
    rng = random.Random(cfg.get("seed", 0) * 1000 + 100 + gen)
    rows = []
    for r in rng.sample(train, cfg["design_contexts"]):
        # pair supply: sample several seeds per spec — a converged model
        # proposes the same structure at every seed, and identical candidates
        # can never form a preference pair (the gen-214117 campaign starved
        # DPO for exactly this reason)
        cands = [c for s in range(cfg.get("design_seeds", 2))
                 for c in [generate(model, tok, r["prompt"], sample_seed=s)]
                 if c["valid"]]
        if cands:
            rows.append({"context_id": r["context_id"], "prompt": r["prompt"],
                         "topology_id": r["topology_id"],
                         "target_variant": r["variant_hash"],
                         "target_stages": r["stages"],
                         "target_comp": r["comp"],
                         "candidates": cands, "valid": len(cands),
                         "unique_valid": len({c["graph_hash"] for c in cands})})
    del model
    torch.cuda.empty_cache()
    chans = set(cfg.get("channels", ["replay", "surrogate", "ranker", "l4",
                                     "value", "selfearn"]))
    # raw proposer output, counted BEFORE the search edits the candidate set:
    # "generated" means how many samples the LLM produced, and ranking must
    # not silently redefine it
    raw_c = [c for r in rows for c in r["candidates"]]
    raw_generated, raw_valid = len(raw_c), sum(1 for c in raw_c if c["valid"])
    raw_unique = len({c["graph_hash"] for c in raw_c if c["valid"]})
    # AlphaZero selects BEFORE realisation: diversity_sft.json is what
    # realise_sample reads, so the ranked set has to be in place here or the
    # search's choice never reaches ngspice
    rows, rank_summary = enumerate_and_rank(rows, chans)
    all_c = [c for r in rows for c in r["candidates"]]
    (O4 / "diversity_sft.json").write_text(json.dumps(
        {"rows": rows,
         # proposer output (unchanged meaning: contexts x candidates_per_ctx)
         "generated": raw_generated, "valid": raw_valid,
         "unique_valid_canonical": raw_unique,
         "contexts": cfg["design_contexts"],
         "candidates_per_ctx": cfg.get("design_seeds", 2),
         "iso_classes_generated": raw_unique,
         # what actually goes forward after the search ranks and adds
         # structures the proposer never emitted
         "candidates_after_ranking": len(all_c),
         "unique_after_ranking": len({c["graph_hash"] for c in all_c}),
         "structures_added_by_search": len(all_c) - len(
             {c["graph_hash"] for c in raw_c if c["valid"]})},
        indent=0), encoding="utf-8")
    meas = realise_sample(n=cfg["realise_n"])
    # Task 6: a structure is judged only AFTER a fair sizing budget — every
    # realised structure is SAC-sized under its originating spec and all
    # downstream verdicts (self-earned tiers, pairs, L4 memory) use the best
    # POST-SIZING ngspice result, never the nominal one.
    post = postsize_designs(gen, rows, meas, camp, cfg)
    trace_report = write_execution_traces(gen, rows, post, camp, chans)
    dec = record_search_decisions(gen, rows, post, camp, chans)
    if rank_summary is not None:
        (camp["dirs"]["reports"] /
         f"search_ranking_gen{gen}.json").write_text(
            json.dumps(rank_summary, indent=1, default=str),
            encoding="utf-8")
        if dec:
            dec["ranking"] = {k: rank_summary[k] for k in
                              ("contexts", "top_k", "llm_pick_in_top_k",
                               "structures_added")}
    l4 = Path("datasets/simulation_memory/self_improvement_runs.jsonl")
    with l4.open("a", encoding="utf-8") as f:
        if "l4" not in chans:
            f = open(os.devnull, "w")   # F-battery: L4 channel disabled
        for m in meas["realised"]:
            f.write(json.dumps({"generation": gen, "level": "L4",
                                "campaign_id": camp["campaign_id"],
                                "testbench_hash": ig.TESTBENCH_HASH,
                                **{k: m.get(k) for k in
                                   ("variant", "device_hash", "stages",
                                    "electrical", "stability", "pm",
                                    "context_id")}}) + "\n")
        # post-sizing entries come LAST so measurement lookups prefer them
        for h, p in post.items():
            b = p["sz"]["best"]
            f.write(json.dumps({"generation": gen, "level": "L4",
                                "campaign_id": camp["campaign_id"],
                                "testbench_hash": ig.TESTBENCH_HASH,
                                "postsizing": True, "variant": h,
                                "context_id": p["context_id"],
                                "stages": p["stages"],
                                "electrical": b["electrical"],
                                "stability": b["stability"],
                                "pm": b["pm_deg"], "gain_db": b["gain_db"],
                                "ugbw_hz": b["ugbw_hz"],
                                "spice_calls": p["sz"]["spice_calls"],
                                "outcome_tier":
                                    p["sz"]["outcome"]["outcome_tier"]})
                    + "\n")
    # Task 7: AlphaZero value targets from FINAL post-sizing outcomes, with
    # full conditioning fields (spec, family, budget, uncertainty)
    az = Path("datasets/simulation_memory/az_value_targets.jsonl")
    with az.open("a", encoding="utf-8") as f:
        if "value" not in chans:
            f = open(os.devnull, "w")   # F-battery: value channel disabled
        for h, p in post.items():
            f.write(json.dumps({
                "graph_hash": h, "topology_family":
                    f"{p['stages']}s_" + {"miller_cap": "miller",
                                          "rc_nulling": "rc"}.get(
                                              p["comp"], p["comp"]),
                "spec": p["spec"], "context_id": p["context_id"],
                **p["sz"]["value"],
                "value_source": "REAL_POST_SIZING_SPICE",
                "sizing_budget": p["sz"]["budget"],
                "spice_calls": p["sz"]["spice_calls"],
                "uncertainty": 0.0 if p["sz"]["best"]["stability"] and
                    str(p["sz"]["best"]["stability"]).startswith("verified")
                    else 1.0,
                "campaign_id": camp["campaign_id"], "generation": gen})
                + "\n")
    meas_map = load_measurement_map()
    # G/A10: tiered self-earned examples — only verified enters normal SFT
    tiers = {"verified_self_earned": 0, "provisional_self_earned": 0,
             "failed_self_earned": 0, "quarantined_self_earned": 0}
    for row in rows:
        seen_row = set()
        for c in row["candidates"]:
            if c["graph_hash"] in seen_row:
                continue
            seen_row.add(c["graph_hash"])
            tier, why = ig.classify_earned(row, c, meas_map.get(c["graph_hash"]),
                                           exam_manifest)
            tiers[tier] += 1
            ps = post.get(c["graph_hash"])
            rec = {"prompt": row["prompt"],
                   "response": json.dumps(c["obj"], separators=(",", ":"))
                   if c.get("obj") else None,
                   "context_id": row["context_id"],
                   "topology_id": row["topology_id"],
                   "tier": tier, "tier_reasons": why,
                   # Task 6 lineage: proposal -> executed topology -> sizing
                   "topology_hash": c["graph_hash"],
                   "topology_action": "keep_llm_proposal",
                   "nominal_metrics": next(
                       (m for m in meas["realised"]
                        if m.get("variant") == c["graph_hash"]), None),
                   "postsizing": ({"best_knobs": ps["sz"]["best"]["knobs"],
                                   "final_metrics": {
                                       k: ps["sz"]["best"][k] for k in
                                       ("gain_db", "pm_deg", "ugbw_hz",
                                        "stable", "stability")},
                                   "outcome_tier":
                                       ps["sz"]["outcome"]["outcome_tier"],
                                   "sac_transitions":
                                       len(ps["sz"]["transitions"]),
                                   "spice_calls": ps["sz"]["spice_calls"],
                                   "reward_policy": ps["sz"]["reward_policy"],
                                   "detail_ref":
                                       f"reports/postsizing_gen{gen}.json"}
                                  if ps else None),
                   "provenance": f"self_earned_gen{gen}", "lineage": lineage}
            # SE battery: the policy decides what enters the SFT textbook
            # (verified_self_earned.jsonl is the ONLY file SFT consumes).
            # Records excluded by policy still land in their audit tier file.
            se_arm = cfg.get("se_arm", "se6")
            if "selfearn" not in chans:
                se_arm = "se0"
            nominal_by_variant = {m.get("variant"): m
                                  for m in meas["realised"]}
            admitted = se_admits(se_arm, tier, c, row, meas_map,
                                 nominal_by_variant)
            audit_tier = (tier if (tier != "verified_self_earned"
                                   or admitted)
                          else "verified_excluded_by_policy")
            with (camp["dirs"]["self_earned"] / f"{audit_tier}.jsonl").open(
                    "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            if admitted and tier != "verified_self_earned":
                with (camp["dirs"]["self_earned"]
                      / "verified_self_earned.jsonl").open(
                          "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(rec, se_policy_admitted=se_arm),
                                       default=str) + "\n")
    # spec-conditioned preference pairs via the integrity engine (D).
    # Compensation-probe results join PAIR BUILDING as same-context
    # alternatives (a measured none-vs-miller comparison under one spec is
    # exactly the preference signal DPO needs — a converged model proposes
    # one structure per spec, starving the pair supply otherwise). Probes
    # stay OUT of the self-earned tier loop above: the model did not propose
    # them, and they may conflict with the corpus rule the exam grades.
    pair_rows = []
    for row in rows:
        extra = [{"valid": True, "parseable": True, "probe": True,
                  "obj": json.loads(p["proposal"]), "graph_hash": h}
                 for h, p in post.items()
                 if p.get("probe") and p["context_id"] == row["context_id"]
                 and all(c["graph_hash"] != h for c in row["candidates"])]
        pair_rows.append(dict(row, candidates=row["candidates"] + extra)
                         if extra else row)
    built = ig.build_context_pairs(pair_rows, meas_map, lineage,
                                   quarantine_fn=ig.quarantine_writer(camp))
    pf = camp["dirs"]["pairs"] / f"pairs_gen{gen}.jsonl"
    pf.write_text("\n".join(json.dumps(p, default=str)
                            for p in built["pairs"]), encoding="utf-8")
    return {"designs": len(all_c), "contexts_designed": len(rows),
            "measured": len(meas["realised"]),
            "spice_calls": meas["real_spice_calls"],
            "electrical": electrical_curve(post),
            "earned_tiers": tiers, "raw_pairs_built": len(built["pairs"]),
            "pair_drops": built["drops"]}


# ----------------------- DPO dataset gate + training --------------------------
def prepare_dpo_dataset(camp, corpus, exam_manifest, lineage_ok, lineage_note):
    pairs = []
    for pf in sorted(camp["dirs"]["pairs"].glob("pairs_gen*.jsonl")):
        pairs += [json.loads(x) for x in pf.read_text().splitlines()
                  if x.strip()]
    deduped = ig.dedupe_pairs(pairs)
    balanced = ig.balance_pairs(deduped["pairs"])
    exam_unchanged = (ig.build_exam_manifest(corpus)["frozen_exam_hash"]
                      == exam_manifest["frozen_exam_hash"])
    ready = ig.readiness_report(balanced["pairs"], exam_manifest,
                                balanced["distribution"],
                                balanced["collapse_flags"],
                                lineage_ok, lineage_note, exam_unchanged)
    report = {"integrity": deduped["report"],
              "drop_examples": deduped["examples"],
              "distribution": balanced["distribution"],
              "collapse_flags": balanced["collapse_flags"],
              "readiness": ready}
    (camp["dirs"]["reports"] / "dpo_dataset_report.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")
    (camp["dirs"]["pairs"] / "dpo_train.jsonl").write_text(
        "\n".join(json.dumps(p, default=str) for p in balanced["pairs"]),
        encoding="utf-8")
    return balanced["pairs"], ready, report


def dpo_generation(gen, camp, sft_ckpt, pairs, cfg):
    import torch
    tok = _tok()
    policy = load_adapter(sft_ckpt, trainable=True)
    ref = load_adapter(sft_ckpt, trainable=False, cpu=True)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref_ck = float(sum(p.abs().sum().float() for p in ref.parameters()))
    opt = torch.optim.AdamW([p for p in policy.parameters()
                             if p.requires_grad], lr=cfg["dpo_lr"])
    rng = random.Random(gen)
    losses = []
    for _ in range(cfg["dpo_epochs"]):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), 4):
            loss = 0.0
            batch = pairs[i:i + 4]
            for pr in batch:
                # A1: real prompt or no pair; same format dropout as SFT so
                # preferences also transfer to evidence-free exam prompts
                prompt = format_dropout(pr["prompt"], rng)
                lp_p, _ = seq_logprob(policy, tok, prompt, pr["chosen"])
                lp_r, _ = seq_logprob(policy, tok, prompt, pr["rejected"])
                with torch.no_grad():
                    lr_p, _ = seq_logprob(ref, tok, prompt, pr["chosen"])
                    lr_r, _ = seq_logprob(ref, tok, prompt, pr["rejected"])
                lr_p, lr_r = lr_p.to(lp_p.device), lr_r.to(lp_r.device)
                loss = loss - pr.get("confidence", 0.9) * \
                    torch.nn.functional.logsigmoid(
                        cfg["beta"] * ((lp_p - lr_p) - (lp_r - lr_r)))
            opt.zero_grad(); (loss / len(batch)).backward(); opt.step()
            losses.append(float((loss / len(batch)).detach()))
    ref_unchanged = float(sum(p.abs().sum().float()
                              for p in ref.parameters())) == ref_ck
    ckpt = camp["dirs"]["models"] / f"gen{gen}_dpo"
    policy.save_pretrained(str(ckpt))
    del policy, ref
    torch.cuda.empty_cache()
    return {"pairs": len(pairs), "epochs": cfg["dpo_epochs"],
            "loss_first_last": [round(losses[0], 3), round(losses[-1], 3)]
            if losses else None,
            "frozen_reference_unchanged": ref_unchanged,
            "output_dpo_checkpoint": str(ckpt),
            "output_dpo_hash": ig.sha_checkpoint(ckpt)}


# ------------------------------ one generation --------------------------------
def run_generation(gen, corpus, camp, parent, parent_hash, exam_manifest, cfg,
                   parent_exam=None):
    t0 = time.time()
    lineage_ok, note = True, "ok"
    if parent is not None:                  # H: parent must be the accepted one
        actual = ig.sha_checkpoint(parent)
        lineage_ok = actual == parent_hash
        note = f"parent {parent} hash {actual} vs accepted {parent_hash}"
        assert lineage_ok, f"generation lineage broken: {note}"
    lineage = {"campaign_id": camp["campaign_id"], "generation_id": gen,
               "parent_checkpoint": str(parent) if parent else "base+fresh_lora",
               "parent_adapter_hash": parent_hash or "none",
               "corpus_hash": corpus["split_manifest"]["corpus_hash"],
               "split_hash": corpus["split_manifest"]["split_hash"],
               "source_script": "run_self_improvement.py",
               "source_run_id": camp["campaign_id"], "random_seed": gen,
               "registry_version": ig.SCHEMA,
               "vocabulary_version": MODEL_ID}
    rec = {"generation_id": gen, "parent_generation_id": gen - 1,
           "parent_checkpoint_path": str(parent) if parent else None,
           "parent_checkpoint_hash": parent_hash, "campaign_id": camp["campaign_id"]}
    rec["sft"] = sft_generation(gen, corpus, camp, parent, exam_manifest, cfg)
    sft_ckpt = Path(rec["sft"]["output_sft_checkpoint"])
    rec["sft_exam"] = run_exam(sft_ckpt, corpus, exam_manifest, camp,
                               f"gen{gen}_sft_exam")
    # Repair 1 (v2): SFT itself is gated — a materially regressed SFT
    # checkpoint never becomes the working model; the parent carries forward
    rec["sft_accepted"], sft_reasons = True, []
    if parent is not None and parent_exam is not None:
        rec["sft_accepted"], sft_reasons = ig.decide_acceptance(
            parent_exam, rec["sft_exam"])
        rec["sft_rejection_reason"] = ("" if rec["sft_accepted"]
                                       else "; ".join(sft_reasons))
    if not rec["sft_accepted"]:
        sft_ckpt = parent
        rec["sft_exam_used"] = "parent (SFT regressed: "
        rec["sft_exam"] = dict(parent_exam,
                               note="parent metrics; new SFT rejected")
    rec["design"] = design_and_measure(gen, corpus, camp, sft_ckpt,
                                       exam_manifest, cfg, lineage)
    arm = cfg.get("arm", "full")
    pairs, ready, _ = prepare_dpo_dataset(camp, corpus, exam_manifest,
                                          lineage_ok, note)
    if arm == "dpo_no_integrity":
        # ABLATION NEGATIVE CONTROL: raw pairs, no dedup/balance/readiness,
        # plus the archived promptless queue with the historical fallback
        # prompt. Never available outside --arm.
        raw = []
        for pf in sorted(camp["dirs"]["pairs"].glob("pairs_gen*.jsonl")):
            raw += [json.loads(x) for x in pf.read_text().splitlines()
                    if x.strip()]
        fallback = ("### SPEC gain>=60dB pm>=45deg tech=sky130\n"
                    "### PROPOSAL\n")
        for camp_dir in sorted(SI.glob("camp_*")):
            q = camp_dir / "archive" / "v2_mechanical.jsonl"
            if q.is_file():
                for x in q.read_text().splitlines():
                    if x.strip():
                        pp = json.loads(x)
                        raw.append({"pair_id": pp.get("pair_id", "arch"),
                                    "prompt": pp.get("prompt") or fallback,
                                    "chosen": pp["preferred"],
                                    "rejected": pp["rejected"],
                                    "confidence": 0.9})
                break
        pairs = raw
        ready = {"training_ready": bool(raw), "pair_count": len(raw),
                 "blockers": ["ABLATION: integrity controls disabled"]}
    rec["arm"] = arm
    rec["se_arm"] = cfg.get("se_arm", "se6")
    rec["channels"] = cfg.get("channels")
    rec["engine"] = "sac_v2_pretrained"   # repaired production tuner
    rec["profile"] = cfg.get("profile", "full")
    rec["campaign_seed"] = cfg.get("seed", 0)
    rec["dpo_readiness"] = {"training_ready": ready["training_ready"],
                            "blockers": ready["blockers"],
                            "pair_count": ready["pair_count"]}
    rec["input_dpo_dataset_hash"] = ig.sha_json(
        [p.get("pair_id") for p in pairs])
    if arm == "sft_only":
        accept, reasons = False, ["arm sft_only: DPO disabled"]
        dpo_ckpt = None
    elif ready["training_ready"] and pairs:
        rec["dpo"] = dpo_generation(gen, camp, sft_ckpt, pairs, cfg)
        dpo_ckpt = Path(rec["dpo"]["output_dpo_checkpoint"])
        rec["dpo_exam"] = run_exam(dpo_ckpt, corpus, exam_manifest, camp,
                                   f"gen{gen}_dpo_exam")
        accept, reasons = ig.decide_acceptance(rec["sft_exam"],
                                               rec["dpo_exam"])
        if not rec["dpo"]["frozen_reference_unchanged"]:
            accept, reasons = False, reasons + ["frozen reference mutated"]
        if arm == "dpo_no_gate":
            rec["gate_would_have_said"] = {"accept": accept,
                                           "reasons": reasons}
            accept, reasons = True, ["ABLATION: acceptance gate bypassed"]
    else:
        accept, reasons = False, ["dpo skipped: " + "; ".join(
            ready["blockers"] or ["no pairs"])]
        dpo_ckpt = None
    accepted = dpo_ckpt if accept else sft_ckpt
    rec.update({"dpo_update_accepted": accept, "dpo_update_rejected": not accept,
                "acceptance_reason": ("dpo passed frozen-exam acceptance"
                                      if accept else "; ".join(reasons)),
                "accepted_checkpoint": str(accepted),
                "accepted_checkpoint_hash": ig.sha_checkpoint(accepted),
                "wall_clock_s": round(time.time() - t0, 1)})
    with (camp["dirs"]["logs"] / "generations.jsonl").open(
            "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("sft_exam", "dpo_exam")}, indent=1,
                     default=str))
    for k in ("sft_exam", "dpo_exam"):
        if k in rec:
            e = rec[k]
            print(f"  {k}: valid {e['valid_rate']} | spec-match "
                  f"{e['spec_match_rate']} | unique {e['unique_structures']} "
                  f"| top-response {e['most_common_response_fraction']}")
    return rec


# ------------------------------ Part N dry run --------------------------------
def dry_run(camp, corpus, exam_manifest):
    print("=== DRY RUN (no GPU): corpus, split, leakage, pair pipeline ===")
    sm = corpus["split_manifest"]
    print(f"corpus_hash {sm['corpus_hash']}  split_hash {sm['split_hash']}  "
          f"split_seed {sm['split_seed']}")
    print(f"families  train {len(sm['train_family_ids'])} | validation "
          f"{len(sm['validation_family_ids'])} | blind "
          f"{len(sm['blind_test_family_ids'])}")
    overlap = (set(sm["train_family_ids"]) & set(sm["validation_family_ids"])) \
        | (set(sm["train_spec_ids"]) & set(sm["validation_spec_ids"])) \
        | (set(sm["train_family_ids"]) & set(sm["blind_test_family_ids"])) \
        | (set(sm["blind_test_family_ids"]) & set(sm["validation_family_ids"]))
    assert not overlap, f"split overlap: {overlap}"
    print("split overlap: none")
    # design rows WITHOUT the model: real train prompts + the measured
    # structure classes as candidates. Rehearses the DATA pipeline only —
    # these pairs are written to reports/ and are never used for training.
    meas_map = load_measurement_map()
    classes = {}
    for h, m in meas_map.items():
        st = m.get("stages")
        if st:
            classes[h] = st
    train = [r for r in corpus["records"] if r["split"] == "train"]
    rng = random.Random(0)
    rows = []
    variants = {(2, "none"), (2, "miller"), (3, "none"), (3, "miller"),
                (3, "rc")}
    for r in rng.sample(train, min(12, len(train))):
        cands = []
        for stages, comp in sorted(variants):
            obj = json.loads(variant_text(stages, comp, False, False))
            ident = ig.candidate_identity(obj)
            if ident["canonical_graph_hash"] in meas_map:
                cands.append({"valid": True, "obj": obj,
                              "graph_hash": ident["canonical_graph_hash"],
                              "parseable": True})
        rows.append({"context_id": r["context_id"], "prompt": r["prompt"],
                     "topology_id": r["topology_id"],
                     "target_variant": r["variant_hash"],
                     "target_stages": r["stages"], "candidates": cands,
                     "dry_run": True})
    lineage = {"campaign_id": camp["campaign_id"], "generation_id": "dry_run",
               "source_script": "run_self_improvement.py --dry-run",
               "corpus_hash": sm["corpus_hash"], "split_hash": sm["split_hash"]}
    built = ig.build_context_pairs(rows, meas_map, lineage,
                                   quarantine_fn=ig.quarantine_writer(camp))
    deduped = ig.dedupe_pairs(built["pairs"])
    balanced = ig.balance_pairs(deduped["pairs"])
    ready = ig.readiness_report(balanced["pairs"], exam_manifest,
                                balanced["distribution"],
                                balanced["collapse_flags"], True,
                                "dry run — no checkpoints", True)
    report = {"drops": built["drops"], "dedup": deduped["report"],
              "distribution": balanced["distribution"],
              "collapse_flags": balanced["collapse_flags"],
              "readiness": ready}
    (camp["dirs"]["reports"] / "dry_run_report.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")
    (camp["dirs"]["reports"] / "dry_run_pairs.jsonl").write_text(
        "\n".join(json.dumps(p, default=str) for p in balanced["pairs"]),
        encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("drops", "dedup", "collapse_flags")}, indent=1))
    print(json.dumps(report["distribution"], indent=1))
    print(f"\ntraining_ready = {ready['training_ready']}")
    if ready["blockers"]:
        print("blockers:", *ready["blockers"], sep="\n  - ")
    shown = 0
    seen_specs = set()
    print("\n=== INSPECTABLE RETAINED PAIRS ===")
    for p in balanced["pairs"]:
        if shown >= 10 and p["spec_id"] in seen_specs:
            continue
        seen_specs.add(p["spec_id"])
        shown += 1
        print(f"\n--- pair {shown}: {p['pair_id']} ---")
        print("prompt          :", p["prompt"].splitlines()[0])
        print("structured spec :", {k: p["structured_spec"][k] for k in
                                    ("gain_target_db", "phase_margin_target_deg",
                                     "load_capacitance_pf", "ugbw_target_hz")})
        print("eval context id :", p["evaluation_context_id"])
        print("chosen          :", p["chosen_identity"]["topology_family_id"],
              p["chosen_topology_hash"])
        print("rejected        :", p["rejected_identity"]["topology_family_id"],
              p["rejected_topology_hash"])
        print("chosen metrics  :", {k: p["chosen_metrics"][k] for k in
                                    ("stability_status", "phase_margin_deg",
                                     "exact_structure_match",
                                     "specs_passed_count")})
        print("rejected metrics:", {k: p["rejected_metrics"][k] for k in
                                    ("stability_status", "phase_margin_deg",
                                     "exact_structure_match",
                                     "specs_passed_count")})
        print("reason          :", p["preference_reason"])
        print("lineage         :", p["lineage"]["campaign_id"],
              "gen", p["lineage"]["generation_id"])
        if shown >= 12:
            break
    if shown == 0:
        print("NO PAIRS RETAINED — inspect the drop report")
    return ready


# --------------------------------- main ---------------------------------------
def blind_eval(ckpt: Path, corpus):
    """Task 1/10: ONE-TIME evaluation on the untouched blind test set. Run
    only after every campaign decision is frozen — never for training, DPO,
    rollback or debugging. Appends to an audit log so reuse is visible."""
    import torch
    manifest = json.loads((O4 / "blind_test.json").read_text())
    blind = [r for r in corpus["records"] if r["split"] == "blindtest"]
    live = ig.sha_json(sorted((r["prompt"], r["variant_hash"])
                              for r in blind))
    assert live == manifest["frozen_blind_hash"], "blind set drifted"
    audit = Path("artifacts/blind_eval_audit.jsonl")
    prior = len(audit.read_text().splitlines()) if audit.is_file() else 0
    if prior:
        print(f"WARNING: blind set was already evaluated {prior} time(s) — "
              f"results below are no longer statistically blind")
    tok = _tok()
    model = load_adapter(ckpt)
    valid, match, uniq, resp = 0, 0, set(), {}
    for r in blind:
        c = generate(model, tok, r["prompt"], sample_seed=0)
        if c["valid"]:
            valid += 1
            uniq.add(c["graph_hash"])
            match += int(c["graph_hash"] == r["variant_hash"])
            rh = ig.candidate_identity(c["obj"])["normalized_response_hash"]
            resp[rh] = resp.get(rh, 0) + 1
    del model
    torch.cuda.empty_cache()
    n = len(blind)
    out = {"checkpoint": str(ckpt), "checkpoint_hash": ig.sha_checkpoint(ckpt),
           "blind_hash": manifest["frozen_blind_hash"], "contexts": n,
           "valid_rate": round(valid / n, 3),
           "spec_match_rate": round(match / n, 3),
           "unique_structures": len(uniq),
           "most_common_response_fraction": round(
               max(resp.values()) / max(1, valid), 3) if resp else 0.0,
           "evaluation_number": prior + 1, "timestamp": time.time()}
    with audit.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("generations", nargs="?", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--resume-campaign", metavar="CAMP_NAME", default=None,
                    help="continue an interrupted campaign from its last "
                         "completed generation (arm/seed read from its log)")
    ap.add_argument("--arm", default="full",
                    choices=["full", "sft_only", "dpo_no_integrity",
                             "dpo_no_gate"],
                    help="ablation arm (publication study); default full")
    ap.add_argument("--seed", type=int, default=0,
                    help="campaign seed offset (multi-seed study)")
    ap.add_argument("--fast", action="store_true",
                    help="uniform reduced profile (battery studies)")
    ap.add_argument("--se-arm", default="se6",
                    choices=["se0", "se1", "se2", "se3", "se4", "se5", "se6"],
                    help="self-earned qualification policy (SE battery)")
    ap.add_argument("--channels", default="replay,surrogate,ranker,l4,value,selfearn",
                    help="comma set of feedback channels (F battery); "
                         "'none' disables all")
    ap.add_argument("--blind-eval", metavar="CHECKPOINT",
                    help="one-time blind-test evaluation of a checkpoint; "
                         "no campaign, no training")
    args = ap.parse_args(argv)

    if args.blind_eval:
        build_corpus()
        corpus = json.loads((O4 / "corpus.json").read_text())
        blind_eval(Path(args.blind_eval), corpus)
        return

    prior_gens = []
    if args.resume_campaign:
        root = SI / args.resume_campaign
        logf = root / "logs" / "generations.jsonl"
        assert logf.is_file(), f"nothing to resume in {root}"
        prior_gens = [json.loads(x) for x in
                      logf.read_text(encoding="utf-8").splitlines()]
        assert prior_gens, "campaign log is empty — start a fresh campaign"
        camp = {"campaign_id": args.resume_campaign, "root": root,
                "dirs": {n: root / n for n in
                         ("corpus", "pairs", "quarantine", "models",
                          "evaluations", "logs", "manifests", "reports",
                          "self_earned", "archive")},
                "archived": []}
        args.arm = prior_gens[0].get("arm", "full")
        args.seed = prior_gens[0].get("campaign_seed", 0)
        args.se_arm = prior_gens[0].get("se_arm", "se6")
        _ch = prior_gens[0].get("channels")
        args.channels = ",".join(_ch) if _ch else "none"
        print(f"resuming {args.resume_campaign} at generation "
              f"{len(prior_gens)} (arm={args.arm}, seed={args.seed})")
    else:
        camp = ig.new_campaign(SI, STALE_PATHS,
                               "pre-repair artifact (promptless pairs / old split)")
        print(f"campaign: {camp['campaign_id']}  "
              f"(archived {len(camp['archived'])} stale artifacts)")
    build_corpus()
    corpus = json.loads((O4 / "corpus.json").read_text())
    exam_manifest = json.loads((O4 / "frozen_exam.json").read_text())
    if args.resume_campaign:
        saved = json.loads((camp["dirs"]["corpus"] /
                            "frozen_exam.json").read_text())
        assert saved["frozen_exam_hash"] ==             exam_manifest["frozen_exam_hash"],             "frozen exam drifted since the campaign started — cannot resume"
    else:
        for name in ("corpus.json", "split_manifest.json",
                     "frozen_exam.json"):
            (camp["dirs"]["corpus"] / name).write_text(
                (O4 / name).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"frozen_exam_hash: {exam_manifest['frozen_exam_hash']}  "
          f"({exam_manifest['contexts']} contexts)")

    if args.dry_run:
        dry_run(camp, corpus, exam_manifest)
        return

    channels = (set() if args.channels == "none"
                else set(args.channels.split(",")))
    base_cfg = (CFG_VALIDATE if args.validate
                else CFG_FAST if args.fast else CFG_FULL)
    cfg = dict(base_cfg, arm=args.arm, seed=args.seed, se_arm=args.se_arm,
               channels=sorted(channels),
               profile=("fast" if args.fast else
                        "validate" if args.validate else "full"))
    n_gen = 1 if args.validate else args.generations
    if args.arm != "full" or args.seed:
        print(f"ablation arm: {args.arm} | campaign seed: {args.seed}")
    parent, parent_hash = None, None
    curve = list(prior_gens)
    start_gen = len(prior_gens)
    if prior_gens:
        parent = Path(prior_gens[-1]["accepted_checkpoint"])
        parent_hash = prior_gens[-1]["accepted_checkpoint_hash"]
    for gen in range(start_gen, n_gen):
        if gen:     # refresh TRAIN prompts with the latest L4 measured
            build_corpus()      # evidence (### KNOWN lines); the held-out
            corpus = json.loads((O4 / "corpus.json").read_text())
            live = ig.build_exam_manifest(corpus)   # exam must stay frozen
            assert live["frozen_exam_hash"] == \
                exam_manifest["frozen_exam_hash"], "exam drifted on rebuild"
        parent_exam = (curve[-1].get("dpo_exam")
                       if curve and curve[-1].get("dpo_update_accepted")
                       else curve[-1]["sft_exam"]) if curve else None
        rec = run_generation(gen, corpus, camp, parent, parent_hash,
                             exam_manifest, cfg, parent_exam=parent_exam)
        parent = Path(rec["accepted_checkpoint"])
        parent_hash = rec["accepted_checkpoint_hash"]
        curve.append(rec)

    print("\n=== LEARNING CURVE (frozen exam, this campaign only) ===")
    for r in curve:
        s, d = r["sft_exam"], r.get("dpo_exam")
        line = (f"gen {r['generation_id']}: SFT valid {s['valid_rate']} "
                f"match {s['spec_match_rate']} unique {s['unique_structures']}")
        if d:
            line += (f" | DPO valid {d['valid_rate']} match "
                     f"{d['spec_match_rate']} unique {d['unique_structures']}")
        line += (" | accepted: "
                 + ("DPO" if r["dpo_update_accepted"] else "SFT (rollback)"))
        print(line)
    # AlphaZero continuous learning: refresh the value net on this
    # campaign's new post-sizing outcomes (CPU, ~1 min, backed-up ckpt)
    try:
        from agentic_raptor.topology_rl.value_refresh import refresh
        vr = refresh()
        print("value-net refresh:",
              {k: vr.get(k) for k in ("targets_total", "improved")}
              if "skipped" not in vr else vr["skipped"])
    except Exception as exc:
        print(f"value-net refresh skipped: {exc}")
    print("=== ELECTRICAL CURVE (real post-sizing ngspice, per generation) ===")
    for r in curve:
        e = r["design"].get("electrical") or {}
        if not e.get("designs_sized"):
            print(f"gen {r['generation_id']}: no designs sized")
            continue
        print(f"gen {r['generation_id']}: exact-pass {e['exact_spec_pass_rate']}"
              f" | stable {e['stable_rate']} | gain-pass {e['gain_pass_rate']}"
              f" | pm-pass {e['pm_pass_rate']}"
              f" | avg-constraints {e['avg_hard_constraints_passed']}"
              f" | med-dist {e['median_normalized_distance']}"
              f" | sized {e['designs_sized']}"
              f" (spice {e['postsizing_spice_calls']})")

    if args.validate:
        r = curve[0]
        s, d = r["sft_exam"], r.get("dpo_exam")
        ok = (s["valid_rate"] >= 0.85 and s["unique_structures"] > 1
              and (d is None or (d["spec_match_rate"]
                                 >= s["spec_match_rate"] - 0.02
                                 and d["most_common_response_fraction"] <= 0.6))
              and r["dpo_readiness"]["pair_count"] >= 0)
        print(f"\nvalidation acceptable: {ok}")
        if ok:
            print("full campaign command:\n  "
                  "python run_self_improvement.py 3")
        else:
            print("validation failed — do NOT launch the full campaign yet")


if __name__ == "__main__":
    main()
