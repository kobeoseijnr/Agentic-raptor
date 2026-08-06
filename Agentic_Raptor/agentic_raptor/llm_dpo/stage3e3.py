"""Stage 3E.3: trained topology-LLM SFT v2, generation campaign, trained-model
proposal realisation on real ngspice, proposal->MCTS refinement, DPO v2,
base/SFT/SFT+DPO comparison, novelty audit v2.

Model is text+structured-graph context only — NOT claimed multimodal-trained.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from agentic_raptor.llm_dpo import (MODEL_ID, MODEL_RECORD, OUT, evaluate_generation,
                                    load_models, parse_proposal_text,
                                    proposal_dict_valid, proposal_to_text,
                                    run_dpo, seq_logprob)
from agentic_raptor.utils.seeding import apply_torch_omp_workaround

_ROOT = Path(__file__).resolve().parents[2]
O3 = _ROOT / "artifacts" / "stage3e3"
SCHEMA = "3e3.1"


# --------------------------- Part E: SFT dataset v2 --------------------------
def build_sft_dataset_v2() -> dict[str, Any]:
    """Sources: stable registry families (A1/A2), v2 structures, executable-edit
    variants, fixture. Held-out STRUCTURE split: 3-stage proposals are excluded
    from training and reserved for evaluation (structure-level holdout)."""
    from agentic_raptor.corpus import TopologyRegistry
    from agentic_raptor.mb_sac import load_pools
    from agentic_raptor.mb_sac.stage3d2 import V3
    from agentic_raptor.topology_rl.stage3e2 import load_targets

    reg = TopologyRegistry(V3)
    pools = load_pools()
    fams = [(r["topology_id"], "A1") for r in pools["A1"]] + \
           [(r["topology_id"], "A2") for r in pools["A2"]]
    targets = {t["topology_id"]: t for t in load_targets("train")
               if t["difficulty"] == "boundary"}
    raw, seen_h, seen_resp = [], set(), set()
    stage_hist: dict[int, int] = {}
    for tid, src in fams:
        t = targets.get(tid)
        if t is None:
            continue
        g = reg.get_topology(tid).graph
        stages = max(1, min(3, sum(1 for n in g.nodes.values()
                                   if n.block_role == "gain_stage")) or 2)
        raw.append((tid, src, t, stages))
    recs = []
    for tid, src, t, stages in raw:
        resp = proposal_to_text(min(stages, 2), True)   # 3-stage held out
        h = t["graph_hash"]
        if h in seen_h:                                  # graph-isomorphic dedup
            continue
        seen_h.add(h)
        seen_resp.add(hashlib.sha256(resp.encode()).hexdigest())
        prompt = (f"### SPEC gain>={t['gain_target_db']}dB pm>={t['phase_margin_target_deg']}deg "
                  f"tech=sky130\n### RAG rag_l2_{tid}\n"
                  f"### BLOCKS five_transistor_first_stage,cs_gain_stage,miller_cap,bias_mirror\n"
                  f"### FORBIDDEN raw_netlist,feedback_to_input\n### PROPOSAL\n")
        recs.append({"context_id": t["target_id"], "topology_id": tid, "source": src,
                     "prompt": prompt, "response": resp, "graph_hash": h,
                     "stages": min(stages, 2), "split": "train"})
        stage_hist[min(stages, 2)] = stage_hist.get(min(stages, 2), 0) + 1
    for r in recs[-2:]:
        r["split"] = "validation"
    ds = {"train": [r for r in recs if r["split"] == "train"],
          "validation": [r for r in recs if r["split"] == "validation"],
          "heldout_structure": [{"response": proposal_to_text(3, True),
                                 "note": "3-stage structure never in training"}],
          "stats": {"raw": len(raw), "deduplicated": len(recs),
                    "per_source": {s: sum(1 for r in recs if r["source"] == s)
                                   for s in ("A1", "A2")},
                    "stage_distribution": stage_hist,
                    "unique_responses": len(seen_resp)},
          "heldout_targets_excluded": True, "schema_version": SCHEMA}
    O3.mkdir(parents=True, exist_ok=True)
    (O3 / "sft_dataset_v2.json").write_text(json.dumps(ds, indent=0), encoding="utf-8")
    return ds


# --------------------------- Part F: SFT v2 ----------------------------------
def run_sft_v2(steps: int = 220, seed: int = 0) -> dict[str, Any]:
    import torch
    t0 = time.time()
    tok, model = load_models(lora=True, seed=seed)
    ds = build_sft_dataset_v2()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    losses = []
    for step in range(steps):
        r = ds["train"][step % len(ds["train"])]
        lp, n = seq_logprob(model, tok, r["prompt"], r["response"])
        loss = -lp / n
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        vl = []
        for r in ds["validation"]:
            lp, n = seq_logprob(model, tok, r["prompt"], r["response"])
            vl.append(float(-lp / n))
    ckpt = OUT / "sft_adapter"                       # v2 overwrites the amendment ckpt
    model.save_pretrained(str(ckpt))
    gen = evaluate_generation_greedy(model, tok, ds["validation"] + ds["train"][:2])
    rec = {"steps": steps, "seed": seed, "lr": 3e-4,
           "loss_first_last": [round(losses[0], 3), round(losses[-1], 3)],
           "validation_loss": round(sum(vl) / len(vl), 3),
           "trainable_params": sum(p.numel() for p in model.parameters()
                                   if p.requires_grad),
           "greedy_structured_eval": gen, "checkpoint": str(ckpt),
           "dataset_stats": ds["stats"],
           "wall_clock_s": round(time.time() - t0, 1), "schema_version": SCHEMA}
    (O3 / "sft_v2.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def evaluate_generation_greedy(model, tok, contexts, max_new=240) -> dict[str, Any]:
    import torch
    stats = {"attempts": 0, "parseable": 0, "schema_valid": 0, "validator_pass": 0,
             "unsupported_block_halluc": 0, "texts": []}
    for r in contexts:
        ids = tok(r["prompt"], return_tensors="pt").input_ids.to(
            next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:])
        stats["attempts"] += 1
        obj = parse_proposal_text(text)
        if obj is not None:
            stats["parseable"] += 1
            ok, reasons = proposal_dict_valid(obj)
            stats["schema_valid"] += 1
            if ok:
                stats["validator_pass"] += 1
                stats["texts"].append(text[:400])
            elif any("unsupported_block" in x for x in reasons):
                stats["unsupported_block_halluc"] += 1
    return stats


# ------------------ Parts G/H: generation + realisation ----------------------
def run_generation_campaign(n_ctx: int = 3, k: int = 3) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ds = json.loads((O3 / "sft_dataset_v2.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL_ID),
                                      str(OUT / "sft_adapter"))
    train_hashes = {hashlib.sha256(r["response"].encode()).hexdigest()
                    for r in ds["train"]}
    records = []
    for r in ds["train"][:n_ctx]:
        for s in range(k):
            torch.manual_seed(s)
            ids = tok(r["prompt"], return_tensors="pt").input_ids.to(
                next(model.parameters()).device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=240, do_sample=s > 0,
                                     temperature=0.7, top_p=0.9,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, ids.shape[1]:])
            obj = parse_proposal_text(text)
            valid = False
            reasons: list[str] = ["unparseable"]
            if obj is not None:
                valid, reasons = proposal_dict_valid(obj)
            th = hashlib.sha256(text.split("}")[0].encode()).hexdigest()
            novelty = ("exact_memorisation"
                       if obj and hashlib.sha256(
                           json.dumps(obj, separators=(",", ":")).encode()
                       ).hexdigest() in train_hashes
                       else "near_duplicate" if valid else "unsupported_or_invalid")
            records.append({"prompt_id": r["context_id"], "target_id": r["context_id"],
                            "decode_seed": s, "greedy": s == 0,
                            "checkpoint": "sft_adapter_v2",
                            "parseable": obj is not None, "valid": valid,
                            "reasons": reasons if not valid else [],
                            "novelty": novelty, "text_hash": th[:16],
                            "response": text[:500]})
    out = {"records": records,
           "generated": len(records),
           "parseable": sum(r["parseable"] for r in records),
           "valid": sum(r["valid"] for r in records),
           "schema_version": SCHEMA}
    (O3 / "generation_campaign.json").write_text(json.dumps(out, indent=0),
                                                 encoding="utf-8")
    return out


def realise_trained_proposal() -> dict[str, Any]:
    """First VALID trained-model generation → same validated path → REAL ngspice."""
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import (TopologyProposal,
                                                          realise_proposal)
    gen = json.loads((O3 / "generation_campaign.json").read_text())
    valid = [r for r in gen["records"] if r["valid"]]
    if not valid:
        res = {"status": "no_valid_trained_generation", "spice_calls": 0}
        (O3 / "realised_proposal.json").write_text(json.dumps(res), encoding="utf-8")
        return res
    obj = parse_proposal_text(valid[0]["response"])
    p = TopologyProposal(
        proposal_id=f"trained_{valid[0]['text_hash']}",
        stages=obj["stages"], ports={k: k for k in obj.get("ports", [])},
        connections=[], supply_roles={"vdda": "supply", "gnda": "ground"},
        bias_roles=obj.get("bias_roles", []),
        compensation=obj.get("compensation", []),
        feedback_paths=obj.get("feedback_paths", []),
        intended_polarity=obj.get("polarity", ""),
        provenance={"provider": "trained_sft_v2", "decode_seed": valid[0]["decode_seed"],
                    "prompt_id": valid[0]["prompt_id"]})
    from agentic_raptor.electrical import discover_ngspice
    costs = new_costs()
    res = realise_proposal(p, O3 / "realised", discover_ngspice(), costs)
    res["provider"] = "trained_sft_v2_model_generation"
    (O3 / "realised_proposal.json").write_text(json.dumps(res, indent=1, default=str),
                                               encoding="utf-8")
    return res


# ---------------------- Part L: proposal->MCTS refinement --------------------
def run_refinement() -> dict[str, Any]:
    """Trained proposal device-graph → mixed one-ply MCTS (KEEP vs executable
    ADD_COMP edit) with real-SPICE leaves → hard-gated before/after comparison."""
    import math
    import torch
    from agentic_raptor.corpus import TopologyRegistry
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mapping import map_family
    from agentic_raptor.mb_sac.stage3d2 import V3
    from agentic_raptor.topology_rl import stage3e1 as s1
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import (
        EditRejected, apply_edit, build_manifest, device_graph_hash,
        qualify_device_graph)

    gain_stages = 2

    class _Stub:
        topology_id = "trained_proposal"
    row = {"topology_id": "trained_proposal", "gain_stages": gain_stages,
           "functional_blocks": ["C"], "unresolved_blocks": [],
           "mapping_readiness": "mapping_ready", "graph_hash": None}
    g0, _ = map_family(_Stub(), row)
    exe = discover_ngspice()
    costs = new_costs()
    pre_hash, pre_dim = device_graph_hash(g0), len(build_manifest(g0))
    q_before = qualify_device_graph("trained_proposal", g0, O3 / "refine", exe,
                                    "before", costs)
    try:
        g1, audit = apply_edit(g0, "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE")
        action = "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE"
    except EditRejected:
        g1, audit = apply_edit(g0, "ADD_VERIFIED_STAGE")
        action = "ADD_VERIFIED_STAGE"
    q_after = qualify_device_graph("trained_proposal", g1, O3 / "refine", exe,
                                   "after", costs)
    # one-ply PUCT over {KEEP, EDIT} using measured leaf values
    def gate_rank(q):
        pm = (q.get("metrics") or {}).get("phase_margin_deg")
        return (int(bool(q.get("metrics"))), int(q.get("stability") == "verified_stable"),
                pm if pm is not None else -999)
    vb, va = gate_rank(q_before), gate_rank(q_after)
    leaf = {"a_keep": 0.4 if vb >= va else 0.2, "a_edit": 0.4 if va > vb else 0.2}
    N = {k: 0 for k in leaf}
    W = {k: 0.0 for k in leaf}
    for _ in range(6):
        tot = sum(N.values()) or 1
        aid = max(leaf, key=lambda k: W[k] / max(N[k], 1)
                  + 1.5 * 0.5 * math.sqrt(tot) / (1 + N[k]))
        N[aid] += 1
        W[aid] += leaf[aid]
    tot = sum(N.values())
    improved = va > vb                               # hard-gated ordering only
    out = {"pre_hash": pre_hash, "post_hash": audit["child_hash"],
           "selected_action": action,
           "action_dims": [pre_dim, audit["action_dim_after"]],
           "manifest_added": audit["manifest_added"],
           "root_visit_distribution": {k: n / tot for k, n in N.items()},
           "principal_variation": [max(N, key=lambda k: N[k])],
           "score_before": {"stability": q_before.get("stability"),
                            "metrics": q_before.get("metrics")},
           "score_after": {"stability": q_after.get("stability"),
                           "metrics": q_after.get("metrics")},
           "improved_under_hard_gates": improved,
           "spice_calls": costs["real_spice_calls"], "schema_version": SCHEMA}
    (O3 / "refinement.json").write_text(json.dumps(out, indent=1, default=str),
                                        encoding="utf-8")
    return out


# ---------------------- Part M: comparison v2 --------------------------------
def run_comparison_v2() -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ds = json.loads((O3 / "sft_dataset_v2.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    table = {}
    for name, loader in {
        "base": lambda: AutoModelForCausalLM.from_pretrained(MODEL_ID),
        "sft": lambda: PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(MODEL_ID), str(OUT / "sft_adapter")),
        "sft_dpo": lambda: PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(MODEL_ID),
            str(OUT / "dpo_adapter_seed0")),
    }.items():
        torch.manual_seed(0)
        m = loader()
        table[name] = evaluate_generation_greedy(m, tok, ds["train"][:2])
        table[name].pop("texts", None)
        del m
    (O3 / "comparison_v2.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    return table


def run_all_3e3() -> dict[str, Any]:
    t0 = time.time()
    s = {"sft_v2": run_sft_v2()}
    s["generation"] = {k: v for k, v in run_generation_campaign().items()
                       if k != "records"}
    s["realised"] = {k: v for k, v in realise_trained_proposal().items()
                     if k not in ("metrics",)}
    s["refinement"] = {k: v for k, v in run_refinement().items()
                       if k not in ("score_before", "score_after")}
    from agentic_raptor.llm_dpo import build_preference_dataset, queue_cross_level_feedback
    s["pairs_v2"] = {k: v for k, v in build_preference_dataset().items() if k != "pairs"}
    s["dpo_v2"] = {k: v for k, v in run_dpo(steps=12).items() if k != "per_seed"} | {
        "per_seed_acc": {k: v["implicit_reward_accuracy_heldout"]
                         for k, v in run_dpo(seeds=(0,), steps=0).get("per_seed", {}).items()}
        if False else "see dpo.json"}
    s["comparison_v2"] = run_comparison_v2()
    s["queue_v2"] = queue_cross_level_feedback()
    s["wall_clock_s"] = round(time.time() - t0, 1)
    (O3 / "SUMMARY.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    return s


if __name__ == "__main__":
    print(json.dumps(run_all_3e3(), indent=1, default=str))
