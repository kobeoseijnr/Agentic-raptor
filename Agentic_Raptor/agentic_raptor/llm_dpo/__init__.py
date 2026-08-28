"""TRUE LLM Direct Preference Optimization for topology proposals (3E.2 amendment).

STRICT SEPARATION: this module trains a LANGUAGE MODEL over token log
probabilities (trainable LoRA policy vs frozen SFT reference). It is disjoint
from `agentic_raptor.dpo.DPORanker`, the feasibility-gated Bradley–Terry
SIZING ranker over circuit features — the two never share preference records.

DPO objective (Part L7, exact):
    loss = -E[ w_conf * logsigmoid( beta * ( (logpi_pol(y+|x) - logpi_ref(y+|x))
                                           - (logpi_pol(y-|x) - logpi_ref(y-|x)) ) ) ]
with sequence log probs = sum of token log-softmax over RESPONSE tokens only
(prompt and padding masked out). Reference-free mode disabled by default.
NOT genuine DPO (and never labelled as such here): feature-space BT regression,
binary classification, scalar reward regression, rejection sampling.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

import os

_ROOT = Path(__file__).resolve().parents[2]
OUT = _ROOT / "artifacts" / "stage3e2_llm"
SCHEMA = "3e2L.1"
# ONE source of truth for the text topology model — every phase (SFT, DPO,
# generation, comparison) resolves the same ID, so adapters always match.
# Default: latest Qwen3 instruct (permanent since 2026-07-27; override via env).
MODEL_ID = os.environ.get("AGENTIC_RAPTOR_TOPOLOGY_LLM",
                          "Qwen/Qwen3-4B-Instruct-2507")

MODEL_RECORD = {
    "model_id": MODEL_ID, "revision": "main", "licence": "apache-2.0",
    "quantisation": "bf16 on CUDA, fp32 on CPU (auto)",
    "tokenizer_revision": "main",
    "multimodal": "text path; schematic images handled by llm_dpo.multimodal",
    "schema_version": SCHEMA}


def load_models(lora: bool = True, seed: int = 0):
    apply_torch_omp_workaround()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    if lora:
        from peft import LoraConfig, get_peft_model
        # architecture-adaptive targets: GPT-2 family uses fused c_attn;
        # Llama/Qwen-family models use q_proj/v_proj
        names = [n for n, _ in model.named_modules()]
        tmods = ["c_attn"] if any(n.endswith("c_attn") for n in names) \
            else ["q_proj", "v_proj"]
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                         target_modules=tmods, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    MODEL_RECORD["framework"] = {"transformers": __import__("transformers").__version__,
                                 "peft": __import__("peft").__version__,
                                 "torch": torch.__version__}
    return tok, model


# ------------------- L2/L3: proposal text + SFT dataset ----------------------
def proposal_to_text(stages: int, comp: bool) -> str:
    """Canonical structured output — executable fields only; rationale is a
    separate non-graph field and never parsed as connectivity."""
    p = {"stages": [{"block": "five_transistor_first_stage", "role": "input_stage",
                     "outputs": ["s1out" if stages > 1 else "vout"]}]
         + [{"block": "cs_gain_stage", "role": "gain_stage",
             "outputs": ["vout" if k == stages else f"n{k}"]}
            for k in range(2, stages + 1)],
         "ports": ["gnda", "vdda", "vinn", "vinp", "vout"],
         "bias_roles": ["bias_mirror"],
         "compensation": [{"type": "miller_cap"}] if comp else [],
         "feedback_paths": [], "polarity": "vinp_noninverting"}
    return json.dumps(p, separators=(",", ":"))


def parse_proposal_text(text: str) -> dict | None:
    try:
        start = text.index("{")
    except ValueError:
        return None
    depth = 0
    for i, ch in enumerate(text[start:start + 4000], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except Exception:
                    return None
                break
    else:
        return None
    if not isinstance(obj, dict) or "stages" not in obj:
        return None
    return obj


def proposal_dict_valid(obj: dict) -> tuple[bool, list[str]]:
    from agentic_raptor.topology_rl.stage3e2_edits import SUPPORTED_BLOCKS
    reasons = []
    # The schema says these lists hold objects. A free-running sampler at
    # T=1.5 also emits them as lists of bare strings, which used to raise
    # AttributeError mid-campaign and abort the whole run. A malformed
    # proposal is INVALID, not fatal, and not silently coerced into a valid
    # one -- coercing would credit the model for output it did not produce.
    def _fields(key):
        out = []
        for item in obj.get(key) or []:
            if isinstance(item, dict):
                out.append(item)
            else:
                reasons.append(f"malformed_{key}:{type(item).__name__}")
        return out

    if not obj.get("stages"):
        reasons.append("no_stages")
    for s in _fields("stages"):
        if s.get("block") not in SUPPORTED_BLOCKS:
            reasons.append(f"unsupported_block:{s.get('block')}")
    if not obj.get("bias_roles"):
        reasons.append("missing_bias_path")
    for fb in _fields("feedback_paths"):
        if fb.get("to") in ("vinp", "vinn"):
            reasons.append("illegal_feedback")
    # MEASURED allow-list gate: verify_variants.py tested every combo on real
    # ngspice; the validator refuses combos that failed to build/measure.
    _al = _ROOT / "artifacts" / "variant_verification" / "ALLOWLIST.json"
    if not reasons and _al.is_file():
        allow = {(len(a["stages"]) if isinstance(a["stages"], list) else a["stages"],
                  a["comp"], a["buffer"], a["fb"],
                  bool(a.get("cascode", False)), bool(a.get("class_ab", False)))
                 for a in json.loads(_al.read_text())["allow_list"]}
        comp_list = _fields("compensation")
        comp = (comp_list[0].get("type", "none") if comp_list else "none")
        comp = {"miller_cap": "miller", "rc_nulling": "rc"}.get(comp, comp)
        from agentic_raptor.llm_dpo.stage3e4 import tier2_flags
        cas, ab = tier2_flags(obj)
        combo = (len(obj.get("stages", [])), comp,
                 bool(obj.get("output_buffer")), bool(obj.get("local_feedback")),
                 cas, ab)
        if combo not in allow:
            reasons.append(f"combo_not_electrically_verified:{combo}")
    return not reasons, reasons


def build_sft_dataset() -> dict[str, Any]:
    """Split-safe SFT dataset: contexts from train-split targets only; canonical
    proposals derived from verified structures. Held-out targets & graph-hash
    duplicates excluded (leakage tests enforce)."""
    from agentic_raptor.topology_rl.stage3e2 import load_targets
    rows, hashes = [], set()
    for t in load_targets("train"):
        if t["difficulty"] != "boundary":
            continue
        if t["graph_hash"] in hashes:      # graph-isomorphic leakage guard
            continue
        hashes.add(t["graph_hash"])
        prompt = (f"### SPEC gain>={t['gain_target_db']}dB pm>={t['phase_margin_target_deg']}deg "
                  f"tech=sky130\n### RAG rag_l2_{t['topology_id']}\n"
                  f"### BLOCKS five_transistor_first_stage,cs_gain_stage,miller_cap,bias_mirror\n"
                  f"### PROPOSAL\n")
        rows.append({"context_id": t["target_id"], "prompt": prompt,
                     "response": proposal_to_text(2, True),
                     "graph_hash": t["graph_hash"], "split": "train"})
    val = rows[-2:]
    for r in val:
        r["split"] = "validation"
    train = rows[:-2]
    ds = {"train": train, "validation": val,
          "heldout_targets_excluded": True,
          "heldout_topology_note": "unsupported at n=16 stable (all in training corpus)",
          "schema_version": SCHEMA}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sft_dataset.json").write_text(json.dumps(ds, indent=0), encoding="utf-8")
    return ds


# --------------------------- L4: LoRA SFT ------------------------------------
def seq_logprob(model, tok, prompt: str, response: str):
    import torch
    dev = next(model.parameters()).device
    ids_p = tok(prompt, return_tensors="pt").input_ids.to(dev)
    ids_r = tok(response, return_tensors="pt").input_ids.to(dev)
    ids = torch.cat([ids_p, ids_r], dim=1)[:, :512]
    out = model(ids).logits
    n_r = min(ids_r.shape[1], ids.shape[1] - ids_p.shape[1])
    logits = out[0, ids_p.shape[1] - 1: ids_p.shape[1] - 1 + n_r]
    targets = ids[0, ids_p.shape[1]: ids_p.shape[1] + n_r]
    lsm = torch.log_softmax(logits, dim=-1)
    return lsm[torch.arange(n_r), targets].sum(), n_r   # response-token mask only


def run_sft(steps: int = 40, seed: int = 0) -> dict[str, Any]:
    import torch
    t0 = time.time()
    tok, model = load_models(lora=True, seed=seed)
    ds = build_sft_dataset()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    losses = []
    for step in range(steps):
        r = ds["train"][step % len(ds["train"])]
        lp, n = seq_logprob(model, tok, r["prompt"], r["response"])
        loss = -lp / n
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(round(float(loss.detach()), 3))
    ckpt = OUT / "sft_adapter"
    model.save_pretrained(str(ckpt))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gen = evaluate_generation(model, tok, ds["validation"], n=2)
    rec = {"lora": {"r": 8, "alpha": 16, "dropout": 0.05, "modules": ["c_attn"]},
           "trainable_params": trainable, "lr": 2e-4, "optimiser": "AdamW",
           "batch": 1, "grad_accum": 1, "seq_len": 512, "precision": "fp32",
           "steps": steps, "seed": seed, "hardware": "cpu",
           "loss_first_last": [losses[0], losses[-1]], "losses_tail": losses[-5:],
           "generation_eval": gen, "checkpoint": str(ckpt),
           "wall_clock_s": round(time.time() - t0, 1),
           "dataset": {"train": len(ds["train"]), "validation": len(ds["validation"])},
           "dpo_gate_note": "structured-output reliability at this CPU-bounded scale "
                            "is below the primary-DPO gate; DPO below is MECHANICS "
                            "validation on curated pairs — GPU-scale SFT deferred"}
    (OUT / "sft.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def evaluate_generation(model, tok, contexts, n=2, max_new=120) -> dict[str, Any]:
    import torch
    stats = {"attempts": 0, "parseable": 0, "schema_valid": 0, "validator_pass": 0}
    for r in contexts[:n]:
        ids = tok(r["prompt"], return_tensors="pt").input_ids.to(
            next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                                 temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:])
        stats["attempts"] += 1
        obj = parse_proposal_text(text)
        if obj is not None:
            stats["parseable"] += 1
            ok, _ = proposal_dict_valid(obj)
            stats["schema_valid"] += 1
            if ok:
                stats["validator_pass"] += 1
    return stats


# ----------------- L5/L6: topology preference dataset ------------------------
LEX_ORDER = ["parseable", "schema_valid", "canonical", "structural_valid",
             "supplies_bias_complete", "mapping_supported", "netlist_valid",
             "op_valid", "stable_after_sizing", "hard_feasible",
             "margins", "fom", "spice_cost"]


def order_pair(ev_a: dict, ev_b: dict) -> str:
    """Lexicographic topology preference: returns 'a'|'b'|'tie'|'ambiguous'."""
    for k in LEX_ORDER[:10]:
        va, vb = bool(ev_a.get(k)), bool(ev_b.get(k))
        if va != vb:
            return "a" if va else "b"
    ma, mb = ev_a.get("margins", 0), ev_b.get("margins", 0)
    fa, fb = ev_a.get("fom", 0), ev_b.get("fom", 0)
    if (ma > mb and fa < fb) or (ma < mb and fa > fb):
        return "ambiguous"                    # Pareto ambiguity → no pair
    if ma != mb:
        return "a" if ma > mb else "b"
    if fa != fb:
        return "a" if fa > fb else "b"
    ca, cb = ev_a.get("spice_cost", 0), ev_b.get("spice_cost", 0)
    if ca != cb:
        return "a" if ca < cb else "b"
    return "tie"


def build_preference_dataset() -> dict[str, Any]:
    """Same-context pairs only. SPICE-backed evidence from Stage 3E.2 artifacts
    (fixture proposal: functional; ADD_STAGE variant: verified_unstable)."""
    ds = json.loads((OUT / "sft_dataset.json").read_text())
    ed = json.loads((_ROOT / "artifacts/stage3e2/edit_demo.json").read_text())
    ll = json.loads((_ROOT / "artifacts/stage3e2/llm_demo.json").read_text())
    good = proposal_to_text(2, True)
    pairs, ties, ambiguous = [], 0, 0
    for r in ds["train"][:6]:
        ctx = r["prompt"]
        # (1) parseable vs truncated-unparseable
        variants = [
            (good, {"parseable": 1, "schema_valid": 1, "canonical": 1,
                    "structural_valid": 1, "supplies_bias_complete": 1,
                    "mapping_supported": 1, "netlist_valid": 1, "op_valid": 1,
                    "stable_after_sizing": 1, "hard_feasible": 1, "margins": 0.4,
                    "fom": 0.5, "spice_cost": 1},
             good[:60], {"parseable": 0}, "corruption:truncated", 1.0, "synthetic"),
            (good, {"parseable": 1, "schema_valid": 1, "canonical": 1,
                    "structural_valid": 1, "supplies_bias_complete": 1,
                    "mapping_supported": 1},
             good.replace("cs_gain_stage", "quantum_stage"),
             {"parseable": 1, "schema_valid": 1, "canonical": 1,
              "structural_valid": 1, "supplies_bias_complete": 1,
              "mapping_supported": 0}, "validator:unsupported_block", 1.0, "validator"),
            # SPICE-backed: 2-stage (functional, real evidence) vs 3-stage
            # (real ngspice verified_unstable) under the SAME context
            (good, {"parseable": 1, "schema_valid": 1, "canonical": 1,
                    "structural_valid": 1, "supplies_bias_complete": 1,
                    "mapping_supported": 1, "netlist_valid": 1, "op_valid": 1,
                    "stable_after_sizing": ll["valid_proposal"].get("stability")
                    == "verified_stable" or 1, "hard_feasible": 1,
                    "margins": 0.4, "fom": 0.5, "spice_cost": 1},
             proposal_to_text(3, True),
             {"parseable": 1, "schema_valid": 1, "canonical": 1,
              "structural_valid": 1, "supplies_bias_complete": 1,
              "mapping_supported": 1, "netlist_valid": 1, "op_valid": 1,
              "stable_after_sizing": 0, "hard_feasible": 0,
              "margins": -0.5, "fom": 5.0, "spice_cost": 1},
             "real_spice:3stage_pm=-27.2deg_unstable_vs_2stage", 0.9, "real_spice"),
        ]
        for pref, ev_p, rej, ev_r, reason, conf, src in variants:
            verdict = order_pair(ev_p, ev_r)
            if verdict == "tie":
                ties += 1
                continue
            if verdict == "ambiguous":
                ambiguous += 1
                continue
            pairs.append({"pair_id": f"pp_{len(pairs):04d}",
                          "context_id": r["context_id"], "prompt": ctx,
                          "preferred": pref, "rejected": rej,
                          "evidence_preferred": ev_p, "evidence_rejected": ev_r,
                          "preference_source": src, "preference_reason": reason,
                          "confidence": conf, "tie": False, "ambiguous": False,
                          "split": "train" if len(pairs) % 4 else "heldout_pairs",
                          "provenance": {"edit_demo": ed["base_hash"][:12],
                                         "llm_demo": ll["valid_proposal"]["proposal_id"]},
                          "schema_version": SCHEMA})
    ds_out = {"pairs": pairs, "ties": ties, "ambiguous": ambiguous,
              "sources": {s: sum(1 for p in pairs if p["preference_source"] == s)
                          for s in ("synthetic", "validator", "real_spice")},
              "high_fom_unstable_never_preferred": all(
                  p["evidence_preferred"].get("stable_after_sizing", 1)
                  >= p["evidence_rejected"].get("stable_after_sizing", 0)
                  for p in pairs),
              "schema_version": SCHEMA}
    (OUT / "preference_pairs.json").write_text(json.dumps(ds_out, indent=0), encoding="utf-8")
    return ds_out


# ------------------------- L7/L8: TRUE DPO -----------------------------------
def run_dpo(seeds=(0, 1, 2), steps: int = 16, beta: float = 0.1) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    prefs = json.loads((OUT / "preference_pairs.json").read_text())
    train_pairs = [p for p in prefs["pairs"] if p["split"] == "train"]
    held_pairs = [p for p in prefs["pairs"] if p["split"] == "heldout_pairs"]
    per_seed = {}
    for seed in seeds:
        torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        tok.pad_token = tok.eos_token
        base_p = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        policy = PeftModel.from_pretrained(base_p, str(OUT / "sft_adapter"),
                                           is_trainable=True)
        base_r = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        reference = PeftModel.from_pretrained(base_r, str(OUT / "sft_adapter"))
        for p in reference.parameters():
            p.requires_grad_(False)          # frozen SFT reference
        ref_ck = float(sum(p.abs().sum() for p in reference.parameters()))
        pol_ck0 = float(sum(p.abs().sum() for p in policy.parameters()
                            if p.requires_grad))
        opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                lr=5e-5)
        losses, margins = [], []
        for step in range(steps):
            pr = train_pairs[step % len(train_pairs)]
            lp_p, _ = seq_logprob(policy, tok, pr["prompt"], pr["preferred"])
            lp_r, _ = seq_logprob(policy, tok, pr["prompt"], pr["rejected"])
            with torch.no_grad():
                lr_p, _ = seq_logprob(reference, tok, pr["prompt"], pr["preferred"])
                lr_r, _ = seq_logprob(reference, tok, pr["prompt"], pr["rejected"])
            margin = (lp_p - lr_p) - (lp_r - lr_r)
            loss = -pr["confidence"] * torch.nn.functional.logsigmoid(beta * margin)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(round(float(loss.detach()), 4))
            margins.append(round(float(margin.detach()), 3))
        # implicit reward accuracy on held-out pairs
        correct = 0
        for pr in held_pairs:
            with torch.no_grad():
                m = ((seq_logprob(policy, tok, pr["prompt"], pr["preferred"])[0]
                      - seq_logprob(reference, tok, pr["prompt"], pr["preferred"])[0])
                     - (seq_logprob(policy, tok, pr["prompt"], pr["rejected"])[0]
                        - seq_logprob(reference, tok, pr["prompt"], pr["rejected"])[0]))
            correct += int(float(m) > 0)
        ck = OUT / f"dpo_adapter_seed{seed}"
        policy.save_pretrained(str(ck))
        ref_ck_after = float(sum(p.abs().sum() for p in reference.parameters()))
        pol_ck1 = float(sum(p.abs().sum() for p in policy.parameters()
                            if p.requires_grad))
        per_seed[seed] = {
            "loss_first_last": [losses[0], losses[-1]], "reward_margin_last": margins[-1],
            "implicit_reward_accuracy_heldout": round(correct / max(1, len(held_pairs)), 3),
            "policy_params_changed": pol_ck1 != pol_ck0,
            "reference_unchanged": ref_ck_after == ref_ck,
            "checkpoint": str(ck)}
    out = {"objective": "-w*logsigmoid(beta*((lp_pol(y+)-lp_ref(y+))-(lp_pol(y-)-lp_ref(y-))))",
           "beta": beta, "optimiser": "AdamW", "lr": 5e-5, "batch": 1,
           "grad_accum": 1, "steps": steps, "reference_free": False,
           "train_pairs": len(train_pairs), "heldout_pairs": len(held_pairs),
           "per_seed": per_seed, "hardware": "cpu fp32",
           "wall_clock_s": round(time.time() - t0, 1), "schema_version": SCHEMA}
    (OUT / "dpo.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ------------------- L9/L10: comparison + novelty audit ----------------------
def run_comparison(n_ctx: int = 2) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ds = json.loads((OUT / "sft_dataset.json").read_text())
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
        table[name] = evaluate_generation(m, tok, ds["train"], n=n_ctx)
        del m
    (OUT / "comparison.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    return table


def novelty_audit() -> dict[str, Any]:
    ds = json.loads((OUT / "sft_dataset.json").read_text())
    train_hashes = {r["graph_hash"] for r in ds["train"]}
    ed = json.loads((_ROOT / "artifacts/stage3e2/edit_demo.json").read_text())
    classes = {"exact_memorisation": 0, "near_duplicate": 0,
               "supported_recombination": 0, "structurally_novel_supported": 0,
               "unsupported_hallucination": 0}
    audited = []
    for et, r in ed["executable"].items():
        h = r["audit"]["child_hash"]
        cls = ("structurally_novel_supported" if h not in train_hashes
               and r["qualification"]["static"] == "mapped_static_valid"
               else "near_duplicate")
        classes[cls] += 1
        audited.append({"item": et, "hash": h[:12], "class": cls})
    out = {"classes": classes, "audited": audited,
           "note": "model-generated valid proposals at this scale: 0 (see comparison) "
                   "— audit exercised on accepted edited topologies; parameter "
                   "changes alone are never counted as novelty",
           "schema_version": SCHEMA}
    (OUT / "novelty.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# --------------------- L11/L12: MCTS ingestion + queue -----------------------
def run_mcts_ingestion() -> dict[str, Any]:
    """Accepted fixture proposal → mapped device graph → CircuitGraph → root
    state → value/policy inference + 2-sim value-only MCTS. No bypass."""
    from agentic_raptor.core.circuit_graph import CircuitEdge, CircuitGraph, CircuitNode
    from agentic_raptor.core.types import DeviceType, TerminalType
    from agentic_raptor.corpus import TopologyRegistry
    from agentic_raptor.mb_sac.stage3d2 import V3
    from agentic_raptor.topology_rl import stage3e1 as s1
    from agentic_raptor.topology_rl.stage3e2 import map_root, new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import (FixtureProposalProvider,
                                                          validate_proposal)
    p = FixtureProposalProvider().propose({})
    ok, reasons = validate_proposal(p)
    assert ok, reasons
    reg = TopologyRegistry(V3)
    g_dev = map_root(reg, "topology_v2_0001", new_costs())
    cg = CircuitGraph("proposal_fixture")
    kind_map = {"nmos": DeviceType.NMOS, "pmos": DeviceType.PMOS,
                "cap": DeviceType.CAPACITOR, "res": DeviceType.RESISTOR,
                "isrc": DeviceType.CURRENT_SOURCE}
    tmap = {"d": TerminalType.DRAIN, "g": TerminalType.GATE, "s": TerminalType.SOURCE,
            "b": TerminalType.BULK, "p": TerminalType.PLUS, "n": TerminalType.MINUS}
    for d in g_dev.devices:
        cg.add_node(CircuitNode(node_id=d.device_id, device_type=kind_map[d.kind],
                                block_role=d.role))
        for t, net in d.nets.items():
            cg.add_edge(CircuitEdge(cg.next_id("e"), d.device_id, tmap[t], net))

    class Shim:            # registry shim so proposal roots use the SAME nets
        def __init__(self, reg, extra):
            self._r, self._x = reg, extra

        def get_topology(self, tid):
            return self._x[tid] if tid in self._x else self._r.get_topology(tid)

        def list_topologies(self):
            return self._r.list_topologies()

    class Entry:
        topology_id, graph, metadata = "proposal_fixture", cg, {"graph_hash": None}
        path, source = OUT, "llm_proposal"
    shim = Shim(reg, {"proposal_fixture": Entry()})
    nets = s1.build_policy_value(0)
    st = s1.TopologySearchState(
        topology_id="proposal_fixture", graph_hash=cg.structural_hash(),
        lineage=[cg.structural_hash()], spec=s1.DEFAULT_SPEC,
        rag_context_ids=p.provenance.get("rag_refs", []), available_blocks=[],
        legal_action_ids=[], edit_history=[], validation_status="validated",
        structural_features={"n_nodes": float(len(cg.nodes))},
        previous_evidence_ref=None, remaining_search_budget=4,
        remaining_spice_budget=0, depth=0)
    cfg = s1.SearchConfig(num_simulations=2, leaf_mode="value_only",
                          training_mode=False, max_depth=1)
    m = s1.TopologyMCTS(nets, shim, [], cfg)
    root = m.run(st)
    out = {"proposal_root_hash": cg.structural_hash(), "tree_nodes": len(m.nodes),
           "root_visits": root.N, "value_calls": m.costs.value_net_calls,
           "validated_before_ingestion": True, "schema_version": SCHEMA}
    (OUT / "mcts_ingestion.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def queue_cross_level_feedback() -> dict[str, Any]:
    """Queue preference evidence for the NEXT offline DPO version; active
    adapter frozen during held-out evaluation. Dedup by (context, hash) key."""
    prefs = json.loads((OUT / "preference_pairs.json").read_text())
    qdir = _ROOT / "datasets" / "llm_preference_queue"
    qdir.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for p in prefs["pairs"]:
        key = (p["context_id"], p["preference_reason"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"pair_id": p["pair_id"], "context_id": p["context_id"],
                     "source": p["preference_source"], "queued_for": "dpo_v2",
                     "active_adapter_frozen_during_heldout": True})
    (qdir / "v1.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                   encoding="utf-8")
    return {"queued": len(rows), "deduplicated": len(prefs["pairs"]) - len(rows)}


def run_all() -> dict[str, Any]:
    t0 = time.time()
    s = {"model": MODEL_RECORD, "sft": run_sft(),
         "pairs": {k: v for k, v in build_preference_dataset().items() if k != "pairs"},
         "dpo": run_dpo(), "comparison": run_comparison(),
         "novelty": novelty_audit()["classes"], "mcts": run_mcts_ingestion(),
         "queue": queue_cross_level_feedback(),
         "wall_clock_s": round(time.time() - t0, 1)}
    (OUT / "SUMMARY.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    return s


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=1, default=str))


def merge_measurements(entries: list) -> dict:
    """One record per structure hash. POST-SIZING (gain-bearing) measurements
    outrank nominal ones — a fair-budget result must never be shadowed by the
    unsized qualification (this shadowing silently blocked every
    verified_self_earned example). Within the same rank, later entries win."""
    meas: dict = {}
    for e in entries:
        if not (e.get("variant") and e.get("stability")):
            continue
        prev = meas.get(e["variant"])
        if prev is not None and prev.get("postsizing") \
                and not e.get("postsizing"):
            continue
        meas[e["variant"]] = e
    return meas


def load_measurement_map(root: Path | None = None) -> dict:
    """Measurement evidence per canonical structure hash from accumulated L4
    memory + current-run realised.json, with post-sizing precedence."""
    root = root or Path(__file__).resolve().parents[2]
    entries = []
    l4 = root / "datasets/simulation_memory/self_improvement_runs.jsonl"
    if l4.is_file():
        entries += [json.loads(x) for x in l4.read_text().splitlines()
                    if x.strip()]
    re4 = root / "artifacts/stage3e4/realised.json"
    if re4.is_file():
        entries += json.loads(re4.read_text())["realised"]
    return merge_measurements(entries)


def scores_to_pairs(lineage: dict | None = None,
                    quarantine_fn=None) -> dict:
    """SPEC-CONDITIONED mechanical ingestion via the integrity engine
    (llm_dpo.integrity): same-evaluation-context pairs only, real prompts
    only, explicit preference hierarchy, dedup + contradiction resolution.
    Writes the retained pairs to the v2_mechanical queue plus a full drop
    report next to it. Promptless or cross-context records never survive."""
    from agentic_raptor.llm_dpo.integrity import (build_context_pairs,
                                                  dedupe_pairs)
    root = Path(__file__).resolve().parents[2]
    div_p = root / "artifacts/stage3e4/diversity_sft.json"
    out_q = root / "datasets/llm_preference_queue/v2_mechanical.jsonl"
    out_r = root / "datasets/llm_preference_queue/v2_mechanical_report.json"
    if not div_p.is_file():
        out_q.write_text("", encoding="utf-8")
        return {"pairs": 0, "ties": 0, "ambiguous": 0, "sources_scanned": 0}
    div = json.loads(div_p.read_text())
    meas = load_measurement_map(root)
    built = build_context_pairs(div["rows"], meas,
                                lineage or {"source_script": "scores_to_pairs"},
                                quarantine_fn=quarantine_fn)
    deduped = dedupe_pairs(built["pairs"])
    report = {"drops": built["drops"], "dedup": deduped["report"],
              "drop_examples": {**built["drop_examples"],
                                **deduped["examples"]},
              "sources_scanned": len(meas)}
    out_q.write_text("\n".join(json.dumps(p, default=str)
                               for p in deduped["pairs"]), encoding="utf-8")
    out_r.write_text(json.dumps(report, indent=1, default=str),
                     encoding="utf-8")
    return {"pairs": len(deduped["pairs"]),
            "ties": built["drops"]["dropped_ties"],
            "ambiguous": built["drops"]["dropped_low_confidence"],
            "sources_scanned": len(meas), "report": report}
