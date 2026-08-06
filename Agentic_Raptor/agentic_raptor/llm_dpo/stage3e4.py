"""Stage 3E.4: multi-structure corpus, diversity-aware generation,
model-generated DPO pairs with unique physical comparison groups, equal-budget
comparison with bootstrap CIs, multi-proposal realisation + MCTS refinement.

HARDWARE: no CUDA device on this host. The GPU 1B-7B campaign is BLOCKED
(exact blocker: torch reports cuda=False). MODEL_ID is configurable via
AGENTIC_RAPTOR_TOPOLOGY_LLM env var; this run executes the identical pipeline
at CPU pilot scale (distilgpt2) and every gate is reported honestly.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from random import Random
from typing import Any

from agentic_raptor.llm_dpo import (MODEL_ID as _DEFAULT_MODEL, OUT, load_models,
                                    parse_proposal_text, proposal_dict_valid,
                                    seq_logprob)
from agentic_raptor.utils.seeding import apply_torch_omp_workaround

_ROOT = Path(__file__).resolve().parents[2]
O4 = _ROOT / "artifacts" / "stage3e4"
SCHEMA = "3e4.1"
MODEL_ID = os.environ.get("AGENTIC_RAPTOR_TOPOLOGY_LLM", _DEFAULT_MODEL)

#: structure variants realisable through the EXISTING executable infrastructure
VARIANTS = [v for v in itertools.product((1, 2, 3), ("none", "miller", "rc"),
                                         (False, True), (False, True))
            if not (v[0] == 1 and v[1] != "none")]     # comp needs >=2 stages


def variant_text(stages: int, comp: str, buf: bool, fb: bool) -> str:
    p = {"stages": [{"block": "five_transistor_first_stage", "role": "input_stage",
                     "outputs": ["s1out" if stages > 1 else "vout"]}]
         + [{"block": "cs_gain_stage", "role": "gain_stage",
             "outputs": ["vout" if k == stages else f"n{k}"]}
            for k in range(2, stages + 1)],
         "ports": ["gnda", "vdda", "vinn", "vinp", "vout"],
         "bias_roles": ["bias_mirror"],
         "compensation": ([] if comp == "none" else
                          [{"type": "miller_cap" if comp == "miller" else "rc_nulling"}]),
         "output_buffer": buf, "local_feedback": fb,
         "feedback_paths": [], "polarity": "vinp_noninverting"}
    return json.dumps(p, separators=(",", ":"))


def variant_hash(obj: dict) -> str:
    key = (len(obj.get("stages", [])),
           (obj.get("compensation") or [{}])[0].get("type", "none")
           if obj.get("compensation") else "none",
           bool(obj.get("output_buffer")), bool(obj.get("local_feedback")))
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]


def build_corpus() -> dict[str, Any]:
    """Spec-conditioned multi-structure corpus: gain tier deterministically
    selects structure (higher gain -> more stages/compensation)."""
    from agentic_raptor.topology_rl.stage3e2 import load_targets
    rng = Random(3e4 and 34)
    recs, seen = [], set()
    all_t = load_targets()
    # RAG L4 evidence: measured outcomes retrieved into TRAIN prompts only
    # (held-out prompts stay evidence-free — frozen exam, no leakage)
    l4p = _ROOT / "datasets/simulation_memory/self_improvement_runs.jsonl"
    #: retrieval pool = every measured record. Selection is by relevance, so
    #: a recency window only limits what can ever be found (the store holds
    #: ~750 measurements; the old window saw the last 20 of them).
    l4 = ([json.loads(x) for x in l4p.read_text().splitlines()]
          if l4p.is_file() else [])
    # Fix 1: belt-and-braces. Evaluation specs are already excluded from the
    # design pool so they should never reach L4 -- but RAG injects evidence
    # straight into training prompts, so it re-checks rather than trusting.
    try:
        from agentic_raptor.publication.eval_sets import excluded_context_ids
        _frozen = excluded_context_ids()
        l4 = [e for e in l4 if e.get("context_id") not in _frozen]
    except Exception:           # no evaluation sets built yet
        pass

    def known_line(stages: int, pm_target: float, gain_target: float,
                   cl_pf: float) -> str:
        """Evidence for THIS spec, ranked by closeness to it.

        Previously: filter on stage count, take the last 2 by recency. Two
        problems. The stage count came from the gain-tier rule that
        check_stage_rule.py measured false (2-stage landed closer on 6 of 8
        high-gain specs), so retrieval bucketed by a broken proxy. And it
        ignored the phase-margin target entirely -- the constraint that
        actually fails on every design -- so a 60 deg request could be shown
        a 34 deg result purely because it ran most recently.

        Now every measured outcome competes, ranked by distance in the
        dimensions the spec states. Stage count is a mild preference, not a
        filter, so evidence from a different structure class can surface
        when it is genuinely the closest match.
        """
        cands = [e for e in l4 if e.get("stability")]
        if not cands:
            return ""

        def distance(e):
            """Scales matter more than weights here. dB gaps run to tens
            while degree gaps run to ones, so dividing gain by 20 and PM by
            45 let gain dominate ~10x -- the opposite of the intent. Each
            term is normalised by what counts as a BIG miss in its own unit
            (15 deg of phase margin, 30 dB of gain), then PM is weighted up
            because it is the constraint that actually fails."""
            d = 0.0
            if e.get("pm") is not None and pm_target:
                d += 2.0 * abs(e["pm"] - pm_target) / 15.0
            else:
                d += 2.0
            if e.get("gain_db") is not None and gain_target:
                d += abs(e["gain_db"] - gain_target) / 30.0
            else:
                d += 0.5                    # unmeasured gain: mild penalty
            d += 0.25 * (e.get("stages") != stages)         # preference only
            d += 0.15 * (not e.get("postsizing"))   # tuned results preferred
            return d
        seen_ev, hits = set(), []
        for e in sorted(cands, key=distance):
            # two identical lines waste the context window and teach nothing
            line = (f"{e['stages']}stage {e['stability']}"
                    + (f" pm={round(e['pm'])}deg"
                       if e.get("pm") is not None else ""))
            if line in seen_ev:
                continue
            seen_ev.add(line)
            hits.append(line)
            if len(hits) == 2:
                break
        return "### KNOWN " + "; ".join(hits) + "\n" if hits else ""
    enriched = []
    for t in all_t:
        for draw in (0, 1):     # two spec-enrichment draws per target
            g = t["gain_target_db"]
            pm = t["phase_margin_target_deg"]
            # Spec ENRICHMENT v4: load/UGBW drawn by a seeded PER-TARGET RNG —
            # independent of record ordering. Structure remains a learnable
            # function of VISIBLE spec only.
            _r = Random(f"{t['target_id']}:enrich{draw}")
            cl_pf = _r.choice((50, 100, 200, 500, 1000))
            ugbw = _r.choice((1e4, 1e5, 1e6))
            stages = 1 if g < 30 else 2 if g < 70 else 3
            comp = ("none" if stages == 1
                    else "none" if cl_pf <= 100         # light load: parasitic-stable
                    else "rc" if pm >= 55               # tight margin -> nulling
                    else "miller")
            enriched.append((t, draw, cl_pf, ugbw, stages, comp))
    # GROUPED STRATIFIED splits (Part C): the assignment unit is the CANONICAL
    # TOPOLOGY FAMILY (topology_id), never the target — a family in two splits
    # would leak graph-equivalent circuits and RAG references across the exam
    # boundary. Families are stratified by the profile of structure classes
    # their targets produce, then cycled train/train/train/heldout/validation.
    # A deterministic offset search guarantees every structure class reaches
    # every split; the whole assignment is seed-fixed so the exam is frozen.
    fam_classes: dict[str, list] = {}
    fam_targets: dict[str, set] = {}
    for t, draw, cl_pf, ugbw, stages, comp in enriched:
        fam_classes.setdefault(t["topology_id"], []).append((stages, comp))
        fam_targets.setdefault(t["topology_id"], set()).add(t["target_id"])
    profile_groups: dict[tuple, list[str]] = {}
    for fam in sorted(fam_classes):
        profile_groups.setdefault(
            tuple(sorted(set(fam_classes[fam]))), []).append(fam)
    ordered_fams = [f for key in sorted(profile_groups)
                    for f in profile_groups[key]]
    all_classes = {c for v in fam_classes.values() for c in v}
    # the (i+offset)%5==4 slot is the BLIND TEST set (Task 1): family-disjoint
    # from train and from the frozen held-out validation exam, never touched
    # by training, DPO, rollback decisions or debugging — evaluated exactly
    # once after all campaign decisions are frozen. Position 3 (heldout) is
    # UNCHANGED, so the frozen validation exam and its hash stay identical.
    fam_split, split_seed = None, None
    for offset in range(5):     # deterministic search for full class coverage
        trial = {f: ("heldout" if (i + offset) % 5 == 3
                     else "blindtest" if (i + offset) % 5 == 4 else "train")
                 for i, f in enumerate(ordered_fams)}
        cover = {s: set() for s in ("train", "blindtest", "heldout")}
        for f, cls in fam_classes.items():
            cover[trial[f]].update(cls)
        if all(cover[s] == all_classes for s in cover):
            fam_split, split_seed = trial, offset
            break
    if fam_split is None:       # fall back to best offset, still deterministic
        fam_split, split_seed = ({f: ("heldout" if i % 5 == 3 else
                                      "blindtest" if i % 5 == 4 else "train")
                                  for i, f in enumerate(ordered_fams)}, 0)
    split_of = {tid: fam_split[fam]
                for fam, tids in fam_targets.items() for tid in tids}
    for t, draw, cl_pf, ugbw, stages, comp in enriched:
        g = t["gain_target_db"]
        # Output buffer DISABLED in corpus until the follower template passes
        # electrical verification (NMOS variant: below threshold, -76dB; PMOS
        # variant: rails high, -129dB — DC bias debug in progress, see
        # artifacts/buf_fix). Honest gate: don't teach what we can't build.
        buf = False
        # feedback edit failed the measured sweep in ALL combos -> gated off
        # (like buffer) until its template is fixed and re-verified
        fb = False
        t = dict(t, load_pf=cl_pf, ugbw_spec_hz=ugbw)
        resp = variant_text(stages, comp, buf, fb)
        vh = variant_hash(json.loads(resp))
        split_v = split_of[t["target_id"]]
        prompt = (f"### SPEC gain>={g}dB pm>={t['phase_margin_target_deg']}deg "
                  f"cl={cl_pf}pF ugbw>={ugbw:.0e}Hz tech=sky130\n"
                  f"### RAG rag_l2_{t['topology_id']}\n"
                  + (known_line(stages, t["phase_margin_target_deg"], g,
                                 cl_pf) if split_v == "train" else "")
                  + f"### BLOCKS five_transistor_first_stage,cs_gain_stage,miller_cap,bias_mirror\n"
                  f"### FORBIDDEN raw_netlist,feedback_to_input\n### PROPOSAL\n")
        key = hashlib.sha256((prompt + resp).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        recs.append({"context_id": t["target_id"], "topology_id": t["topology_id"],
                     "prompt": prompt, "response": resp, "variant_hash": vh,
                     "stages": stages, "comp": comp, "buffer": buf, "fb": fb,
                     "split": split_v})
    # SPEC-line collision guard (Part C/J): enrichment draws can produce a
    # byte-identical SPEC line across splits. Precedence: heldout (frozen
    # validation exam) wins over everything; blindtest wins over train — the
    # colliding lower-precedence record is dropped, so spec leakage becomes
    # structurally impossible instead of merely detected downstream.
    held_spec_lines = {r["prompt"].splitlines()[0] for r in recs
                       if r["split"] == "heldout"}
    collisions = [r for r in recs if r["split"] != "heldout"
                  and r["prompt"].splitlines()[0] in held_spec_lines]
    recs = [r for r in recs if r["split"] == "heldout"
            or r["prompt"].splitlines()[0] not in held_spec_lines]
    blind_spec_lines = {r["prompt"].splitlines()[0] for r in recs
                        if r["split"] == "blindtest"}
    collisions += [r for r in recs if r["split"] == "train"
                   and r["prompt"].splitlines()[0] in blind_spec_lines]
    recs = [r for r in recs if r["split"] != "train"
            or r["prompt"].splitlines()[0] not in blind_spec_lines]
    classes = {r["variant_hash"] for r in recs}
    # HARD leak assertions (Part C): stop execution on any overlap
    fam_by_split: dict[str, set] = {}
    ctx_by_split: dict[str, set] = {}
    for r in recs:
        fam_by_split.setdefault(r["split"], set()).add(r["topology_id"])
        ctx_by_split.setdefault(r["split"], set()).add(r["context_id"])
    for a in fam_by_split:
        for b in fam_by_split:
            if a < b:
                assert not (fam_by_split[a] & fam_by_split[b]), \
                    f"topology family leaked across splits {a}/{b}"
                assert not (ctx_by_split[a] & ctx_by_split[b]), \
                    f"spec context leaked across splits {a}/{b}"
    assert all(fam_by_split.get(s) for s in ("train", "blindtest", "heldout")), \
        "a split is empty"
    from agentic_raptor.llm_dpo.integrity import build_exam_manifest, sha_json
    stats = {"raw": len(all_t), "valid": len(recs), "deduplicated": len(recs),
             "iso_classes": len(classes),
             "stage_distribution": {s: sum(1 for r in recs if r["stages"] == s)
                                    for s in (1, 2, 3)},
             "comp_distribution": {c: sum(1 for r in recs if r["comp"] == c)
                                   for c in ("none", "miller", "rc")},
             "buffer": sum(r["buffer"] for r in recs),
             "feedback": sum(r["fb"] for r in recs),
             "cross_split_spec_collisions_dropped": len(collisions),
             "splits": {s: sum(1 for r in recs if r["split"] == s)
                        for s in ("train", "blindtest", "heldout")}}
    corpus_hash = sha_json([(r["prompt"], r["response"], r["split"])
                            for r in recs])
    split_manifest = {
        "split_scheme": "family_grouped_stratified_v2_blind",
        "split_seed": split_seed,
        "corpus_hash": corpus_hash,
        "split_hash": sha_json(sorted(split_of.items())),
        **{f"{s}_spec_ids": sorted(ctx_by_split.get(s, set()))
           for s in ("train", "blindtest", "heldout")},
        **{f"{s}_family_ids": sorted(fam_by_split.get(s, set()))
           for s in ("train", "blindtest", "heldout")}}
    # required manifest field names: heldout = the frozen VALIDATION exam
    # (used for checkpoint acceptance); blindtest = untouched final test
    split_manifest["validation_spec_ids"] = split_manifest.pop("heldout_spec_ids")
    split_manifest["validation_family_ids"] = \
        split_manifest.pop("heldout_family_ids")
    split_manifest["blind_test_spec_ids"] = split_manifest.pop("blindtest_spec_ids")
    split_manifest["blind_test_family_ids"] = \
        split_manifest.pop("blindtest_family_ids")
    O4.mkdir(parents=True, exist_ok=True)
    corpus_doc = {"records": recs, "stats": stats,
                  "split_manifest": split_manifest, "schema_version": SCHEMA}
    (O4 / "corpus.json").write_text(json.dumps(corpus_doc, indent=0),
                                    encoding="utf-8")
    (O4 / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=1), encoding="utf-8")
    (O4 / "frozen_exam.json").write_text(
        json.dumps(build_exam_manifest(corpus_doc), indent=1),
        encoding="utf-8")
    blind = [r for r in recs if r["split"] == "blindtest"]
    (O4 / "blind_test.json").write_text(json.dumps(
        {"frozen_blind_hash": sha_json(sorted(
            (r["prompt"], r["variant_hash"]) for r in blind)),
         "contexts": len(blind),
         "family_ids": sorted({r["topology_id"] for r in blind}),
         "class_counts": {f"{r0[0]}s_{r0[1]}": sum(
             1 for r in blind if (r["stages"], r["comp"]) == r0)
             for r0 in sorted({(r["stages"], r["comp"]) for r in blind})},
         "policy": "evaluate exactly once, after all campaign decisions are "
                   "frozen; never for training, DPO, rollback or debugging",
         "schema_version": SCHEMA}, indent=1), encoding="utf-8")
    return stats


def run_sft(steps: int = 400, seed: int = 0, corpus_path: Path | None = None,
            out_dir: Path | None = None) -> dict[str, Any]:
    """Supervised fine-tune of the topology proposer.

    corpus_path / out_dir default to the originals, so existing callers are
    unaffected. They exist so the repaired corpus can be trained to a NEW
    adapter: overwriting artifacts/stage3e4/sft_adapter in place would destroy
    the baseline checkpoint every diversity comparison is measured against.
    """
    import torch
    t0 = time.time()
    tok, model = load_models(lora=True, seed=seed)
    corpus_path = Path(corpus_path) if corpus_path else (O4 / "corpus.json")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    train = [r for r in corpus["records"] if r["split"] == "train"]
    if not train:
        raise SystemExit(f"no train records in {corpus_path}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    losses = []
    rng = Random(seed)
    for step in range(steps):
        r = train[rng.randrange(len(train))]
        lp, n = seq_logprob(model, tok, r["prompt"], r["response"])
        loss = -lp / n
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    ck = Path(out_dir) if out_dir else (O4 / "sft_adapter")
    ck.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ck))
    rec = {"steps": steps, "loss_first_last": [round(losses[0], 3), round(losses[-1], 3)],
           "checkpoint": str(ck), "wall_clock_s": round(time.time() - t0, 1),
           "model_id": MODEL_ID, "corpus": str(corpus_path),
           "train_records": len(train),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "schema_version": SCHEMA}
    (ck.parent / f"{ck.name}_sft.json" if out_dir else O4 / "sft.json").write_text(
        json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def generate(model, tok, prompt: str, sample_seed: int) -> dict[str, Any]:
    import torch
    torch.manual_seed(sample_seed)
    ids = tok(prompt, return_tensors="pt").input_ids.to(
        next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=260, do_sample=sample_seed > 0,
                             temperature=0.8, top_p=0.92,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:])
    obj = parse_proposal_text(text)
    valid, reasons = (proposal_dict_valid(obj) if obj else (False, ["unparseable"]))
    return {"seed": sample_seed, "parseable": obj is not None, "valid": valid,
            "reasons": [] if valid else reasons,
            "graph_hash": variant_hash(obj) if obj and valid else None,
            "obj": obj if valid else None,
            "temperature": 0.8, "top_p": 0.92}


#: escalating sample ladder for diverse proposal. A converged proposer is
#: extremely peaked -- measured: 4 samples at T=0.8 collapse to ONE structure
#: class per spec (3 unique across a whole 40-sample generation) -- so a
#: fixed temperature cannot surface the design space. Each rung widens the
#: distribution only as far as needed; the ladder stops the moment the target
#: number of DISTINCT valid structures is reached.
DIVERSITY_LADDER = ((0.8, 0.92, 4), (1.0, 0.95, 4),
                    (1.2, 0.98, 6), (1.5, 0.99, 6))


def propose_diverse(model, tok, prompt: str, target_k: int = 5,
                    ladder=DIVERSITY_LADDER, seed0: int = 0) -> dict[str, Any]:
    """Sample until `target_k` DISTINCT valid structures are found.

    Everything returned is the model's own output -- this widens sampling,
    it does not enumerate or substitute candidates. Telemetry records how
    hard the model had to be pushed, which is itself the finding: if the
    ladder exhausts below target_k the proposer genuinely cannot cover the
    space, and that must be reported rather than papered over.
    """
    import torch
    ids = tok(prompt, return_tensors="pt").input_ids.to(
        next(model.parameters()).device)
    seen: dict[str, dict] = {}
    attempts, rungs_used = 0, []
    for temp, top_p, n in ladder:
        used = False
        for j in range(n):
            if len(seen) >= target_k:
                break
            used = True
            attempts += 1
            torch.manual_seed(seed0 * 1000 + attempts)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=260, do_sample=True,
                                     temperature=temp, top_p=top_p,
                                     pad_token_id=tok.eos_token_id)
            obj = parse_proposal_text(tok.decode(out[0, ids.shape[1]:]))
            if not obj:
                continue
            valid, reasons = proposal_dict_valid(obj)
            if not valid:
                continue
            h = variant_hash(obj)
            if h in seen:
                continue
            seen[h] = {"seed": attempts, "parseable": True, "valid": True,
                       "reasons": [], "graph_hash": h, "obj": obj,
                       "temperature": temp, "top_p": top_p,
                       "source": "llm_diverse"}
        if used:
            rungs_used.append({"temperature": temp, "top_p": top_p})
        if len(seen) >= target_k:
            break
    return {"candidates": list(seen.values()),
            "distinct": len(seen), "target_k": target_k,
            "reached_target": len(seen) >= target_k,
            "attempts": attempts, "rungs_used": rungs_used,
            "max_temperature": rungs_used[-1]["temperature"]
            if rungs_used else None}


#: the corpus teaches exclusion-conditioned proposal with this marker
EXCLUDE_LINE = "### EXCLUDE "


def _exclusion_prompt(prompt: str, seen: dict) -> str:
    """Rebuild the prompt asking for something OTHER than what we have.

    Byte-identical to build_diverse_corpus.py: '<fam>/<hash[:12]>' joined by
    ', ', inserted before '### PROPOSAL'.
    """
    ex = ", ".join(f"{c['family']}/{h[:12]}" for h, c in seen.items())
    return prompt.replace("### PROPOSAL", f"{EXCLUDE_LINE}{ex}\n### PROPOSAL")


def propose_diverse_excl(model, tok, prompt: str, target_k: int = 5,
                         ladder=DIVERSITY_LADDER, seed0: int = 0,
                         family_fn=None) -> dict[str, Any]:
    """Diverse proposal that USES the exclusion conditioning it was trained on.

    propose_diverse() widens temperature only. The repaired corpus also
    contains 340 exclusion-conditioned examples -- "given these structures,
    propose a different one" -- and nothing at serve time ever asked for that,
    so a trained capability sat unused. Measured consequence: 2s_miller has
    139 training records and was emitted ZERO times in 120 temperature-only
    samples, while mean distinct plateaued at 3.5/5.

    Every candidate is still the model's own output; this changes what the
    model is ASKED, not what it is credited with.
    """
    import torch
    seen: dict[str, dict] = {}
    attempts, rungs_used = 0, []

    def _family(obj):
        if family_fn:
            return family_fn(obj)
        from agentic_raptor.llm_dpo import integrity as ig
        try:
            return f"{len(obj['stages'])}s_" + ig.compensation_class(obj)
        except Exception:
            return "unknown"

    for temp, top_p, n in ladder:
        used = False
        for _j in range(n):
            if len(seen) >= target_k:
                break
            used = True
            attempts += 1
            # condition on what we already have, exactly as trained
            cur = _exclusion_prompt(prompt, seen) if seen else prompt
            # pass the mask explicitly: pad_token == eos_token, so transformers
            # cannot infer it. Single un-batched prompt means it is all-ones
            # either way -- this silences the warning, it does not change
            # generation.
            enc = tok(cur, return_tensors="pt").to(
                next(model.parameters()).device)
            ids = enc.input_ids
            torch.manual_seed(seed0 * 1000 + attempts)
            with torch.no_grad():
                out = model.generate(ids,
                                     attention_mask=enc.attention_mask,
                                     max_new_tokens=260, do_sample=True,
                                     temperature=temp, top_p=top_p,
                                     pad_token_id=tok.eos_token_id)
            obj = parse_proposal_text(tok.decode(out[0, ids.shape[1]:]))
            if not obj:
                continue
            valid, _reasons = proposal_dict_valid(obj)
            if not valid:
                continue
            h = variant_hash(obj)
            if h in seen:
                continue
            seen[h] = {"seed": attempts, "parseable": True, "valid": True,
                       "reasons": [], "graph_hash": h, "obj": obj,
                       "family": _family(obj),
                       "temperature": temp, "top_p": top_p,
                       "excluded_count": len(seen),
                       "source": "llm_diverse_exclusion"}
        if used:
            rungs_used.append({"temperature": temp, "top_p": top_p})
        if len(seen) >= target_k:
            break
    return {"candidates": list(seen.values()),
            "distinct": len(seen), "target_k": target_k,
            "reached_target": len(seen) >= target_k,
            "attempts": attempts, "rungs_used": rungs_used,
            "conditioning": "exclusion",
            "max_temperature": rungs_used[-1]["temperature"]
            if rungs_used else None}


def run_diversity_campaign(adapter: str, k: int = 4, label: str = "sft") -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    corpus = json.loads((O4 / "corpus.json").read_text())
    held = [r for r in corpus["records"] if r["split"] == "heldout"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    ctx_rows = []
    for r in held:
        cands = [generate(model, tok, r["prompt"], s) for s in range(k)]
        uniq = {c["graph_hash"] for c in cands if c["valid"]}
        ctx_rows.append({"context_id": r["context_id"], "prompt": r["prompt"],
                         "topology_id": r["topology_id"],
                         "target_variant": r["variant_hash"],
                         "target_stages": r["stages"],
                         "candidates": cands, "valid": sum(c["valid"] for c in cands),
                         "unique_valid": len(uniq)})
    all_valid = [c for row in ctx_rows for c in row["candidates"] if c["valid"]]
    uniq_all = {c["graph_hash"] for c in all_valid}
    out = {"label": label, "contexts": len(ctx_rows), "candidates_per_ctx": k,
           "generated": len(ctx_rows) * k,
           "parseable": sum(c["parseable"] for r in ctx_rows for c in r["candidates"]),
           "valid": len(all_valid), "unique_valid_canonical": len(uniq_all),
           "iso_classes_generated": len(uniq_all),
           "rows": ctx_rows, "schema_version": SCHEMA}
    (O4 / f"diversity_{label}.json").write_text(json.dumps(out, indent=0), encoding="utf-8")
    return out


# build_model_pairs REMOVED from the training path (Part D): it ranked
# candidates by spec-match label alone with no physical measurement and could
# emit rejected="INVALID" placeholder rows. All pair building now goes through
# agentic_raptor.llm_dpo.integrity (same evaluation context, full measurement
# vector, explicit hierarchy, dedup/contradiction/balance controls).


def realise_sample(n: int = 3) -> dict[str, Any]:
    """Map+simulate distinct-structure generated proposals on REAL ngspice via
    base-template mapping + executable edits (buffer/rc/fb variants)."""
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mapping import map_family
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import (
        EditRejected, apply_edit, device_graph_hash, qualify_device_graph)
    div = json.loads((O4 / "diversity_sft.json").read_text())
    seen, results = set(), []
    exe = discover_ngspice()
    costs = new_costs()
    for row in div["rows"]:
        for c in row["candidates"]:
            if not c["valid"] or c["graph_hash"] in seen or len(results) >= n:
                continue
            seen.add(c["graph_hash"])
            obj = c["obj"]

            class _S:
                topology_id = f"gen_{c['graph_hash'][:8]}"
            g, _ = map_family(_S(), {
                "topology_id": _S.topology_id,
                "gain_stages": len(obj["stages"]),
                "functional_blocks": (["C"] if obj.get("compensation") else []),
                "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
                "graph_hash": None})
            try:
                if obj.get("output_buffer"):
                    g, _a = apply_edit(g, "ADD_SUPPORTED_OUTPUT_STAGE")
                if obj.get("compensation") and obj["compensation"][0]["type"] == "rc_nulling":
                    g, _a = apply_edit(g, "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE")
                if obj.get("local_feedback"):
                    g, _a = apply_edit(g, "CONNECT_VERIFIED_FEEDBACK_PATH")
            except EditRejected as exc:
                results.append({"variant": c["graph_hash"],
                                "context_id": row["context_id"],
                                "status": f"edit_rejected:{exc}"})
                continue
            q = qualify_device_graph(_S.topology_id, g, O4 / "realised", exe,
                                     c["graph_hash"][:12], costs)
            results.append({"variant": c["graph_hash"],
                            "context_id": row["context_id"],
                            "proposal": json.dumps(obj, separators=(",", ":")),
                            "device_hash": device_graph_hash(g),
                            "stages": len(obj["stages"]),
                            "electrical": q.get("electrical"),
                            "stability": q.get("stability"),
                            "pm": (q.get("metrics") or {}).get("phase_margin_deg")})
    out = {"realised": results, "real_spice_calls": costs["real_spice_calls"],
           "schema_version": SCHEMA}
    (O4 / "realised.json").write_text(json.dumps(out, indent=1, default=str),
                                      encoding="utf-8")
    return out


def bootstrap_ci(vals: list[float], n: int = 500, seed: int = 0) -> list[float]:
    rng = Random(seed)
    if not vals:
        return [0.0, 0.0]
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)]


def run_comparison() -> dict[str, Any]:
    table = {}
    for label, adapter in (("base", None), ("sft", str(O4 / "sft_adapter")),
                           ("sft_dpo", str(OUT / "dpo_adapter_seed0"))):
        try:
            d = run_diversity_campaign(adapter, k=3, label=f"cmp_{label}")
        except ValueError as exc:   # adapter built for a different architecture
            table[label] = {"skipped": f"adapter_incompatible: {exc}"[:120]}
            continue
        per_ctx_valid = [r["valid"] / 3 for r in d["rows"]]
        table[label] = {"valid_rate": round(d["valid"] / d["generated"], 3),
                        "unique_valid": d["unique_valid_canonical"],
                        "iso_classes": d["iso_classes_generated"],
                        "valid_rate_ci95": bootstrap_ci(per_ctx_valid)}
    (O4 / "comparison.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    return table


def run_all() -> dict[str, Any]:
    t0 = time.time()
    apply_torch_omp_workaround()
    import torch
    s = {"hardware": (f"CUDA: {torch.cuda.get_device_name(0)}"
                      if torch.cuda.is_available() else "CPU only")
         + f" | model: {MODEL_ID}",
         "corpus": build_corpus(), "sft": run_sft()}
    s["diversity_sft"] = {k: v for k, v in
                          run_diversity_campaign(str(O4 / "sft_adapter")).items()
                          if k != "rows"}
    s["realised"] = realise_sample(n=10)
    from agentic_raptor.llm_dpo import scores_to_pairs
    ingest = scores_to_pairs()                     # measurements -> DPO queue
    s["auto_pair_ingestion"] = {k: v for k, v in ingest.items()
                                if k != "report"}
    s["comparison"] = run_comparison()
    s["wall_clock_s"] = round(time.time() - t0, 1)
    (O4 / "SUMMARY.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    return s


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=1, default=str))
