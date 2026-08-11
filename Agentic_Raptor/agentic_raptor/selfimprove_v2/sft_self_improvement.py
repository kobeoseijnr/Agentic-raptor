"""The SFT self-improvement loop's queue -> dataset -> generation pathway.

Everything upstream of this file already exists and is correct:

    proposal -> PUCT -> SAC/MB-SAC sizing -> DPO/hard-safety ranker
    -> a FRESH authoritative ngspice call (run_raptor_v2.verify(),
       mode="final_verification"/"backup", never a reused sizing call)
    -> harvest_run() -> sft_queue.jsonl  (agentic_raptor.selfimprove_v2.streams)

sft_admission_reasons() in that file is, and remains, the primary gate: a
row only ever reaches sft_queue.jsonl if it was authoritatively measured,
provenance-matched to its own topology/sizing/spec, and passed the spec
outright. This module does NOT replace that logic -- everything below is a
SECOND, independent pass over already-admitted rows (defense in depth, the
same layering harvest_run() itself already uses for protected-spec checks),
plus the machinery the admission gate was never responsible for: turning
admitted rows into a clean, deduplicated, family-balanced, replay-augmented
training corpus, and turning a training run into a versioned, gated SFT
generation.

Design decisions, stated explicitly rather than left implicit:

  * TARGET = canonical topology graph text (the same family-canonicalised
    text agentic_raptor.selfimprove_v2.corpus._canon_text already produces
    for build_corpus_v2), never sizing values. Performance metrics
    (gain/PM/UGBW/IDD/FoM/margins) are carried as METADATA for filtering,
    weighting and provenance only -- never inserted into the response text
    the model is trained to reproduce.

  * INPUT prompts are rebuilt CLEAN: any baked-in "### KNOWN ..." evidence
    line is stripped (strip_known_line) and, for genuinely new verified
    examples, replaced with a freshly retrieved line sourced ONLY from a
    POST_CLOAD_FIX_V1 RAG memory file (run_raptor_v2.rag_stage/retrieve,
    the same function inference itself calls) -- never from the archived
    datasets/simulation_memory/self_improvement_runs.jsonl, which predates
    both the VCM fix and the C_LOAD fix. See build_clean_prompt().

  * Continual-SFT strategy: REPLAY, not sequential fine-tuning. load_models()
    (agentic_raptor.llm_dpo) always initialises a FRESH LoRA adapter over
    the base model -- there is no "resume adapter" path -- so every SFT
    self-improvement generation trains from scratch on (clean structural
    replay examples + new verified-good examples), exactly the strategy
    train_proposer_diverse.py / build_corpus_v2 already use for the
    existing corpus_diverse.json + measured-target mix. This module keeps
    that strategy but fixes a real, previously-undetected defect in the
    prompt half of it: build_corpus_v2 was carrying corpus_diverse.json's
    765 stored prompts through UNCHANGED, and EVERY one of them has a
    "### KNOWN ..." line baked in at build time from the now-archived L4
    file (verified directly: 765/765 records) -- so both the "replay" set
    AND any new measured target sharing a spec_id with a base record were
    silently carrying stale pre-fix evidence into every retrain. Fixed
    here by stripping that line from both halves (build_replay_examples,
    build_clean_prompt) before anything is written to a training corpus.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from agentic_raptor.publication.artifact_provenance import POST_CLOAD_FIX_V1
from agentic_raptor.selfimprove_v2.streams import structurally_valid

ROOT = Path(__file__).resolve().parents[2]

#: SFT-generation checkpoints, versioned independently of
#: run_self_improvement_v2.py's own gen_NNN loop counter (which advances
#: every run whether or not the proposer retrains). G0 is the pre-existing
#: structural adapter and is never overwritten -- see ensure_g0_manifest().
GENERATIONS_ROOT = ROOT / "artifacts/publication_v2/proposer_repair/generations"
BASE_ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"
BASE_CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"

#: quality-tier margins, in the SAME normalised units
#: agentic_raptor.mb_sac.spec_sizing.postsizing_outcome() already uses for
#: its own margin scaling (gain/20dB, PM/45deg) -- declared here explicitly
#: as policy, not derived from any prior calibration. A row's tier is the
#: WORST (most conservative) of its measured margins; feasibility is
#: always primary because every tier requires exact_spec_pass, enforced
#: upstream by sft_eligible(), not by this function.
STRONG_PASS_MARGIN = 0.15
HIGH_QUALITY_PASS_MARGIN = 0.30

#: balancing policy defaults -- explicit, overridable by the training
#: trigger script's CLI flags, never silently invented at call time.
DEFAULT_PER_FAMILY_CAP = 8
DEFAULT_PER_SPEC_CAP = 4

_KNOWN_LINE_RE = re.compile(r"^### KNOWN [^\n]*\n", re.M)


# ---------------------------------------------------------------------------
# 2. eligibility
# ---------------------------------------------------------------------------
def sft_eligible(row: dict, *, protected_context_ids: set,
                 protected_evaluation_context_ids: set | None = None) -> list:
    """Every reason `row` (an already-admitted agentic_raptor.selfimprove_v2.
    streams.harvest_run sft_queue row) may NOT enter the training dataset.

    Empty list = eligible. This re-derives eligibility from the row's own
    fields rather than trusting `admitted: True` at face value -- a second,
    independent pass, exactly the pattern harvest_run() itself already uses
    (both `protected_ids` display-name AND `evaluation_context_id`
    content-hash checks, belt and braces) so a stale/hand-edited/merged
    queue file is caught here too, not just at original harvest time.
    """
    bad = []
    if row.get("electrical_environment_version") != POST_CLOAD_FIX_V1:
        bad.append("not_post_cload_fix_v1:"
                   f"{row.get('electrical_environment_version')!r}")
    if not row.get("admitted"):
        bad.append("not_admitted")
    if not row.get("exact_spec_pass"):
        bad.append("not_exact_spec_pass")
    if row.get("spec_index") is None:
        bad.append("missing_spec_index")
    if not row.get("spec_hash"):
        bad.append("missing_spec_hash")
    if not row.get("canonical_graph_hash"):
        bad.append("missing_canonical_graph_hash")
    obj = row.get("obj")
    if not obj or not structurally_valid(obj):
        bad.append("structurally_invalid_or_missing_topology_graph")

    spec_id = row.get("generation_spec_id")
    if spec_id is not None and spec_id in protected_context_ids:
        bad.append("protected_evaluation_record")
    spec = row.get("spec") or {}
    if spec and protected_evaluation_context_ids is not None:
        try:
            from agentic_raptor.llm_dpo.integrity import evaluation_context_id
            if evaluation_context_id(spec) in protected_evaluation_context_ids:
                if "protected_evaluation_record" not in bad:
                    bad.append("protected_evaluation_record")
        except Exception:
            pass

    req = row.get("requested_c_load_f")
    sim = row.get("simulated_c_load_f")
    if req is None or sim is None:
        bad.append("missing_c_load_provenance")
    elif req != sim:
        # both are, by construction, resolved through the SAME
        # effective_c_load() call -- any difference is a real, unexplained
        # override, not floating-point noise from two independent paths.
        bad.append(f"unexplained_load_override:requested={req!r}"
                   f",simulated={sim!r}")

    call_id = row.get("verification_spice_call_id")
    mode = row.get("verification_mode")
    if not call_id:
        bad.append("missing_verification_call_id")
    if mode not in ("final_verification", "backup"):
        bad.append("verification_call_reused_from_sizing_or_unknown_mode:"
                   f"{mode!r}")
    return bad


# ---------------------------------------------------------------------------
# 6. deduplication
# ---------------------------------------------------------------------------
def deduplicate(rows: list) -> tuple:
    """Dedup by (spec_hash, canonical_topology_hash), first-seen wins (rows
    are expected to already be in a deterministic order -- callers that
    care about WHICH duplicate survives should sort first). Also reports
    global canonical-topology repetition so one structure succeeding across
    many specs is visible even though each (spec, topology) pair is unique.
    """
    seen: dict = {}
    kept = []
    topology_counts: dict = {}
    for r in rows:
        th = r.get("canonical_graph_hash")
        topology_counts[th] = topology_counts.get(th, 0) + 1
        key = (r.get("spec_hash"), th)
        if key in seen:
            continue
        seen[key] = r
        kept.append(r)
    stats = {
        "raw": len(rows),
        "unique_spec_topology_pairs": len(kept),
        "unique_topology_hashes": len({r.get("canonical_graph_hash")
                                       for r in kept}),
        "duplicates_removed": len(rows) - len(kept),
        "topology_repetition_counts": dict(
            sorted(topology_counts.items(), key=lambda kv: -kv[1])),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# 7. family / spec balancing
# ---------------------------------------------------------------------------
def balance_by_family(rows: list, *, per_family_cap: int | None = None,
                      per_spec_cap: int | None = None) -> tuple:
    """Deterministic cap: sorted by (family, spec_id, topology hash) so
    which rows survive a cap never depends on queue-file write order, then
    a single pass drops anything past either cap. Unlike
    run_raptor_v2.retrieve()'s two-pass cap-then-lift (which must fill an
    exact k slots), balancing here has no fixed target size -- its only
    job is capping frequency, so a single deterministic pass is both
    correct and simpler.
    """
    ordered = sorted(rows, key=lambda r: (r.get("family") or "",
                                          r.get("generation_spec_id") or "",
                                          r.get("canonical_graph_hash") or ""))
    fam_counts: dict = {}
    spec_counts: dict = {}
    kept, dropped = [], []
    for r in ordered:
        fam, sid = r.get("family"), r.get("generation_spec_id")
        if per_family_cap is not None and fam_counts.get(fam, 0) >= per_family_cap:
            dropped.append(r)
            continue
        if per_spec_cap is not None and spec_counts.get(sid, 0) >= per_spec_cap:
            dropped.append(r)
            continue
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        spec_counts[sid] = spec_counts.get(sid, 0) + 1
        kept.append(r)
    policy = {
        "strategy": "deterministic_per_family_and_per_spec_cap",
        "per_family_cap": per_family_cap, "per_spec_cap": per_spec_cap,
        "family_distribution_before": dict(Counter(r.get("family")
                                                    for r in rows)),
        "family_distribution_after": dict(Counter(r.get("family")
                                                   for r in kept)),
        "spec_distribution_before": dict(Counter(r.get("generation_spec_id")
                                                  for r in rows)),
        "spec_distribution_after": dict(Counter(r.get("generation_spec_id")
                                                 for r in kept)),
        "dropped_for_balance": len(dropped),
    }
    return kept, policy


# ---------------------------------------------------------------------------
# 8. quality tiers
# ---------------------------------------------------------------------------
def classify_quality_tier(row: dict) -> str:
    """PASS / STRONG_PASS / HIGH_QUALITY_PASS. See STRONG_PASS_MARGIN /
    HIGH_QUALITY_PASS_MARGIN docstring above for the declared (not fitted)
    thresholds. Falls back to PASS when a margin cannot be computed (e.g.
    the spec has no gain/PM target recorded) -- an unmeasurable margin is
    never treated as a high-quality signal."""
    spec = row.get("spec") or {}
    gain_t, pm_t = spec.get("gain_target_db"), spec.get("phase_margin_target_deg")
    margins = []
    if row.get("gain_db") is not None and gain_t is not None:
        margins.append((row["gain_db"] - gain_t) / 20.0)
    if row.get("pm_deg") is not None and pm_t is not None:
        margins.append((row["pm_deg"] - pm_t) / 45.0)
    if not margins:
        return "PASS"
    worst = min(margins)
    if worst >= HIGH_QUALITY_PASS_MARGIN:
        return "HIGH_QUALITY_PASS"
    if worst >= STRONG_PASS_MARGIN:
        return "STRONG_PASS"
    return "PASS"


# ---------------------------------------------------------------------------
# 3/4. clean input construction (target = topology graph, never sizing;
# input = clean prompt, never stale pre-fix evidence)
# ---------------------------------------------------------------------------
def strip_known_line(prompt: str) -> str:
    """Remove any baked-in '### KNOWN ...' evidence line. Same regex
    agentic_raptor.llm_dpo.rag.augment_prompt already uses to REPLACE a
    stale KNOWN line with a fresh one; used here to remove it outright
    (callers that want fresh evidence re-augment afterwards via
    build_clean_prompt, which sources it only from a POST_CLOAD_FIX_V1
    memory file)."""
    return _KNOWN_LINE_RE.sub("", prompt)


def build_clean_prompt(spec: dict, base_prompt: str, *,
                       rag_memory_path=None) -> str:
    """Strip stale evidence, then re-augment with FRESH evidence retrieved
    only from `rag_memory_path` (defaults, via run_raptor_v2.rag_stage's
    own default, to RAG_MEMORY_V2 -- the clean, POST_CLOAD_FIX_V1 file
    inference itself reads). Everything else in the template (### SPEC,
    ### RAG topology-id tag, ### BLOCKS, ### FORBIDDEN, ### PROPOSAL) is
    spec-derived, not measurement-derived, so it is safe to carry through
    unchanged -- only the ### KNOWN line is ever measurement-derived."""
    from run_raptor_v2 import rag_stage
    stripped = strip_known_line(base_prompt)
    out = rag_stage(spec, stripped, use_rag=True,
                    memory_path=str(rag_memory_path) if rag_memory_path else None)
    return out["prompt"]


def build_target_text(obj: dict) -> str:
    """The TARGET is the canonical structured topology graph for the
    family the design realises (never the exact sampled byte string, never
    sizing values) -- same canonicalisation build_corpus_v2 already uses
    for measured targets."""
    from agentic_raptor.selfimprove_v2.corpus import _canon_text
    return _canon_text(obj)


def build_replay_examples(base_corpus: dict) -> list:
    """The clean structural replay set: every record from the existing
    canonical corpus (corpus_diverse.json by default), with its stale
    '### KNOWN' line stripped. This is what prevents catastrophic
    forgetting of topology-language/schema competence (Section 9) without
    carrying stale pre-fix evidence into the replay half of training
    (Section 4)."""
    out = []
    for r in base_corpus.get("records") or []:
        rr = dict(r)
        rr["prompt"] = strip_known_line(r.get("prompt") or "")
        rr["target_source"] = "structural_replay_clean"
        out.append(rr)
    return out


def build_example(row: dict, *, base_prompt: str, rag_memory_path,
                  generation_id: str, quality_tier: str,
                  proposer_checkpoint: str | None = None,
                  adapter_generation: str | None = None) -> dict:
    """One clean, inference-compatible {prompt, response} training example
    plus a separate `metadata` block (Section 5) carrying every
    provenance/performance field -- never merged into prompt/response."""
    spec = row.get("spec") or {}
    prompt = build_clean_prompt(spec, base_prompt, rag_memory_path=rag_memory_path)
    response = build_target_text(row["obj"])
    return {
        "context_id": row.get("generation_spec_id"),
        "prompt": prompt, "response": response,
        "variant_hash": row.get("canonical_graph_hash"),
        "canonical_graph_hash": row.get("canonical_graph_hash"),
        "topology_signature": row.get("family"),
        "split": "train", "target_source": "sft_self_improvement_verified",
        "metadata": {
            "generation_id": generation_id,
            "source": "authoritative_verified_success",
            "spec_index": row.get("spec_index"),
            "spec_hash": row.get("spec_hash"),
            "canonical_topology_hash": row.get("canonical_graph_hash"),
            "topology_family": row.get("family"),
            "verification_spice_call_id": row.get("verification_spice_call_id"),
            "requested_c_load_f": row.get("requested_c_load_f"),
            "simulated_c_load_f": row.get("simulated_c_load_f"),
            "gain_db": row.get("gain_db"), "pm_deg": row.get("pm_deg"),
            "ugbw_hz": row.get("ugbw_hz"), "idd_a": row.get("idd_a"),
            "exact_spec_pass": row.get("exact_spec_pass"),
            "electrical_environment_version":
                row.get("electrical_environment_version"),
            "quality_tier": quality_tier,
            "proposer_checkpoint": proposer_checkpoint,
            "sft_adapter_generation": adapter_generation,
            "seed": row.get("seed"), "budget": row.get("budget"),
            "timestamp": time.time(),
            "training_eligible": True,
        }}


# ---------------------------------------------------------------------------
# 5. dataset assembly + versioned write
# ---------------------------------------------------------------------------
def build_dataset(queue_rows: list, base_corpus: dict, *,
                  generation_id: str, protected_context_ids: set,
                  protected_evaluation_context_ids: set | None = None,
                  rag_memory_path=None, per_family_cap=DEFAULT_PER_FAMILY_CAP,
                  per_spec_cap=DEFAULT_PER_SPEC_CAP,
                  proposer_checkpoint: str | None = None,
                  adapter_generation: str | None = None,
                  replay: bool = True) -> dict:
    """Orchestrates Sections 2/3/4/5/6/7/8/9: filter -> dedup -> balance ->
    quality-tier -> clean example -> replay mix. Returns
    {"records": [...], "manifest": {...}} -- `records` is directly the
    shape agentic_raptor.llm_dpo.stage3e4.run_sft expects
    (corpus["records"] with a "split" key per row), so the return value can
    be written straight to a corpus_path.
    """
    raw_n = len(queue_rows)
    eligible, rejected = [], []
    for r in queue_rows:
        reasons = sft_eligible(
            r, protected_context_ids=protected_context_ids,
            protected_evaluation_context_ids=protected_evaluation_context_ids)
        if reasons:
            rejected.append({"spec_hash": r.get("spec_hash"),
                             "canonical_graph_hash": r.get("canonical_graph_hash"),
                             "reasons": reasons})
        else:
            eligible.append(r)

    deduped, dedup_stats = deduplicate(eligible)
    balanced, balance_policy = balance_by_family(
        deduped, per_family_cap=per_family_cap, per_spec_cap=per_spec_cap)

    prompts_by_spec = {r.get("context_id"): r.get("prompt")
                       for r in base_corpus.get("records") or []}
    tiers = Counter()
    new_examples, skipped_no_prompt = [], []
    for r in balanced:
        base_prompt = prompts_by_spec.get(r.get("generation_spec_id"))
        if not base_prompt:
            skipped_no_prompt.append(r.get("generation_spec_id"))
            continue
        tier = classify_quality_tier(r)
        tiers[tier] += 1
        new_examples.append(build_example(
            r, base_prompt=base_prompt, rag_memory_path=rag_memory_path,
            generation_id=generation_id, quality_tier=tier,
            proposer_checkpoint=proposer_checkpoint,
            adapter_generation=adapter_generation))

    replay_examples = build_replay_examples(base_corpus) if replay else []
    records = replay_examples + [
        {k: v for k, v in ex.items() if k != "metadata"} for ex in new_examples]

    manifest = {
        "generation_id": generation_id,
        "raw_sft_queue_records": raw_n,
        "rejected_records": len(rejected),
        "rejected_reasons_sample": rejected[:20],
        "eligible_records": len(eligible),
        "dedup": dedup_stats,
        "balance_policy": balance_policy,
        "quality_tier_counts": dict(tiers),
        "skipped_no_prompt_for_spec": skipped_no_prompt,
        "structural_replay_included": replay,
        "structural_replay_count": len(replay_examples),
        "new_verified_count": len(new_examples),
        "total_records": len(records),
        "mixture_ratio_replay_to_new": (
            round(len(replay_examples) / len(new_examples), 3)
            if new_examples else None),
        "known_evidence_policy": (
            "'### KNOWN' evidence lines are stripped from every replay AND "
            "new-example prompt, then (new examples only) re-augmented with "
            "evidence retrieved solely from a POST_CLOAD_FIX_V1 RAG memory "
            "file; datasets/simulation_memory/self_improvement_runs.jsonl "
            "(archived, pre-VCM-fix and pre-C_LOAD-fix) is never read by "
            "this module."),
        "electrical_environment_version": POST_CLOAD_FIX_V1,
    }
    return {"records": records, "manifest": manifest,
           "new_examples_with_metadata": new_examples}


def write_versioned_dataset(dataset: dict, out_dir: Path) -> dict:
    """Writes a NEW versioned dataset directory -- never overwrites
    BASE_CORPUS/corpus_diverse.json. Returns the paths + content hashes
    written, for the generation manifest to reference."""
    import hashlib
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = {"records": dataset["records"], "schema_version": "sft_si.1"}
    corpus_path = out_dir / "corpus.json"
    corpus_path.write_text(json.dumps(corpus, indent=1, default=str),
                           encoding="utf-8")
    meta_path = out_dir / "examples_metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for ex in dataset["new_examples_with_metadata"]:
            f.write(json.dumps({"context_id": ex["context_id"],
                                "canonical_graph_hash": ex["canonical_graph_hash"],
                                **ex["metadata"]}, default=str) + "\n")
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(dataset["manifest"], indent=1,
                                        default=str), encoding="utf-8")
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()[:16]
    return {"corpus_path": corpus_path, "metadata_path": meta_path,
           "manifest_path": manifest_path, "corpus_hash": corpus_hash}


# ---------------------------------------------------------------------------
# 10/16/17. generation checkpointing + A0-A8 freeze / A9 lineage guard
# ---------------------------------------------------------------------------
def generation_dir(gen_id: str) -> Path:
    return GENERATIONS_ROOT / gen_id


def read_generation_manifest(gen_id: str) -> dict | None:
    p = generation_dir(gen_id) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write_generation_manifest(gen_id: str, manifest: dict) -> Path:
    """G0 is immutable: this refuses to overwrite an existing G0 manifest.
    Every other generation may be written once by the trigger script that
    just trained/validated it (also not designed to be called twice for
    the same gen_id, but only G0 gets a hard runtime guard, since G0 is
    the one generation every lineage must always be able to fall back to)."""
    d = generation_dir(gen_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "manifest.json"
    if gen_id == "G0" and p.is_file():
        raise RuntimeError("G0 is immutable and already has a manifest -- "
                           "refusing to overwrite")
    p.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return p


def ensure_g0_manifest(base_adapter: Path = BASE_ADAPTER,
                       base_corpus_path: Path = BASE_CORPUS) -> dict:
    """Bootstraps G0's manifest, pointing read-only at the pre-existing
    structural adapter -- never copies, moves, or retrains it. Idempotent:
    returns the existing manifest unchanged if one is already there."""
    existing = read_generation_manifest("G0")
    if existing is not None:
        return existing
    from agentic_raptor.ranking.types import directory_sha256
    manifest = {
        "generation_id": "G0", "parent_generation": None,
        "source_adapter": str(base_adapter), "adapter_path": str(base_adapter),
        "training_dataset_path": str(base_corpus_path),
        "checkpoint_hash": (directory_sha256(str(base_adapter))
                            if Path(base_adapter).is_dir() else None),
        "electrical_environment_version": POST_CLOAD_FIX_V1,
        "immutable": True, "rejected": False,
        "note": ("G0 is the pre-existing structural SFT adapter, trained "
                "before this self-improvement pathway existed. This "
                "manifest is a read-only provenance record, not a "
                "retrain -- see ensure_g0_manifest().")}
    write_generation_manifest("G0", manifest)
    return manifest


def next_generation_id(parent_gen_id: str) -> str:
    n = int(parent_gen_id.lstrip("G")) + 1
    return f"G{n}"


# ---------------------------------------------------------------------------
# 12/13. validation: format capability (cheap, no SPICE/PUCT/ranker) +
# capability-regression gate. The small controlled downstream SPICE subset
# (Section 13's second half) reuses run_self_improvement_v2.eval_proposer,
# which already runs the full real pipeline and computes proposer_gates'
# exact required metrics -- not duplicated here.
# ---------------------------------------------------------------------------
def capability_probe(adapter: str | None, eval_items: list, *,
                     target_k: int = 5, seed: int = 0,
                     known_family_hashes: set | None = None) -> dict:
    """Proposer-level capability check: schema/parse/graph-construction
    success (folded together, since run_raptor_v2.propose_and_validate only
    ever returns parsed + schema-valid + structurally-valid candidates --
    an unparseable/invalid sample is silently one more `attempt` that did
    not become a `candidate`, exactly what schema_graph_construction_
    success_rate = distinct/attempts measures), Valid@K, Unique@K, family
    diversity, RUN_REHIT rate (repeat-hash fraction across the WHOLE probe
    -- a global mode-collapse signal, not per-spec), and novel valid yield
    (candidates whose hash was never in `known_family_hashes`, i.e.
    genuinely new coverage rather than a re-derivation of what the parent
    generation could already produce).

    `eval_items`: [{"spec": dict, "prompt": str, "spec_hash": str}, ...],
    prompts already clean (see build_clean_prompt) -- this function is
    agnostic to how they were built. Uses the same production
    propose_and_validate() Stage 2/3/4 diagnostics and run_pipeline() all
    call, conditioning="exclusion" (the real A4 production default), so
    this measures the actual serving mechanism, not a reimplementation.
    """
    from run_qwen_ablation import _load
    from run_raptor_v2 import propose_and_validate
    tok, model = _load(adapter)
    known_family_hashes = known_family_hashes or set()
    per_spec, all_hashes = [], []
    for item in eval_items:
        prop = propose_and_validate(model, tok, item["prompt"], target_k=target_k,
                                    conditioning="exclusion", seed0=seed,
                                    use_llm=True, spec=item.get("spec"))
        cands = prop["candidates"]
        hashes = [c["canonical_graph_hash"] for c in cands]
        all_hashes.extend(hashes)
        per_spec.append({
            "spec_hash": item.get("spec_hash"),
            "attempts": prop["attempts"], "distinct": prop["distinct"],
            "candidate_generation_status": prop["candidate_generation_status"],
            "valid_at_k": prop["distinct"] / target_k,
            "unique_at_k": len(set(hashes)) / max(1, len(hashes)),
            "families": [c["canonical_family"] for c in cands],
            "novel_valid": sum(1 for h in hashes if h not in known_family_hashes)})
    n = max(1, len(eval_items))
    total_attempts = sum(r["attempts"] for r in per_spec)
    total_distinct = sum(r["distinct"] for r in per_spec)
    return {
        "n_specs": len(eval_items),
        "schema_graph_construction_success_rate": (
            total_distinct / total_attempts if total_attempts else 0.0),
        "mean_valid_at_k": sum(r["valid_at_k"] for r in per_spec) / n,
        "mean_unique_at_k": sum(r["unique_at_k"] for r in per_spec) / n,
        "family_diversity_count": len({f for r in per_spec for f in r["families"]}),
        "run_rehit_rate": (1 - len(set(all_hashes)) / len(all_hashes)
                           if all_hashes else None),
        "novel_valid_yield": sum(r["novel_valid"] for r in per_spec),
        "per_spec": per_spec}


def capability_gate(cand: dict, parent: dict, *,
                    max_relative_regression: float = 0.10) -> dict:
    """Section 12: block promotion on a material regression in schema/
    graph-construction capability from the PARENT generation (for G1, this
    is G0's own known 100%/100% result). max_relative_regression=0.10 --
    declared explicitly, not invented at call time -- allows the
    candidate's schema_graph_construction_success_rate and mean_unique_at_k
    to fall at most 10% relative to the parent before this fails.
    family_diversity_count and mean_valid_at_k must not decline AT ALL --
    any drop blocks promotion, since that is exactly what "self-improvement
    simply creating stronger mode collapse" (the failure mode Section 12
    names explicitly) would show up as first.
    """
    failures = []

    def _rel_ok(key):
        p, c = parent.get(key), cand.get(key)
        if p is None or c is None:
            return True
        if p == 0:
            return c >= 0
        return c >= p * (1 - max_relative_regression)

    if not _rel_ok("schema_graph_construction_success_rate"):
        failures.append(
            "schema_graph_construction_success_rate regressed: "
            f"{parent.get('schema_graph_construction_success_rate')} -> "
            f"{cand.get('schema_graph_construction_success_rate')}")
    if not _rel_ok("mean_unique_at_k"):
        failures.append(f"mean_unique_at_k regressed: "
                        f"{parent.get('mean_unique_at_k')} -> "
                        f"{cand.get('mean_unique_at_k')}")
    pfd, cfd = parent.get("family_diversity_count"), cand.get("family_diversity_count")
    if pfd is not None and cfd is not None and cfd < pfd:
        failures.append(f"family_diversity_count declined: {pfd} -> {cfd}")
    pvk, cvk = parent.get("mean_valid_at_k"), cand.get("mean_valid_at_k")
    if pvk is not None and cvk is not None and cvk < pvk - 1e-9:
        failures.append(f"mean_valid_at_k declined: {pvk} -> {cvk}")
    return {"passed": not failures, "failures": failures,
           "max_relative_regression": max_relative_regression}


def assert_lineage_may_train(lineage: str) -> None:
    """The in-process A9 static/adaptive guard (agentic_raptor.publication.
    generation_state defines the lineage concept: "adaptive" may
    harvest/retrain, "static" runs the identical workload but never
    updates RAG/SFT/PUCT/SAC/surrogate/DPO). A0-A8 ablation code
    (run_ablation_v3.py) never imports this module at all -- that is the
    FIRST guard, structural, verified by
    tests/test_sft_self_improvement.py. This is the SECOND: even a caller
    that does reach this module must explicitly claim lineage="adaptive"
    before train_sft_self_improvement.py will touch any weights."""
    if lineage != "adaptive":
        raise RuntimeError(
            f"SFT self-improvement training refused for lineage={lineage!r}."
            " Only the 'adaptive' A9 lineage may train a new SFT "
            "generation -- A0-A8 and the 'static' A9 lineage must keep "
            "the SFT checkpoint frozen at its parent generation's hash.")
