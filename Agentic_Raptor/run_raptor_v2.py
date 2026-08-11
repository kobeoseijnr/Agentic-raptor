"""Canonical Agentic RAPTOR inference pipeline (publication v2).

    Specification
    -> RAG retrieves structured verified evidence
    -> SFT-trained LLM generates 5 DISTINCT valid topology graphs
    -> deterministic validator canonicalises and deduplicates them
    -> PUCT ranks the 5 LLM-generated graphs and selects exactly 2
    -> SAC/C9 sizes both selected topologies INDEPENDENTLY
    -> surrogate predicts post-sizing metrics for both
    -> DPO-trained post-SAC ranker compares the 2 sized designs, selects 1
    -> ngspice authoritatively verifies the selected design
    -> measured outcomes route to the correct component and branch

Written as a new entry point rather than an edit to run_full_raptor.py, so
the historical script and every artifact it produced stay reproducible. That
older path is a DIFFERENT architecture (1 LLM proposal, 5 code-enumerated
candidates, 1 selected, 1 sized) and its results must not be pooled with
results from this one.

Every count in the flow is asserted, not assumed. The failure this guards
against is silent: a pipeline that computes advice, executes something else,
and attributes the measurement to the advice would look like it worked.

Run:  python run_raptor_v2.py [--spec-index 0] [--budget 32] [--calibrate]
"""
import argparse
import json
import time
from pathlib import Path

from agentic_raptor.electrical.fom import compute_fom
from agentic_raptor.electrical.pvt_eval import (PvtConfig, aggregate_pvt,
                                                run_pvt_sweep)
from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.publication import ROOT
from agentic_raptor.ranking import (PostSACDesign, compare, netlist_hash_of,
                                    outcome_from_sizing, predict_post_sac,
                                    record_pair)

OUT = ROOT / "artifacts/publication_v2/raptor_v2_runs"
MEM = ROOT / "datasets/simulation_memory"
TARGET_K = 5
SELECT_K = 2
#: the only honest candidate provenances. "llm" (generated) and "retrieval"
#: (A1: K unique EXISTING topologies, no generative model -- see
#: retrieve_topology_candidates) are both real, non-fabricated pools.
#: Anything else (in particular code-ENUMERATION, the defect the old
#: run_full_raptor.py architecture had) must never enter this pipeline.
VALID_CANDIDATE_SOURCES = ("llm", "retrieval")


class ArchitectureViolation(AssertionError):
    """A stage produced the wrong count or lost candidate provenance."""


# ----------------------------- Stage 2: RAG ----------------------------------
#: v2 RAG memory: measured evidence produced BY the v2 pipeline. The old
#: self_improvement_runs.jsonl is archived -- it came from a different
#: architecture and was measured before the input-common-mode fix, so its
#: phase margins describe circuits running at 1/79th of design current.
#:
#: Stage 1.6: versioned again, same reasoning as the VCM-fix archival above
#: -- rag_memory_v2.jsonl was populated (partly) under the pre-C_LOAD-fix
#: environment (see artifact_provenance.py). New writes go to a fresh
#: _post_cload_v1 file rather than appending onto a file that mixes both
#: eras; the original stays exactly where it was as PRE_CLOAD_FIX evidence.
RAG_MEMORY_V2 = (ROOT / "artifacts/publication_v2/selfimprove"
                 / "rag_memory_v2_post_cload_v1.jsonl")


def rag_stage(spec: dict, prompt: str, *, use_rag: bool = True,
             memory_path=None) -> dict:
    """A2 (no-RAG) isolation point: use_rag=False is an explicit no-op --
    zero records, the UNMODIFIED prompt, and `retrieve()` is never called at
    all (no bogus-path trick, no eval_sets import, no hidden fallback
    retrieval of any kind)."""
    if not use_rag:
        return {"records": [], "retrieval_ids": [], "prompt": prompt}
    return retrieve(spec, prompt, memory_path=memory_path)


def retrieve(spec: dict, prompt: str, k: int = 6,
             memory_path=None, max_per_family: int = 2) -> dict:
    """Spec-conditioned retrieval of measured evidence.

    Retrieval provides CONTEXT ONLY. It must never become a PUCT candidate --
    that conflation is what the audit found in the old path.

    Reads the v2 memory only. An empty memory yields NO evidence line, which
    is the honest state before the loop has produced any: inventing context
    would put unmeasured claims in front of the proposer.

    Stage 2 diagnostic finding (2026-08-10): the OLD dedup keyed `seen` by
    the rendered text line (stage count + stability + ROUNDED pm), so
    multiple measurements of the SAME graph with slightly different pm
    values all counted as "distinct" -- confirmed on real post-repair data:
    every one of 36 retrieved records across 6 real diagnostic pairs came
    from ONE topology variant, and WITH_RAG/NO_RAG produced byte-identical
    candidate sets in 5/6 pairs as a direct result (no family diversity in
    the injected evidence -> nothing for the proposer to react to).
    max_per_family caps how many records from the SAME family can occupy
    the k retrieval slots, so a memory pool dominated by one common family
    can no longer crowd out every other family's evidence.
    """
    l4p = Path(memory_path) if memory_path else RAG_MEMORY_V2
    records = []
    if l4p.is_file():
        ev = [json.loads(x) for x in
              l4p.read_text(encoding="utf-8").splitlines()]
        ev = [e for e in ev if e.get("stability")]
        try:
            from agentic_raptor.publication.eval_sets import \
                excluded_context_ids
            frozen = excluded_context_ids()
            ev = [e for e in ev if e.get("context_id") not in frozen]
        except Exception:
            pass

        def closeness(e):
            d = 0.0
            if e.get("pm") is not None and spec.get("phase_margin_target_deg"):
                d += abs(e["pm"] - spec["phase_margin_target_deg"]) / 15.0
            if e.get("gain_db") is not None and spec.get("gain_target_db"):
                d += abs(e["gain_db"] - spec["gain_target_db"]) / 30.0
            return d
        seen = set()
        family_counts: dict = {}
        # pass 1: closest-first, respecting the per-family cap
        # pass 2 (only if pass 1 under-filled k): closest-first, cap lifted
        # -- a sparse memory pool must still return k records rather than
        # fewer, it just must not let one family monopolize them when
        # genuine alternatives exist.
        for enforce_cap in (True, False):
            if len(records) >= k:
                break
            for e in sorted(ev, key=closeness):
                if len(records) >= k:
                    break
                line = (f"{e.get('stages','?')}stage {e['stability']}"
                        + (f" pm={round(e['pm'])}deg"
                           if e.get("pm") is not None else ""))
                fam = e.get("family") or f"{e.get('stages','?')}s_unknown"
                key = e.get("variant") or line
                if key in seen:
                    continue
                if enforce_cap and family_counts.get(fam, 0) >= max_per_family:
                    continue
                seen.add(key)
                family_counts[fam] = family_counts.get(fam, 0) + 1
                records.append({"retrieval_id": e.get("variant") or line,
                                "line": line, "stages": e.get("stages"),
                                "family": fam, "pm": e.get("pm"),
                                "gain_db": e.get("gain_db"),
                                "outcome": ("success" if e.get("postsizing")
                                            else "observation")})
    evidence = "; ".join(r["line"] for r in records)
    # evidence goes BEFORE the proposal instruction
    enriched = (prompt.replace("### BLOCKS", f"### KNOWN {evidence}\n### BLOCKS")
                if evidence else prompt)
    return {"records": records, "retrieval_ids": [r["retrieval_id"]
                                                  for r in records],
            "prompt": enriched}


# ------------------- Stages 3 + 4: propose 5, validate, dedupe ---------------
def retrieve_topology_candidates(spec: dict, target_k: int = TARGET_K,
                                 split: str = "train",
                                 exclude_topology_id: str | None = None,
                                 protected_ids: set | None = None,
                                 seed: int = 0) -> dict:
    """A1 (no-LLM): K unique EXISTING topology graphs, no generative model.

    Draws from the SAME corpus the SFT model was trained on -- each record's
    `response` field is the structural `obj` the model was trained to
    reproduce for that prompt, so this is genuinely "retrieval of existing
    topologies," not a disguised second generator. Deduplicated by the exact
    same `canonical_graph_hash` (`variant_hash`) the LLM path uses, so
    Unique@K/family-coverage are comparable across A0 and A1. Never draws
    from `heldout`/`blindtest`, and never from the frozen evaluation subset
    of `heldout` -- retrieval must not "find" the answer to a spec it is
    later evaluated against.
    """
    import random

    from agentic_raptor.llm_dpo.stage3e4 import variant_hash
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    protected_ids = protected_ids or set()
    pool = [r for r in corpus["records"] if r["split"] == split
           and r.get("topology_id") != exclude_topology_id
           and r.get("context_id") not in protected_ids]
    rng = random.Random(seed)
    rng.shuffle(pool)
    seen_hashes, cands = set(), []
    scanned = 0
    for r in pool:
        scanned += 1
        try:
            obj = json.loads(r["response"])
        except (KeyError, ValueError):
            continue
        fam_hash = variant_hash(obj)
        if fam_hash in seen_hashes:
            continue
        seen_hashes.add(fam_hash)
        cands.append({
            "llm_proposal_id": f"p{len(cands):02d}",
            "canonical_graph_hash": fam_hash,
            "canonical_family": f"{len(obj['stages'])}s_"
                                + ig.compensation_class(obj),
            "obj": obj, "temperature": None, "top_p": None,
            "attempt_index": scanned, "source": "retrieval",
            "retrieved_from_topology_id": r.get("topology_id"),
            "retrieved_from_context_id": r.get("context_id")})
        if len(cands) >= target_k:
            break
    status = ("ok" if len(cands) >= target_k
              else "below_target_k" if len(cands) >= SELECT_K
              else "insufficient_model_diversity")
    return {"candidates": cands, "distinct": len(cands),
            "target_k": target_k, "select_k": SELECT_K,
            "attempts": scanned, "max_temperature": None,
            "distinct_family_count": len({c["canonical_family"]
                                          for c in cands}),
            "conditioning": "retrieval",
            "candidate_generation_status": status,
            "retrieval_pool_size": len(pool)}


def propose_and_validate(model, tok, prompt: str, target_k: int = TARGET_K,
                         max_attempts: int = 20,
                         conditioning: str = "exclusion",
                         seed0: int = 0, use_llm: bool = True,
                         spec: dict | None = None) -> dict:
    """LLM generates; the validator canonicalises and deduplicates.

    Every returned candidate carries source="llm". Nothing is enumerated, and
    a short set is reported as insufficient_model_diversity rather than
    quietly topped up -- silently filling would make the architecture claim
    false while the numbers looked fine.

    use_llm=False is A1 (no-LLM): dispatches to `retrieve_topology_candidates`
    instead -- same return contract, no generative model call of any kind.
    """
    if not use_llm:
        assert spec is not None, "A1 (no-LLM) requires `spec` for retrieval"
        try:
            from agentic_raptor.publication.eval_sets import \
                excluded_context_ids
            protected = set(excluded_context_ids())
        except Exception:
            protected = set()
        return retrieve_topology_candidates(
            spec, target_k=target_k, exclude_topology_id=spec.get("topology_id"),
            protected_ids=protected, seed=seed0)
    # Exclusion conditioning, not temperature alone. The repaired corpus
    # contains 340 "given these, propose a DIFFERENT one" examples; sampling
    # the same prompt hotter never invokes that training. Measured on the
    # heldout split: temperature-only 3.5 distinct with 2s_miller NEVER
    # emitted in 120 samples; exclusion-conditioned 4.17 with every family
    # reachable. Same 20-attempt budget, same ladder, same model.
    from agentic_raptor.llm_dpo.stage3e4 import (DIVERSITY_LADDER,
                                                 propose_diverse,
                                                 propose_diverse_excl)
    if conditioning == "exclusion":
        res = propose_diverse_excl(model, tok, prompt, target_k=target_k,
                                   ladder=DIVERSITY_LADDER, seed0=seed0)
    else:                       # ablation arm: temperature widening only
        res = propose_diverse(model, tok, prompt, target_k=target_k,
                              ladder=DIVERSITY_LADDER, seed0=seed0)
    cands = []
    for i, c in enumerate(res["candidates"]):
        cands.append({
            "llm_proposal_id": f"p{i:02d}",
            "canonical_graph_hash": c["graph_hash"],
            "canonical_family": f"{len(c['obj']['stages'])}s_"
                                + ig.compensation_class(c["obj"]),
            "obj": c["obj"], "temperature": c["temperature"],
            "top_p": c["top_p"], "attempt_index": c["seed"],
            "source": "llm"})
    hashes = [c["canonical_graph_hash"] for c in cands]
    assert len(hashes) == len(set(hashes)), "validator returned duplicates"
    # A short set is REPORTED, not padded. Below SELECT_K the pipeline cannot
    # proceed at all (PUCT needs two distinct candidates to choose between);
    # between SELECT_K and target_k it proceeds and records the true count, so
    # "the proposer supplied N" is a measurement rather than a crash. Nothing
    # is ever enumerated to make up the difference.
    status = ("ok" if len(cands) >= target_k
              else "below_target_k" if len(cands) >= SELECT_K
              else "insufficient_model_diversity")
    return {"candidates": cands, "distinct": len(cands),
            "target_k": target_k, "select_k": SELECT_K,
            "attempts": res["attempts"],
            "max_temperature": res["max_temperature"],
            "distinct_family_count": len({c["canonical_family"]
                                          for c in cands}),
            "conditioning": conditioning,
            "candidate_generation_status": status}


# --------------------- Stage 5: TRUE_ALPHAZERO selects 2 ---------------------
# RETIRED 2026-08-11: puct_select_two() -- root-level candidate selection
# only; the registry it constructed had no derive_edited(), so its one real
# structural-edit action was rejected every call (confirmed both by static
# trace and by a live call whose root_action_ids never contained "a_comp").
# Could never produce a topology outside the initial LLM candidate pool.
# Full retirement provenance (SHA-256, training data, reason):
# artifacts/publication_v3/ROOT_LEVEL_PUCT_RETIRED.json
# Superseded by agentic_raptor.topology_rl.alphazero.alphazero_select_two()
# (genuine multi-depth topology-edit MCTS) and .direct_prior_select_two()
# (A5/NO_ALPHAZERO's fair baseline, extracted from this function's own
# search="none" branch) -- see run_pipeline's stage5_alphazero block below.
# The deleted checkpoint's bytes are preserved for historical reference at
# artifacts/code_snapshots/pre_alphazero_cutover/retired_checkpoint/ (not on
# any runtime-resolvable path).


# ------------- Stages 6 + 7: size both independently, predict both -----------
def _non_rl_size(method: str, tid: str, g, spec: dict, exe, out_dir: Path,
                 costs, budget: int, seed: int,
                 c_load_f: float | None = None) -> dict:
    """Adapts sizing_baselines.py's stateless samplers (already run under the
    identical measure()/postsizing_outcome() harness, at equal real-SPICE
    budgets) to sac_size()'s return contract -- A6 (no-SAC) swaps the sizer
    without size_and_predict's caller knowing the difference. `tpe_lite` is
    the strongest non-RL sizer already implemented in v2 (perturbs the
    top-third of measured history by reward; a heuristic imitation of TPE,
    not a real Tree-Parzen-Estimator -- no BO/CMA-ES library is installed in
    this environment, see the ablation report)."""
    from agentic_raptor.mb_sac.spec_sizing import postsizing_outcome
    from agentic_raptor.publication.sizing_baselines import (_grid, _random,
                                                              _sample_loop,
                                                              _tpe)
    sampler = {"tpe_lite": _tpe, "grid": _grid, "random": _random}[method]
    results = _sample_loop(tid, g, spec, exe, out_dir, costs, budget, seed,
                           sampler, c_load_f=c_load_f)
    best = max(results, key=lambda r: r["reward"])
    return {"outcome": postsizing_outcome(best, spec), "best": best,
           "results": results, "spice_calls": len(results), "seed": seed,
           "schema_version": f"non_rl_{method}.1",
           "action_space": [f"non_rl_{method}"],
           "reward_policy": "spec_reward", "surrogate_checkpoint": None,
           "sac_algorithm": f"non_rl_{method}"}


def size_and_predict(selected: list, spec: dict, exe, budget: int,
                     sizing_repeats: int = 1, use_surrogate: bool = True,
                     use_sizing_ranker: bool = True,
                     sizing_method: str = "sac",
                     c_load_f: float | None = None) -> list:
    """Independent sizing per branch, then a genuine surrogate estimate.

    sizing_method: "sac" (default, MB-SAC) | "tpe_lite" | "grid" | "random"
    -- non-"sac" values are A6 (no-SAC) ablation arms, routed through
    `_non_rl_size` instead of `sac_size`; sizing_repeats/use_surrogate/
    use_sizing_ranker are then not applicable (SAC-specific) and ignored.

    Separate costs objects per branch: sharing one would let branch A's SPICE
    accounting and budget consumption contaminate branch B.

    The previous version built the "prediction" from `sac_size(..., exe, ...)`
    -- real ngspice measurements -- and handed it to the ranker under field
    names like `predicted_feasible`. The ranker was therefore choosing which
    design to verify while already holding the verification answer, so every
    downstream number looked excellent and meant nothing. That is the exact
    defect `ranking/types.py` was written to make impossible, and it is why
    predictions now come from `predict_post_sac`, which never touches ngspice
    and returns explicit UNKNOWNs when the surrogate is unavailable.

    `sizing_repeats` > 1 sizes each branch multiple times (seeds 17, 18, ...)
    and keeps the best attempt. `sac_size` is NOT reproducible at a fixed
    seed -- three identical calls measured UGBW at 6.9k / 20.0k / 8.2k -- so
    one attempt reports the sizer's luck as much as the topology's true
    reach. Default is 1 (off): this changes cost and must be opted into, not
    silently applied to a run that was sized to run once.
    """
    from agentic_raptor.mb_sac.spec_sizing import apply_knobs, sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from run_puct_ablation import _realise
    out = []
    for label, c in zip("AB", selected):
        # AlphaZero edited descendants carry their OWN realised
        # DeviceCircuitGraph (c["device_graph"]) -- an edit applied to a
        # candidate's graph has no natural LLM `obj` JSON form, so
        # re-deriving via _realise(c["obj"]) would silently rebuild the
        # ORIGINAL FAMILY TEMPLATE and discard the edit entirely. Only
        # fall back to _realise(c["obj"]) for candidates that never went
        # through AlphaZero (e.g. A5's direct-selection baseline, or any
        # legacy caller), which still only ever carry an `obj`.
        g = c["device_graph"] if c.get("device_graph") is not None else _realise(c["obj"])
        tid = f"v2_{label}_{c['llm_proposal_id']}"
        attempts = []
        if sizing_method != "sac":
            costs = new_costs()
            attempts.append(_non_rl_size(sizing_method, tid, g, spec, exe,
                                         OUT / "sizing", costs, budget,
                                         seed=17, c_load_f=c_load_f))
        else:
            for i in range(max(1, sizing_repeats)):
                costs = new_costs()               # isolated per branch/attempt
                attempts.append(sac_size(
                    tid, g, spec, exe, OUT / "sizing", costs,
                    budget=budget, seed=17 + i, persist=False,
                    use_surrogate=use_surrogate,
                    use_ranker=use_sizing_ranker, c_load_f=c_load_f))

        def _rank(a):
            o = a["outcome"]
            d = o.get("normalized_distance_to_feasibility")
            return (not o.get("exact_spec_pass"),
                    d if d is not None else float("inf"))
        sz = min(attempts, key=_rank)
        o, best = sz["outcome"], sz["best"]
        if sizing_repeats > 1:
            o = dict(o, sizing_attempts=len(attempts),
                     sizing_attempt_distances=[
                         a["outcome"].get("normalized_distance_to_feasibility")
                         for a in attempts],
                     sizing_attempt_passes=[
                         bool(a["outcome"].get("exact_spec_pass"))
                         for a in attempts])
        knobs = best.get("knobs") or {}
        manifest = sz.get("schema_version")
        # every ngspice call the WINNING attempt's sizing loop made, tagged
        # with its REAL seed. The previous version hardcoded "sz17" -- fine
        # when every call used seed=17, but with sizing_repeats>1 each
        # attempt uses a different seed, and reusing one string for all of
        # them would collide two different SPICE calls onto one fake id.
        sizing_ids = [f"{tid}:sz{sz['seed']}_{r.get('step')}"
                      for r in sz["results"]]
        d = PostSACDesign(
            label=label, spec_id=spec.get("spec_id", "S"),
            llm_proposal_id=c["llm_proposal_id"],
            canonical_graph_hash=c["canonical_graph_hash"],
            topology_signature=c["canonical_family"],
            topology_family=c["canonical_family"],
            sizing_vector=knobs,
            sizing_manifest_hash=manifest,
            sac_trajectory_id=tid,
            action_space_version=",".join(sz.get("action_space") or []),
            reward_version=sz.get("reward_policy"),
            sizing_budget=budget,
            # the WINNING attempt's own call count -- what it cost to reach
            # THIS sizing. Total cost across discarded attempts is separate
            # (o["sizing_spice_calls_all_attempts"] below), because
            # understating it here would make sizing_repeats look free.
            sizing_spice_calls=sz["spice_calls"],
            sizing_spice_call_ids=sizing_ids,
            puct_rank=c["rank"], puct_visits=c.get("visit_count"),
            final_netlist_hash=netlist_hash_of(g))
        if sizing_repeats > 1:
            o["sizing_spice_calls_all_attempts"] = sum(
                a["spice_calls"] for a in attempts)
            o["sizing_winning_seed"] = sz["seed"]
        # PREDICTION: surrogate only, no ngspice, unknown stays unknown
        pred = predict_post_sac(
            spec, topology_hash=c["canonical_graph_hash"],
            topology_family=c["canonical_family"],
            sizing_vector=knobs, sizing_manifest_hash=manifest,
            surrogate_path=sz.get("surrogate_checkpoint"))
        # Phase 7 invariant: the sized branch is the topology PUCT selected
        if d.topology_hash != c["canonical_graph_hash"]:
            raise ArchitectureViolation(
                f"branch {label} sized a topology PUCT did not select")
        # the graph is carried so stage 9 can re-emit the FINAL sized netlist
        # and measure it in a NEW authoritative call
        out.append((d, pred, sz, o, apply_knobs(g, list(
            knobs.values()) if isinstance(knobs, dict) else knobs)))
    if out[0][0].topology_hash == out[1][0].topology_hash:
        raise ArchitectureViolation("both branches sized the same topology")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-index", type=int, default=0)
    ap.add_argument("--split", default="heldout",
                    choices=("train", "heldout", "blindtest"),
                    help="which corpus split to draw the spec from. Ranker "
                         "TRAINING pairs must come from 'train': 27 of 29 "
                         "heldout specs sit in the protected evaluation set, "
                         "so their pairs are refused as training data (and "
                         "should be). 'blindtest' is evaluated once, at the "
                         "end, and never for training or debugging.")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--sizing-repeats", type=int, default=1,
                    help="size each selected branch this many times (seeds "
                         "17, 18, ...) and keep the best attempt. sac_size "
                         "is not reproducible at a fixed seed -- default 1 "
                         "keeps existing behaviour; >1 trades N x sizing "
                         "cost for recovering designs lost to sizer luck "
                         "rather than a genuine capability gap")
    ap.add_argument("--calibrate", action="store_true",
                    help="verify BOTH designs to build a trusted ranker pair")
    ap.add_argument("--arm", default="L8")
    ap.add_argument("--adapter", default=None,
                    help="explicit proposer checkpoint, overriding --arm. "
                         "resolve_arms() only knows campaign checkpoints, so "
                         "an adapter trained outside a campaign (e.g. the "
                         "diverse-corpus retrain) is unreachable by arm name "
                         "and --arm would silently load the OLD checkpoint")
    ap.add_argument("--pvt", action="store_true",
                    help="run a post-optimization PVT robustness sweep on "
                         "the final selected design (only if it passed "
                         "nominal verification). Off by default: adds "
                         "process_corners x supply_voltages x temperatures_c "
                         "FRESH real ngspice calls on top of the nominal "
                         "pipeline cost.")
    ap.add_argument("--pvt-config", default=None,
                    help="YAML file with a top-level 'pvt:' mapping "
                         "(enabled/process_corners/supply_voltages/"
                         "temperatures_c). Overrides --pvt-corners/"
                         "--pvt-voltages/--pvt-temps-c when given.")
    ap.add_argument("--pvt-corners", default="tt",
                    help="comma-separated process corners, e.g. tt,ff,ss. "
                         "Only corners the configured PDK actually ships "
                         "are accepted -- see "
                         "agentic_raptor.electrical.pvt_eval."
                         "available_process_corners().")
    ap.add_argument("--pvt-voltages", default="1.8",
                    help="comma-separated ABSOLUTE supply voltages in "
                         "volts, e.g. 1.62,1.8,1.98")
    ap.add_argument("--pvt-temps-c", default="27",
                    help="comma-separated ABSOLUTE temperatures in deg C, "
                         "e.g. -40,27,85")
    args = ap.parse_args()
    from run_qwen_ablation import _load, resolve_arms
    if args.adapter:
        adapter = args.adapter
        if not (Path(adapter) / "adapter_config.json").is_file():
            raise SystemExit(f"not a peft adapter directory: {adapter}")
    else:
        adapter = resolve_arms()[args.arm]["adapter"]
        assert adapter, f"{args.arm} checkpoint missing"
    tok, model = _load(adapter)
    pvt_config = None
    if args.pvt or args.pvt_config:
        if args.pvt_config:
            pvt_config = PvtConfig.from_yaml(args.pvt_config)
            pvt_config.enabled = pvt_config.enabled or args.pvt
        else:
            pvt_config = PvtConfig(
                enabled=True,
                process_corners=tuple(c.strip()
                                      for c in args.pvt_corners.split(",")),
                supply_voltages=tuple(float(v)
                                      for v in args.pvt_voltages.split(",")),
                temperatures_c=tuple(float(t)
                                     for t in args.pvt_temps_c.split(",")))
    trace = run_pipeline(model, tok, adapter, split=args.split,
                         spec_index=args.spec_index, budget=args.budget,
                         calibrate=args.calibrate,
                         sizing_repeats=args.sizing_repeats,
                         pvt_config=pvt_config)
    print(json.dumps({k: trace[k] for k in
                      ("stage3_propose", "stage5_alphazero", "stage8_ranker",
                       "stage9_verification", "nominal", "fom", "pvt",
                       "spice_usage", "stage11_feedback")
                      if k in trace}, indent=1, default=str))


def run_pipeline(model, tok, adapter, *, split="heldout", spec_index=0,
                 budget=32, calibrate=False, conditioning="exclusion",
                 search="one_root", ranker_mode="dpo", seed=0,
                 out_prefix="TRACE", ranker_ckpt=None, value_ckpt=None,
                 rag_memory=None, harvest=False, sizing_repeats=1,
                 pvt_config: PvtConfig | None = None,
                 learning_mode="adaptive", use_surrogate=True,
                 use_sizing_ranker=True, sizing_method="sac",
                 use_llm=True, use_rag=True,
                 c_load_override_f=None, c_load_override_reason=None):
    """One full pipeline execution. Returns the trace.

    Separated from main() so an ablation can hold ONE loaded model across many
    runs -- reloading a 4B model per run costs more than the pipeline itself.

    conditioning / search / ranker_mode are the ablation arms; the defaults
    are the production configuration.

    learning_mode: "adaptive" (default -- everything persists exactly as
    before) | "frozen" | "static". Frozen/static evaluation runs (A0-A8, and
    A9's static lineage) must NEVER mutate shared learning state: this skips
    both the ranker-training-pair write (record_pair's `persist=False`) AND
    the RAG/SFT/SAC-replay/PUCT/proposer-DPO stream harvest entirely, while
    still computing and returning the same diagnostic fields in the trace.

    c_load_override_f: Stage 1.5 repair (2026-08-09). Normal runs leave this
    None, so the load actually simulated is the SPEC's own requested
    load_capacitance_pf (agentic_raptor.electrical.effective_c_load) -- not
    the old unconditional NOMINAL_CLOAD_F fallback. Pass an explicit value
    ONLY for a deliberate experiment (e.g. the fixed-topology physical
    sanity sweep); c_load_override_reason should say why, and both get
    recorded in trace["stage1_spec"] so paper-mode can hard-fail on an
    override that has no recorded reason.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from agentic_raptor.electrical import discover_ngspice, effective_c_load

    # ---- Stage 1: specification -------------------------------------------
    corpus = json.loads(
        (ROOT / "artifacts/stage3e4/corpus.json").read_text())
    pool = [r for r in corpus["records"] if r["split"] == split]
    if not pool:
        raise SystemExit(f"no records in split {split!r}")
    rec = pool[spec_index % len(pool)]
    spec = ig.parse_spec(rec["prompt"])
    spec["spec_id"] = rec["context_id"]
    spec_hash = ig.sha_json(spec)
    # carried so record_pair can verify the outcome belongs to THIS spec
    spec["spec_hash"] = spec_hash
    # added AFTER the hash so it can never change spec_hash / break
    # comparability with any run recorded before this field existed
    spec["topology_id"] = rec.get("topology_id")
    trace_name = (f"{out_prefix}_{split}_{spec_index % len(pool):03d}"
                  f"_{spec['spec_id']}.json")
    # Stage 1.5: ONE canonical load for this entire run, resolved once and
    # threaded to every real measurement below (SAC/non-RL sizing, final
    # verify(), and from there PVT/FoM) -- never re-derived independently
    # per stage, so nominal and PVT cannot silently disagree on what load
    # they simulated.
    run_cl = effective_c_load(spec, override=c_load_override_f)
    trace = {"stage1_spec": {"spec_id": spec["spec_id"], "split": split,
                             "spec_index": spec_index % len(pool),
                             "spec_hash": spec_hash, "spec": spec,
                             "requested_c_load_f": (
                                 spec.get("load_capacitance_pf") * 1e-12
                                 if spec.get("load_capacitance_pf") is not None
                                 else None),
                             "effective_c_load_f": run_cl,
                             "c_load_override": c_load_override_f is not None,
                             "c_load_override_value": c_load_override_f,
                             "c_load_override_reason": c_load_override_reason}}

    # ---- Stage 2: RAG ------------------------------------------------------
    rag = rag_stage(spec, rec["prompt"], use_rag=use_rag, memory_path=rag_memory)
    trace["stage2_rag"] = {"retrieval_ids": rag["retrieval_ids"],
                           "records": len(rag["records"]),
                           "use_rag": use_rag}

    # ---- Stages 3 + 4: LLM generates 5, validator dedupes ------------------
    trace["proposer_checkpoint"] = str(adapter)
    trace["proposer_checkpoint_source"] = "explicit adapter"
    trace["arms"] = {"conditioning": conditioning, "search": search,
                     "ranker": ranker_mode, "seed": seed,
                     "learning_mode": learning_mode,
                     "sizing_method": sizing_method,
                     "use_surrogate": use_surrogate,
                     "use_sizing_ranker": use_sizing_ranker,
                     "use_llm": use_llm, "use_rag": use_rag}
    prop = propose_and_validate(model, tok, rag["prompt"],
                                conditioning=conditioning, seed0=seed,
                                use_llm=use_llm, spec=spec)
    trace["stage3_propose"] = {
        k: prop[k] for k in ("distinct", "target_k", "select_k", "attempts",
                             "max_temperature", "distinct_family_count",
                             "conditioning",
                             "candidate_generation_status")}
    trace["stage3_propose"]["proposal_hashes"] = [
        c["canonical_graph_hash"] for c in prop["candidates"]]
    # evidence the gate verifies: every candidate's provenance, and the full
    # canonical hash list (not just a count, which cannot prove uniqueness)
    trace["stage3_propose"]["sources"] = sorted(
        {c["source"] for c in prop["candidates"]})
    trace["stage3_propose"]["canonical_graph_hashes"] = [
        c["canonical_graph_hash"] for c in prop["candidates"]]
    from agentic_raptor.ranking import directory_sha256
    trace["models"] = {
        "proposer": {"training_method": "SFT", "checkpoint": str(adapter),
                     "hash": directory_sha256(str(adapter))}}
    trace["stage3_propose"]["reached_target_k"] = (
        prop["distinct"] >= TARGET_K)
    # Abort ONLY when the proposer cannot supply enough candidates for PUCT to
    # choose between (< SELECT_K). A short-but-usable set proceeds and the
    # true count is recorded, because "the proposer supplied 4 of 5" is a
    # measurement worth having and a crash is not. Nothing is enumerated to
    # close the gap either way -- that guarantee is unchanged.
    if prop["candidate_generation_status"] == "insufficient_model_diversity":
        trace["result"] = "ABORTED: insufficient_model_diversity"
        (OUT / trace_name).write_text(
            json.dumps(trace, indent=1, default=str), encoding="utf-8")
        print(json.dumps(trace["stage3_propose"], indent=1))
        raise ArchitectureViolation(
            f"proposer returned {prop['distinct']} distinct graphs, need at "
            f"least {SELECT_K} for PUCT to select from; enumeration is NOT "
            f"used to fill the gap")

    # ---- Stage 5: TRUE_ALPHAZERO selects 2 (retired root-level PUCT / --
    # puct_select_two() -- 2026-08-11) ---------------------------------------
    # search="one_root" (still the historical parameter name/default, kept
    # for ablation_v3.to_run_pipeline_kwargs() compatibility -- use_mcts
    # maps to it unchanged) now dispatches to genuine multi-depth
    # AlphaZero topology-edit search; search="none" (A5/NO_ALPHAZERO)
    # dispatches to direct_prior_select_two(), the retired selector's own
    # search="none" branch extracted verbatim -- pure proposer-prior
    # ranking, zero AlphaZero policy/value/search involvement, zero
    # dependency on any retired machinery.
    from agentic_raptor.topology_rl.alphazero import (
        AlphaZeroConfig, alphazero_select_two, direct_prior_select_two)
    if search == "none":
        sel = direct_prior_select_two(prop["candidates"], spec)
        stage5_trace = {
            "search_topology": "a5_direct_prior_no_alphazero",
            "input_count": len(prop["candidates"]),
            "selected_count": len(sel["selected"]),
            "candidate_visits": sel["candidate_visits"],
            "ranking": [{k: c[k] for k in
                         ("llm_proposal_id", "canonical_graph_hash",
                          "canonical_family", "policy_prior",
                          "value_prediction", "visit_count", "rank",
                          "selected_top2")} for c in sel["ranked"]]}
    else:
        az_cfg = AlphaZeroConfig(seed=seed)
        sel = alphazero_select_two(prop["candidates"], spec, spec["spec_id"],
                                   value_ckpt=value_ckpt, seed=seed,
                                   config=az_cfg)
        stage5_trace = {
            "search_topology": "true_alphazero_multi_depth_edit_search",
            "input_count": len(prop["candidates"]),
            "selected_count": len(sel["selected"]),
            "tree_nodes": sel["tree_nodes"],
            "max_depth_reached": sel["max_depth_reached"],
            "seed_topology_hashes": sel["seed_topology_hashes"],
            "electrical_environment_version": sel["electrical_environment_version"],
            "ranking": [{k: c[k] for k in
                         ("llm_proposal_id", "canonical_graph_hash",
                          "canonical_family", "rank", "visit_count",
                          "originating_seed_id", "originating_seed_hash",
                          "edit_history", "edit_depth", "is_edited_descendant")}
                        for c in sel["selected"]]}
    trace["stage5_alphazero"] = stage5_trace

    # ---- Stages 6 + 7: size both, predict both ----------------------------
    exe = discover_ngspice()
    branches = size_and_predict(sel["selected"], spec, exe, budget,
                                sizing_repeats=sizing_repeats,
                                use_surrogate=use_surrogate,
                                use_sizing_ranker=use_sizing_ranker,
                                sizing_method=sizing_method,
                                c_load_f=run_cl)
    (da, pa, sza, oa, ga), (db, pb, szb, ob, gb) = branches

    def _calls_to_first_pass(sz):
        # 1-indexed count of the WINNING sizing attempt's own real SPICE
        # calls up to and including the first exact_spec_pass step; None if
        # it never passed within budget. Computed unconditionally (not only
        # when harvesting) -- this is a diagnostic metric every A0-A9 result
        # needs (PRIMARY PAPER METRICS #3), not training data.
        from agentic_raptor.mb_sac.spec_sizing import postsizing_outcome
        for i, r in enumerate(sz.get("results") or []):
            if postsizing_outcome(r, spec).get("exact_spec_pass"):
                return i + 1
        return None
    trace["stage6_sizing"] = {
        lbl: {"topology_hash": d.topology_hash,
              "sizing_manifest_hash": d.sizing_manifest_hash,
              "spice_calls": d.sizing_spice_calls,
              "sizing_vector": d.sizing_vector,
              "calls_to_first_pass": _calls_to_first_pass(sz),
              # only present when sizing_repeats > 1: how many attempts were
              # tried, which one won, and their spread -- so a passing design
              # is visibly "found on attempt 3 of 3" rather than looking like
              # a first-try success
              "sizing_attempts": o.get("sizing_attempts"),
              "sizing_attempt_distances": o.get("sizing_attempt_distances"),
              "sizing_attempt_passes": o.get("sizing_attempt_passes"),
              "sizing_winning_seed": o.get("sizing_winning_seed"),
              "sizing_spice_calls_all_attempts":
              o.get("sizing_spice_calls_all_attempts")}
        for lbl, d, o, sz in (("A", da, oa, sza), ("B", db, ob, szb))}
    trace["stage7_surrogate"] = {
        lbl: {"gain_db": p.gain_db, "pm_deg": p.pm_deg,
              "ugbw_hz": p.ugbw_hz,
              "margins": p.normalized_margins,
              "worst_violation": p.worst_predicted_violation,
              "feasible_for_spec": p.predicted_feasible_for(spec),
              "missing_constraints": list(p.missing_constraints(spec)),
              "uncertainty": p.predictive_uncertainty,
              "stability_probability": p.stability_probability,
              "authoritative": p.authoritative,
              "source": p.source,
              "spice_result_id": p.spice_result_id,
              "surrogate_checkpoint_hash": p.surrogate_checkpoint_hash}
        for lbl, p in (("A", pa), ("B", pb))}

    # ---- Stage 8: post-SAC ranker selects 1 -------------------------------
    # A missing checkpoint is explicit, never a silent fallback: the arm name
    # follows the model that actually decided, so a run made before the
    # ranker was trained can never be reported as DPO-ranked.
    from agentic_raptor.ranking.model import PostSACRanker
    ranker = (PostSACRanker.load(ranker_ckpt)
              if ranker_mode == "dpo" else None)
    arm = "dpo_ranker" if ranker else "explicit_baseline"
    decision = compare(da, db, pa, pb, spec=spec,
                       model=ranker,          # compare() calls model.score()
                       ranker_arm=arm,
                       ranker_checkpoint_hash=(ranker.checkpoint_hash
                                               if ranker else None))
    trace["models"]["ranker"] = {
        "training_method": "DPO" if ranker else None,
        "frozen": True if ranker else None,
        "hash": ranker.checkpoint_hash if ranker else None,
        "arm": arm}
    trace["stage8_ranker"] = {
        "ranker_arm": arm,
        "input_count": 2,
        "both_sized": bool(da.sizing_vector) and bool(db.sizing_vector),
        "backup_design": decision["backup_design"],
        "ranker_checkpoint_hash": decision.get("ranker_checkpoint_hash"),
        "selected_design": decision["selected_design"],
        "decision_basis": decision["decision_basis"],
        "deciding_level": decision["deciding_level"],
        "low_confidence": decision["low_confidence"],
        "ranker_score_A": decision.get("ranker_score_A"),
        "ranker_score_B": decision.get("ranker_score_B"),
        "score_margin": decision.get("score_margin"),
        "ranker_error": decision.get("ranker_error"),
        "hard_safety_tier_A": decision.get("hard_safety_tier_A"),
        "hard_safety_tier_B": decision.get("hard_safety_tier_B"),
        "selected_topology_hash": decision["selected_topology_hash"],
        "backup_topology_hash": decision["backup_topology_hash"]}

    # ---- Stages 9 + 10: authoritative verification, backup on failure -----
    # A NEW ngspice call on the final sized netlist. Reusing the sizing
    # outcome would mean the "verification" happened BEFORE the ranker chose,
    # and `_provenance_ok` rejects such a pair outright
    # (final_call_id_is_a_sizing_call_id).
    from agentic_raptor.mb_sac.spec_sizing import measure, postsizing_outcome
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    designs = {"A": (da, ga), "B": (db, gb)}
    sel_lbl = decision["selected_design"]
    bak_lbl = "B" if sel_lbl == "A" else "A"
    # Stage 1.5 repair: the authoritative final measurement -- the one whose
    # c_load_f becomes trace["nominal"]["c_load_f"], and from there PVT and
    # FoM -- reuses the SAME run_cl every other real measurement in this run
    # resolved (Stage 1, above), not a value re-derived here independently.
    verify_cl = run_cl

    def verify(label: str, mode: str):
        d, graph = designs[label]
        call_id = f"final:{spec['spec_id']}:{label}:{int(time.time()*1000)}"
        # tag must be unique per run: a fixed "ver_A"/"ver_B" overwrote the
        # previous run's ngspice output, leaving only the last one auditable
        tag = f"ver_{spec['spec_id']}_{label}_{int(time.time()*1000)}"
        meas = measure(d.sac_trajectory_id, graph, exe, OUT / "verify",
                       tag, new_costs(), c_load_f=verify_cl)
        oc = postsizing_outcome(meas, spec)
        return oc, outcome_from_sizing(
            oc, call_id=call_id, topology_hash=d.canonical_graph_hash,
            sizing_manifest_hash=d.sizing_manifest_hash,
            netlist_hash=netlist_hash_of(graph), mode=mode, best=meas,
            spec_id=spec["spec_id"], spec_hash=spec_hash)

    oc_sel, auth_sel = verify(sel_lbl, "final_verification")
    verified = {sel_lbl: auth_sel}
    measured = {sel_lbl: {"gain_db": auth_sel.gain_db,
                          "pm_deg": auth_sel.pm_deg,
                          "ugbw_hz": auth_sel.ugbw_hz,
                          "distance": auth_sel.normalized_distance_to_feasibility}}
    reason = None
    dual = calibrate or decision["low_confidence"]
    if not oc_sel.get("exact_spec_pass"):
        dual, reason = True, "selected_design_failed"
    elif calibrate:
        reason = "calibration_sampling"
    elif decision["low_confidence"]:
        reason = "low_ranker_confidence"
    if dual:
        _oc_bak, auth_bak = verify(bak_lbl, "backup")
        verified[bak_lbl] = auth_bak
        measured[bak_lbl] = {"gain_db": auth_bak.gain_db,
                             "pm_deg": auth_bak.pm_deg,
                             "ugbw_hz": auth_bak.ugbw_hz,
                             "distance":
                             auth_bak.normalized_distance_to_feasibility}
    trace["stage9_verification"] = {
        "authoritative_engine": "ngspice",
        "verified_designs": sorted(verified),
        "second_design_reason": reason,
        "selected_exact_pass": bool(oc_sel.get("exact_spec_pass")),
        "selected_failure_reason": oc_sel.get("exact_failure_reason"),
        # achieved vs target: "UGBW below target" alone cannot distinguish a
        # 5% miss from a 5x miss, and those imply different fixes
        "measured": measured,
        "targets": {"gain_db": spec.get("gain_target_db"),
                    "pm_deg": spec.get("phase_margin_target_deg"),
                    "ugbw_hz": spec.get("ugbw_target_hz")},
        "distance_to_feasibility":
            oc_sel.get("normalized_distance_to_feasibility"),
        "worst_constraint": oc_sel.get("worst_failing_constraint"),
        "final_call_ids": {k: v.call_id for k, v in verified.items()},
        "final_spice_call_id": auth_sel.call_id,
        "final_verification_source": auth_sel.source,
        "sizing_spice_call_ids": (list(da.sizing_spice_call_ids or [])
                                  + list(db.sizing_spice_call_ids or [])),
        "second_design_simulated": bak_lbl in verified,
        "final_calls_are_new": all(
            v.call_id not in (designs[k][0].sizing_spice_call_ids or [])
            for k, v in verified.items()),
        "sizing_spice_calls": da.sizing_spice_calls + db.sizing_spice_calls,
        "verification_spice_calls": len(verified)}

    # ---- Nominal metrics + FoM (Parts 1-2) ---------------------------------
    # Reads ONLY the authoritative measurement of the SELECTED, verified
    # design (auth_sel) -- never a prediction, never the sizing knob.
    sel_d, sel_graph = designs[sel_lbl]
    ibias_a = None  # optimizer/design knob -- see agentic_raptor.electrical.fom
    for dev in getattr(sel_graph, "devices", []):
        if getattr(dev, "kind", None) == "isrc":
            ibias_a = (dev.sizing or {}).get("value")
            break
    fail_reasons = (oc_sel.get("exact_failure_reason").split("; ")
                    if oc_sel.get("exact_failure_reason") else [])
    # Stage 1.5 provenance: requested (from the spec text) vs simulated (what
    # verify_cl actually resolved to, i.e. what auth_sel.c_load_f measures
    # under) are recorded SEPARATELY so a mismatch is visible in the trace
    # rather than only inferable by re-deriving both values by hand. Reuses
    # the SAME requested/override values Stage 1 already resolved into
    # trace["stage1_spec"] -- not re-derived independently here.
    requested_cl = trace["stage1_spec"]["requested_c_load_f"]
    cl_override = requested_cl is not None and abs(
        (auth_sel.c_load_f or 0) - requested_cl) > 1e-15
    trace["nominal"] = {
        "gain_db": auth_sel.gain_db, "pm_deg": auth_sel.pm_deg,
        "ugbw_hz": auth_sel.ugbw_hz,
        "idd_a": auth_sel.idd_a,            # TOTAL measured supply current
        "ibias_a": ibias_a,                 # design knob -- NOT idd_a
        "power_w": auth_sel.power_w,
        "c_load_f": auth_sel.c_load_f,
        "requested_c_load_f": requested_cl,
        "simulated_c_load_f": auth_sel.c_load_f,
        "c_load_override": cl_override,
        "c_load_override_reason": (
            trace["stage1_spec"]["c_load_override_reason"] if cl_override
            else ("spec parsed no load_capacitance_pf -- fell back to "
                 "NOMINAL_CLOAD_F" if requested_cl is None else None)),
        "c_load_unexplained_mismatch": bool(
            cl_override and not trace["stage1_spec"]["c_load_override_reason"]),
        "complete_pass": bool(oc_sel.get("exact_spec_pass")),
        "failure_reasons": fail_reasons,
        "specification_margins": oc_sel.get("margin_vector"),
        "ugbw_mhz": (auth_sel.ugbw_hz / 1e6)
        if auth_sel.ugbw_hz is not None else None,
        "idd_ma": (auth_sel.idd_a / 1e-3)
        if auth_sel.idd_a is not None else None,
        "c_load_pf": (auth_sel.c_load_f / 1e-12)
        if auth_sel.c_load_f is not None else None}
    trace["fom"] = compute_fom(auth_sel.ugbw_hz, auth_sel.c_load_f,
                               auth_sel.idd_a)

    # ---- PVT robustness evaluation (Parts 3-9), post-optimization only ----
    # `pvt_result` is a local block written ONLY into `trace["pvt"]` below --
    # structurally separate from `hv`/record_pair/harvest_run further down,
    # so PVT evaluation data cannot reach RAG/SFT/SAC-replay/PUCT/DPO
    # training streams regardless of split (Part 11).
    pvt_result: dict = {"enabled": bool(pvt_config and pvt_config.enabled)}
    pvt_spice_calls = 0
    if pvt_config and pvt_config.enabled:
        pvt_eligible = bool(oc_sel.get("exact_spec_pass"))
        pvt_result["eligible"] = pvt_eligible
        if pvt_eligible:
            corner_records = run_pvt_sweep(
                sel_d.sac_trajectory_id, sel_graph, spec, exe, OUT / "pvt",
                pvt_config, label=sel_lbl, c_load_f=auth_sel.c_load_f)
            pvt_spice_calls = sum(r.get("real_spice_calls", 1)
                                  for r in corner_records)
            pvt_result.update(aggregate_pvt(corner_records,
                                            pvt_config.required_corner_ids))
            pvt_result["corners"] = corner_records
        else:
            pvt_result["reason_not_run"] = "nominal_design_failed_spec"
    trace["pvt"] = pvt_result

    # ---- SPICE call accounting (Part 9) ------------------------------------
    # PVT calls are post-hoc robustness-evaluation cost, never counted as
    # cost required to FIND the nominal design.
    optimization_calls = da.sizing_spice_calls + db.sizing_spice_calls
    final_verification_calls = len(verified)
    trace["spice_usage"] = {
        "optimization_spice_calls": optimization_calls,
        "final_nominal_verification_calls": final_verification_calls,
        "pvt_spice_calls": pvt_spice_calls,
        "total_spice_calls": (optimization_calls + final_verification_calls
                              + pvt_spice_calls)}

    # ---- Stage 11: ranker training data (trusted only if both measured) ----
    # predictions are recorded too: the ranker trainer needs the FEATURES the
    # ranker saw, paired with the MEASURED label
    # FROZEN/STATIC evaluation (A0-A8, A9's static lineage) must never grow
    # the ranker's training queue -- persist=False still computes and
    # returns status/ranker_correct/provenance for the trace below, it just
    # skips the trusted_pairs.jsonl/provisional_pairs.jsonl append.
    pair = record_pair(spec, da, db,
                       verified.get("A"), verified.get("B"),
                       pred_a=pa, pred_b=pb,
                       ranker_choice=sel_lbl,
                       persist=(learning_mode == "adaptive"))
    # Each measured outcome must route back to the branch that produced it.
    # Asserted, not assumed: mis-routed feedback would train every component
    # on another branch's result and still look well-formed.
    routed, problems = {}, []
    for lbl, auth in verified.items():
        d = designs[lbl][0]
        routed[lbl] = {"topology_hash": auth.topology_hash,
                       "llm_proposal_id": d.llm_proposal_id,
                       "call_id": auth.call_id,
                       "exact_spec_pass": auth.exact_spec_pass}
        if auth.topology_hash != d.canonical_graph_hash:
            problems.append(f"{lbl}:topology_hash_mismatch")
        if auth.spec_id != spec["spec_id"]:
            problems.append(f"{lbl}:spec_id_mismatch")
        if auth.sizing_manifest_hash != d.sizing_manifest_hash:
            problems.append(f"{lbl}:manifest_mismatch")
        if auth.call_id in (d.sizing_spice_call_ids or []):
            problems.append(f"{lbl}:call_id_was_a_sizing_call")
    trace["stage11_feedback"] = {
        "ranker_pair_status": pair["status"],
        "both_measured": bool(verified.get("A") and verified.get("B")),
        "provenance_problems": pair.get("provenance_problems"),
        "routed": routed,
        "branch_assertions_passed": not problems,
        "branch_assertion_problems": problems,
        "ranker_correct": pair.get("ranker_correct")}

    # Built UNCONDITIONALLY, same as record_pair() above. Previously this only
    # ran when a caller explicitly passed harvest=True, and the direct CLI
    # (main(), below) never did -- so 66 real calibration runs fixed the
    # ranker while feeding NOTHING to RAG, the SFT queue, SAC replay, PUCT
    # examples or the proposer's DPO queue. Every measured outcome, pass or
    # fail, must reach these on every real run, not only when routed through
    # the self-improvement loop wrapper.
    #
    # Everything the learning streams need, in REPLAYABLE form. The PUCT
    # trainer rebuilds states from action ids like "a_sel_p00", which mean
    # nothing outside this run -- so the candidate OBJECTS are stored, not
    # just their hashes. Without them the policy/value trainer silently sees
    # zero usable examples and still reports success.
    #
    # learning_mode != "adaptive" (FROZEN A0-A8 evaluation, or A9's STATIC
    # lineage): this whole block -- including the harvest payload build and
    # every stream append -- is skipped outright. Not "harvest then discard":
    # the payload is never constructed and harvest_run() is never called, so
    # there is no route by which a frozen/static evaluation outcome can reach
    # RAG/SFT/SAC-replay/PUCT/proposer-DPO training data.
    if learning_mode == "adaptive":
        hv = {
            "spec": spec, "spec_hash": spec_hash,
            "split": split, "spec_index": spec_index % len(pool),
            "seed": seed, "budget": budget,
            # Stage 1.6: carried into every harvested stream record's
            # provenance block by harvest_run() -- requested comes from the
            # spec text, simulated is read per-branch from `authoritative.
            # c_load_f` above (both now flow through as of this change).
            "requested_c_load_f": trace["stage1_spec"]["requested_c_load_f"],
            # sel["ranked"] (A5/direct_prior_select_two) covers every
            # ORIGINAL LLM candidate. alphazero_select_two has no
            # equivalent -- its "ranked_all" ranks TREE STATES (seeds AND
            # edited descendants together), not original LLM proposals,
            # so equating the two would be misleading. Falls back to just
            # the two SELECTED entries in that case: an edited
            # descendant's `obj` is honestly None (it was never LLM-
            # proposed -- see streams.py's sft_queue construction, which
            # correctly then skips it: crediting an AlphaZero-discovered
            # structural edit to "the LLM proposed this" would be exactly
            # the provenance violation this project has never allowed
            # elsewhere). rag_memory/ranker_pairs are unaffected -- those
            # key off the SIZED DESIGN's own canonical_graph_hash/family,
            # not this candidate list.
            "candidates": [{k: c[k] for k in
                            ("llm_proposal_id", "canonical_graph_hash",
                             "canonical_family", "obj", "rank",
                             "visit_count", "selected_top2")}
                           for c in (sel.get("ranked") or sel["selected"])],
            "root_visits": sel.get("visits") or {},
            "candidate_visits": sel.get("candidate_visits") or {},
            "root_action_ids": sel.get("root_action_ids") or [],
            "root_state": sel.get("root_state"),
            "candidate_manifests": sel.get("candidate_manifests") or {},
            "search": sel.get("search"),
            "branches": {
                lbl: {"design": {f: getattr(d, f) for f in
                                 ("label", "spec_id", "llm_proposal_id",
                                  "canonical_graph_hash", "topology_family",
                                  "sizing_vector", "sizing_manifest_hash",
                                  "sizing_spice_calls", "puct_rank",
                                  "final_netlist_hash")},
                      "prediction": {f: getattr(p, f) for f in
                                     ("gain_db", "pm_deg", "normalized_margins",
                                      "predictive_uncertainty",
                                      "stability_probability",
                                      "surrogate_checkpoint_hash")},
                      "sac": {"spice_calls": sz["spice_calls"],
                              "algorithm": sz.get("sac_algorithm"),
                              "reward_policy": sz.get("reward_policy"),
                              "trajectory": [
                                  {"step": r.get("step"),
                                   "knobs": r.get("knobs"),
                                   "reward": r.get("reward"),
                                   "pm_deg": r.get("pm_deg"),
                                   "gain_db": r.get("gain_db"),
                                   "ugbw_hz": r.get("ugbw_hz")}
                                  for r in sz["results"]]},
                      "authoritative": (
                          {f: getattr(verified[lbl], f) for f in
                           ("call_id", "topology_hash", "sizing_manifest_hash",
                            "netlist_hash", "mode", "exact_spec_pass",
                            "operating_point_valid", "spice_converged",
                            "verified_stable", "spec_id", "spec_hash",
                            "gain_db", "pm_deg", "ugbw_hz", "power_w",
                            # Stage 1.6: c_load_f is what makes a stream
                            # record reconstructible as PRE_/POST_CLOAD_FIX
                            # -- omitted before, silently losing this
                            # provenance for every RAG/PUCT/ranker record.
                            "idd_a", "c_load_f",
                            "normalized_distance_to_feasibility",
                            "exact_failure_reason")}
                          if lbl in verified else None)}
                for lbl, (d, p, sz) in
                (("A", (da, pa, sza)), ("B", (db, pb, szb)))},
            "ranker": {"arm": arm,
                       "selected_design": decision["selected_design"],
                       "backup_design": decision["backup_design"],
                       "decision_basis": decision["decision_basis"],
                       "deciding_level": decision["deciding_level"],
                       "low_confidence": decision["low_confidence"],
                       "score_A": decision.get("ranker_score_A"),
                       "score_B": decision.get("ranker_score_B"),
                       "checkpoint_hash": decision.get(
                           "ranker_checkpoint_hash")},
            "proposer_checkpoint": str(adapter)}

        # ---- route the measured outcome: TRUE -> SFT/PUCT/SAC/DPO queues,
        # FALSE -> RAG only. The per-stream admission rules already live in
        # harvest_run() (streams.py) -- rag_memory takes every authoritative
        # outcome regardless of pass/fail; sft_queue only ADMITS a passing,
        # provenance-clean, train-split, non-protected result; proposer_dpo
        # pairs need a clean pass-vs-fail split; sac_replay and puct_examples
        # take both, since a failure is still a training signal for THOSE.
        from agentic_raptor.selfimprove_v2.streams import append, harvest_run
        try:
            from agentic_raptor.publication.eval_sets import \
                excluded_context_ids
            protected = set(excluded_context_ids())
        except Exception:
            protected = set()
        streams = harvest_run(hv, split=split, protected_ids=protected)
        # Stage 1.6: versioned for the same reason as RAG_MEMORY_V2 above --
        # puct_examples.jsonl / ranker_pairs.jsonl / etc. under the OLD path
        # mix pre- and post-repair rows; new writes go to a fresh directory.
        LIVE = ROOT / "artifacts/publication_v2/live_streams_post_cload_v1"
        counts = {}
        for name, rows in streams.items():
            if name == "ranker_pairs":
                # record_pair() above is the proven mechanism for this one
                # (it fixed the deployed checkpoint); routing the SAME pair
                # through harvest_run's separate winner/loser computation
                # too would create two parallel, possibly disagreeing
                # sources of ranker training data for one measured outcome.
                continue
            path = (RAG_MEMORY_V2 if name == "rag_memory"
                    else LIVE / f"{name}.jsonl")
            counts[name] = append(path, rows)
        trace["stage11_feedback"]["routed_to_streams"] = counts
        if harvest:
            trace["_harvest"] = hv
    else:
        trace["stage11_feedback"]["routed_to_streams"] = {
            "skipped": True, "reason": f"learning_mode={learning_mode!r}"}

    trace["provenance_chain"] = {
        "spec_hash": spec_hash,
        "llm_proposal_hashes": trace["stage3_propose"]["proposal_hashes"],
        "puct_selected": [da.topology_hash, db.topology_hash],
        "sizing_manifests": [da.sizing_manifest_hash, db.sizing_manifest_hash],
        "ranker_selected": decision["selected_topology_hash"],
        "verified": sorted(verified)}
    trace["seconds"] = round(time.time() - t0, 1)
    trace["result"] = "OK"
    hv = trace.pop("_harvest", None)
    (OUT / trace_name).write_text(
        json.dumps(trace, indent=1, default=str), encoding="utf-8")
    if hv is not None:
        trace["_harvest"] = hv          # returned in-memory, not in the trace
    return trace


if __name__ == "__main__":
    main()
