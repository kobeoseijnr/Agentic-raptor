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

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.publication import ROOT
from agentic_raptor.ranking import (PostSACDesign, compare, netlist_hash_of,
                                    outcome_from_sizing, predict_post_sac,
                                    record_pair)

OUT = ROOT / "artifacts/publication_v2/raptor_v2_runs"
MEM = ROOT / "datasets/simulation_memory"
TARGET_K = 5
SELECT_K = 2


class ArchitectureViolation(AssertionError):
    """A stage produced the wrong count or lost candidate provenance."""


# ----------------------------- Stage 2: RAG ----------------------------------
#: v2 RAG memory: measured evidence produced BY the v2 pipeline. The old
#: self_improvement_runs.jsonl is archived -- it came from a different
#: architecture and was measured before the input-common-mode fix, so its
#: phase margins describe circuits running at 1/79th of design current.
RAG_MEMORY_V2 = ROOT / "artifacts/publication_v2/selfimprove/rag_memory_v2.jsonl"


def retrieve(spec: dict, prompt: str, k: int = 6,
             memory_path=None) -> dict:
    """Spec-conditioned retrieval of measured evidence.

    Retrieval provides CONTEXT ONLY. It must never become a PUCT candidate --
    that conflation is what the audit found in the old path.

    Reads the v2 memory only. An empty memory yields NO evidence line, which
    is the honest state before the loop has produced any: inventing context
    would put unmeasured claims in front of the proposer.
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
        for e in sorted(ev, key=closeness):
            line = (f"{e.get('stages','?')}stage {e['stability']}"
                    + (f" pm={round(e['pm'])}deg"
                       if e.get("pm") is not None else ""))
            if line in seen:
                continue
            seen.add(line)
            records.append({"retrieval_id": e.get("variant") or line,
                            "line": line, "stages": e.get("stages"),
                            "pm": e.get("pm"), "gain_db": e.get("gain_db"),
                            "outcome": ("success" if e.get("postsizing")
                                        else "observation")})
            if len(records) >= k:
                break
    evidence = "; ".join(r["line"] for r in records)
    # evidence goes BEFORE the proposal instruction
    enriched = (prompt.replace("### BLOCKS", f"### KNOWN {evidence}\n### BLOCKS")
                if evidence else prompt)
    return {"records": records, "retrieval_ids": [r["retrieval_id"]
                                                  for r in records],
            "prompt": enriched}


# ------------------- Stages 3 + 4: propose 5, validate, dedupe ---------------
def propose_and_validate(model, tok, prompt: str, target_k: int = TARGET_K,
                         max_attempts: int = 20,
                         conditioning: str = "exclusion",
                         seed0: int = 0) -> dict:
    """LLM generates; the validator canonicalises and deduplicates.

    Every returned candidate carries source="llm". Nothing is enumerated, and
    a short set is reported as insufficient_model_diversity rather than
    quietly topped up -- silently filling would make the architecture claim
    false while the numbers looked fine.
    """
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


# --------------------- Stage 5: PUCT ranks 5, selects 2 ----------------------
def puct_select_two(candidates: list, spec: dict, ctx_id: str,
                    search: str = "one_root", seed: int = 0,
                    value_ckpt=None) -> dict:
    """Rank the LLM's candidates by PUCT visit counts; return the top 2.

    PUCT ranks what it is given. It does not synthesise or substitute
    candidates -- the pool is exactly the validated LLM set.
    """
    # PUCT needs at least SELECT_K distinct candidates to make a choice. It
    # used to demand exactly TARGET_K, which aborted every spec where the
    # proposer returned 3 or 4 -- roughly two thirds of them at the measured
    # 4.17 mean, so the pipeline mostly produced crashes. The count is
    # recorded in the trace instead of being asserted away.
    if len(candidates) < SELECT_K:
        raise ArchitectureViolation(
            f"PUCT needs >= {SELECT_K} LLM candidates to select from, got "
            f"{len(candidates)}")
    if any(c["source"] != "llm" for c in candidates):
        raise ArchitectureViolation(
            "PUCT candidate without LLM provenance -- enumeration must never "
            "enter the production pool")
    from agentic_raptor.topology_rl import stage3e1 as s1
    from run_puct_ablation import _realise, topology_priors
    priors = topology_priors(spec)

    if search == "none":
        # ABLATION CONTROL: no search at all. Order by the structural prior
        # alone, exactly what the pipeline would do if PUCT contributed
        # nothing, so the search arm measures the search and not the prior.
        cand_visits = {c["llm_proposal_id"]: 0 for c in candidates}
        ranked = sorted(candidates,
                        key=lambda c: (-(priors.get(c["canonical_family"])
                                         or 0.0), c["canonical_graph_hash"]))
        for rank, c in enumerate(ranked):
            c["policy_prior"] = priors.get(c["canonical_family"])
            c["value_prediction"] = priors.get(c["canonical_family"]) or 0.0
            c["visit_count"] = 0
            c["rank"] = rank
            c["selected_top2"] = rank < SELECT_K
        selected = ranked[:SELECT_K]
        if len(selected) != SELECT_K:
            raise ArchitectureViolation(
                f"must select {SELECT_K}, selected {len(selected)}")
        return {"ranked": ranked, "selected": selected, "visits": {},
                "candidate_visits": cand_visits, "root_action_ids": [],
                "distinct_visit_counts": 0, "root_state": None,
                "candidate_manifests": {}, "search": "none"}


    # ONE root whose actions are the LLM's OWN graphs.
    #
    # The previous version ran a SEPARATE search per candidate and scored each
    # by the visit share its structure CLASS attracted. Two consequences, both
    # measured: every candidate came back with an identical visit count (256
    # of 256 in the first canonical trace), so the search contributed nothing
    # to the ordering; and two candidates in the same family were literally
    # indistinguishable, leaving the choice to a hash string comparison.
    #
    # Here the candidates are the actions, so their visit counts differ and
    # mean what PUCT visit counts are supposed to mean.
    # the search/value net operates on CircuitGraph, the sizer on
    # DeviceCircuitGraph. Convert with the SAME function the corpus families
    # use, or the structural hashes the search ranks by are not comparable.
    from agentic_raptor.topology_rl.value_refresh import \
        device_graph_to_circuit_graph
    graphs = {c["llm_proposal_id"]:
              device_graph_to_circuit_graph(_realise(c["obj"]),
                                            c["llm_proposal_id"])
              for c in candidates}
    by_id = {c["llm_proposal_id"]: c for c in candidates}

    class _Entry:
        def __init__(self, g):
            self.graph = g

    class _LLMRegistry:
        """Resolves proposal ids to the LLM's own graphs, nothing else.

        A registry over corpus families would put structures the model never
        proposed into the action set -- the enumeration this pipeline forbids.
        """
        def get_topology(self, tid):
            if tid in graphs:
                return _Entry(graphs[tid])
            return _Entry(next(iter(graphs.values())))   # neutral root

        def list_topologies(self):
            return sorted(graphs)

    reg = _LLMRegistry()
    pool = sorted(graphs)
    nets = s1.build_policy_value(0)
    # generation N+1 must load the checkpoint generation N had ACCEPTED
    ck = Path(value_ckpt) if value_ckpt else (
        ROOT / "artifacts/stage3e1/policy_value_ep0.pt")
    if ck.is_file():
        try:
            s1.load_checkpoint(nets, ck)
        except Exception:
            pass
    rg = reg.get_topology("proposal_root").graph
    st = s1.TopologySearchState(
        topology_id="proposal_root", graph_hash=rg.structural_hash(),
        # lineage starts EMPTY: it seeds the cycle check, and the neutral root
        # borrows a candidate's graph, so a non-empty lineage would reject
        # that candidate as "already in ancestry" and silently drop it
        lineage=[],
        spec={"target_gain_db": spec["gain_target_db"],
              "target_gbw_hz": spec.get("ugbw_target_hz") or 1e4,
              "minimum_phase_margin_deg": spec["phase_margin_target_deg"],
              "load_capacitance_f": spec["load_capacitance_pf"] * 1e-12,
              "supply_voltage": 1.8},
        rag_context_ids=[ctx_id], available_blocks=[], legal_action_ids=[],
        edit_history=[], validation_status="validated",
        structural_features={"n_nodes": float(len(rg.nodes))},
        previous_evidence_ref=None, remaining_search_budget=6,
        remaining_spice_budget=0, depth=0)
    # max_children must admit EVERY candidate, or the default cap of 4 drops
    # the fifth proposal without saying so
    cfg = s1.SearchConfig(num_simulations=256, leaf_mode="value_only",
                          training_mode=False, max_depth=3, seed=seed,
                          max_children=max(4, len(candidates) + 2))
    root = s1.TopologyMCTS(nets, reg, pool, cfg).run(st)
    visits_all = {c.action.action_id: c.N for c in root.children}
    # The training-time reconstruction must rebuild THIS state, not an
    # approximation of it. Persist the real root (its structural hash and
    # node/edge counts come from the actual graph) plus a manifest per
    # candidate, so a replayed example can be verified against what the
    # search actually saw instead of being silently rebuilt from constants.
    from dataclasses import asdict as _asdict
    root_state = _asdict(st)
    manifests = {}
    for c in candidates:
        pid = c["llm_proposal_id"]
        g = graphs[pid]
        manifests[pid] = {
            "canonical_graph_hash": c["canonical_graph_hash"],
            "canonical_family": c["canonical_family"],
            "structural_hash": g.structural_hash(),
            "n_nodes": float(len(g.nodes)), "n_edges": float(len(g.edges)),
            "policy_prior": priors.get(c["canonical_family"]),
            "visits": visits_all.get(f"a_sel_{pid}", 0)}
    # only SELECT actions name a candidate; a_keep/a_term/a_comp do not
    cand_visits = {pid: visits_all.get(f"a_sel_{pid}", 0) for pid in graphs}
    total = sum(cand_visits.values()) or 1
    scores = {pid: n / total for pid, n in cand_visits.items()}
    # Decide by VISIT COUNT -- the standard PUCT/AlphaZero rule. The search
    # already applies the policy network's priors internally
    # (use_policy_prior=True), so multiplying by the hand-set family prior
    # afterwards applies a prior twice and lets a constant overrule the
    # search: measured on four candidates, a proposal with 18 visits
    # outranked one with 27 purely on family weight. The family prior is
    # retained only to break exact visit ties, ahead of the hash.
    ranked = sorted(candidates,
                    key=lambda c: (-cand_visits[c["llm_proposal_id"]],
                                   -(priors.get(c["canonical_family"]) or 0.0),
                                   c["canonical_graph_hash"]))
    for rank, c in enumerate(ranked):
        c["policy_prior"] = priors.get(c["canonical_family"])
        c["value_prediction"] = scores[c["llm_proposal_id"]]
        c["visit_count"] = cand_visits[c["llm_proposal_id"]]
        c["rank"] = rank
        c["selected_top2"] = rank < SELECT_K
    selected = ranked[:SELECT_K]
    if len(selected) != SELECT_K:
        raise ArchitectureViolation(f"PUCT must select {SELECT_K}, selected "
                                    f"{len(selected)}")
    if selected[0]["canonical_graph_hash"] == \
            selected[1]["canonical_graph_hash"]:
        raise ArchitectureViolation("PUCT selected the same graph twice")
    return {"ranked": ranked, "selected": selected, "visits": visits_all,
            "candidate_visits": cand_visits,
            "root_action_ids": sorted(visits_all),
            "distinct_visit_counts": len(set(cand_visits.values())),
            "root_state": root_state, "candidate_manifests": manifests,
            "search": "one_root"}


# ------------- Stages 6 + 7: size both independently, predict both -----------
def size_and_predict(selected: list, spec: dict, exe, budget: int) -> list:
    """Independent sizing per branch, then a genuine surrogate estimate.

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
    """
    from agentic_raptor.mb_sac.spec_sizing import apply_knobs, sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from run_puct_ablation import _realise
    out = []
    for label, c in zip("AB", selected):
        costs = new_costs()                       # isolated per branch
        g = _realise(c["obj"])
        tid = f"v2_{label}_{c['llm_proposal_id']}"
        sz = sac_size(tid, g, spec, exe, OUT / "sizing", costs,
                      budget=budget, seed=17, persist=False)
        o, best = sz["outcome"], sz["best"]
        knobs = best.get("knobs") or {}
        manifest = sz.get("schema_version")
        # every ngspice call the SIZING loop made; a final verification that
        # reuses one of these ids is rejected by the pair provenance check
        sizing_ids = [f"{tid}:sz17_{r.get('step')}" for r in sz["results"]]
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
            sizing_spice_calls=sz["spice_calls"],
            sizing_spice_call_ids=sizing_ids,
            puct_rank=c["rank"], puct_visits=c.get("visit_count"),
            final_netlist_hash=netlist_hash_of(g))
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
    ap.add_argument("--calibrate", action="store_true",
                    help="verify BOTH designs to build a trusted ranker pair")
    ap.add_argument("--arm", default="L8")
    ap.add_argument("--adapter", default=None,
                    help="explicit proposer checkpoint, overriding --arm. "
                         "resolve_arms() only knows campaign checkpoints, so "
                         "an adapter trained outside a campaign (e.g. the "
                         "diverse-corpus retrain) is unreachable by arm name "
                         "and --arm would silently load the OLD checkpoint")
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
    trace = run_pipeline(model, tok, adapter, split=args.split,
                         spec_index=args.spec_index, budget=args.budget,
                         calibrate=args.calibrate)
    print(json.dumps({k: trace[k] for k in
                      ("stage3_propose", "stage5_puct", "stage8_ranker",
                       "stage9_verification", "stage11_feedback")
                      if k in trace}, indent=1, default=str))


def run_pipeline(model, tok, adapter, *, split="heldout", spec_index=0,
                 budget=32, calibrate=False, conditioning="exclusion",
                 search="one_root", ranker_mode="dpo", seed=0,
                 out_prefix="TRACE", ranker_ckpt=None, value_ckpt=None,
                 rag_memory=None, harvest=False):
    """One full pipeline execution. Returns the trace.

    Separated from main() so an ablation can hold ONE loaded model across many
    runs -- reloading a 4B model per run costs more than the pipeline itself.

    conditioning / search / ranker_mode are the ablation arms; the defaults
    are the production configuration.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from agentic_raptor.electrical import discover_ngspice

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
    trace_name = (f"{out_prefix}_{split}_{spec_index % len(pool):03d}"
                  f"_{spec['spec_id']}.json")
    trace = {"stage1_spec": {"spec_id": spec["spec_id"], "split": split,
                             "spec_index": spec_index % len(pool),
                             "spec_hash": spec_hash, "spec": spec}}

    # ---- Stage 2: RAG ------------------------------------------------------
    rag = retrieve(spec, rec["prompt"], memory_path=rag_memory)
    trace["stage2_rag"] = {"retrieval_ids": rag["retrieval_ids"],
                           "records": len(rag["records"])}

    # ---- Stages 3 + 4: LLM generates 5, validator dedupes ------------------
    trace["proposer_checkpoint"] = str(adapter)
    trace["proposer_checkpoint_source"] = "explicit adapter"
    trace["arms"] = {"conditioning": conditioning, "search": search,
                     "ranker": ranker_mode, "seed": seed}
    prop = propose_and_validate(model, tok, rag["prompt"],
                                conditioning=conditioning, seed0=seed)
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

    # ---- Stage 5: PUCT ranks 5, selects 2 ---------------------------------
    sel = puct_select_two(prop["candidates"], spec, spec["spec_id"],
                          search=search, seed=seed, value_ckpt=value_ckpt)
    trace["stage5_puct"] = {
        "input_count": len(prop["candidates"]),
        "selected_count": len(sel["selected"]),
        # ONE search whose actions are the LLM's graphs. root_action_count is
        # the number of candidate-selecting actions at that root, and
        # distinct_visit_counts records whether the search actually separated
        # them -- if every candidate came back with the same N, the visit
        # counts did no work and the ordering came from the prior alone.
        "root_action_count": len([a for a in sel["root_action_ids"]
                                  if a.startswith("a_sel_")]),
        "single_root": True,
        "search_topology": "single_root_over_llm_candidates",
        "candidate_visits": sel["candidate_visits"],
        "distinct_visit_counts": sel["distinct_visit_counts"],
        "root_action_ids": sel["root_action_ids"],
        "ranking": [{k: c[k] for k in
                     ("llm_proposal_id", "canonical_graph_hash",
                      "canonical_family", "policy_prior",
                      "value_prediction", "visit_count", "rank",
                      "selected_top2")} for c in sel["ranked"]]}

    # ---- Stages 6 + 7: size both, predict both ----------------------------
    exe = discover_ngspice()
    branches = size_and_predict(sel["selected"], spec, exe, budget)
    (da, pa, sza, oa, ga), (db, pb, szb, ob, gb) = branches
    trace["stage6_sizing"] = {
        lbl: {"topology_hash": d.topology_hash,
              "sizing_manifest_hash": d.sizing_manifest_hash,
              "spice_calls": d.sizing_spice_calls,
              "sizing_vector": d.sizing_vector}
        for lbl, d in (("A", da), ("B", db))}
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

    def verify(label: str, mode: str):
        d, graph = designs[label]
        call_id = f"final:{spec['spec_id']}:{label}:{int(time.time()*1000)}"
        meas = measure(d.sac_trajectory_id, graph, exe, OUT / "verify",
                       f"ver_{label}", new_costs())
        oc = postsizing_outcome(meas, spec)
        return oc, outcome_from_sizing(
            oc, call_id=call_id, topology_hash=d.canonical_graph_hash,
            sizing_manifest_hash=d.sizing_manifest_hash,
            netlist_hash=netlist_hash_of(graph), mode=mode, best=meas,
            spec_id=spec["spec_id"], spec_hash=spec_hash)

    oc_sel, auth_sel = verify(sel_lbl, "final_verification")
    verified = {sel_lbl: auth_sel}
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
    trace["stage9_verification"] = {
        "authoritative_engine": "ngspice",
        "verified_designs": sorted(verified),
        "second_design_reason": reason,
        "selected_exact_pass": bool(oc_sel.get("exact_spec_pass")),
        "selected_failure_reason": oc_sel.get("exact_failure_reason"),
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

    # ---- Stage 11: ranker training data (trusted only if both measured) ----
    # predictions are recorded too: the ranker trainer needs the FEATURES the
    # ranker saw, paired with the MEASURED label
    pair = record_pair(spec, da, db,
                       verified.get("A"), verified.get("B"),
                       pred_a=pa, pred_b=pb,
                       ranker_choice=sel_lbl)
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

    if harvest:
        # Everything the learning streams need, in REPLAYABLE form.
        # The PUCT trainer rebuilds states from action ids like "a_sel_p00",
        # which mean nothing outside this run -- so the candidate OBJECTS are
        # stored, not just their hashes. Without them the policy/value trainer
        # silently sees zero usable examples and still reports success.
        trace["_harvest"] = {
            "spec": spec, "spec_hash": spec_hash,
            "split": split, "spec_index": spec_index % len(pool),
            "seed": seed, "budget": budget,
            "candidates": [{k: c[k] for k in
                            ("llm_proposal_id", "canonical_graph_hash",
                             "canonical_family", "obj", "rank",
                             "visit_count", "selected_top2")}
                           for c in sel["ranked"]],
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
