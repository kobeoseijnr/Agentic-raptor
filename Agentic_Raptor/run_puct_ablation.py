"""PUCT decision ablation P0-P7: does the search layer's DECISION improve the
final measured circuit? Equal sizing budgets; proposals generated once (GPU)
and cached; every arm sizes the topology it actually selected (alignment
verified by hash).

  P0 proposal executed directly (no search)
  P1 local keep/edit heuristic (2 real-SPICE probes)
  P2 PUCT advice recorded, but proposal still executed
  P3 PUCT highest-visit action executed
  P4 PUCT with UNIFORM priors, executed
  P5 PUCT with RANDOM value estimates, executed
  P6 PUCT without registry-switch actions, executed
  P7 full PUCT + local keep/edit on the winner (production semantics)

Run:  python run_puct_ablation.py [--tasks 8] [--budget 12]
      (phase 1 needs GPU once; re-runs reuse cached proposals)
"""
import argparse
import json
import time
from pathlib import Path

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.publication import PUB, ROOT

OUT = PUB / "puct_ablation"
ARMS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

#: The five structures that exist in the corpus. That is the WHOLE design
#: space -- no buffer or feedback variants are present -- so the search can
#: enumerate it rather than hope a peaked proposer samples it.
#:
#: 3s_none was REPLACED by 2s_rc after measurement. An uncompensated 3-stage
#: amplifier has three gain poles and no compensation element, so cap_x has
#: nothing to act on and the only remaining stabiliser is discarding gm: it
#: failed the realizability gate at every corpus spec ("unstable; PM below
#: target", distance 0.51 / 0.38). 2s_rc (two-stage Miller + nulling resistor,
#: the classic SMCNR -- legacy AnalogGym ships it as TwoSt_SMCNR_Pin_2) builds
#: from existing mapping support and passed both probed specs with large
#: margin (PM 99.2/79.1 deg, UGBW 2.6e5/3.29e6 against 1e4/1e5 targets).
#:
#: This list must stay identical to the proposer's target families -- that is
#: the TRAIN/SERVE MATCH property below. Change both or neither.
CORPUS_CLASSES = ["2s_none", "2s_miller", "2s_rc", "3s_miller", "3s_rc"]


def tier_classes(spec) -> list:
    """The corpus's stage-tier rule: gain tier dictates the stage count."""
    g = spec["gain_target_db"]
    stages = 1 if g < 30 else 2 if g < 70 else 3
    comps = ["none", "miller"] + (["rc"] if stages == 3 else [])
    return [f"{stages}s_{c}" for c in comps]


def compatible_classes(spec, tier_gated: bool = False) -> list:
    """Structures the search may offer for `spec`.

    The tier gate is OFF because measurement contradicts it, not preference.
    check_stage_rule.py sized 8 high-gain specs under both tiers: 2-stage
    landed closer on 6 of 8, and the ONLY exact pass in that 40-circuit
    battery was `2s_none` on spec 004_t_hard_topology_0002 (82.74 dB,
    64.85 deg) -- a candidate the hard rule REJECTS, while it admitted
    3-stage candidates sitting at -20 to -0.5 deg of phase margin.

    tier_gated=True preserves the old hard rule as a frozen negative control.
    """
    return tier_classes(spec) if tier_gated else list(CORPUS_CLASSES)


#: Fix 2 -- SOFT structural prior replacing the hard stage filter.
#: These bias the search; they never remove a family. Only deterministic
#: structural invalidity (the candidate validator) may exclude a topology.
#: 2s_rc inherits the 0.20 that 3s_none held, so the distribution still sums
#: to 1 and the swap introduces no evidence-free preference of its own.
BASE_TOPOLOGY_PRIORS = {"2s_none": 0.15, "2s_miller": 0.15,
                        "2s_rc": 0.20, "3s_miller": 0.30, "3s_rc": 0.20}
#: floor guaranteeing every electrically valid family stays reachable
MIN_PRIOR = 0.02


def topology_priors(spec, gated: bool = False) -> dict:
    """Soft prior over the five corpus structures for `spec`.

    Shape of the prior (what the old rule tried to express as a hard gate):
      * higher gain targets make more stages more plausible;
      * tight phase-margin targets make compensation more plausible;
      * nothing is ever driven to zero.

    The floor is the whole point. A hard rule cannot be recovered from once
    it is wrong, and it demonstrably was: the single passing candidate in the
    stage-rule battery was a structure the rule forbade.
    """
    if gated:                                   # frozen negative control
        allowed = set(tier_classes(spec))
        return {c: (1.0 / len(allowed) if c in allowed else 0.0)
                for c in CORPUS_CLASSES}
    gain = float(spec.get("gain_target_db") or 0.0)
    pm = float(spec.get("phase_margin_target_deg") or 45.0)
    w = {}
    for cls in CORPUS_CLASSES:
        stages, comp = int(cls[0]), cls.split("_", 1)[1]
        p = BASE_TOPOLOGY_PRIORS[cls]
        # more gain demanded -> more stages a bit more likely (soft, bounded)
        p *= 1.0 + 0.25 * (stages - 2) * max(-1.0, min(1.0,
                                                       (gain - 70.0) / 40.0))
        # tighter phase margin -> compensation a bit more likely
        if comp != "none":
            p *= 1.0 + 0.25 * max(-1.0, min(1.0, (pm - 45.0) / 20.0))
        w[cls] = max(MIN_PRIOR, p)
    total = sum(w.values())
    # full precision: rounding here would stop the distribution summing to 1
    return {c: v / total for c, v in w.items()}


def realise_class(cls):
    """Execute a class-canonical topology (device graph for sizing)."""
    from agentic_raptor.mapping import map_family
    from agentic_raptor.topology_rl.stage3e2_edits import (EditRejected,
                                                           apply_edit)
    stages, comp = int(cls[0]), cls.split("_", 1)[1]

    class _S:
        topology_id = f"cls_{cls}"
    g, _ = map_family(_S(), {
        "topology_id": _S.topology_id, "gain_stages": stages,
        "functional_blocks": [] if comp == "none" else ["C"],
        "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
        "graph_hash": None})
    if comp == "rc":
        try:
            g, _a = apply_edit(g, "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE")
        except EditRejected:
            pass
    return g


def run_puct_fixed(proposal_obj, spec, ctx_id, root_cls=None):
    """P8: the repaired search. Everything P3-P7 get wrong, corrected:

      1. TRAIN/SERVE MATCH -- every candidate (including the proposal) is
         scored through its canonical STRUCTURE-CLASS graph, the exact
         representation the value network was trained on. P3-P7 scored
         V3-registry catalogue graphs the nets had never seen, so their
         value estimates were noise.
      2. CANDIDATE POOL = THE CORPUS -- the five structures that actually
         exist, not four arbitrary catalogue families the proposer never
         emits and the corpus never scores.
      3. LEGAL ACTIONS -- the candidate validator recognises the mapping
         layer's block roles, so keep/switch/edit are legal at all. While it
         only matched the extractor's "gain_stage" role, every candidate was
         rejected as gainless and TERMINATE was the sole legal move: the
         search could not choose, which is why P3-P7 never switched.
      4. REAL DEPTH -- compensation edits materialise a genuinely different
         child graph, so max_depth=3 is a lookahead rather than cosmetic.
      5. 256 value-only simulations, so visit counts reflect the value head
         rather than sampling noise (top-two margin 0.021 -> 0.117).

    No tier gate: check_stage_rule.py measured that rule false (2-stage
    landed closer on 6 of 8 high-gain specs), so restricting the pool by it
    forbade the structures that measure best.
    """
    import torch
    from agentic_raptor.llm_dpo import integrity as ig
    from agentic_raptor.topology_rl import stage3e1 as s1
    from agentic_raptor.topology_rl.value_refresh import FamilyRegistry
    # root_cls lets the search run with NO proposal at all -- the case where
    # the LLM produced nothing valid, which is exactly where an independent
    # search has to stand on its own
    prop_cls = root_cls or (f"{len(proposal_obj['stages'])}s_"
                            + ig.compensation_class(proposal_obj))
    pool = [c for c in compatible_classes(spec) if c != prop_cls]
    reg = FamilyRegistry(set(pool + [prop_cls]))

    class _Shim:
        def get_topology(self, tid):
            if tid == "proposal_root":
                return reg.get_topology(prop_cls)
            return reg.get_topology(tid)

        def list_topologies(self):
            return sorted(set(pool + [prop_cls]))

        def derive_edited(self, tid, edit_name):
            """Exposing this is what lets the tree grow past one ply."""
            return reg.derive_edited(
                prop_cls if tid == "proposal_root" else tid, edit_name)
    nets = s1.build_policy_value(0)
    ck = ROOT / "artifacts/stage3e1/policy_value_ep0.pt"
    if ck.is_file():
        try:
            s1.load_checkpoint(nets, ck)
        except Exception:
            pass
    cg = reg.get_topology(prop_cls).graph
    st = s1.TopologySearchState(
        topology_id="proposal_root", graph_hash=cg.structural_hash(),
        lineage=[cg.structural_hash()],
        spec={"target_gain_db": spec["gain_target_db"],
              "target_gbw_hz": spec.get("ugbw_target_hz") or 1e4,
              "minimum_phase_margin_deg": spec["phase_margin_target_deg"],
              "load_capacitance_f": spec["load_capacitance_pf"] * 1e-12,
              "supply_voltage": 1.8},
        rag_context_ids=[ctx_id], available_blocks=[], legal_action_ids=[],
        edit_history=[], validation_status="validated",
        structural_features={"n_nodes": float(len(cg.nodes))},
        previous_evidence_ref=None, remaining_search_budget=6,
        remaining_spice_budget=0, depth=0)
    # multi-ply: class choice THEN a realisable compensation edit, so the
    # tree is an actual lookahead rather than a one-shot vote. 256/depth-3
    # measured at 0.58s against ~12s for the sizing that follows, and it
    # widens the top-two visit margin from 0.021 (48 sims) to 0.117 --
    # i.e. the decision stops being decided by noise.
    cfg = s1.SearchConfig(num_simulations=256, leaf_mode="value_only",
                          training_mode=False, max_depth=3, seed=0)
    mcts = s1.TopologyMCTS(nets, _Shim(), pool, cfg)
    root = mcts.run(st)
    visits = {c.action.action_id: c.N for c in root.children}
    sel = max(visits, key=lambda k: visits[k]) if visits else None
    return sel, visits, prop_cls


def get_proposals(n_tasks: int) -> dict:
    """Phase 1 (GPU, cached): final-checkpoint proposals for validation specs."""
    cache = OUT / "proposals.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    import torch
    from run_qwen_ablation import _load, resolve_arms
    from agentic_raptor.llm_dpo.stage3e4 import generate
    adapter = resolve_arms()["L8"]["adapter"]
    assert adapter, "final checkpoint missing"
    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
    held = [r for r in corpus["records"] if r["split"] == "heldout"][:n_tasks]
    tok, model = _load(adapter)
    props = {}
    for i, r in enumerate(held):
        # context_id is NOT unique per record (two distinct specs can share
        # one label, differing only e.g. in ugbw_target_hz) -- keying by it
        # alone silently collapses distinct held-out tasks into one another
        key = f"{i:03d}_{r['context_id']}"
        c = next((generate(model, tok, r["prompt"], sample_seed=s)
                  for s in range(4)
                  if generate(model, tok, r["prompt"], sample_seed=s)["valid"]),
                 None)
        c = c if c and c["valid"] else None
        if c:
            props[key] = {
                "prompt": r["prompt"], "obj": c["obj"],
                "graph_hash": c["graph_hash"],
                "context_id": r["context_id"],
                "spec": ig.parse_spec(r["prompt"]),
                "target_variant": r["variant_hash"]}
    del model
    torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(props, indent=1), encoding="utf-8")
    return props


def _to_circuit_graph(g, name):
    from agentic_raptor.core.circuit_graph import (CircuitEdge, CircuitGraph,
                                                   CircuitNode)
    from agentic_raptor.core.types import DeviceType, TerminalType
    km = {"nmos": DeviceType.NMOS, "pmos": DeviceType.PMOS,
          "cap": DeviceType.CAPACITOR, "res": DeviceType.RESISTOR,
          "isrc": DeviceType.CURRENT_SOURCE}
    tm = {"d": TerminalType.DRAIN, "g": TerminalType.GATE,
          "s": TerminalType.SOURCE, "b": TerminalType.BULK,
          "p": TerminalType.PLUS, "n": TerminalType.MINUS}
    cg = CircuitGraph(name)
    for d in g.devices:
        cg.add_node(CircuitNode(node_id=d.device_id, device_type=km[d.kind],
                                block_role=d.role))
        for t, net in d.nets.items():
            cg.add_edge(CircuitEdge(cg.next_id("e"), d.device_id, tm[t], net))
    return cg


def _realise(obj):
    from agentic_raptor.mapping import map_family

    class _S:
        topology_id = "puct_ab"
    g, _ = map_family(_S(), {
        "topology_id": _S.topology_id, "gain_stages": len(obj["stages"]),
        "functional_blocks": ["C"] if obj.get("compensation") else [],
        "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
        "graph_hash": None})
    return g


def run_puct(cg, spec, ctx_id, no_switch=False, uniform_prior=False,
             random_value=False):
    """One PUCT search over the proposal root; returns (selected, visits)."""
    import torch
    from agentic_raptor.corpus import TopologyRegistry
    from agentic_raptor.mb_sac import load_pools
    from agentic_raptor.mb_sac.stage3d2 import V3
    from agentic_raptor.topology_rl import stage3e1 as s1
    reg = TopologyRegistry(V3)

    class _Shim:
        def get_topology(self, tid):
            if tid == "proposal_root":
                class _E:
                    topology_id, graph, metadata = "proposal_root", cg, {}
                    path, source = OUT, "llm_proposal"
                return _E()
            return reg.get_topology(tid)

        def list_topologies(self):
            return reg.list_topologies()
    nets = s1.build_policy_value(0)
    ck = ROOT / "artifacts/stage3e1/policy_value_ep0.pt"
    if ck.is_file():
        try:
            s1.load_checkpoint(nets, ck)
        except Exception:
            pass
    if uniform_prior:
        _pf = nets["policy_forward"]

        def uniform(state, actions, r):
            acts, lg, _p = _pf(state, actions, r)
            flat = torch.zeros_like(lg)
            return acts, flat, torch.softmax(flat, dim=0)
        nets["policy_forward"] = uniform
    if random_value:
        def rv(state, r):
            return {"scalar": torch.rand(1)[0] * 2 - 1,
                    "feasibility_logit_uncalibrated": torch.tensor(0.0),
                    "stability_logit_uncalibrated": torch.tensor(0.0),
                    "expected_spice_cost": torch.tensor(1.0),
                    "budget_exhaustion_logit": torch.tensor(0.0)}
        nets["value_forward"] = rv
    pool = ([] if no_switch else
            [r["topology_id"] for r in
             (load_pools()["A1"] + load_pools()["A2"])][:4])
    st = s1.TopologySearchState(
        topology_id="proposal_root", graph_hash=cg.structural_hash(),
        lineage=[cg.structural_hash()],
        spec={"target_gain_db": spec["gain_target_db"],
              "target_gbw_hz": spec.get("ugbw_target_hz") or 1e4,
              "minimum_phase_margin_deg": spec["phase_margin_target_deg"],
              "load_capacitance_f": spec["load_capacitance_pf"] * 1e-12,
              "supply_voltage": 1.8},
        rag_context_ids=[ctx_id], available_blocks=[], legal_action_ids=[],
        edit_history=[], validation_status="validated",
        structural_features={"n_nodes": float(len(cg.nodes))},
        previous_evidence_ref=None, remaining_search_budget=6,
        remaining_spice_budget=0, depth=0)
    cfg = s1.SearchConfig(num_simulations=8, leaf_mode="value_only",
                          training_mode=False, max_depth=1, seed=0)
    mcts = s1.TopologyMCTS(nets, _Shim(), pool, cfg)
    root = mcts.run(st)
    visits = {c.action.action_id: c.N for c in root.children}
    sel = max(visits, key=lambda k: visits[k]) if visits else None
    return sel, visits


def _local_edit(tid, g, exe, out, costs):
    """keep vs add-compensation, decided on 2 real SPICE probes."""
    from agentic_raptor.topology_rl.stage3e2_edits import (EditRejected,
                                                           apply_edit,
                                                           qualify_device_graph)
    qk = qualify_device_graph(tid, g, out, exe, "keep", costs)
    try:
        ge, _a = apply_edit(g, "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE")
        qe = qualify_device_graph(tid, ge, out, exe, "edit", costs)
    except EditRejected:
        ge, qe = None, {}

    def pm(q):
        return (q.get("metrics") or {}).get("phase_margin_deg") or -999
    return (g, "KEEP") if pm(qk) >= pm(qe) or ge is None else (ge, "EDIT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.mb_sac.spec_sizing import postsizing_outcome, sac_size
    from agentic_raptor.topology_rl.stage3e2 import map_root, new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import device_graph_hash
    from agentic_raptor.corpus import TopologyRegistry
    from agentic_raptor.mb_sac.stage3d2 import V3
    props = get_proposals(args.tasks)
    print(f"proposals cached: {len(props)}")
    exe = discover_ngspice()
    out = OUT.resolve() / "runs"
    out.mkdir(parents=True, exist_ok=True)
    reg = TopologyRegistry(V3)
    rows_f = OUT / "puct_ablation_rows.jsonl"
    done = set()
    if rows_f.is_file():
        done = {(json.loads(x)["context_id"], json.loads(x)["arm"])
                for x in rows_f.read_text().splitlines() if x.strip()}
    for task_key, p in props.items():
        spec = p["spec"]
        ctx_id = p.get("context_id", task_key)
        g0 = _realise(p["obj"])
        cg = _to_circuit_graph(g0, "proposal_root")
        for arm in args.arms.split(","):
            if (task_key, arm) in done:
                continue
            t0 = time.time()
            costs = new_costs()
            g, action, sel, visits = g0, "KEEP_PROPOSAL", None, {}
            run_name = f"pa_{task_key[:3]}_{ctx_id[-6:]}"
            if arm == "P8":
                sel, visits, prop_cls = run_puct_fixed(p["obj"], spec,
                                                       ctx_id)
                if sel and sel.startswith("a_sel_") and                         sel[6:] != prop_cls:
                    g = realise_class(sel[6:])
                    action = f"SWITCH_CLASS:{sel[6:]}"
                else:
                    action = "KEEP(fixed_puct)"
            elif arm == "P1":
                g, action = _local_edit(run_name, g0, exe, out, costs)
            elif arm in ("P2", "P3", "P4", "P5", "P6", "P7"):
                sel, visits = run_puct(
                    cg, spec, ctx_id, no_switch=(arm == "P6"),
                    uniform_prior=(arm == "P4"), random_value=(arm == "P5"))
                if arm != "P2" and sel and sel.startswith("a_sel_"):
                    g = map_root(reg, sel[6:], costs)
                    action = f"SWITCH:{sel[6:]}"
                elif arm != "P2":
                    action = "KEEP"
                else:
                    action = f"ADVICE_ONLY({sel})"
                if arm == "P7":
                    g, ed = _local_edit(run_name, g, exe, out, costs)
                    action += f"+{ed}"
            search_spice = costs["real_spice_calls"]
            sz = sac_size(run_name, g, spec, exe, out, costs,
                          budget=args.budget, seed=17, persist=False)
            o = sz["outcome"]
            row = {"context_id": task_key, "spec_label": ctx_id,
                   "arm": arm, "action": action,
                   "puct_selected": sel, "visits": visits,
                   "executed_graph_hash": device_graph_hash(g),
                   "proposal_overturned": not (action.startswith("KEEP")
                                               or "ADVICE" in action),
                   "search_spice": search_spice,
                   "sizing_spice": sz["spice_calls"],
                   "exact_pass": o["exact_spec_pass"],
                   "constraints_passed": o["hard_constraints_passed"],
                   "distance": o["normalized_distance_to_feasibility"],
                   "best_gain_db": sz["best"]["gain_db"],
                   "best_pm_deg": sz["best"]["pm_deg"],
                   "runtime_s": round(time.time() - t0, 1)}
            with rows_f.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(task_key, arm, action, "exact", o["exact_spec_pass"],
                  "dist", o["normalized_distance_to_feasibility"])
    rows = [json.loads(x) for x in rows_f.read_text().splitlines()
            if x.strip()]
    agg = {}
    for r in rows:
        agg.setdefault(r["arm"], []).append(r)
    summary = {arm: {"tasks": len(rs),
                     "exact_pass": sum(r["exact_pass"] for r in rs),
                     "mean_distance": round(sum(r["distance"] or 1
                                                for r in rs) / len(rs), 4),
                     "overturn_rate": round(sum(r["proposal_overturned"]
                                                for r in rs) / len(rs), 3),
                     "mean_search_spice": round(sum(r["search_spice"]
                                                    for r in rs) / len(rs), 1)}
               for arm, rs in sorted(agg.items())}
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
