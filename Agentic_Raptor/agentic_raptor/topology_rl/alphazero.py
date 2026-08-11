"""TRUE_ALPHAZERO topology search -- replaces the retired root-level PUCT
selector (run_raptor_v2.puct_select_two -> agentic_raptor.topology_rl.
stage3e1's one-root wiring, which could only ever KEEP/TERMINATE/SELECT
among complete LLM candidates, never edit one).

This module does NOT reimplement MCTS. It reuses agentic_raptor.topology_rl.
stage3e1's TopologyMCTS engine (PUCT selection, progressive expansion, leaf
evaluation, single-agent backup, visit-count final selection) UNCHANGED,
via the generate_actions_fn/apply_action_fn injection points added to
TopologyMCTS.__init__ for exactly this purpose. What's new here is the
missing piece: a registry whose derive_edited() is real, and an
action-generation/application pair that offers the FULL real structural
edit catalog (agentic_raptor.topology_rl.stage3e2_edits.EDIT_TEMPLATES --
8 types share an exact name with Stage3E1ActionType and are used here;
see AZ_EDIT_ACTION_TYPES) instead of stage3e1.generate_actions()'s single
hardcoded ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE action.

Architecture decision (Section 15, Option A: virtual super-root):
    root ("proposal_root", a neutral node whose graph borrows the first
    LLM seed) -> SELECT_EXISTING_TOPOLOGY commits to one LLM seed (or
    KEEP stays on the neutral root's own seed) -> from THAT point on,
    real structural edits (ADD_VERIFIED_STAGE, ADD_EXISTING_SUPPORTED_
    COMPENSATION_STRUCTURE, ...) and SELECT_EXISTING_TOPOLOGY (jump to a
    sibling seed) are BOTH legal at every subsequent depth, up to
    max_depth.
Chosen over Option B (independent trees per seed, split budget) because:
  1. it reuses the EXACT one-root wiring already proven in
     run_raptor_v2.puct_select_two (same TopologySearchState/lineage
     machinery, same neutral-root convention) -- less new, untested
     surface area than N independent trees;
  2. ONE shared visit-count budget lets PUCT allocate MORE simulations to
     whichever seed's value estimate looks more promising, rather than an
     even a-priori split across seeds that wastes budget on a seed the
     value net already discounts;
  3. it is what makes a single coherent principal variation
     (seed choice + edit sequence) meaningful, matching AlphaZero's own
     single-tree convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

from agentic_raptor.topology_rl.stage3e1 import (
    Stage3E1Action, Stage3E1ActionType, TopologySearchState,
    TopologyValidationResult, _has_gain_device)
from agentic_raptor.topology_rl.stage3e2_edits import (EDIT_TEMPLATES,
                                                        EditRejected, apply_edit)

#: the 8 real, already-implemented structural edits that share an EXACT
#: name with a Stage3E1ActionType member (verified: stage3e2_edits.
#: EDIT_TEMPLATES has 10 entries; 8 match a Stage3E1ActionType exactly,
#: the other 2 -- ADD_NESTED_MILLER, REPLACE_SIMPLE_MILLER_WITH_NESTED --
#: have no corresponding enum member and are intentionally left out of the
#: legal-action set rather than guess a remapping).
AZ_EDIT_ACTION_TYPES: frozenset = frozenset({
    Stage3E1ActionType.ADD_VERIFIED_STAGE,
    Stage3E1ActionType.REPLACE_STAGE_WITH_COMPATIBLE_BLOCK,
    Stage3E1ActionType.REPLACE_LOAD_WITH_COMPATIBLE_BLOCK,
    Stage3E1ActionType.ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE,
    Stage3E1ActionType.REPLACE_SUPPORTED_COMPENSATION_STRUCTURE,
    Stage3E1ActionType.ADD_SUPPORTED_OUTPUT_STAGE,
    Stage3E1ActionType.REMOVE_OPTIONAL_SUPPORTED_STAGE,
    Stage3E1ActionType.CONNECT_VERIFIED_FEEDBACK_PATH,
})
assert {t.value for t in AZ_EDIT_ACTION_TYPES} <= set(EDIT_TEMPLATES), (
    "AZ_EDIT_ACTION_TYPES must only name real, implemented edits")


class _Entry:
    __slots__ = ("topology_id", "graph", "metadata")


class LLMSeededEditRegistry:
    """A registry over the LLM's OWN proposed candidate graphs, whose
    derive_edited() is real: it applies stage3e2_edits.apply_edit()
    DIRECTLY to a candidate's own realised DeviceCircuitGraph, not to a
    re-derived family template (agentic_raptor.topology_rl.value_refresh.
    FamilyRegistry.derive_edited() does the latter -- correct for its own
    offline value-training use, but "the family this candidate belongs
    to, edited" is a weaker claim than "this candidate, edited," which is
    what a real state-transition needs). Every entry this registry ever
    returns is either an original LLM candidate or a genuine, validated
    structural derivative of one -- enumeration never enters this pool.
    """

    def __init__(self, candidates: list[dict]):
        from run_puct_ablation import _realise
        self._device_graphs: dict[str, Any] = {}
        self._entries: dict[str, _Entry] = {}
        self.seed_ids: list[str] = []
        for c in candidates:
            tid = c["llm_proposal_id"]
            dg = _realise(c["obj"])
            self._device_graphs[tid] = dg
            self._register(tid, dg)
            self.seed_ids.append(tid)
        if not self.seed_ids:
            raise ValueError("LLMSeededEditRegistry needs >= 1 candidate")

    def _register(self, tid: str, dg) -> None:
        from agentic_raptor.topology_rl.value_refresh import \
            device_graph_to_circuit_graph
        cg = device_graph_to_circuit_graph(dg, tid)
        e = _Entry()
        e.topology_id, e.graph, e.metadata = tid, cg, {}
        self._entries[tid] = e

    def _resolve(self, tid: str) -> str:
        return self.seed_ids[0] if tid == "proposal_root" else tid

    def get_topology(self, tid: str) -> _Entry:
        return self._entries[self._resolve(tid)]

    def list_topologies(self) -> list[str]:
        return sorted(self._entries)

    def derive_edited(self, tid: str, edit_type: str) -> str:
        """Real transition: current DeviceCircuitGraph -> typed structural
        edit -> new DeviceCircuitGraph -> canonicalised CircuitGraph entry.
        Raises EditRejected (propagated from stage3e2_edits.apply_edit) if
        the edit is not legal for this specific graph -- the SAME
        exception the legality probe in generate_alphazero_actions()
        catches, so "is this legal" and "apply it" are one code path, not
        two that could silently disagree."""
        real_tid = self._resolve(tid)
        parent_dg = self._device_graphs.get(real_tid)
        if parent_dg is None:
            raise KeyError(f"no device graph cached for {real_tid!r}")
        child_tid = f"{real_tid}~{edit_type}"
        if child_tid in self._entries:
            return child_tid
        ndg, _audit = apply_edit(parent_dg, edit_type)
        self._device_graphs[child_tid] = ndg
        self._register(child_tid, ndg)
        return child_tid


# ---------------------------------------------------------------------------
def validate_alphazero_candidate(reg: LLMSeededEditRegistry, tid: str,
                                 action: Stage3E1Action,
                                 ancestry: set) -> TopologyValidationResult:
    """Central validator for the AlphaZero action set -- same shape as
    stage3e1.validate_candidate(), but the structural-edit gate gets
    proven, not assumed: it actually calls reg.derive_edited() (the real
    apply_edit() path) rather than checking membership in a fixed
    "mapping supported" allow-list, so legality here is exactly what will
    happen on application."""
    reasons: list[str] = []
    e = reg.get_topology(tid)
    g = e.graph
    h = g.structural_hash()
    if action.action_type in AZ_EDIT_ACTION_TYPES:
        try:
            reg.derive_edited(tid, action.action_type.value)
        except EditRejected as exc:
            reasons.append(f"edit_rejected:{exc}")
        except Exception as exc:              # pragma: no cover - defensive
            reasons.append(f"edit_error:{type(exc).__name__}:{exc}")
    if (h in ancestry
            and action.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY):
        reasons.append("recreates_ancestor_graph")
    mapping_ok = not any(r.startswith(("edit_rejected", "edit_error")) for r in reasons)
    nodes = list(g.nodes.values())
    has_gain = _has_gain_device(nodes) or tid.split("~", 1)[0] in reg.seed_ids
    if not has_gain:
        reasons.append("no_gain_stage")
    return TopologyValidationResult(
        structurally_valid=len(nodes) > 0, semantically_valid=has_gain,
        mapping_supported=mapping_ok and "recreates_ancestor_graph" not in reasons,
        bias_complete=True, supply_complete=True, io_complete=True,
        no_floating_nodes=True,
        no_illegal_cycles="recreates_ancestor_graph" not in reasons,
        feedback_status="open_loop_adm", compensation_status="source_defined",
        transistor_realisation_supported=mapping_ok, reasons=reasons)


def generate_alphazero_actions(state: TopologySearchState,
                               reg: LLMSeededEditRegistry,
                               pool_ids: list[str], max_alternatives: int = 4
                               ) -> tuple[list[Stage3E1Action], list[dict]]:
    """Strict two-phase action space (Stage 5 Campaign 01B, Section 1 --
    postmortem on CAMPAIGN_01_ATTEMPT_1):

        SUPER_ROOT (state.topology_id == "proposal_root"):
            legal: SELECT_EXISTING_TOPOLOGY(seed_0), SELECT_EXISTING_TOPOLOGY(
            seed_1), ... -- exactly one seed commit, nothing else.
        AFTER SEED SELECTION (any other state):
            legal: real structural edits (AZ_EDIT_ACTION_TYPES), TERMINATE.
            SELECT_EXISTING_TOPOLOGY is NEVER offered again.

    The previous version offered a_sel_* at EVERY depth (gated only by
    `alt != state.topology_id`, true at every non-root state too), which
    let a real self-play trajectory repeatedly jump between complete LLM
    seeds mid-search (SELECT(p03) -> SELECT(p00)) -- that behavior belongs
    to the retired root-level selector (run_raptor_v2.puct_select_two),
    not to sequential topology-edit search, and was the mechanical root
    cause of CAMPAIGN_01_ATTEMPT_1's degenerate, spec-insensitive replay.
    a_keep (KEEP_TOPOLOGY) is dropped entirely: it was a redundant no-op
    ("stay uncommitted at the neutral root") that neither phase above
    needs -- TERMINATE already covers "stop here" once committed, and the
    super-root's only job now is to commit to a seed.
    """
    ancestry = set(state.lineage)
    at_super_root = state.topology_id == "proposal_root"

    cands: list[Stage3E1Action] = []
    if at_super_root:
        alts = [a for a in sorted(pool_ids) if a != state.topology_id][:max_alternatives]
        for alt in alts:
            cands.append(Stage3E1Action(
                f"a_sel_{alt}", Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                source_ref=alt, target_location="root",
                preconditions=("target_in_registry",),
                compatibility=("same_spec_class",), provenance="registry"))
    else:
        cands.append(Stage3E1Action("a_term", Stage3E1ActionType.TERMINATE_SEARCH,
                                    provenance="self"))
        for edit_type in sorted(AZ_EDIT_ACTION_TYPES, key=lambda t: t.value):
            cands.append(Stage3E1Action(
                f"a_edit_{edit_type.value}", edit_type,
                source_ref=state.topology_id, target_location="graph",
                preconditions=("edit_legal_for_current_graph",),
                provenance="stage3e2_edits"))

    legal, rejections = [], []
    for a in cands:
        if a.action_type == Stage3E1ActionType.TERMINATE_SEARCH:
            legal.append(a)
            continue
        if state.remaining_search_budget <= 0:
            rejections.append({"action_id": a.action_id,
                               "reason": "search_budget_exhausted"})
            continue
        tgt = a.source_ref or state.topology_id
        v = validate_alphazero_candidate(reg, tgt, a, ancestry)
        if v.ok:
            legal.append(a)
        else:
            rejections.append({"action_id": a.action_id,
                               "reason": ";".join(v.reasons)})
    return legal, rejections


def apply_alphazero_action(state: TopologySearchState, action: Stage3E1Action,
                           reg: LLMSeededEditRegistry) -> TopologySearchState:
    """apply_topology_action() extended to route any AZ_EDIT_ACTION_TYPES
    member through reg.derive_edited() -- a REAL graph transition, not a
    candidate swap. SELECT_EXISTING_TOPOLOGY/KEEP/TERMINATE behave exactly
    as stage3e1.apply_topology_action() already does."""
    if action.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY:
        tid = action.source_ref
    elif action.action_type in AZ_EDIT_ACTION_TYPES:
        tid = reg.derive_edited(state.topology_id, action.action_type.value)
    else:
        tid = state.topology_id
    g = reg.get_topology(tid).graph
    h = g.structural_hash()
    if (h in set(state.lineage)
            and action.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY):
        raise ValueError(f"cycle: {h[:12]} already in ancestry")
    audit = {"action": action.to_dict(), "parent_hash": state.graph_hash,
             "child_hash": h, "depth": state.depth + 1}
    return TopologySearchState(
        topology_id=tid, graph_hash=h, lineage=state.lineage + [h],
        spec=dict(state.spec), rag_context_ids=list(state.rag_context_ids),
        available_blocks=list(state.available_blocks), legal_action_ids=[],
        edit_history=state.edit_history + [audit], validation_status="validated",
        structural_features={"n_nodes": float(len(g.nodes)),
                             "n_edges": float(len(g.edges))},
        previous_evidence_ref=None,
        remaining_search_budget=state.remaining_search_budget - 1,
        remaining_spice_budget=state.remaining_spice_budget,
        depth=state.depth + 1,
        terminal_reason=("terminate_action"
                        if action.action_type == Stage3E1ActionType.TERMINATE_SEARCH
                        else None))


# ---------------------------------------------------------------------------
@dataclass
class AlphaZeroConfig:
    """Renamed/reorganised per the migration's requested naming
    (Section 6) -- translates to stage3e1.SearchConfig internally rather
    than duplicating the engine."""
    alphazero_simulations_per_move: int = 256
    alphazero_c_puct: float = 1.5
    alphazero_max_edit_depth: int = 4
    alphazero_max_children: int = 12
    alphazero_temperature: float = 1.0     # visit-count exponent 1/tau at inference
    seed: int = 0
    training_mode: bool = False
    #: Section 4 (Campaign 01B): root exploration noise, P_noisy(a) =
    #: (1-eps)*P(a) + eps*eta(a), eta ~ Dirichlet(alpha) -- applied by the
    #: REUSED stage3e1.TopologyMCTS._expand() machinery (it already had
    #: this exact formula, gated on cfg.training_mode; only wiring these
    #: two values through was missing). ONLY takes effect when
    #: training_mode=True (episode collection); alphazero_select_two()'s
    #: and validate_az_candidate()'s default AlphaZeroConfig() already has
    #: training_mode=False, so FULL selection and paper inference are
    #: unaffected without any extra guard.
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.30
    leaf_mode: str = "value_only"          # "value_only" for Part-A-style diagnostics


def build_root_state(spec: dict, ctx_id: str, seed_ids: list[str],
                     search_budget: int = 8) -> TopologySearchState:
    n_nodes = 0.0
    return TopologySearchState(
        topology_id="proposal_root", graph_hash="proposal_root", lineage=[],
        spec={"target_gain_db": spec["gain_target_db"],
             "target_gbw_hz": spec.get("ugbw_target_hz") or 1e4,
             "minimum_phase_margin_deg": spec["phase_margin_target_deg"],
             "load_capacitance_f": spec["load_capacitance_pf"] * 1e-12,
             "supply_voltage": 1.8},
        rag_context_ids=[ctx_id], available_blocks=[], legal_action_ids=[],
        edit_history=[], validation_status="validated",
        structural_features={"n_nodes": n_nodes}, previous_evidence_ref=None,
        remaining_search_budget=search_budget, remaining_spice_budget=0, depth=0)


def load_alphazero_nets(value_ckpt=None, seed: int = 0):
    """The SAME policy/value network stage3e1.build_policy_value() already
    builds -- its policy head is sized over the FULL Stage3E1ActionType
    enum (n_types = len(Stage3E1ActionType)), so it already had capacity
    for every real edit type before this module existed; nothing about
    the network architecture needed to change, only what actions
    generate_alphazero_actions() offers it."""
    from pathlib import Path

    from agentic_raptor.topology_rl import stage3e1 as s1
    nets = s1.build_policy_value(seed)
    if value_ckpt is not None:
        ck = Path(value_ckpt)
        if ck.is_file():
            s1.load_checkpoint(nets, ck)
    return nets


def _search_cfg_from(cfg: AlphaZeroConfig):
    from agentic_raptor.topology_rl import stage3e1 as s1
    return s1.SearchConfig(
        num_simulations=cfg.alphazero_simulations_per_move,
        c_puct=cfg.alphazero_c_puct, max_depth=cfg.alphazero_max_edit_depth,
        max_children=cfg.alphazero_max_children, leaf_mode=cfg.leaf_mode,
        training_mode=cfg.training_mode, root_noise_eps=cfg.dirichlet_epsilon,
        root_dirichlet_alpha=cfg.dirichlet_alpha, seed=cfg.seed)


def _search_from_state(root_state: TopologySearchState,
                       reg: LLMSeededEditRegistry, pool_ids: list[str],
                       nets: dict, cfg: AlphaZeroConfig):
    """The reusable core: construct a TopologyMCTS wired to the AlphaZero
    action functions and run it from an ARBITRARY starting state -- used
    both by run_alphazero_search() (single search rooted at the neutral
    multi-seed root) and run_alphazero_episode() (one fresh search PER
    REAL step, rooted at wherever the episode's real trajectory currently
    is)."""
    from agentic_raptor.topology_rl import stage3e1 as s1
    search_cfg = _search_cfg_from(cfg)
    mcts = s1.TopologyMCTS(nets, reg, pool_ids, search_cfg,
                           generate_actions_fn=generate_alphazero_actions,
                           apply_action_fn=apply_alphazero_action)
    root = mcts.run(root_state)
    return root, mcts


def run_alphazero_search(candidates: list[dict], spec: dict, ctx_id: str, *,
                         value_ckpt=None, seed: int = 0,
                         config: AlphaZeroConfig | None = None) -> dict:
    """The TRUE_ALPHAZERO replacement for run_raptor_v2.puct_select_two.
    Same candidate-pool contract (validated LLM candidates in, ranked-by-
    visit-count candidates out) but the search can now genuinely produce
    topology hashes absent from `candidates` -- see the `tree_topology_
    hashes`/`novel_vs_seed_hashes` fields in the return value.
    """
    cfg = config or AlphaZeroConfig(seed=seed)
    reg = LLMSeededEditRegistry(candidates)
    root_state = build_root_state(spec, ctx_id, reg.seed_ids)
    nets = load_alphazero_nets(value_ckpt, seed=0)
    root, mcts = _search_from_state(root_state, reg, reg.seed_ids, nets, cfg)

    all_hashes = {n.state.graph_hash for n in mcts.nodes} - {"proposal_root"}
    seed_hashes = {reg.get_topology(sid).graph.structural_hash() for sid in reg.seed_ids}
    novel_vs_seed = sorted(all_hashes - seed_hashes)
    depths = [n.state.depth for n in mcts.nodes]

    from agentic_raptor.topology_rl import stage3e1 as s1
    result = s1.search_result(root, mcts)
    # "proposal_root" is never independently meaningful -- it is always an
    # alias for reg.seed_ids[0] (the neutral root's borrowed graph, see
    # LLMSeededEditRegistry._resolve). The search CAN legitimately settle
    # on a_term at the root (this is a real, observed outcome, not a bug:
    # a small simulation budget or a value net that scores the root's own
    # seed competitively can make "stop here" win the visit count) --
    # resolved here so a caller never has to special-case the sentinel id.
    selected_topology_id = result["selected_topology"]
    if selected_topology_id == "proposal_root":
        selected_topology_id = reg.seed_ids[0]
    return {
        "selected_topology_id": selected_topology_id,
        "selected_action": result["selected_action"],
        "root_visit_distribution": result["root_visit_distribution"],
        "candidate_rankings": result["candidate_rankings"],
        "principal_variation": result["principal_variation"],
        "costs": result["costs"],
        "validator_rejections": result["validator_rejections"],
        "terminal_reason": result["terminal_reason"],
        "tree_nodes": len(mcts.nodes),
        "max_depth_reached": max(depths) if depths else 0,
        "mean_depth": (sum(depths) / len(depths)) if depths else 0.0,
        "seed_ids": reg.seed_ids,
        "seed_topology_hashes": sorted(seed_hashes),
        "tree_topology_hashes": sorted(all_hashes),
        "novel_vs_seed_hashes": novel_vs_seed,
        "novel_vs_seed_count": len(novel_vs_seed),
        "nodes": [n.record() for n in mcts.nodes],
        "config": {"alphazero_simulations_per_move": cfg.alphazero_simulations_per_move,
                  "alphazero_c_puct": cfg.alphazero_c_puct,
                  "alphazero_max_edit_depth": cfg.alphazero_max_edit_depth,
                  "alphazero_max_children": cfg.alphazero_max_children,
                  "leaf_mode": cfg.leaf_mode, "seed": cfg.seed},
    }


# ---------------------------------------------------------------------------
# Episode / replay -- real self-play: FRESH MCTS at every real step, not a
# principal-variation readout of one big tree. Each step re-plans with
# everything the PREVIOUS real transition changed, matching genuine
# AlphaZero self-play.
# ---------------------------------------------------------------------------
def stable_episode_rng_seed(campaign_seed: int, generation_id: str,
                            spec_hash: str, rollout_seed: int) -> int:
    """Section 3 (Campaign 01B): a deterministic, process-stable per-
    episode RNG seed. The previous code called `random.Random(seed)`
    where `seed` was ONLY the rollout seed (e.g. 0) -- every episode in a
    single-seed campaign run therefore replayed the IDENTICAL stochastic-
    sampling sequence regardless of spec, which (combined with an
    untrained G0's low-differentiation visit distributions) plausibly
    contributed to CAMPAIGN_01_ATTEMPT_1's 8/8 identical trajectories.
    Deliberately uses hashlib (NOT Python's built-in hash(), which is
    salted per-process for strings unless PYTHONHASHSEED is fixed) so the
    same four inputs always produce the same seed across processes/runs.
    """
    import hashlib
    payload = f"{campaign_seed}|{generation_id}|{spec_hash}|{rollout_seed}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def visit_policy(root_node, tau: float = 1.0) -> dict:
    """pi(a|s) proportional to N(s,a)^(1/tau), normalised over the root
    node's own legal children. tau<=1e-3 is treated as deterministic
    (one-hot on the max-N child, ties broken by lowest node_id -- the
    same tie-break stage3e1.search_result()'s own best_child selection
    uses, so deterministic inference here always agrees with it)."""
    children = root_node.children
    if not children:
        return {}
    if tau <= 1e-3:
        best = max(children, key=lambda c: (c.N, -c.node_id))
        return {c.action.action_id: (1.0 if c is best else 0.0) for c in children}
    weighted = {c.action.action_id: c.N ** (1.0 / tau) for c in children}
    total = sum(weighted.values()) or 1.0
    return {k: v / total for k, v in weighted.items()}


def run_alphazero_episode(candidates: list[dict], spec: dict, ctx_id: str,
                          spec_hash: str, *, spec_index: int | None = None,
                          value_ckpt=None, seed: int = 0,
                          campaign_seed: int = 0,
                          config: AlphaZeroConfig | None = None,
                          max_episode_depth: int = 4,
                          deterministic: bool = True,
                          generation_id: str = "AZ_G0",
                          nets: dict | None = None) -> dict:
    """One real self-play episode: S0 -> pi0 -> A0 -> S1 -> pi1 -> ... ->
    terminal. `deterministic=True` (max-visit action every step) is what
    Section 15's smoke episode and inference both use; `deterministic=
    False` samples proportional to pi (temperature =
    config.alphazero_temperature) for training-time exploration.
    Returns everything build_replay_rows()/terminal_evaluation() need --
    including the LIVE registry (`registry`), since the terminal
    topology's DeviceCircuitGraph lives only there.

    The stochastic-sampling RNG is seeded via stable_episode_rng_seed()
    (Section 3, Campaign 01B) from (campaign_seed, generation_id,
    spec_hash, seed) rather than `seed` alone -- so two episodes that
    differ in spec or rollout seed never silently replay the same
    sampling sequence (see that function's docstring for why this
    mattered in practice). `episode_rng_seed` is returned for replay
    provenance/auditing.

    `nets` may be passed in (default None -> load_alphazero_nets(
    value_ckpt)) so a controlled/mocked policy can be substituted for
    testing -- e.g. proving TERMINATE is reachable before max_episode_
    depth (Section 2) by feeding a policy that overwhelmingly prefers it.
    """
    import random as _random
    from dataclasses import asdict

    cfg = config or AlphaZeroConfig(seed=seed)
    reg = LLMSeededEditRegistry(candidates)
    nets = nets or load_alphazero_nets(value_ckpt, seed=0)
    episode_rng_seed = stable_episode_rng_seed(campaign_seed, generation_id,
                                               spec_hash, seed)
    rng = _random.Random(episode_rng_seed)

    state = build_root_state(spec, ctx_id, reg.seed_ids)
    steps = []
    for step_idx in range(max_episode_depth):
        root, mcts = _search_from_state(state, reg, reg.seed_ids, nets, cfg)
        if not root.children:
            break
        tau = 0.0 if deterministic else cfg.alphazero_temperature
        pi = visit_policy(root, tau=tau)
        if deterministic:
            chosen = max(root.children, key=lambda c: (c.N, -c.node_id))
        else:
            r, acc, chosen = rng.random(), 0.0, root.children[-1]
            for c in root.children:
                acc += pi.get(c.action.action_id, 0.0)
                if r <= acc:
                    chosen = c
                    break
        cur_hash = (reg.get_topology(reg.seed_ids[0]).graph.structural_hash()
                   if state.topology_id == "proposal_root"
                   else reg.get_topology(state.topology_id).graph.structural_hash())
        steps.append({
            "step": step_idx, "state": asdict(state),
            "state_topology_id": state.topology_id, "state_graph_hash": cur_hash,
            "depth": state.depth, "edit_history": list(state.edit_history),
            "legal_action_ids": sorted(pi), "raw_visit_counts":
                {c.action.action_id: c.N for c in root.children},
            "pi": pi, "selected_action_id": chosen.action.action_id,
            "selected_action_type": chosen.action.action_type.value,
            "tree_nodes_this_step": len(mcts.nodes)})
        state = chosen.state
        if chosen.action.action_type == Stage3E1ActionType.TERMINATE_SEARCH:
            break

    terminal_topology_id = (reg.seed_ids[0] if state.topology_id == "proposal_root"
                            else state.topology_id)
    terminal_topology_hash = reg.get_topology(terminal_topology_id).graph.structural_hash()
    # `steps` only ever records each state BEFORE its own action was
    # applied -- the truly terminal state (after the LAST action, or the
    # state reached at max_episode_depth) is never itself a step's
    # "state_graph_hash", so it must be added explicitly here or both
    # novelty accounting and replay's terminal_topology_hash silently
    # undercount/mis-point to the second-to-last state instead of the
    # real terminal one.
    step_hashes = {s["state_graph_hash"] for s in steps} | {terminal_topology_hash}
    seed_hashes = {reg.get_topology(sid).graph.structural_hash() for sid in reg.seed_ids}
    return {"ctx_id": ctx_id, "spec_hash": spec_hash, "spec_index": spec_index,
           "seed_ids": reg.seed_ids, "seed_topology_hashes": sorted(seed_hashes),
           "steps": steps, "terminal_topology_id": terminal_topology_id,
           "terminal_topology_hash": terminal_topology_hash,
           "terminal_came_from_edit": terminal_topology_id not in reg.seed_ids,
           "seed_novel_hashes": sorted(step_hashes - seed_hashes),
           "seed_novel_count": len(step_hashes - seed_hashes),
           "registry": reg, "generation_id": generation_id, "seed": seed,
           "campaign_seed": campaign_seed, "episode_rng_seed": episode_rng_seed,
           "deterministic": deterministic,
           "dirichlet_epsilon": cfg.dirichlet_epsilon if cfg.training_mode else None,
           "dirichlet_alpha": cfg.dirichlet_alpha if cfg.training_mode else None}


def terminal_evaluation(episode: dict, spec: dict, *, budget: int = 16,
                        seed: int = 0, out_dir=None) -> dict:
    """MB-SAC sizing on the AlphaZero terminal topology, THEN a fresh,
    SEPARATE authoritative ngspice call on the winning sizing vector --
    mirrors run_raptor_v2.py's verify() exactly (a NEW call tagged
    mode="final_verification", never the sizing exploration's own
    measurement), so z carries the same provenance guarantee every other
    authoritative outcome in this project has.

    z reuses the EXACT feasibility-first scalar this project already uses
    for puct_examples.jsonl's value_target (agentic_raptor.selfimprove_v2.
    streams.harvest_run): 1.0 for an exact spec pass, else max(-1.0,
    1.0 - 2*normalized_distance_to_feasibility) -- not a new,
    AlphaZero-specific objective.
    """
    import time
    from pathlib import Path

    from agentic_raptor.electrical import discover_ngspice, effective_c_load
    from agentic_raptor.mb_sac.spec_sizing import (apply_knobs, measure,
                                                    postsizing_outcome,
                                                    sac_size)
    from agentic_raptor.topology_rl.stage3e2 import new_costs

    reg = episode["registry"]
    tid = episode["terminal_topology_id"]
    graph = reg._device_graphs[tid]
    ctx_id = episode["ctx_id"]
    exe = discover_ngspice()
    out_dir = (Path(out_dir) if out_dir
              else ROOT / "artifacts/publication_v3/alphazero_smoke/sizing")
    cl = effective_c_load(spec)

    from agentic_raptor.mb_sac.spec_sizing import KNOB_NAMES

    sz = sac_size(f"az_{ctx_id}", graph, spec, exe, out_dir, new_costs(),
                 budget=budget, seed=seed, persist=False, c_load_f=cl)
    # sac_size stores the winning knob vector as a {KNOB_NAME: value} dict
    # (rounded for persistence), not the plain list apply_knobs() expects --
    # convert back via KNOB_NAMES' canonical order.
    knob_vector = [sz["best"]["knobs"][name] for name in KNOB_NAMES]
    sized_graph = apply_knobs(graph, knob_vector)
    call_id = f"az_final:{ctx_id}:{int(time.time() * 1000)}"
    tag = f"az_ver_{ctx_id}_{int(time.time() * 1000)}"
    meas = measure(f"az_final_{ctx_id}", sized_graph, exe, out_dir, tag,
                   new_costs(), c_load_f=cl)
    oc = postsizing_outcome(meas, spec)
    d = oc.get("normalized_distance_to_feasibility")
    z = (1.0 if oc.get("exact_spec_pass")
        else max(-1.0, 1.0 - 2.0 * float(d)) if d is not None else None)
    return {
        "call_id": call_id, "mode": "final_verification", "spec_id": ctx_id,
        "spec_hash": episode["spec_hash"],
        "terminal_topology_id": tid,
        "requested_c_load_f": cl, "simulated_c_load_f": meas.get("c_load_f"),
        "electrical_environment_version": "POST_CLOAD_FIX_V1",
        "exact_spec_pass": bool(oc.get("exact_spec_pass")),
        "normalized_distance_to_feasibility": d,
        "gain_db": meas.get("gain_db"), "pm_deg": meas.get("pm_deg"),
        "ugbw_hz": meas.get("ugbw_hz"), "idd_a": meas.get("idd_a"),
        "sizing_spice_calls": sz["spice_calls"], "sizing_budget": budget,
        "z": z}


def build_replay_rows(episode: dict, terminal: dict, *,
                      checkpoint_hash: str | None = None,
                      protected_context_ids: frozenset = frozenset()) -> list:
    """(state, pi, z) rows for every REAL step in the episode -- never the
    old candidate->scalar-value puct_examples.jsonl schema (that schema
    has no `pi`/visit-distribution over a real edit-inclusive action set
    at all -- it cannot be relabelled into this one). Hard-rejects
    protected specs and any non-POST_CLOAD_FIX_V1 / CLOAD-mismatched
    terminal measurement -- z backfills into every step with NO sign
    flips (Section 5: single-agent, no adversary)."""
    if episode["ctx_id"] in protected_context_ids:
        return []
    if terminal.get("electrical_environment_version") != "POST_CLOAD_FIX_V1":
        return []
    if terminal.get("requested_c_load_f") != terminal.get("simulated_c_load_f"):
        return []
    z = terminal.get("z")
    if z is None:
        return []
    terminal_hash = episode["terminal_topology_hash"]
    rows = []
    for s in episode["steps"]:
        rows.append({
            "generation": episode["generation_id"], "spec_index": episode.get("spec_index"),
            "spec_hash": episode["spec_hash"], "context_id": episode["ctx_id"],
            "step": s["step"], "state": s["state"],
            "state_topology_id": s["state_topology_id"],
            "state_graph_hash": s["state_graph_hash"], "edit_depth": s["depth"],
            "edit_history": s["edit_history"], "legal_action_ids": s["legal_action_ids"],
            "raw_visit_counts": s["raw_visit_counts"], "pi": s["pi"],
            "selected_action_id": s["selected_action_id"], "z": float(z),
            "terminal_topology_hash": terminal_hash,
            "terminal_authoritative_call_id": terminal["call_id"],
            "requested_c_load_f": terminal["requested_c_load_f"],
            "simulated_c_load_f": terminal["simulated_c_load_f"],
            "electrical_environment_version": terminal["electrical_environment_version"],
            "policy_value_checkpoint_hash": checkpoint_hash,
            "seed": episode["seed"], "campaign_seed": episode.get("campaign_seed", 0),
            "episode_rng_seed": episode.get("episode_rng_seed"),
            "dirichlet_epsilon": episode.get("dirichlet_epsilon"),
            "dirichlet_alpha": episode.get("dirichlet_alpha"),
            "split": "train"})
    return rows


AZ_REPLAY_ROOT = ROOT / "artifacts/publication_v3/alphazero_replay"


def write_replay_rows(rows: list, generation_id: str, out_root=None) -> "Path":
    from pathlib import Path
    out_root = Path(out_root) if out_root else AZ_REPLAY_ROOT
    p = out_root / f"{generation_id}_replay.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(_json.dumps(r, default=str) + "\n")
    return p


# ---------------------------------------------------------------------------
# Training -- reuses stage3e1.train_step() (already computes the exact
# policy/value losses this project specifies) unmodified; the only new
# code is converting a replay row into train_step()'s expected per-example
# shape.
# ---------------------------------------------------------------------------
def train_az_generation(replay_rows: list, reg: LLMSeededEditRegistry, *,
                        parent_checkpoint=None, lr: float = 1e-3,
                        weight_decay: float = 1e-4, epochs: int = 1,
                        seed: int = 0, nets: dict | None = None,
                        opt=None) -> dict:
    """`nets`/`opt` may be passed in (and are then also returned) so a
    caller collecting replay from MULTIPLE episodes -- each with its OWN
    LLMSeededEditRegistry, whose topology_ids (e.g. "p00") are only
    locally unique within that one episode and WOULD collide if merged
    into a single registry -- can call this once per episode's own
    (rows, registry) pair while accumulating gradients on the SAME
    network across the whole batch, instead of building one fresh network
    per episode (which would silently discard everything learned from
    every earlier episode)."""
    from agentic_raptor.topology_rl import stage3e1 as s1
    import torch
    nets = nets or load_alphazero_nets(parent_checkpoint, seed=0)
    opt = opt or torch.optim.Adam(nets["params"], lr=lr, weight_decay=weight_decay)
    examples = [{"state": r["state"], "legal_action_ids": r["legal_action_ids"],
                "visit_distribution": r["pi"], "value_target": r["z"]}
               for r in replay_rows]
    epoch_reports = [s1.train_step(nets, examples, reg, lr=lr,
                                   weight_decay=weight_decay, opt=opt)
                     for _ in range(epochs)]
    return {"nets": nets, "opt": opt, "epoch_reports": epoch_reports,
           "examples_used": epoch_reports[-1]["examples_used"] if epoch_reports else 0,
           "final_policy_loss": epoch_reports[-1]["policy_loss"] if epoch_reports else None,
           "final_value_loss": epoch_reports[-1]["value_loss"] if epoch_reports else None}


# ---------------------------------------------------------------------------
# Generational checkpointing -- same pattern as agentic_raptor.
# selfimprove_v2.sft_self_improvement's G0/G1 generations, applied to the
# AlphaZero policy/value network instead of the SFT proposer.
# ---------------------------------------------------------------------------
AZ_GENERATIONS_ROOT = ROOT / "artifacts/publication_v3/alphazero_generations"

#: Section 8: three checkpoint statuses, never conflated.
#:   SMOKE     -- mechanically valid (training path proven) but not
#:                performance-qualified; may be used in development/smoke
#:                integration testing ONLY.
#:   CANDIDATE -- trained on a meaningful adaptation campaign, awaiting
#:                promotion.
#:   PROMOTED  -- passed the defined AlphaZero validation/promotion gates;
#:                the ONLY status paper/A0-A8 mode may use.
AZ_CHECKPOINT_STATUSES = ("SMOKE", "CANDIDATE", "PROMOTED")


def require_promoted_az_checkpoint() -> "Path":
    """Section 9: paper/A0-A8 mode's hard gate. Scans every AlphaZero
    generation manifest for checkpoint_status == "PROMOTED" and returns
    its checkpoint path -- raises AlphaZeroSelectionError if none exists,
    rather than silently falling back to AZ_G1 SMOKE, a fresh random G0,
    the retired root-PUCT checkpoint, or the clean 91-example value-only
    checkpoint. Callers preparing a real A0-A8/paper campaign must call
    this explicitly and pass its result as run_pipeline's
    alphazero_value_ckpt -- run_pipeline itself does not call this (it
    would make every ordinary dev/smoke run in this checkout require a
    promoted checkpoint that does not exist yet)."""
    if not AZ_GENERATIONS_ROOT.is_dir():
        raise AlphaZeroSelectionError(
            "no AlphaZero generations exist yet -- paper mode requires a "
            "PROMOTED checkpoint")
    for gen_dir in sorted(AZ_GENERATIONS_ROOT.iterdir()):
        manifest = read_az_generation_manifest(gen_dir.name)
        if manifest and manifest.get("checkpoint_status") == "PROMOTED":
            return Path(manifest["checkpoint_path"])
    raise AlphaZeroSelectionError(
        "paper/A0-A8 mode requires an explicitly PROMOTED AlphaZero "
        "checkpoint -- none exists (current generations: "
        f"{sorted(p.name for p in AZ_GENERATIONS_ROOT.iterdir())}). "
        "This is EXPECTED immediately after the FULL cutover migration -- "
        "a real training campaign must produce and promote one before "
        "paper mode can run.")


def az_generation_dir(gen_id: str):
    return AZ_GENERATIONS_ROOT / gen_id


def read_az_generation_manifest(gen_id: str) -> dict | None:
    import json as _json
    p = az_generation_dir(gen_id) / "manifest.json"
    return _json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write_az_generation_manifest(gen_id: str, manifest: dict):
    import json as _json
    d = az_generation_dir(gen_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "manifest.json"
    if gen_id == "AZ_G0" and p.is_file():
        raise RuntimeError("AZ_G0 is immutable and already has a manifest -- "
                           "refusing to overwrite")
    p.write_text(_json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return p


def next_az_generation_id(parent_gen_id: str) -> str:
    n = int(parent_gen_id.replace("AZ_G", "")) + 1
    return f"AZ_G{n}"


def ensure_az_g0_manifest() -> dict:
    """AZ_G0 = a FRESH, randomly-initialised policy/value network.

    Explicitly NOT the retired root-PUCT checkpoint (policy_value_ep0.pt)
    and NOT a silent reuse of the clean 91-example value-only checkpoint
    either -- that head was cross-validated against puct_examples.jsonl
    rows whose legal_action_ids were always a subset of {a_keep, a_sel_*,
    a_term}; it was never exercised on a real a_edit_* action, so treating
    it as a warm start for AlphaZero's policy head (scored over the FULL
    Stage3E1ActionType space) would silently claim validated competence
    the checkpoint never demonstrated. If a warm start is wanted later,
    it must be evaluated on AlphaZero replay first, not assumed.
    """
    existing = read_az_generation_manifest("AZ_G0")
    if existing is not None:
        return existing
    from agentic_raptor.topology_rl import stage3e1 as s1
    nets = load_alphazero_nets(value_ckpt=None, seed=0)
    ck_path = az_generation_dir("AZ_G0") / "checkpoint" / "policy_value.pt"
    meta = {"generation_id": "AZ_G0", "parent_generation": None,
           "initialization": "random_untrained",
           "note": ("AZ_G0 is a freshly-initialised policy/value network -- "
                    "NOT the retired root-PUCT checkpoint, NOT the clean "
                    "91-example value-only checkpoint (trained against a "
                    "narrower action space that never included real "
                    "structural edits). See ensure_az_g0_manifest().")}
    s1.save_checkpoint(nets, ck_path, meta)
    manifest = {**meta, "checkpoint_path": str(ck_path), "immutable": True,
               "rejected": False, "checkpoint_status": "SMOKE",
               "checkpoint_status_reason": "randomly-initialised, zero training"}
    write_az_generation_manifest("AZ_G0", manifest)
    return manifest


def validate_az_candidate(nets: dict, candidates: list[dict], spec: dict,
                          ctx_id: str, *, seed: int = 0) -> dict:
    """Section 13's minimum promotion checks -- mechanics, not statistical
    superiority. A candidate that fails ANY of these is rejected
    regardless of loss curves: correct AlphaZero mechanics come before
    "is it better yet." """
    import math

    failures = []
    try:
        cfg = AlphaZeroConfig(alphazero_simulations_per_move=32,
                              alphazero_max_edit_depth=3, seed=seed)
        reg = LLMSeededEditRegistry(candidates)
        root_state = build_root_state(spec, ctx_id, reg.seed_ids)
        root, mcts = _search_from_state(root_state, reg, reg.seed_ids, nets, cfg)
    except Exception as exc:
        return {"passed": False, "failures": [f"search_raised:{type(exc).__name__}:{exc}"]}
    if not root.children:
        failures.append("no_legal_actions_at_root")
        return {"passed": False, "failures": failures}
    depths = [n.state.depth for n in mcts.nodes]
    if max(depths, default=0) <= 1:
        failures.append(f"max_depth_not_greater_than_1:{max(depths, default=0)}")
    pi = visit_policy(root, tau=1.0)
    total = sum(pi.values())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
        failures.append(f"policy_not_normalised:sum={total}")
    if any((not math.isfinite(v)) for v in pi.values()):
        failures.append("policy_contains_nan_or_inf")
    for n in mcts.nodes:
        if not math.isfinite(n.Q) or not math.isfinite(n.W):
            failures.append(f"node_{n.node_id}_nan_or_inf_Q_or_W")
            break
    # deterministic inference must be reproducible
    pi_det_a = visit_policy(root, tau=0.0)
    root2, _ = _search_from_state(root_state, reg, reg.seed_ids, nets,
                                  AlphaZeroConfig(alphazero_simulations_per_move=32,
                                                 alphazero_max_edit_depth=3, seed=seed))
    pi_det_b = visit_policy(root2, tau=0.0)
    if max(pi_det_a, key=pi_det_a.get, default=None) != max(pi_det_b, key=pi_det_b.get, default=None):
        failures.append("deterministic_inference_not_reproducible")
    return {"passed": not failures, "failures": failures,
           "max_depth_reached": max(depths, default=0), "tree_nodes": len(mcts.nodes)}


# ---------------------------------------------------------------------------
# The live FULL contract: the TRUE_ALPHAZERO replacement for the retired
# puct_select_two()'s one_root branch.
# ---------------------------------------------------------------------------
class AlphaZeroSelectionError(Exception):
    """Hard-fail conditions for alphazero_select_two() -- never silently
    substitutes root-PUCT, prior-only, or an arbitrary candidate."""


def alphazero_select_two(candidates: list[dict], spec: dict, ctx_id: str, *,
                         value_ckpt=None, seed: int = 0,
                         config: AlphaZeroConfig | None = None,
                         nets: dict | None = None) -> dict:
    """Same external contract as the retired puct_select_two(): validated
    LLM candidates in, exactly two topology objects out, ready for
    size_and_predict(). Unlike the retired selector, the two outputs can
    genuinely be edited descendants -- each carries its own realised
    `device_graph` so size_and_predict() sizes the ACTUAL edited graph,
    never a re-derived family template (see run_raptor_v2.size_and_
    predict's `c.get("device_graph")` branch).

    Top-2 rule (Section 3 -- decided and documented, not an undocumented
    heuristic): rank EVERY distinct canonical topology state visited
    anywhere in the search tree (not just root-level children) by that
    state's own total MCTS visit count N, ties broken by canonical hash.
    This is the exact visit-count-ranking convention the retired selector
    already used (rank candidates by N, take the top SELECT_K), applied
    over the FULL tree instead of only root children -- which is what
    lets an edited descendant compete for a slot at all. Visits are
    SUMMED per canonical hash (not just the max taken) so a hash reached
    via more than one node is not undercounted -- since Campaign 01B's
    Section 1 fix, SELECT is legal only once (at the super-root), so this
    mainly matters for a hash two different seeds' edit sequences happen
    to converge on, not repeated re-selection. Uses ONLY search-internal
    evidence (N) -- never MB-SAC or authoritative SPICE, which would leak
    downstream truth backward into topology selection.

    This chosen rule already satisfies every "preferred behavior" Section
    3 lists (search evidence, visit-count semantics, validated states,
    distinct hashes, no aliasing, full ancestry) directly, via the
    single-shared-tree architecture (Section 15's Option A) -- the
    alternative per-seed-independent-trees clause in the same section
    does not apply here, since this architecture never produces "one
    final trajectory per seed" to begin with.

    `nets` may be passed in (default None -> load_alphazero_nets(
    value_ckpt)) so a controlled/mocked network can be substituted for
    testing -- e.g. proving an edited descendant CAN win a top-2 slot
    given a value net that genuinely prefers one (Section 1's two-phase
    action space makes an edited node structurally deeper than an
    unedited seed, hence visit-capped by its own parent's N, so this is
    no longer something an untrained/near-random G0 reliably produces
    within a small simulation budget -- the mechanism itself still needs
    proving directly).
    """
    if len(candidates) < 2:
        raise AlphaZeroSelectionError(
            f"AlphaZero needs >= 2 LLM candidates to select from, got "
            f"{len(candidates)}")
    cfg = config or AlphaZeroConfig(seed=seed)
    reg = LLMSeededEditRegistry(candidates)
    root_state = build_root_state(spec, ctx_id, reg.seed_ids)
    nets = nets or load_alphazero_nets(value_ckpt, seed=0)
    root, mcts = _search_from_state(root_state, reg, reg.seed_ids, nets, cfg)

    by_hash: dict[str, dict] = {}
    for n in mcts.nodes:
        tid = n.state.topology_id
        h = n.state.graph_hash
        if tid == "proposal_root":
            tid = reg.seed_ids[0]
            h = reg.get_topology(tid).graph.structural_hash()
        if h not in by_hash:
            by_hash[h] = {"hash": h, "N": 0, "topology_id": tid,
                         "depth": n.state.depth, "edit_history": n.state.edit_history}
        by_hash[h]["N"] += n.N
        if n.state.depth < by_hash[h]["depth"]:
            by_hash[h]["depth"] = n.state.depth
            by_hash[h]["topology_id"] = tid
            by_hash[h]["edit_history"] = n.state.edit_history

    ranked = sorted(by_hash.values(), key=lambda r: (-r["N"], r["hash"]))
    if len(ranked) < 2:
        raise AlphaZeroSelectionError(
            f"AlphaZero search found only {len(ranked)} distinct topology "
            "state(s) -- cannot select two DISTINCT outputs. Hard failure "
            "by design: no fallback to root-PUCT, prior-only, or an "
            "arbitrary candidate.")
    top2 = ranked[:2]
    assert top2[0]["hash"] != top2[1]["hash"], "top-2 must be canonical-distinct"

    orig_by_id = {c["llm_proposal_id"]: c for c in candidates}

    def _to_candidate(entry: dict, rank: int) -> dict:
        tid = entry["topology_id"]
        seed_id = tid.split("~", 1)[0]
        is_edited = tid != seed_id
        seed_family = orig_by_id[seed_id]["canonical_family"]
        return {
            "llm_proposal_id": tid, "canonical_graph_hash": entry["hash"],
            # a never-before-seen label for an edited descendant, so any
            # family-conditioned surrogate/prediction naturally reports
            # UNKNOWN for it rather than silently reusing the SEED
            # family's learned statistics for a structurally different
            # graph (predict_post_sac already returns explicit UNKNOWNs
            # for an unrecognised family -- this relies on that existing
            # behavior rather than inventing new UNKNOWN-handling).
            "canonical_family": (f"{seed_family}~edited" if is_edited
                                 else seed_family),
            "obj": (orig_by_id[seed_id]["obj"] if not is_edited else None),
            "device_graph": reg._device_graphs[tid],
            "source": "alphazero", "rank": rank, "visit_count": entry["N"],
            "policy_prior": None, "selected_top2": True,
            "originating_seed_id": seed_id,
            "originating_seed_hash": reg.get_topology(seed_id).graph.structural_hash(),
            "edit_history": entry["edit_history"], "edit_depth": entry["depth"],
            "is_edited_descendant": is_edited}

    selected = [_to_candidate(e, i) for i, e in enumerate(top2)]
    return {
        "selected": selected,
        "ranked_all": [{"topology_id": e["topology_id"], "hash": e["hash"],
                        "N": e["N"], "depth": e["depth"]} for e in ranked],
        "tree_nodes": len(mcts.nodes),
        "max_depth_reached": max((n.state.depth for n in mcts.nodes), default=0),
        "seed_ids": reg.seed_ids,
        "seed_topology_hashes": sorted(
            reg.get_topology(sid).graph.structural_hash() for sid in reg.seed_ids),
        "search": "true_alphazero", "electrical_environment_version": "POST_CLOAD_FIX_V1"}


def direct_prior_select_two(candidates: list[dict], spec: dict) -> dict:
    """A5 NO_ALPHAZERO's baseline: the retired puct_select_two()'s own
    search="none" branch, extracted verbatim (not reimplemented) -- pure
    proposer/family-prior ranking, zero AlphaZero policy/value/search
    involvement, zero dependency on the retired root-selector machinery.
    Selects exactly two of the ORIGINAL validated LLM candidates (never
    an edited descendant -- there is no search here to produce one),
    matching FULL's MB-SAC count.
    """
    from run_puct_ablation import topology_priors
    if len(candidates) < 2:
        raise AlphaZeroSelectionError(
            f"A5 direct-selection needs >= 2 LLM candidates, got {len(candidates)}")
    priors = topology_priors(spec)
    ranked = sorted(candidates,
                    key=lambda c: (-(priors.get(c["canonical_family"]) or 0.0),
                                   c["canonical_graph_hash"]))
    for rank, c in enumerate(ranked):
        c["policy_prior"] = priors.get(c["canonical_family"])
        c["value_prediction"] = priors.get(c["canonical_family"]) or 0.0
        c["visit_count"] = 0
        c["rank"] = rank
        c["selected_top2"] = rank < 2
        c["device_graph"] = None
        c["source"] = c.get("source") or "llm"
    selected = ranked[:2]
    if selected[0]["canonical_graph_hash"] == selected[1]["canonical_graph_hash"]:
        raise AlphaZeroSelectionError("A5 selected the same graph twice")
    return {"selected": selected, "ranked": ranked, "search": "a5_direct_prior",
           "candidate_visits": {c["llm_proposal_id"]: 0 for c in candidates}}
