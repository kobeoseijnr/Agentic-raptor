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

import dataclasses
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


#: Stage 5D Section 6: replay schema versioning. The ORIGINAL AlphaZero
#: replay rows (build_replay_rows() before this change, e.g. Campaign 01
#: Retry's AZ_G1_REAL_replay.jsonl, 115 rows) never persisted the
#: DeviceCircuitGraph a row's pi/z were computed against -- only scalar/
#: structural summaries -- so they cannot be used to reconstruct the
#: exact policy/value input and are LEGACY_NONRECONSTRUCTABLE for
#: training purposes (still valid for the pure replay-metadata analyses
#: Stage 5C ran). New rows carry "replay_schema_version": "az_replay.2"
#: and a "state_graph" field; old rows have neither key at all.
AZ_REPLAY_SCHEMA_LEGACY = "az_replay.1_LEGACY_NONRECONSTRUCTABLE"
AZ_REPLAY_SCHEMA_GRAPH_COMPLETE = "az_replay.2"


#: AlphaZero improvement task, Part 2/16: action-schema versioning for the
#: PER-SEED architecture. AZ_ACTION_SCHEMA_SUPERROOT retroactively labels
#: the existing Option A scheme (generate_alphazero_actions() at
#: "proposal_root": SELECT_EXISTING_TOPOLOGY only; everywhere else: edits +
#: TERMINATE only) so search/replay/manifest records can declare which
#: scheme produced them without guessing from shape alone.
#:
#: AZ_ACTION_SCHEMA_PER_SEED requires NO new action-generation function:
#: generate_alphazero_actions()'s `at_super_root` branch is keyed on
#: `state.topology_id == "proposal_root"`, and a per-seed tree's root state
#: (build_seed_root_state()) is never given that sentinel id -- it is
#: rooted directly at a real seed's own topology_id from simulation zero.
#: SELECT_EXISTING_TOPOLOGY is therefore structurally unreachable in a
#: per-seed tree, not filtered out after being offered -- there is no
#: dormant SELECT action id anywhere in a per-seed policy head's live
#: action space. This constant exists purely so callers/replay rows/tests
#: can assert which regime produced a given search without re-deriving it
#: from context.
AZ_ACTION_SCHEMA_SUPERROOT = "az_actions.superroot.1"
AZ_ACTION_SCHEMA_PER_SEED = "az_actions.per_seed.1"


def serialize_device_graph(dg) -> dict:
    """DeviceCircuitGraph -> a plain JSON-safe dict. DeviceCircuitGraph
    and its nested DeviceRecord entries are both plain dataclasses with
    only str/int/list/dict fields (verified against agentic_raptor.
    mapping.DeviceCircuitGraph/DeviceRecord's definitions), so
    dataclasses.asdict() round-trips losslessly through json.dumps/loads
    -- no custom encoder needed."""
    from dataclasses import asdict
    return asdict(dg)


def deserialize_device_graph(d: dict):
    """The inverse of serialize_device_graph() -- rebuilds the real
    DeviceCircuitGraph/DeviceRecord dataclass instances (not just nested
    dicts), so the result is usable anywhere a live DeviceCircuitGraph is
    expected (device_graph_to_circuit_graph, apply_edit, sac_size, ...)."""
    from agentic_raptor.mapping import DeviceCircuitGraph, DeviceRecord
    devices = [DeviceRecord(**dr) for dr in d["devices"]]
    return DeviceCircuitGraph(
        topology_id=d["topology_id"], mapping_candidate_id=d["mapping_candidate_id"],
        stage_count=d["stage_count"], devices=devices, ports=d["ports"],
        support_bias=d["support_bias"], polarity=d["polarity"],
        block_assignments=d["block_assignments"], unresolved=d["unresolved"])


def graph_round_trip_hash_matches(dg, tid: str) -> tuple[bool, str, str]:
    """Section 5's mandatory proof: serialize -> deserialize -> recompute
    the canonical structural hash -> compare against the ORIGINAL graph's
    own hash. Returns (matches, original_hash, reconstructed_hash)."""
    from agentic_raptor.topology_rl.value_refresh import \
        device_graph_to_circuit_graph
    original_hash = device_graph_to_circuit_graph(dg, tid).structural_hash()
    reconstructed = deserialize_device_graph(serialize_device_graph(dg))
    reconstructed_hash = device_graph_to_circuit_graph(reconstructed, tid).structural_hash()
    return original_hash == reconstructed_hash, original_hash, reconstructed_hash


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

    def __init__(self, candidates: list[dict], edit_templates: dict | None = None):
        from run_puct_ablation import _realise
        self._device_graphs: dict[str, Any] = {}
        self._entries: dict[str, _Entry] = {}
        self.seed_ids: list[str] = []
        #: EDIT-OPERATOR REPAIR (2026-08-13): optional EXPERIMENTAL operator
        #: template override -- None (default) = live EDIT_TEMPLATES,
        #: byte-identical behavior for every existing caller.
        self._edit_templates = edit_templates
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

    def get_device_graph(self, tid: str):
        """Stage 5D Section 4: the real DeviceCircuitGraph backing `tid`
        (resolving "proposal_root" the same way get_topology() does) --
        the object build_replay_rows() serialises into graph-complete
        replay rows."""
        return self._device_graphs[self._resolve(tid)]

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
        ndg, _audit = apply_edit(parent_dg, edit_type, templates=self._edit_templates)
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


def _convert_spec(spec: dict) -> dict:
    """The AlphaZero-internal spec representation, shared by every root-
    state constructor (super-root and per-seed alike) so the two
    architectures are never accidentally fed differently-shaped specs."""
    return {"target_gain_db": spec["gain_target_db"],
           "target_gbw_hz": spec.get("ugbw_target_hz") or 1e4,
           "minimum_phase_margin_deg": spec["phase_margin_target_deg"],
           "load_capacitance_f": spec["load_capacitance_pf"] * 1e-12,
           "supply_voltage": 1.8}


def build_root_state(spec: dict, ctx_id: str, seed_ids: list[str],
                     search_budget: int = 8) -> TopologySearchState:
    return TopologySearchState(
        topology_id="proposal_root", graph_hash="proposal_root", lineage=[],
        spec=_convert_spec(spec),
        rag_context_ids=[ctx_id], available_blocks=[], legal_action_ids=[],
        edit_history=[], validation_status="validated",
        structural_features={"n_nodes": 0.0}, previous_evidence_ref=None,
        remaining_search_budget=search_budget, remaining_spice_budget=0, depth=0)


def build_seed_root_state(spec: dict, ctx_id: str, seed_id: str,
                          reg: "LLMSeededEditRegistry",
                          search_budget: int = 8) -> TopologySearchState:
    """PER-SEED MCTS root: unlike build_root_state()'s neutral
    "proposal_root" sentinel (which borrows seed_ids[0]'s graph only
    because ONE shared tree needs a single starting graph_hash), this
    root state genuinely IS one specific validated seed -- topology_id
    and graph_hash are that seed's own real identity from the first
    simulation. Because topology_id != "proposal_root",
    generate_alphazero_actions()'s `at_super_root` branch is never
    entered for this tree: SELECT_EXISTING_TOPOLOGY is structurally
    unreachable from turn zero, not filtered after the fact. No new
    action-generation function was needed for this reason (see
    AZ_ACTION_SCHEMA_PER_SEED's docstring)."""
    g = reg.get_topology(seed_id).graph
    h = g.structural_hash()
    return TopologySearchState(
        topology_id=seed_id, graph_hash=h, lineage=[h],
        spec=_convert_spec(spec),
        rag_context_ids=[ctx_id], available_blocks=[], legal_action_ids=[],
        edit_history=[], validation_status="validated",
        structural_features={"n_nodes": float(len(g.nodes))},
        previous_evidence_ref=None,
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


def stable_mcts_search_seed(campaign_seed: int, generation_id: str, spec_hash: str,
                            rollout_seed: int, decision_step_index: int) -> int:
    """Stage 5D, Section 1: fixes the SECOND RNG-reuse bug Stage 5C found
    (distinct from stable_episode_rng_seed(), which only covers the
    python `random` action-SAMPLING seed). stage3e1.TopologyMCTS seeds
    its OWN numpy RNG for Dirichlet root-exploration noise from `cfg.
    seed` (`np.random.default_rng(cfg.seed)`); because a fresh
    TopologyMCTS is built for EVERY real step of EVERY episode but all
    were handed the SAME AlphaZeroConfig object with a single fixed
    `.seed`, every search's exploration-noise draw restarted from an
    IDENTICAL numpy RNG state -- confirmed empirically in Campaign 01
    Retry's real replay: a_sel_p00's root target pi was bit-identical
    (0.2421875) in 32/32 real episodes.

    This derives a distinct seed per (campaign, generation, spec,
    rollout, decision step) via SHA-256 (never Python's process-salted
    built-in hash()) -- the same construction as stable_episode_rng_seed,
    with one extra component (decision_step_index) so that even WITHIN
    one episode, each of its several real decision points gets its own
    exploration-noise stream, not just each episode as a whole.
    """
    import hashlib
    payload = (f"{campaign_seed}|{generation_id}|{spec_hash}|{rollout_seed}|"
              f"{decision_step_index}").encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def stable_per_seed_mcts_search_seed(campaign_seed: int, generation_id: str,
                                     spec_hash: str, rollout_seed: int,
                                     seed_topology_hash: str) -> int:
    """PER-SEED analog of stable_mcts_search_seed(). Deliberately a
    DISTINCT function, not stable_mcts_search_seed() with
    decision_step_index repurposed: the two existing RNG axes (episode
    action-sampling vs per-real-step MCTS exploration noise) already have
    a documented history of being accidentally conflated (see
    stable_mcts_search_seed's docstring) -- introducing a THIRD axis
    (which independent per-seed tree this is) by overloading an existing
    parameter would risk exactly that failure mode again. Differentiates
    by seed_topology_hash (not seed index/order) so the seed stream is
    tied to WHICH graph is being searched, not to an arbitrary position
    in a list that could be reordered between runs."""
    import hashlib
    payload = (f"{campaign_seed}|{generation_id}|{spec_hash}|{rollout_seed}|"
              f"per_seed|{seed_topology_hash}").encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def allocate_simulations_per_seed(total_budget: int, seed_ids: list[str], *,
                                  allocation_mode: str = "equal_per_seed") -> dict[str, int]:
    """Part 3, Section 6: total-compute-matched budget allocation across K
    independent seed trees. Only 'equal_per_seed' is implemented -- no
    learned allocator (explicitly deferred). Deterministic remainder
    handling: seed_ids are sorted first (a fixed, reproducible order, not
    insertion order), and the first `total_budget % K` seeds in that
    sorted order each get one extra simulation, so
    sum(allocation.values()) == total_budget exactly, never silently
    dropping or fabricating simulations."""
    if allocation_mode != "equal_per_seed":
        raise ValueError(
            f"unsupported allocation_mode {allocation_mode!r} -- only "
            f"'equal_per_seed' is implemented (Section 6: no learned "
            f"allocator yet)")
    ordered = sorted(seed_ids)
    k = len(ordered)
    if k == 0:
        return {}
    base, remainder = divmod(total_budget, k)
    return {sid: base + (1 if i < remainder else 0) for i, sid in enumerate(ordered)}


@dataclass
class PerSeedAlphaZeroConfig:
    """PER-SEED MCTS configuration (Part 2/3/6). Mirrors AlphaZeroConfig's
    field names/defaults for every field it shares, but replaces
    `alphazero_simulations_per_move` (one shared-tree budget) with
    `total_mcts_simulations` + `allocation_mode` (Section 6: how that
    total is split across the K independent seed trees)."""
    total_mcts_simulations: int = 128
    allocation_mode: str = "equal_per_seed"
    alphazero_c_puct: float = 1.5
    alphazero_max_edit_depth: int = 4
    alphazero_max_children: int = 12
    alphazero_temperature: float = 1.0
    seed: int = 0
    training_mode: bool = False
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.30
    leaf_mode: str = "value_only"


def _build_per_seed_trees(candidates: list[dict], spec: dict, ctx_id: str, *,
                          value_ckpt=None, seed: int = 0,
                          config: PerSeedAlphaZeroConfig | None = None,
                          nets: dict | None = None, campaign_seed: int = 0,
                          generation_id: str = "AZ_G0",
                          spec_hash: str | None = None,
                          edit_templates: dict | None = None):
    """Shared tree-construction core for BOTH per-seed output selectors
    (summed-visits and principal-variation): identical registries, RNG
    streams, budgets, and search dynamics -- so a selector comparison can
    never be confounded by a difference in how the trees themselves were
    built."""
    from agentic_raptor.topology_rl import stage3e1 as s1

    if len(candidates) < 2:
        raise AlphaZeroSelectionError(
            f"per-seed AlphaZero needs >= 2 LLM candidates, got {len(candidates)}")
    cfg = config or PerSeedAlphaZeroConfig(seed=seed)
    reg = LLMSeededEditRegistry(candidates, edit_templates=edit_templates)
    nets = nets or load_alphazero_nets(value_ckpt, seed=0)
    ordered_seed_ids = sorted(reg.seed_ids)
    budgets = allocate_simulations_per_seed(
        cfg.total_mcts_simulations, ordered_seed_ids, allocation_mode=cfg.allocation_mode)

    per_seed_results = {}
    for sid in ordered_seed_ids:
        seed_hash = reg.get_topology(sid).graph.structural_hash()
        tree_seed = stable_per_seed_mcts_search_seed(
            campaign_seed, generation_id, spec_hash or ctx_id, seed, seed_hash)
        search_cfg = s1.SearchConfig(
            num_simulations=budgets[sid], c_puct=cfg.alphazero_c_puct,
            max_depth=cfg.alphazero_max_edit_depth, max_children=cfg.alphazero_max_children,
            leaf_mode=cfg.leaf_mode, training_mode=cfg.training_mode,
            root_noise_eps=cfg.dirichlet_epsilon, root_dirichlet_alpha=cfg.dirichlet_alpha,
            seed=tree_seed)
        # each seed gets its OWN TopologyMCTS instance -- independent
        # .nodes, .rng, .leaf_cache, .costs by construction (fresh
        # __init__ per call, Part 3 Section 7)
        mcts = s1.TopologyMCTS(nets, reg, ordered_seed_ids, search_cfg,
                               generate_actions_fn=generate_alphazero_actions,
                               apply_action_fn=apply_alphazero_action)
        root_state = build_seed_root_state(spec, ctx_id, sid, reg, search_budget=8)
        root = mcts.run(root_state)
        per_seed_results[sid] = {"root": root, "mcts": mcts, "tree_seed": tree_seed,
                                 "n_simulations_allocated": budgets[sid]}
    return cfg, reg, nets, ordered_seed_ids, budgets, per_seed_results


def run_per_seed_alphazero_search(candidates: list[dict], spec: dict, ctx_id: str, *,
                                  value_ckpt=None, seed: int = 0,
                                  config: PerSeedAlphaZeroConfig | None = None,
                                  nets: dict | None = None,
                                  campaign_seed: int = 0,
                                  generation_id: str = "AZ_G0",
                                  spec_hash: str | None = None) -> dict:
    """PART 2/3/4: the candidate per-seed architecture. Every validated
    LLM seed becomes an INDEPENDENT structural-edit-only MCTS root (no
    SELECT action anywhere -- see AZ_ACTION_SCHEMA_PER_SEED), each with
    its own Node tree, visit counts, leaf cache, and Dirichlet/RNG stream
    (a fresh agentic_raptor.topology_rl.stage3e1.TopologyMCTS instance per
    seed already allocates all of that independently; the only shared
    object across trees is the read-only LLMSeededEditRegistry and the
    policy/value network being scored against, both intentionally shared
    -- editing seed A's graph must never mutate seed B's).

    Same external contract as alphazero_select_two()/run_alphazero_search:
    validated LLM candidates in, ranked/pooled candidates out, ready for
    the SAME downstream top-2 selection contract.

    FINAL PERFORMANCE REPAIR, Phase 1 (2026-08-12): the summed-visit-count
    top-2 rule below was audited and classified
    OUTPUT_SELECTION_DEPTH_BIAS_CONFIRMED -- every backup increments the
    root's N, so a depth-0 original seed always outranks every edited
    descendant of its own tree (max descendant summed-N 16 vs min selected
    root N 43 across the 8-spec audit). This function is retained
    UNCHANGED as the audited historical baseline
    (PER_SEED_OLD_SELECTION); run_per_seed_alphazero_search_pv() below is
    the repaired selector.
    """
    cfg, reg, nets, ordered_seed_ids, budgets, per_seed_results = _build_per_seed_trees(
        candidates, spec, ctx_id, value_ckpt=value_ckpt, seed=seed, config=config,
        nets=nets, campaign_seed=campaign_seed, generation_id=generation_id,
        spec_hash=spec_hash)

    # Part 4: pool descendants across ALL independent trees, tracking
    # originating seed / edit path / depth / visit / value / terminal /
    # canonical hash for every distinct topology state found anywhere.
    by_hash: dict[str, dict] = {}
    for sid, r in per_seed_results.items():
        for n in r["mcts"].nodes:
            h = n.state.graph_hash
            if h not in by_hash:
                by_hash[h] = {"hash": h, "N": 0, "topology_id": n.state.topology_id,
                             "originating_seed_id": sid, "depth": n.state.depth,
                             "edit_history": n.state.edit_history,
                             "terminal_reason": n.terminal_reason, "best_Q": n.Q}
            by_hash[h]["N"] += n.N
            by_hash[h]["best_Q"] = max(by_hash[h]["best_Q"], n.Q)
            if n.state.depth < by_hash[h]["depth"]:
                by_hash[h]["depth"] = n.state.depth
                by_hash[h]["topology_id"] = n.state.topology_id
                by_hash[h]["originating_seed_id"] = sid
                by_hash[h]["edit_history"] = n.state.edit_history
                by_hash[h]["terminal_reason"] = n.terminal_reason

    # Part 4, Section 10: FROZEN top-2 rule -- IDENTICAL to
    # alphazero_select_two()'s (rank by summed visit count N per canonical
    # hash, ties broken by hash string) so a super-root vs per-seed
    # comparison isolates the search-topology change, not a confounding
    # difference in selection rule. Search-internal evidence only (N) --
    # never MB-SAC/SPICE/DPO. Decided here, before any electrical result
    # from this architecture has ever been observed.
    ranked = sorted(by_hash.values(), key=lambda r: (-r["N"], r["hash"]))
    if len(ranked) < 2:
        raise AlphaZeroSelectionError(
            f"per-seed AlphaZero search found only {len(ranked)} distinct "
            "topology state(s) across all seed trees -- cannot select two "
            "DISTINCT outputs. Hard failure by design: no fallback.")
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
            "canonical_family": (f"{seed_family}~edited" if is_edited else seed_family),
            "obj": (orig_by_id[seed_id]["obj"] if not is_edited else None),
            "device_graph": reg._device_graphs[tid],
            "source": "alphazero_per_seed", "rank": rank, "visit_count": entry["N"],
            "policy_prior": None, "selected_top2": True,
            "originating_seed_id": seed_id,
            "originating_seed_hash": reg.get_topology(seed_id).graph.structural_hash(),
            "edit_history": entry["edit_history"], "edit_depth": entry["depth"],
            "is_edited_descendant": is_edited}

    selected = [_to_candidate(e, i) for i, e in enumerate(top2)]
    all_nodes = [n for r in per_seed_results.values() for n in r["mcts"].nodes]
    return {
        "selected": selected,
        "ranked_all": [{"topology_id": e["topology_id"], "hash": e["hash"], "N": e["N"],
                        "depth": e["depth"], "originating_seed_id": e["originating_seed_id"]}
                       for e in ranked],
        "tree_nodes": len(all_nodes),
        "per_seed_tree_nodes": {sid: len(r["mcts"].nodes) for sid, r in per_seed_results.items()},
        "per_seed_simulations_allocated": budgets,
        "per_seed_tree_seeds": {sid: r["tree_seed"] for sid, r in per_seed_results.items()},
        "per_seed_costs": {sid: dataclasses.asdict(r["mcts"].costs)
                           for sid, r in per_seed_results.items()},
        "max_depth_reached": max((n.state.depth for n in all_nodes), default=0),
        "seed_ids": ordered_seed_ids,
        "seed_topology_hashes": sorted(
            reg.get_topology(sid).graph.structural_hash() for sid in ordered_seed_ids),
        "total_mcts_simulations": cfg.total_mcts_simulations,
        "allocation_mode": cfg.allocation_mode,
        "search": "true_alphazero_per_seed",
        "action_schema": AZ_ACTION_SCHEMA_PER_SEED,
        "electrical_environment_version": "POST_CLOAD_FIX_V1",
        "nodes": {sid: [n.record() for n in r["mcts"].nodes]
                 for sid, r in per_seed_results.items()},
    }


#: FINAL PERFORMANCE REPAIR Phase 2: selection-rule version tags, so every
#: search result/trace/replay row can declare WHICH output-selection rule
#: produced its top-2 without shape-guessing.
AZ_SELECTION_RULE_SUMMED_VISITS = "az_top2.summed_visits.1"
AZ_SELECTION_RULE_PRINCIPAL_VARIATION = "az_top2.principal_variation.1"
AZ_SELECTION_RULE_PV_PORTFOLIO = "az_top2.pv_portfolio.1"


def extract_principal_variation(root_node) -> dict:
    """Phase 2 Section 5: follow the highest-visit legal action from the
    tree root downward (ties by lowest node_id -- the SAME tie-break
    stage3e1.search_result()/visit_policy(tau<=1e-3) already use) until
    TERMINATE, an unexpanded frontier, or no children remain. Returns the
    principal-variation terminal/output state plus full path provenance.

    If the root's most-visited action is TERMINATE, the PV ends
    immediately and the output graph is the unchanged seed -- a legitimate
    outcome, never penalized. If an edit has the highest visit support,
    the output becomes that edited descendant. This is MCTS's own
    decision semantics (argmax-N action selection, exactly what
    run_alphazero_episode's deterministic mode plays), applied to output
    extraction -- instead of ranking every visited state by raw
    accumulated N, which Phase 1 proved can never surface a descendant.
    """
    path_nodes = []
    node = root_node
    while node.expanded and node.children and node.terminal_reason is None:
        node = max(node.children, key=lambda c: (c.N, -c.node_id))
        path_nodes.append(node)
    root_children_n = sum(c.N for c in root_node.children) or 1
    first = path_nodes[0] if path_nodes else None
    return {
        "output_node": node,
        "output_hash": (node.state.graph_hash if node is not root_node
                        else root_node.state.graph_hash),
        "output_topology_id": node.state.topology_id,
        "pv_action_ids": [n.action.action_id for n in path_nodes],
        "pv_length": len(path_nodes),
        "root_action_id": first.action.action_id if first else None,
        "root_action_is_terminate": (
            first is not None
            and first.action.action_type == Stage3E1ActionType.TERMINATE_SEARCH),
        "root_action_visit_fraction": (round(first.N / root_children_n, 4)
                                       if first else None),
        "pv_root_decision_Q": (round(first.Q, 6) if first else None),
        "output_depth": node.state.depth,
        "output_terminal_reason": node.terminal_reason,
        "output_edit_history": node.state.edit_history,
    }


def run_per_seed_alphazero_search_pv(candidates: list[dict], spec: dict, ctx_id: str, *,
                                     value_ckpt=None, seed: int = 0,
                                     config: PerSeedAlphaZeroConfig | None = None,
                                     nets: dict | None = None,
                                     campaign_seed: int = 0,
                                     generation_id: str = "AZ_G0",
                                     spec_hash: str | None = None,
                                     portfolio: bool = False,
                                     edit_templates: dict | None = None) -> dict:
    """PER_SEED_REPAIRED_SELECTION (Phase 2, Sections 5-7): identical
    per-seed tree construction to run_per_seed_alphazero_search (shared
    _build_per_seed_trees -- same registries, RNG streams, budgets), but
    output extraction follows MCTS decision semantics instead of raw
    accumulated visit counts:

      1. ONE primary candidate per seed tree: the tree's
         PRINCIPAL-VARIATION output state (extract_principal_variation --
         argmax-N action traversal from the seed root; a root-level
         TERMINATE legitimately yields the unchanged seed).
      2. Cross-seed ranking -- PREDECLARED (frozen 2026-08-12, before any
         electrical evaluation of this rule), scale-consistent (never
         compares raw visit counts across trees):
           a. backed-up Q of the principal root decision, descending
           b. root-action visit fraction, descending (tie-break)
           c. canonical hash string, ascending (deterministic tie-break)
      3. Top-2 distinct canonical hashes from that ranking. If the K
         primary candidates yield < 2 distinct hashes (duplicate seed
         graphs converging), each tree's root SECOND-most-visited child's
         own PV is appended as a secondary candidate pool, ranked by the
         same rule; hard-fails (AlphaZeroSelectionError) only if still
         < 2 distinct.

    No novelty preference of any kind: an edited descendant wins a slot
    only when MCTS's own argmax-N decision path leads to it. No MB-SAC /
    fresh-SPICE / PVT / DPO signal is consulted.
    """
    cfg, reg, nets, ordered_seed_ids, budgets, per_seed_results = _build_per_seed_trees(
        candidates, spec, ctx_id, value_ckpt=value_ckpt, seed=seed, config=config,
        nets=nets, campaign_seed=campaign_seed, generation_id=generation_id,
        spec_hash=spec_hash, edit_templates=edit_templates)

    def _pv_entry(sid: str, pv: dict, kind: str) -> dict:
        return {"originating_seed_id": sid, "kind": kind, **{
            k: pv[k] for k in ("output_hash", "output_topology_id", "pv_action_ids",
                              "pv_length", "root_action_id", "root_action_is_terminate",
                              "root_action_visit_fraction", "pv_root_decision_Q",
                              "output_depth", "output_terminal_reason",
                              "output_edit_history")}}

    primaries = []
    secondaries = []
    for sid in ordered_seed_ids:
        root = per_seed_results[sid]["root"]
        pv = extract_principal_variation(root)
        primaries.append(_pv_entry(sid, pv, "primary"))
        # secondary: the root's second-most-visited child's own PV (only
        # used if primaries alone cannot supply 2 distinct hashes)
        ranked_children = sorted(root.children, key=lambda c: (-c.N, c.node_id))
        if len(ranked_children) >= 2:
            second = ranked_children[1]
            sub_pv = extract_principal_variation(second)
            entry = _pv_entry(sid, sub_pv, "secondary")
            # provenance: the root decision for a secondary is the SECOND
            # child itself, so its Q/fraction describe that real decision
            root_children_n = sum(c.N for c in root.children) or 1
            entry["root_action_id"] = second.action.action_id
            entry["root_action_is_terminate"] = (
                second.action.action_type == Stage3E1ActionType.TERMINATE_SEARCH)
            entry["root_action_visit_fraction"] = round(second.N / root_children_n, 4)
            entry["pv_root_decision_Q"] = round(second.Q, 6)
            entry["pv_action_ids"] = [second.action.action_id] + entry["pv_action_ids"]
            entry["pv_length"] = len(entry["pv_action_ids"])
            secondaries.append(entry)

    rank_key = lambda e: (-(e["pv_root_decision_Q"] if e["pv_root_decision_Q"] is not None
                            else -10.0),
                          -(e["root_action_visit_fraction"] or 0.0),
                          e["output_hash"])   # noqa: E731
    primaries.sort(key=rank_key)
    secondaries.sort(key=rank_key)

    if portfolio:
        # CHAMPION + CHALLENGER (portfolio) selection -- 2026-08-13, from
        # the combined-package post-mortem: on the one real electrical loss
        # (spec 2), BOTH top-2 slots had gone to edited candidates, so when
        # the downstream selector picked the broken sibling there was no
        # safe fallback -- even though the other slot's edited chain
        # actually PASSED. The two-candidate pipeline exists to hedge;
        # correlated picks waste the hedge.
        #   slot 0 (CHAMPION): the UNCHANGED seed whose root state the
        #     current value function scores highest -- always a known
        #     family, so downstream hard-gate/DPO get real predictions;
        #   slot 1 (CHALLENGER): the top-ranked PV candidate (edited or
        #     not) with a distinct canonical hash.
        import torch as _torch
        seed_entries = []
        for sid in ordered_seed_ids:
            root = per_seed_results[sid]["root"]
            with _torch.no_grad():
                v = float(nets["value_forward"](root.state, reg)["scalar"])
            seed_entries.append({"originating_seed_id": sid, "kind": "champion_seed",
                                "output_hash": root.state.graph_hash,
                                "output_topology_id": sid,
                                "pv_action_ids": [], "pv_length": 0,
                                "root_action_id": None,
                                "root_action_is_terminate": None,
                                "root_action_visit_fraction": None,
                                "pv_root_decision_Q": round(v, 6),
                                "output_depth": 0, "output_terminal_reason": None,
                                "output_edit_history": []})
        seed_entries.sort(key=lambda e: (-e["pv_root_decision_Q"], e["output_hash"]))
        champion = seed_entries[0]
        chosen = [champion]
        seen_hashes = {champion["output_hash"]}
        for pool in (primaries, secondaries, seed_entries[1:]):
            for e in pool:
                if e["output_hash"] not in seen_hashes:
                    chosen.append(e)
                    seen_hashes.add(e["output_hash"])
                    break
            if len(chosen) >= 2:
                break
    else:
        chosen, seen_hashes = [], set()
        for pool in (primaries, secondaries):
            for e in pool:
                if e["output_hash"] not in seen_hashes:
                    chosen.append(e)
                    seen_hashes.add(e["output_hash"])
                if len(chosen) >= 2:
                    break
            if len(chosen) >= 2:
                break
    if len(chosen) < 2:
        raise AlphaZeroSelectionError(
            f"per-seed PV selection found only {len(chosen)} distinct output "
            "hash(es) across all seed trees (primaries + secondaries) -- "
            "cannot select two DISTINCT outputs. Hard failure by design.")
    assert chosen[0]["output_hash"] != chosen[1]["output_hash"]

    orig_by_id = {c["llm_proposal_id"]: c for c in candidates}

    def _to_candidate(entry: dict, rank: int) -> dict:
        tid = entry["output_topology_id"]
        seed_id = tid.split("~", 1)[0]
        is_edited = tid != seed_id
        seed_family = orig_by_id[seed_id]["canonical_family"]
        return {
            "llm_proposal_id": tid, "canonical_graph_hash": entry["output_hash"],
            "canonical_family": (f"{seed_family}~edited" if is_edited else seed_family),
            "obj": (orig_by_id[seed_id]["obj"] if not is_edited else None),
            "device_graph": reg._device_graphs[tid],
            "source": "alphazero_per_seed_pv", "rank": rank,
            "visit_count": None,   # deliberately not a cross-tree N -- see selection_rule
            "pv_root_decision_Q": entry["pv_root_decision_Q"],
            "root_action_visit_fraction": entry["root_action_visit_fraction"],
            "pv_action_ids": entry["pv_action_ids"],
            "policy_prior": None, "selected_top2": True,
            "originating_seed_id": seed_id,
            "originating_seed_hash": reg.get_topology(seed_id).graph.structural_hash(),
            "edit_history": entry["output_edit_history"],
            "edit_depth": entry["output_depth"],
            "is_edited_descendant": is_edited}

    selected = [_to_candidate(e, i) for i, e in enumerate(chosen)]
    all_nodes = [n for r in per_seed_results.values() for n in r["mcts"].nodes]
    return {
        "selected": selected,
        "pv_primaries": primaries, "pv_secondaries_used": len(chosen) > 0 and any(
            c["kind"] == "secondary" for c in chosen),
        "ranked_all": [{k: e[k] for k in ("originating_seed_id", "kind", "output_hash",
                                          "output_depth", "pv_root_decision_Q",
                                          "root_action_visit_fraction",
                                          "root_action_is_terminate")}
                       for e in primaries + secondaries],
        "tree_nodes": len(all_nodes),
        "per_seed_tree_nodes": {sid: len(r["mcts"].nodes) for sid, r in per_seed_results.items()},
        "per_seed_simulations_allocated": budgets,
        "per_seed_tree_seeds": {sid: r["tree_seed"] for sid, r in per_seed_results.items()},
        "per_seed_costs": {sid: dataclasses.asdict(r["mcts"].costs)
                           for sid, r in per_seed_results.items()},
        "max_depth_reached": max((n.state.depth for n in all_nodes), default=0),
        "seed_ids": ordered_seed_ids,
        "seed_topology_hashes": sorted(
            reg.get_topology(sid).graph.structural_hash() for sid in ordered_seed_ids),
        "total_mcts_simulations": cfg.total_mcts_simulations,
        "allocation_mode": cfg.allocation_mode,
        "search": ("true_alphazero_per_seed_pv_portfolio" if portfolio
                   else "true_alphazero_per_seed_pv"),
        "action_schema": AZ_ACTION_SCHEMA_PER_SEED,
        "selection_rule": (AZ_SELECTION_RULE_PV_PORTFOLIO if portfolio
                           else AZ_SELECTION_RULE_PRINCIPAL_VARIATION),
        "electrical_environment_version": "POST_CLOAD_FIX_V1",
        "nodes": {sid: [n.record() for n in r["mcts"].nodes]
                 for sid, r in per_seed_results.items()},
    }


#: Phase 5: the per-seed-native action-type vocabulary -- the 8 real
#: structural edits plus TERMINATE, nothing else. Deterministic order
#: (edits sorted by enum value, TERMINATE last). SELECT_EXISTING_TOPOLOGY
#: and KEEP_TOPOLOGY have NO slot at all in a per-seed-native policy head.
AZ_PER_SEED_ACTION_TYPES: tuple = tuple(
    sorted(AZ_EDIT_ACTION_TYPES, key=lambda t: t.value)
) + (Stage3E1ActionType.TERMINATE_SEARCH,)


class PerSeedActionNotSupported(Exception):
    """A per-seed-native policy head was asked to score an action type
    outside AZ_PER_SEED_ACTION_TYPES (e.g. a SELECT action) -- hard-fail,
    never silently reinterpret an action index."""


def build_per_seed_policy_value(seed: int = 0):
    """Phase 5: a COMPLETELY FRESH policy/value network natively compatible
    with az_actions.per_seed.1.

      - policy head one-hot spans ONLY AZ_PER_SEED_ACTION_TYPES (9 types:
        8 edits + TERMINATE) -- structurally no SELECT/KEEP slot exists
      - fresh random policy head, fresh random value head, fresh encoder
      - no rejected value bootstrap, no retired root-PUCT weights, no
        hidden checkpoint reuse (torch.manual_seed + fresh construction)
      - same graph/spec encoder ARCHITECTURE as stage3e1.build_policy_value
        (build_mp_conditioner) -- the schema change requires only the
        policy head's action-type one-hot width to shrink (the SELECT
        aemb pathway also disappears: per-seed actions never reference
        another topology, so aemb is structurally zero and is dropped)

    Same {policy_forward, value_forward, params, encoder, heads} contract
    as stage3e1.build_policy_value(), so TopologyMCTS consumes either
    interchangeably."""
    from agentic_raptor.utils.seeding import apply_torch_omp_workaround
    apply_torch_omp_workaround()
    import itertools
    import math as _math

    import torch

    from agentic_raptor.mb_sac.stage3d2 import build_mp_conditioner

    torch.manual_seed(seed)
    encoder, embed = build_mp_conditioner()
    n_types = len(AZ_PER_SEED_ACTION_TYPES)
    type_index = {t: i for i, t in enumerate(AZ_PER_SEED_ACTION_TYPES)}

    class PerSeedHeads(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ctx = 16 + 5 + 3
            self.policy = torch.nn.Sequential(
                torch.nn.Linear(ctx + n_types, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 1))
            self.value = torch.nn.Sequential(
                torch.nn.Linear(ctx, 64), torch.nn.ReLU())
            self.v_scalar = torch.nn.Linear(64, 1)
            self.v_aux = torch.nn.Linear(64, 4)

    heads = PerSeedHeads()
    params = list(itertools.chain(encoder.parameters(), heads.parameters()))

    def spec_vec(spec):
        return torch.tensor([spec.get("target_gain_db", 40) / 100,
                             _math.log10(max(spec.get("target_gbw_hz", 1e4), 1)) / 10,
                             spec.get("minimum_phase_margin_deg", 45) / 90,
                             spec.get("load_capacitance_f", 5e-10) * 1e12 / 1000,
                             spec.get("supply_voltage", 1.8) / 5])

    def ctx_vec(graph, spec, budget3):
        _, gemb = embed(graph)
        return torch.cat([gemb, spec_vec(spec), torch.tensor(budget3, dtype=torch.float32)])

    def policy_forward(state, actions, reg):
        actions = sorted(actions, key=lambda a: a.action_id)
        for a in actions:
            if a.action_type not in type_index:
                raise PerSeedActionNotSupported(
                    f"per-seed-native policy head has no slot for "
                    f"{a.action_type.value!r} (action_id={a.action_id!r})")
        ctx = ctx_vec(reg.get_topology(state.topology_id).graph, state.spec,
                      [state.remaining_search_budget / 8.0,
                       state.remaining_spice_budget / 8.0, state.depth / 4.0])
        logits = []
        for a in actions:
            onehot = torch.zeros(n_types)
            onehot[type_index[a.action_type]] = 1.0
            logits.append(heads.policy(torch.cat([ctx, onehot])))
        lg = torch.cat(logits)
        return actions, lg, torch.softmax(lg, dim=0)

    def value_forward(state, reg):
        ctx = ctx_vec(reg.get_topology(state.topology_id).graph, state.spec,
                      [state.remaining_search_budget / 8.0,
                       state.remaining_spice_budget / 8.0, state.depth / 4.0])
        hdn = heads.value(ctx)
        aux = heads.v_aux(hdn)
        return {"scalar": torch.tanh(heads.v_scalar(hdn))[0],
                "feasibility_logit_uncalibrated": aux[0],
                "stability_logit_uncalibrated": aux[1],
                "expected_spice_cost": torch.relu(aux[2]),
                "budget_exhaustion_logit": aux[3]}

    return {"encoder": encoder, "heads": heads, "params": params,
            "policy_forward": policy_forward, "value_forward": value_forward,
            "encoder_mode": "fresh_per_seed_native",
            "action_schema": AZ_ACTION_SCHEMA_PER_SEED,
            "n_action_types": n_types}


def ensure_az_per_seed_g0_manifest() -> dict:
    """Phase 5 Section 15: AZ_PER_SEED_G0 -- a fresh, UNTRAINED per-seed-
    native baseline. checkpoint_status=SMOKE, never PROMOTED: its purpose
    is the H2 distribution-mismatch test, not live deployment."""
    from agentic_raptor.topology_rl import stage3e1 as s1
    existing = read_az_generation_manifest("AZ_PER_SEED_G0")
    if existing is not None:
        return existing
    nets = build_per_seed_policy_value(seed=0)
    ck_path = az_generation_dir("AZ_PER_SEED_G0") / "checkpoint" / "policy_value.pt"
    meta = {"generation_id": "AZ_PER_SEED_G0", "parent_generation": None,
           "initialization": "random_untrained_per_seed_native",
           "action_schema": AZ_ACTION_SCHEMA_PER_SEED,
           "n_action_types": nets["n_action_types"],
           "note": ("fresh per-seed-native policy/value network: policy head "
                    "spans ONLY the 8 edit types + TERMINATE (no SELECT/KEEP "
                    "slot exists structurally). NOT the rejected value "
                    "bootstrap, NOT retired root-PUCT weights, NOT any "
                    "super-root checkpoint.")}
    s1.save_checkpoint(nets, ck_path, meta)
    manifest = {**meta, "checkpoint_path": str(ck_path), "immutable": True,
               "rejected": False, "checkpoint_status": "SMOKE",
               "checkpoint_status_reason": "randomly-initialised, zero training"}
    write_az_generation_manifest("AZ_PER_SEED_G0", manifest)
    return manifest


def load_per_seed_nets(checkpoint_path=None, seed: int = 0):
    """Load a per-seed-native network (build_per_seed_policy_value
    architecture). checkpoint_path=None -> fresh random."""
    from pathlib import Path as _Path

    from agentic_raptor.topology_rl import stage3e1 as s1
    nets = build_per_seed_policy_value(seed)
    if checkpoint_path is not None:
        ck = _Path(checkpoint_path)
        if ck.is_file():
            s1.load_checkpoint(nets, ck)
    return nets


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
    from dataclasses import asdict, replace as _dc_replace

    cfg = config or AlphaZeroConfig(seed=seed)
    reg = LLMSeededEditRegistry(candidates)
    nets = nets or load_alphazero_nets(value_ckpt, seed=0)
    episode_rng_seed = stable_episode_rng_seed(campaign_seed, generation_id,
                                               spec_hash, seed)
    rng = _random.Random(episode_rng_seed)

    state = build_root_state(spec, ctx_id, reg.seed_ids)
    steps = []
    mcts_search_seeds: list[int] = []
    for step_idx in range(max_episode_depth):
        # Stage 5D Section 1: a fresh, distinct exploration-noise seed per
        # real decision point -- see stable_mcts_search_seed()'s docstring
        # for the bug this replaces (every step of every episode
        # previously reused the SAME cfg.seed, so Dirichlet noise never
        # actually varied). Inert when cfg.training_mode is False (the
        # noise branch in TopologyMCTS._expand() is skipped entirely in
        # that case), so this is safe to apply unconditionally, including
        # during deterministic validation -- see test_alphazero.py's
        # reproducibility tests.
        step_seed = stable_mcts_search_seed(campaign_seed, generation_id,
                                            spec_hash, seed, step_idx)
        step_cfg = _dc_replace(cfg, seed=step_seed)
        mcts_search_seeds.append(step_seed)
        root, mcts = _search_from_state(state, reg, reg.seed_ids, nets, step_cfg)
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
            "tree_nodes_this_step": len(mcts.nodes),
            "mcts_search_seed": step_seed})
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
           "mcts_search_seeds": mcts_search_seeds,
           "deterministic": deterministic,
           "dirichlet_epsilon": cfg.dirichlet_epsilon if cfg.training_mode else None,
           "dirichlet_alpha": cfg.dirichlet_alpha if cfg.training_mode else None}


def run_per_seed_alphazero_episode(candidates: list[dict], spec: dict, ctx_id: str,
                                   spec_hash: str, *, spec_index: int | None = None,
                                   nets: dict, seed: int = 0, campaign_seed: int = 0,
                                   config: PerSeedAlphaZeroConfig | None = None,
                                   max_episode_depth: int = 4,
                                   deterministic: bool = False,
                                   generation_id: str = "AZ_PER_SEED_G0") -> dict:
    """Phase 7: one real per-seed self-play episode for training-data
    collection under az_actions.per_seed.1.

    Step 0: all K seed trees are built (shared _build_per_seed_trees --
    training_mode per config, so Dirichlet noise applies at each tree
    root during collection) and the frozen PV cross-seed rule picks the
    EPISODE'S REALIZED SEED (the top-ranked primary's originating tree).
    The realized tree's own root visit distribution becomes step 0's pi.
    Steps >= 1: a FRESH single-tree search from the current state (same
    per-tree simulation allocation, fresh per-step RNG stream via
    stable_mcts_search_seed) -- genuine sequential re-planning, exactly
    like run_alphazero_episode's convention, minus any SELECT action.

    Returns the same episode-dict shape run_alphazero_episode returns
    (steps/registry/terminal fields), so terminal_evaluation() and
    build_replay_rows() work unchanged; every replay row additionally
    carries action_schema=az_actions.per_seed.1.
    """
    import random as _random
    from dataclasses import asdict, replace as _dc_replace

    from agentic_raptor.topology_rl import stage3e1 as s1

    cfg = config or PerSeedAlphaZeroConfig(seed=seed, training_mode=True)
    episode_rng_seed = stable_episode_rng_seed(campaign_seed, generation_id,
                                               spec_hash, seed)
    rng = _random.Random(episode_rng_seed)

    cfg_all, reg, nets, ordered_seed_ids, budgets, per_seed_results = _build_per_seed_trees(
        candidates, spec, ctx_id, nets=nets, seed=seed, config=cfg,
        campaign_seed=campaign_seed, generation_id=generation_id, spec_hash=spec_hash)

    # frozen PV cross-seed rule picks the realized seed
    primaries = []
    for sid in ordered_seed_ids:
        pv = extract_principal_variation(per_seed_results[sid]["root"])
        primaries.append((sid, pv))
    primaries.sort(key=lambda e: (-(e[1]["pv_root_decision_Q"]
                                    if e[1]["pv_root_decision_Q"] is not None else -10.0),
                                  -(e[1]["root_action_visit_fraction"] or 0.0),
                                  e[1]["output_hash"]))
    realized_seed = primaries[0][0]
    per_step_sims = budgets[realized_seed]

    def _step_row(state, root, mcts, step_idx, step_seed):
        tau = 0.0 if deterministic else cfg.alphazero_temperature
        pi = visit_policy(root, tau=tau)
        if deterministic:
            chosen = max(root.children, key=lambda c: (c.N, -c.node_id))
        else:
            rr, acc, chosen = rng.random(), 0.0, root.children[-1]
            for c in root.children:
                acc += pi.get(c.action.action_id, 0.0)
                if rr <= acc:
                    chosen = c
                    break
        return chosen, {
            "step": step_idx, "state": asdict(state),
            "state_topology_id": state.topology_id,
            "state_graph_hash": (state.graph_hash if state.topology_id != "proposal_root"
                                 else reg.get_topology(reg.seed_ids[0]).graph.structural_hash()),
            "depth": state.depth, "edit_history": list(state.edit_history),
            "legal_action_ids": sorted(pi),
            "raw_visit_counts": {c.action.action_id: c.N for c in root.children},
            "pi": pi, "selected_action_id": chosen.action.action_id,
            "selected_action_type": chosen.action.action_type.value,
            "tree_nodes_this_step": len(mcts.nodes),
            "mcts_search_seed": step_seed}

    steps = []
    mcts_search_seeds = []
    # step 0: the realized tree's own root
    root0 = per_seed_results[realized_seed]["root"]
    state = root0.state
    if root0.children:
        chosen, row = _step_row(state, root0, per_seed_results[realized_seed]["mcts"],
                                0, per_seed_results[realized_seed]["tree_seed"])
        steps.append(row)
        mcts_search_seeds.append(per_seed_results[realized_seed]["tree_seed"])
        state = chosen.state
        terminated = chosen.action.action_type == Stage3E1ActionType.TERMINATE_SEARCH
    else:
        terminated = True

    # steps >= 1: sequential re-planning from the current state
    step_idx = 1
    while not terminated and step_idx < max_episode_depth:
        step_seed = stable_mcts_search_seed(campaign_seed, generation_id,
                                            spec_hash, seed, step_idx)
        scfg = s1.SearchConfig(
            num_simulations=per_step_sims, c_puct=cfg.alphazero_c_puct,
            max_depth=cfg.alphazero_max_edit_depth, max_children=cfg.alphazero_max_children,
            leaf_mode=cfg.leaf_mode, training_mode=cfg.training_mode,
            root_noise_eps=cfg.dirichlet_epsilon, root_dirichlet_alpha=cfg.dirichlet_alpha,
            seed=step_seed)
        mcts = s1.TopologyMCTS(nets, reg, ordered_seed_ids, scfg,
                               generate_actions_fn=generate_alphazero_actions,
                               apply_action_fn=apply_alphazero_action)
        root = mcts.run(state)
        if not root.children:
            break
        chosen, row = _step_row(state, root, mcts, step_idx, step_seed)
        steps.append(row)
        mcts_search_seeds.append(step_seed)
        state = chosen.state
        terminated = chosen.action.action_type == Stage3E1ActionType.TERMINATE_SEARCH
        step_idx += 1

    terminal_topology_id = state.topology_id
    terminal_topology_hash = reg.get_topology(terminal_topology_id).graph.structural_hash()
    step_hashes = {s["state_graph_hash"] for s in steps} | {terminal_topology_hash}
    seed_hashes = {reg.get_topology(sid).graph.structural_hash() for sid in reg.seed_ids}
    return {"ctx_id": ctx_id, "spec_hash": spec_hash, "spec_index": spec_index,
           "seed_ids": ordered_seed_ids, "seed_topology_hashes": sorted(seed_hashes),
           "realized_seed_id": realized_seed,
           "per_seed_simulations_allocated": budgets,
           "steps": steps, "terminal_topology_id": terminal_topology_id,
           "terminal_topology_hash": terminal_topology_hash,
           "terminal_came_from_edit": terminal_topology_id not in reg.seed_ids,
           "seed_novel_hashes": sorted(step_hashes - seed_hashes),
           "seed_novel_count": len(step_hashes - seed_hashes),
           "registry": reg, "generation_id": generation_id, "seed": seed,
           "campaign_seed": campaign_seed, "episode_rng_seed": episode_rng_seed,
           "mcts_search_seeds": mcts_search_seeds,
           "deterministic": deterministic,
           "action_schema": AZ_ACTION_SCHEMA_PER_SEED,
           "selection_rule": AZ_SELECTION_RULE_PRINCIPAL_VARIATION,
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


class GraphRoundTripError(Exception):
    """Stage 5D Section 5: build_replay_rows() refuses to write a row
    whose serialized-then-deserialized graph does not reproduce the
    ORIGINAL canonical structural hash -- a graph-complete replay row is
    worthless (silently wrong training data) if this doesn't hold."""


def build_replay_rows(episode: dict, terminal: dict, *,
                      checkpoint_hash: str | None = None,
                      protected_context_ids: frozenset = frozenset()) -> list:
    """(state, pi, z) rows for every REAL step in the episode -- never the
    old candidate->scalar-value puct_examples.jsonl schema (that schema
    has no `pi`/visit-distribution over a real edit-inclusive action set
    at all -- it cannot be relabelled into this one). Hard-rejects
    protected specs and any non-POST_CLOAD_FIX_V1 / CLOAD-mismatched
    terminal measurement -- z backfills into every step with NO sign
    flips (Section 5: single-agent, no adversary).

    Stage 5D Section 4/5: each row now also carries "state_graph" (the
    REAL, serialized DeviceCircuitGraph -- see serialize_device_graph())
    and "replay_schema_version" = AZ_REPLAY_SCHEMA_GRAPH_COMPLETE, so a
    later offline trainer can reconstruct the EXACT policy/value input
    without any LLM/RAG/corpus lookup. Every row's graph round-trip is
    verified (serialize -> deserialize -> recompute canonical hash ->
    compare) before it is ever written -- raises GraphRoundTripError
    rather than silently persisting a row that couldn't be trusted.
    Requires episode["registry"] (an LLMSeededEditRegistry), always
    present on a real run_alphazero_episode() result.
    """
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
    reg = episode.get("registry")

    def _serialize_checked(tid: str) -> dict:
        dg = reg.get_device_graph(tid)
        ok, orig_hash, recon_hash = graph_round_trip_hash_matches(dg, tid)
        if not ok:
            raise GraphRoundTripError(
                f"graph round-trip failed for topology {tid!r}: "
                f"original hash {orig_hash} != reconstructed hash {recon_hash}")
        return serialize_device_graph(dg)

    # Section 4: a SELECT action's own aemb (the ONLY signal that lets the
    # policy differentiate between candidate seeds) is derived from the
    # TARGET seed's own graph -- at the super-root, EVERY seed_id is a
    # legal SELECT target, not just whichever one this trajectory
    # eventually committed to. Without every seed's graph, an offline
    # trainer/scorer can reconstruct the state actually visited but not
    # the OTHER legal SELECT actions competing against it -- so every row
    # carries the full seed pool's graphs (small: len(seed_ids) <= ~4),
    # not just its own current-topology graph.
    seed_graphs = ({sid: _serialize_checked(sid) for sid in episode["seed_ids"]}
                  if reg is not None else None)

    rows = []
    for s in episode["steps"]:
        tid = s["state_topology_id"]
        state_graph = _serialize_checked(tid) if reg is not None else None
        rows.append({
            "replay_schema_version": (AZ_REPLAY_SCHEMA_GRAPH_COMPLETE if state_graph is not None
                                      else AZ_REPLAY_SCHEMA_LEGACY),
            "generation": episode["generation_id"], "spec_index": episode.get("spec_index"),
            "spec_hash": episode["spec_hash"], "context_id": episode["ctx_id"],
            "step": s["step"], "state": s["state"],
            "state_topology_id": tid, "state_graph": state_graph,
            "seed_graphs": seed_graphs,
            "state_graph_hash": s["state_graph_hash"],
            "state_topology_hash": s["state_graph_hash"],   # Section 6 naming alias
            "edit_depth": s["depth"],
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
            "mcts_search_seed": s.get("mcts_search_seed"),
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


class LegacyReplayRejected(Exception):
    """Stage 5D Section 6/22: the graph-complete trainer refuses rows
    that predate replay_schema_version==AZ_REPLAY_SCHEMA_GRAPH_COMPLETE
    (or otherwise carry no state_graph) -- e.g. Campaign 01 Retry's
    AZ_G1_REAL_replay.jsonl, LEGACY_NONRECONSTRUCTABLE by construction.
    Training on those would silently mismatch each row's graph against a
    DIFFERENT (or missing) topology, which is exactly the failure mode
    this whole schema change exists to prevent."""


class _RowGraphEntry:
    __slots__ = ("graph",)


class _RowGraphRegistry:
    """Stage 5D Section 4: a minimal reg-shaped adapter exposing
    get_topology(tid).graph, backed ENTIRELY by the graph(s) embedded in
    one or more graph-complete replay rows -- no LLM/RAG/corpus lookup,
    no live LLMSeededEditRegistry required. Mirrors LLMSeededEditRegistry
    ._resolve()'s "proposal_root" aliasing so policy_forward/value_forward
    behave identically whether fed a live registry or a reconstructed one."""

    def __init__(self, rows: list[dict]):
        from agentic_raptor.topology_rl.value_refresh import \
            device_graph_to_circuit_graph
        self._graphs: dict[str, Any] = {}

        def _register(tid: str, graph_dict: dict) -> None:
            if tid not in self._graphs:
                dg = deserialize_device_graph(graph_dict)
                self._graphs[tid] = device_graph_to_circuit_graph(dg, tid)

        for r in rows:
            tid = r["state_topology_id"]
            if r.get("state_graph") is not None:
                _register(tid, r["state_graph"])
            # Section 4: every SELECT target at the super-root needs its
            # OWN graph for its aemb -- not just whichever seed this row's
            # own trajectory happened to visit.
            for sid, graph_dict in (r.get("seed_graphs") or {}).items():
                if graph_dict is not None:
                    _register(sid, graph_dict)
        # "proposal_root" rows already carry the RESOLVED seed's own
        # state_graph directly (LLMSeededEditRegistry._resolve's
        # convention is applied at collection time, before serialization)
        # -- no separate alias step is needed here.

    def get_topology(self, tid: str) -> _RowGraphEntry:
        e = _RowGraphEntry()
        e.graph = self._graphs[tid]
        return e


def _reconstruct_action(action_id: str, state_topology_id: str) -> Stage3E1Action:
    """The exact action-id -> Stage3E1Action reconstruction stage3e1.
    train_step() already implements for AlphaZero replay ids, extracted
    here (not duplicated logic-wise) so the minibatch trainer builds
    IDENTICAL Stage3E1Action objects for the same ids."""
    if action_id.startswith("a_sel_"):
        return Stage3E1Action(action_id, Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                              source_ref=action_id.replace("a_sel_", ""))
    if action_id.startswith("a_edit_"):
        return Stage3E1Action(action_id,
                              Stage3E1ActionType[action_id.replace("a_edit_", "")],
                              source_ref=state_topology_id)
    return Stage3E1Action(action_id, Stage3E1ActionType.TERMINATE_SEARCH)


def train_az_generation_minibatch(replay_rows: list[dict], *,
                                  parent_checkpoint=None, lr: float = 3e-4,
                                  weight_decay: float = 1e-4, epochs: int = 3,
                                  batch_size: int = 16, clip: float = 5.0,
                                  generation_age_weights: dict | None = None,
                                  shuffle_seed: int = 0, episode_balanced: bool = True,
                                  policy_weight: float = 1.0, value_weight: float = 1.0,
                                  track_gradient_balance: bool = False,
                                  nets: dict | None = None, opt=None) -> dict:
    """Stage 5D Section 8/9: deterministic shuffled MINIBATCH training on
    graph-complete replay rows -- replaces the single-row-per-optimizer-
    step behavior stage3e1.train_step() has (opt.step() called inside its
    own per-example loop; train_az_generation() calls it once per
    episode, so a real campaign's 32-episode generation did 115
    individual sequential SGD updates, unshuffled, one per replay row --
    see Stage 5C's training_order_recency_analysis). Requires EVERY row
    to be graph-complete (replay_schema_version ==
    AZ_REPLAY_SCHEMA_GRAPH_COMPLETE and state_graph present) -- hard-
    rejects legacy rows rather than silently training on a
    graph/target mismatch.

    Episode-balanced weighting (Section 9, default on): weight(row) =
    1 / n_states_in_that_row's_episode, normalised to sum to the batch
    size within each mini-batch, so a 4-step episode does not contribute
    4x the gradient weight of a 2-step episode purely from being longer.
    z itself is never touched -- only the LOSS weighting changes.

    Stage 5E Section 11/12: `policy_weight`/`value_weight` scale the two
    loss terms (both default 1.0 -- current production behavior,
    unchanged unless a caller explicitly requests otherwise). When
    `track_gradient_balance=True`, EVERY batch does two EXTRA backward
    passes (policy-only, value-only, both retaining the graph) purely to
    measure the shared encoder's gradient norm from each loss component
    separately -- the real (combined, weighted) backward that actually
    updates parameters is unaffected by this measurement. Off by default
    since it roughly triples backward-pass cost.
    """
    import math
    import random as _random

    import torch

    for r in replay_rows:
        if (r.get("replay_schema_version") != AZ_REPLAY_SCHEMA_GRAPH_COMPLETE
                or r.get("state_graph") is None):
            raise LegacyReplayRejected(
                f"row (spec_hash={r.get('spec_hash')!r}, step={r.get('step')!r}) is not "
                f"graph-complete (replay_schema_version={r.get('replay_schema_version')!r}) "
                "-- the minibatch trainer refuses legacy/non-reconstructable replay")

    nets = nets or load_alphazero_nets(parent_checkpoint, seed=0)
    opt = opt or torch.optim.Adam(nets["params"], lr=lr, weight_decay=weight_decay)
    reg = _RowGraphRegistry(replay_rows)

    episode_key = lambda r: (r["spec_hash"], r["seed"])   # noqa: E731
    states_per_episode: dict = {}
    for r in replay_rows:
        states_per_episode[episode_key(r)] = states_per_episode.get(episode_key(r), 0) + 1
    row_weight = ({episode_key(r): 1.0 / states_per_episode[episode_key(r)] for r in replay_rows}
                 if episode_balanced else None)
    # FINAL PERFORMANCE REPAIR Phase 9: bounded cumulative replay across
    # generations of ONE campaign -- generation_age_weights maps a row's
    # "generation" field to a multiplicative factor (e.g. {"AZ_PER_SEED_G1":
    # 0.8, "AZ_PER_SEED_G2": 1.0}, the pre-declared max(0.2, 1-0.2*K)
    # linear-decay schedule from AZ_NEXT_CAMPAIGN_DESIGN). None (default)
    # = no age weighting, byte-identical to prior behavior. Applied on top
    # of episode balancing BEFORE within-batch normalization; z targets
    # are never touched.
    if generation_age_weights is not None:
        base = row_weight or {episode_key(r): 1.0 for r in replay_rows}
        row_weight = {episode_key(r): base[episode_key(r)]
                      * generation_age_weights.get(r.get("generation"), 1.0)
                      for r in replay_rows}

    def _encoder_grad_norm() -> float:
        sq = sum(float((p.grad ** 2).sum()) for p in nets["encoder"].parameters()
                if p.grad is not None)
        return math.sqrt(sq)

    rng = _random.Random(shuffle_seed)
    batch_reports = []
    n = len(replay_rows)
    for _epoch in range(epochs):
        order = list(range(n))
        rng.shuffle(order)
        for b0 in range(0, n, batch_size):
            batch_idx = order[b0:b0 + batch_size]
            batch = [replay_rows[i] for i in batch_idx]
            if not batch:
                continue
            weights = ([row_weight[episode_key(r)] for r in batch] if row_weight
                      else [1.0] * len(batch))
            w_sum = sum(weights) or 1.0
            weights = [len(batch) * w / w_sum for w in weights]   # normalise within batch

            policy_loss_sum = torch.tensor(0.0)
            value_loss_sum = torch.tensor(0.0)
            used = 0
            for r, w in zip(batch, weights):
                if r["z"] is None:
                    continue
                st = TopologySearchState(**{k: v for k, v in r["state"].items()})
                acts = [_reconstruct_action(a, st.topology_id) for a in r["legal_action_ids"]]
                acts_o, logits, _p = nets["policy_forward"](st, acts, reg)
                target = torch.tensor([r["pi"].get(a.action_id, 0.0) for a in acts_o])
                target = target / target.sum().clamp(min=1e-8)
                pl = -(target * torch.log_softmax(logits, dim=0)).sum()
                vout = nets["value_forward"](st, reg)
                vl = (vout["scalar"] - torch.tensor(float(r["z"]))) ** 2
                policy_loss_sum = policy_loss_sum + w * pl
                value_loss_sum = value_loss_sum + w * vl
                used += 1
            if used == 0:
                continue

            policy_encoder_gn = value_encoder_gn = None
            if track_gradient_balance:
                opt.zero_grad()
                (policy_weight * policy_loss_sum / used).backward(retain_graph=True)
                policy_encoder_gn = _encoder_grad_norm()
                opt.zero_grad()
                (value_weight * value_loss_sum / used).backward(retain_graph=True)
                value_encoder_gn = _encoder_grad_norm()

            opt.zero_grad()
            loss = (policy_weight * policy_loss_sum + value_weight * value_loss_sum) / used
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(nets["params"], clip)
            opt.step()
            report = {
                "policy_loss": round(float(policy_loss_sum.detach()) / used, 5),
                "value_loss": round(float(value_loss_sum.detach()) / used, 5),
                "grad_norm": round(float(gn), 5), "batch_size": len(batch),
                "examples_used": used}
            if track_gradient_balance:
                report["policy_encoder_grad_norm"] = round(policy_encoder_gn, 6)
                report["value_encoder_grad_norm"] = round(value_encoder_gn, 6)
                report["policy_to_value_grad_ratio"] = (
                    round(policy_encoder_gn / value_encoder_gn, 4)
                    if value_encoder_gn > 1e-12 else None)
            batch_reports.append(report)

    return {"nets": nets, "opt": opt, "batch_reports": batch_reports,
           "config": {"lr": lr, "weight_decay": weight_decay, "epochs": epochs,
                     "batch_size": batch_size, "clip": clip, "shuffle_seed": shuffle_seed,
                     "episode_balanced": episode_balanced, "policy_weight": policy_weight,
                     "value_weight": value_weight, "optimizer": "Adam",
                     "generation_age_weights": generation_age_weights},
           "final_policy_loss": batch_reports[-1]["policy_loss"] if batch_reports else None,
           "final_value_loss": batch_reports[-1]["value_loss"] if batch_reports else None}


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


class AlphaZeroCheckpointError(Exception):
    """Stage 8 FINAL integration (2026-08-13): any failure to load and
    verify the promoted checkpoint for a FULL/publication AlphaZero
    invocation is a HARD error -- never a silent fallback to a
    random-initialised network. Root cause being fixed: run_pipeline's
    value_ckpt previously defaulted to None, and load_alphazero_nets(None)
    silently returned a fresh deterministic seed-0 random net, so live
    pipeline runs never actually used the promoted checkpoint."""


def load_promoted_alphazero_nets() -> tuple:
    """The ONLY authorized way for FULL/publication runs to obtain
    AlphaZero nets. Returns (nets, provenance) where provenance proves the
    load happened:

      - resolves the PROMOTED checkpoint (require_promoted_az_checkpoint)
      - verifies the file's FULL SHA-256 against the generation manifest's
        own recorded hash (hard fail on mismatch/tamper)
      - loads it and fingerprints the parameters
      - hard-fails if the fingerprint equals a fresh random-init network's
        (proof the load actually changed the weights -- a random net can
        never masquerade as the promoted one)
    """
    import hashlib as _hashlib

    from agentic_raptor.topology_rl.trainer import parameter_checksum

    ckpt_path = require_promoted_az_checkpoint()          # raises if none
    if not ckpt_path.is_file():
        raise AlphaZeroCheckpointError(
            f"promoted checkpoint missing on disk: {ckpt_path}")
    manifest = read_az_generation_manifest(ckpt_path.parent.parent.name)
    recorded = (manifest or {}).get("checkpoint_hash")
    actual = _hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    if recorded is None:
        raise AlphaZeroCheckpointError(
            f"generation manifest for {ckpt_path} records no checkpoint_hash")
    if not actual.startswith(recorded) and actual != recorded:
        raise AlphaZeroCheckpointError(
            f"checkpoint SHA-256 mismatch at {ckpt_path}: file={actual}, "
            f"manifest={recorded} -- refusing to run FULL on an unverified "
            f"checkpoint")

    fresh = load_alphazero_nets(value_ckpt=None, seed=0)
    fresh_fp = (parameter_checksum(fresh["encoder"]), parameter_checksum(fresh["heads"]))
    nets = load_alphazero_nets(value_ckpt=str(ckpt_path), seed=0)
    fp = (parameter_checksum(nets["encoder"]), parameter_checksum(nets["heads"]))
    if fp == fresh_fp:
        raise AlphaZeroCheckpointError(
            f"checkpoint load had NO effect (parameter fingerprint equals a "
            f"fresh seed-0 random network) -- {ckpt_path} was not actually "
            f"loaded; refusing to run FULL on random initialization")
    provenance = {"checkpoint_loaded": True, "checkpoint_path": str(ckpt_path),
                 "checkpoint_sha256": actual,
                 "parameter_fingerprint": {"encoder": fp[0], "heads": fp[1]},
                 "differs_from_random_init": True,
                 "generation_id": ckpt_path.parent.parent.name}
    return nets, provenance


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


def alphazero_champion_challenger_select_two(candidates: list[dict], spec: dict,
                                             ctx_id: str, *, nets: dict,
                                             config: "AlphaZeroConfig | None" = None,
                                             seed: int = 0) -> dict:
    """STAGE 9B ALPHAZERO_CHAMPION_CHALLENGER (2026-08-14, experimental):
    slot 1 (CHAMPION) = strongest validated ORIGINAL seed under the frozen
    deterministic pre-AlphaZero ranking rule (run_puct_ablation.
    topology_priors -- the same predeclared rule A5 uses); slot 2
    (CHALLENGER) = best canonical-distinct topology from the CURRENT
    super-root AlphaZero search (may be a seed or an edited descendant; no
    forced editing). Preserves real AlphaZero search while guaranteeing one
    strong LLM topology always reaches sizing."""
    from run_puct_ablation import topology_priors
    priors = topology_priors(spec)
    ranked_seeds = sorted(candidates,
                          key=lambda c: (-(priors.get(c["canonical_family"]) or 0.0),
                                         c["canonical_graph_hash"]))
    champion_src = ranked_seeds[0]
    az = alphazero_select_two(candidates, spec, ctx_id, nets=nets,
                              config=config, seed=seed)
    champion = {**champion_src, "device_graph": None, "source": "champion_seed",
               "rank": 0, "visit_count": None, "policy_prior":
               priors.get(champion_src["canonical_family"]),
               "selected_top2": True, "originating_seed_id":
               champion_src["llm_proposal_id"],
               "originating_seed_hash": champion_src["canonical_graph_hash"],
               "edit_history": [], "edit_depth": 0,
               "is_edited_descendant": False}
    challenger = next((c for c in az["selected"]
                      if c["canonical_graph_hash"] != champion["canonical_graph_hash"]),
                     None)
    if challenger is None:
        for e in az["ranked_all"]:
            if e["hash"] != champion["canonical_graph_hash"]:
                challenger = next((c for c in az["selected"]), None)
                # rebuild from ranked_all entry via the search registry path
                break
        alt = next((c for c in ranked_seeds[1:]
                   if c["canonical_graph_hash"] != champion["canonical_graph_hash"]), None)
        if challenger is None and alt is not None:
            challenger = {**alt, "device_graph": None, "source": "champion_seed_alt",
                         "rank": 1, "visit_count": None, "policy_prior":
                         priors.get(alt["canonical_family"]), "selected_top2": True,
                         "originating_seed_id": alt["llm_proposal_id"],
                         "originating_seed_hash": alt["canonical_graph_hash"],
                         "edit_history": [], "edit_depth": 0,
                         "is_edited_descendant": False}
    if challenger is None:
        raise AlphaZeroSelectionError("champion-challenger: no distinct challenger")
    challenger = {**challenger, "rank": 1}
    return {**az, "selected": [champion, challenger],
           "search": "true_alphazero_champion_challenger",
           "champion_rule": "frozen deterministic topology_priors seed ranking",
           "champion_hash": champion["canonical_graph_hash"]}
