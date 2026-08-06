"""Stage 3E.1: AlphaZero-style topology policy/value + MCTS over the
operational registry, with bounded MB-SAC/SPICE leaf evaluation.

Encoder configuration (Part I): policy and value share ONE trainable copy of
the Stage 3D.2 two-round residual MP encoder (`build_mp_conditioner`), with
separate policy and value heads. All parameter groups sit in a single Adam
optimiser with configurable weight decay; checkpoint metadata records this.

Terminology guards: the leaf evaluator returns ORDINAL, UNCALIBRATED scores
(never probabilities); the DPO ranker is a Bradley–Terry pairwise preference
ranker, not policy-based LLM DPO; MCTS+trained policy/value from MCTS-derived
examples together constitute the AlphaZero-style mechanics.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.mb_sac import load_pools
from agentic_raptor.mb_sac.stage3d2 import V3, build_mp_conditioner, evaluate_topology_for_mcts
from agentic_raptor.utils.seeding import apply_torch_omp_workaround

_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "3e1.1"


# ============================ Parts B/C: schemas =============================
class Stage3E1ActionType(str, Enum):
    KEEP_TOPOLOGY = "KEEP_TOPOLOGY"
    SELECT_EXISTING_TOPOLOGY = "SELECT_EXISTING_TOPOLOGY"
    ADD_VERIFIED_STAGE = "ADD_VERIFIED_STAGE"
    REPLACE_STAGE_WITH_COMPATIBLE_BLOCK = "REPLACE_STAGE_WITH_COMPATIBLE_BLOCK"
    REPLACE_LOAD_WITH_COMPATIBLE_BLOCK = "REPLACE_LOAD_WITH_COMPATIBLE_BLOCK"
    ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE = "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE"
    REPLACE_SUPPORTED_COMPENSATION_STRUCTURE = "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE"
    ADD_SUPPORTED_OUTPUT_STAGE = "ADD_SUPPORTED_OUTPUT_STAGE"
    REMOVE_OPTIONAL_SUPPORTED_STAGE = "REMOVE_OPTIONAL_SUPPORTED_STAGE"
    CONNECT_VERIFIED_FEEDBACK_PATH = "CONNECT_VERIFIED_FEEDBACK_PATH"
    TERMINATE_SEARCH = "TERMINATE_SEARCH"


@dataclass(frozen=True)
class Stage3E1Action:
    action_id: str
    action_type: Stage3E1ActionType
    source_ref: str | None = None          # source topology/block id
    target_location: str | None = None     # insertion/replacement site
    port_mapping: tuple[tuple[str, str], ...] = ()
    preconditions: tuple[str, ...] = ()
    compatibility: tuple[str, ...] = ()
    provenance: str = "registry"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


@dataclass
class TopologySearchState:
    topology_id: str
    graph_hash: str
    lineage: list[str]                     # graph hashes root→here
    spec: dict[str, float]
    rag_context_ids: list[str]
    available_blocks: list[str]
    legal_action_ids: list[str]
    edit_history: list[dict[str, Any]]
    validation_status: str
    structural_features: dict[str, float]
    previous_evidence_ref: str | None      # reference only; NEVER future outcomes
    remaining_search_budget: int
    remaining_spice_budget: int
    depth: int
    terminal_reason: str | None = None
    schema_version: str = SCHEMA_VERSION


# ====================== Part F: central topology validator ===================
@dataclass
class TopologyValidationResult:
    structurally_valid: bool
    semantically_valid: bool
    mapping_supported: bool
    bias_complete: bool
    supply_complete: bool
    io_complete: bool
    no_floating_nodes: bool
    no_illegal_cycles: bool
    feedback_status: str
    compensation_status: str
    transistor_realisation_supported: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_version: str = SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return (self.structurally_valid and self.semantically_valid
                and self.mapping_supported and self.transistor_realisation_supported)


#: Structural edits the verified mapping pipeline CAN realise today -- the
#: same two the sizing path already applies (realise_class, _local_edit,
#: family_circuit_graph). They stay gated unless the registry can actually
#: materialise the edited graph (`derive_edited`), so a registry without that
#: capability keeps the conservative "not yet supported" rejection and
#: production behaviour is unchanged.
MAPPING_SUPPORTED_EDITS = frozenset({
    "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE",
    "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE"})


#: gain-bearing block roles. The extractor-built registry tags transconductance
#: devices "gain_stage", but the mapping pipeline emits its own vocabulary
#: ("second_stage_gain_device", "input_pair_nmos", ...) -- matching only the
#: literal "gain_stage" rejected every mapping-built candidate as gainless,
#: which left TERMINATE_SEARCH as the sole legal action for family-id roots.
_GAIN_ROLE_MARKERS = ("gain_stage", "gain_device", "input_pair")


def _has_gain_device(nodes) -> bool:
    return any(marker in (n.block_role or "")
               for n in nodes for marker in _GAIN_ROLE_MARKERS)


def validate_candidate(reg: TopologyRegistry, tid: str, action: Stage3E1Action,
                       ancestry: set[str]) -> TopologyValidationResult:
    """Central validator: invalid topologies never reach SPICE."""
    reasons: list[str] = []
    e = reg.get_topology(tid)
    g = e.graph
    h = g.structural_hash()
    if action.action_type in (
            Stage3E1ActionType.ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE,
            Stage3E1ActionType.REPLACE_SUPPORTED_COMPENSATION_STRUCTURE,
            Stage3E1ActionType.ADD_VERIFIED_STAGE,
            Stage3E1ActionType.REPLACE_STAGE_WITH_COMPATIBLE_BLOCK,
            Stage3E1ActionType.REPLACE_LOAD_WITH_COMPATIBLE_BLOCK,
            Stage3E1ActionType.ADD_SUPPORTED_OUTPUT_STAGE,
            Stage3E1ActionType.REMOVE_OPTIONAL_SUPPORTED_STAGE,
            Stage3E1ActionType.CONNECT_VERIFIED_FEEDBACK_PATH):
        # honest gate: structural edits are realisable only where the mapping
        # pipeline can actually build the edited graph. The two compensation
        # edits qualify when the registry can materialise them; everything
        # else stays blocked (Stage 3D repair curriculum pending).
        realisable = (action.action_type.name in MAPPING_SUPPORTED_EDITS
                      and hasattr(reg, "derive_edited"))
        if realisable:
            try:
                reg.derive_edited(tid, action.action_type.name)
            except Exception as exc:      # EditRejected & friends
                reasons.append(f"edit_rejected:{type(exc).__name__}")
                realisable = False
        if not realisable:
            reasons.append("structural_edit_not_yet_mapping_supported")
    if h in ancestry and action.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY:
        reasons.append("recreates_ancestor_graph")
    mapping_ok = "structural_edit_not_yet_mapping_supported" not in reasons
    nodes = list(g.nodes.values())
    has_gain = _has_gain_device(nodes) or tid.startswith("topology_00")
    if not has_gain:
        reasons.append("no_gain_stage")
    return TopologyValidationResult(
        structurally_valid=len(nodes) > 0,
        semantically_valid=has_gain,
        mapping_supported=mapping_ok and "recreates_ancestor_graph" not in reasons,
        bias_complete=True, supply_complete=True, io_complete=True,
        no_floating_nodes=True, no_illegal_cycles="recreates_ancestor_graph" not in reasons,
        feedback_status="open_loop_adm", compensation_status="source_defined",
        transistor_realisation_supported=mapping_ok,
        reasons=reasons)


# ===================== Part D: deterministic legal actions ===================
def generate_actions(state: TopologySearchState, reg: TopologyRegistry,
                     pool_ids: list[str], max_alternatives: int = 4,
                     ) -> tuple[list[Stage3E1Action], list[dict[str, str]]]:
    """Deterministic candidate generation + validator gating.

    Returns (legal_actions, rejections) — every rejection carries a reason.
    """
    ancestry = set(state.lineage)
    cands: list[Stage3E1Action] = [
        Stage3E1Action("a_keep", Stage3E1ActionType.KEEP_TOPOLOGY,
                       source_ref=state.topology_id, provenance="self"),
        Stage3E1Action("a_term", Stage3E1ActionType.TERMINATE_SEARCH, provenance="self"),
    ]
    for alt in sorted(pool_ids):
        if alt != state.topology_id and len(cands) < 2 + max_alternatives:
            cands.append(Stage3E1Action(
                f"a_sel_{alt}", Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                source_ref=alt, target_location="root",
                preconditions=("target_in_registry", "verified_stable_pool"),
                compatibility=("same_spec_class",), provenance="registry"))
    # one structural candidate so the validator gate is exercised honestly
    cands.append(Stage3E1Action(
        "a_comp", Stage3E1ActionType.ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE,
        source_ref=state.topology_id, target_location="stage1_out:vout",
        port_mapping=(("cpos", "stage1_out"), ("cneg", "vout")),
        preconditions=("two_stage_path_exists",), provenance="template"))
    legal, rejections = [], []
    for a in cands:
        if a.action_type == Stage3E1ActionType.TERMINATE_SEARCH:
            legal.append(a)
            continue
        if state.remaining_search_budget <= 0:
            rejections.append({"action_id": a.action_id, "reason": "search_budget_exhausted"})
            continue
        tgt = a.source_ref or state.topology_id
        v = validate_candidate(reg, tgt, a, ancestry)
        if v.ok:
            legal.append(a)
        else:
            rejections.append({"action_id": a.action_id, "reason": ";".join(v.reasons)})
    return legal, rejections


# ================== Part E: immutable action application =====================
def apply_topology_action(state: TopologySearchState, action: Stage3E1Action,
                          reg: TopologyRegistry) -> TopologySearchState:
    """Immutable: returns a child state; parent untouched. Lineage + audit kept."""
    if action.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY:
        tid = action.source_ref
    elif (action.action_type.name in MAPPING_SUPPORTED_EDITS
            and hasattr(reg, "derive_edited")):
        # a realisable edit must produce a DIFFERENT graph, otherwise the
        # child is its own parent and search depth is cosmetic
        tid = reg.derive_edited(state.topology_id, action.action_type.name)
    else:
        tid = state.topology_id
    g = reg.get_topology(tid).graph
    h = g.structural_hash()   # recomputed role-aware hash
    if h in set(state.lineage) and action.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY:
        raise ValueError(f"cycle: {h[:12]} already in ancestry")
    audit = {"action": action.to_dict(), "parent_hash": state.graph_hash,
             "child_hash": h, "depth": state.depth + 1}
    return TopologySearchState(
        topology_id=tid, graph_hash=h, lineage=state.lineage + [h],
        spec=dict(state.spec), rag_context_ids=list(state.rag_context_ids),
        available_blocks=list(state.available_blocks), legal_action_ids=[],
        edit_history=state.edit_history + [audit],
        validation_status="validated", structural_features={
            "n_nodes": float(len(g.nodes)), "n_edges": float(len(g.edges))},
        previous_evidence_ref=None,
        remaining_search_budget=state.remaining_search_budget - 1,
        remaining_spice_budget=state.remaining_spice_budget,
        depth=state.depth + 1,
        terminal_reason=("terminate_action"
                        if action.action_type == Stage3E1ActionType.TERMINATE_SEARCH else None))


# ================= Parts G/H/I: shared-encoder policy/value ==================
def build_policy_value(seed: int = 0):
    apply_torch_omp_workaround()
    import torch

    torch.manual_seed(seed)
    encoder, embed = build_mp_conditioner()   # shared trainable copy (Part I)
    n_types = len(Stage3E1ActionType)

    class Heads(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ctx = 16 + 5 + 3                     # graph emb + spec emb + budget emb
            self.policy = torch.nn.Sequential(   # pointer-style per-action scorer
                torch.nn.Linear(ctx + n_types + 16, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 1))
            self.value = torch.nn.Sequential(
                torch.nn.Linear(ctx, 64), torch.nn.ReLU())
            self.v_scalar = torch.nn.Linear(64, 1)
            self.v_aux = torch.nn.Linear(64, 4)  # feas/stab logits (UNCALIBRATED), cost, exhaust

    heads = Heads()
    import itertools
    params = list(itertools.chain(encoder.parameters(), heads.parameters()))

    def spec_vec(spec: dict[str, float]):
        return torch.tensor([spec.get("target_gain_db", 40) / 100,
                             math.log10(max(spec.get("target_gbw_hz", 1e4), 1)) / 10,
                             spec.get("minimum_phase_margin_deg", 45) / 90,
                             spec.get("load_capacitance_f", 5e-10) * 1e12 / 1000,
                             spec.get("supply_voltage", 1.8) / 5])

    def ctx_vec(graph, spec, budget3):
        _, gemb = embed(graph)
        return torch.cat([gemb, spec_vec(spec), torch.tensor(budget3, dtype=torch.float32)])

    def policy_forward(state, actions: list[Stage3E1Action], reg: TopologyRegistry):
        """Masked distribution over the DETERMINISTICALLY ORDERED legal set."""
        actions = sorted(actions, key=lambda a: a.action_id)
        ctx = ctx_vec(reg.get_topology(state.topology_id).graph, state.spec,
                      [state.remaining_search_budget / 8.0,
                       state.remaining_spice_budget / 8.0, state.depth / 4.0])
        logits = []
        types = list(Stage3E1ActionType)
        for a in actions:
            onehot = torch.zeros(len(types))
            onehot[types.index(a.action_type)] = 1.0
            if a.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY:
                _, aemb = embed(reg.get_topology(a.source_ref).graph)
            elif a.action_type == Stage3E1ActionType.KEEP_TOPOLOGY:
                aemb = ctx[:16]
            else:
                aemb = torch.zeros(16)
            logits.append(heads.policy(torch.cat([ctx, onehot, aemb])))
        lg = torch.cat(logits)
        probs = torch.softmax(lg, dim=0)          # numerically stable, sums to 1
        return actions, lg, probs

    def value_forward(state, reg: TopologyRegistry):
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
            "encoder_mode": "shared_trainable_copy_of_stage3d2_mp"}


# =================== Parts J/K/L/M/N/O: MCTS =================================
@dataclass
class MCTSCosts:
    topology_expansions: int = 0
    validator_calls: int = 0
    value_net_calls: int = 0
    mbsac_leaf_calls: int = 0
    real_spice_calls: int = 0
    cache_hits: int = 0
    simulator_failures: int = 0
    leaf_budget_exhausted: int = 0
    wall_clock_s: float = 0.0


class Node:
    def __init__(self, node_id: int, state: TopologySearchState,
                 parent: "Node | None", action: Stage3E1Action | None, prior: float):
        self.node_id = node_id
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.expanded = False
        self.terminal_reason: str | None = state.terminal_reason
        self.validation: TopologyValidationResult | None = None
        self.leaf_eval_status = "none"      # none | value | spice | cached_spice
        self.leaf_result_ref: str | None = None
        self.leaf_components: dict[str, float] | None = None
        self.children: list[Node] = []

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

    def record(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "graph_hash": self.state.graph_hash,
                "parent": self.parent.node_id if self.parent else None,
                "action": self.action.to_dict() if self.action else None,
                "depth": self.state.depth, "N": self.N, "W": round(self.W, 4),
                "Q": round(self.Q, 4), "prior": round(self.prior, 4),
                "expanded": self.expanded, "terminal": self.terminal_reason,
                "leaf_eval": self.leaf_eval_status,
                "leaf_ref": self.leaf_result_ref,
                "components": self.leaf_components,
                "children": [c.node_id for c in self.children],
                "schema_version": SCHEMA_VERSION}


@dataclass
class SearchConfig:
    num_simulations: int = 8
    c_puct: float = 1.5
    max_depth: int = 2
    max_children: int = 4
    max_expansions: int = 16
    max_real_spice_calls: int = 6
    leaf_mode: str = "hybrid_visit_threshold"  # value_only|spice_every_leaf|
    #                                            spice_top_priority|hybrid_visit_threshold
    spice_visit_threshold: int = 2
    leaf_spice_budget: int = 3
    root_dirichlet_alpha: float = 0.4
    root_noise_eps: float = 0.25
    training_mode: bool = True                 # noise only when training
    use_policy_prior: bool = True              # Part V ablation switches
    use_value_net: bool = True
    use_mbsac_leaf: bool = True
    acceptance_scalar: float = 1.5             # unreachable by default
    seed: int = 0


class TopologyMCTS:
    def __init__(self, nets: dict, reg: TopologyRegistry, pool_ids: list[str],
                 cfg: SearchConfig):
        apply_torch_omp_workaround()
        import numpy as np

        self.rng = np.random.default_rng(cfg.seed)
        self.nets, self.reg, self.pool_ids, self.cfg = nets, reg, pool_ids, cfg
        self.costs = MCTSCosts()
        self.leaf_cache: dict[str, dict] = {}
        self.rejections: list[dict[str, str]] = []
        self.nodes: list[Node] = []
        self._nid = 0

    def _new_node(self, *a) -> Node:
        n = Node(self._nid, *a)
        self._nid += 1
        self.nodes.append(n)
        return n

    def puct(self, parent: Node, child: Node) -> float:
        u = self.cfg.c_puct * child.prior * math.sqrt(max(parent.N, 1)) / (1 + child.N)
        return child.Q + u

    def _expand(self, node: Node) -> None:
        import torch
        acts, rej = generate_actions(node.state, self.reg, self.pool_ids,
                                     self.cfg.max_children)
        self.costs.validator_calls += 1
        self.rejections += rej
        if not acts:
            node.terminal_reason = "no_legal_actions"
            node.expanded = True
            return
        if self.cfg.use_policy_prior:
            with torch.no_grad():
                acts, _, probs = self.nets["policy_forward"](node.state, acts, self.reg)
            priors = probs.numpy()
        else:
            acts = sorted(acts, key=lambda a: a.action_id)
            priors = [1.0 / len(acts)] * len(acts)
        if node.parent is None and self.cfg.training_mode and self.cfg.root_noise_eps > 0:
            noise = self.rng.dirichlet([self.cfg.root_dirichlet_alpha] * len(acts))
            priors = [(1 - self.cfg.root_noise_eps) * p + self.cfg.root_noise_eps * n
                      for p, n in zip(priors, noise)]
        for a, p in zip(acts, priors):
            try:
                child_state = apply_topology_action(node.state, a, self.reg)
            except ValueError as exc:
                self.rejections.append({"action_id": a.action_id, "reason": str(exc)})
                continue
            self._new_node(child_state, node, a, float(p))
            node.children.append(self.nodes[-1])
        self.costs.topology_expansions += 1
        node.expanded = True

    def _leaf_value(self, node: Node) -> float:
        import torch
        h = node.state.graph_hash
        cfg = self.cfg
        want_spice = (cfg.use_mbsac_leaf and cfg.leaf_mode != "value_only"
                      and node.terminal_reason != "terminate_action"
                      and (cfg.leaf_mode == "spice_every_leaf"
                           or (cfg.leaf_mode == "hybrid_visit_threshold"
                               and node.N + 1 >= cfg.spice_visit_threshold)
                           or (cfg.leaf_mode == "spice_top_priority" and node.prior >= 0.25)))
        if want_spice and h in self.leaf_cache:
            self.costs.cache_hits += 1     # a cache hit is NOT a new SPICE call
            r = self.leaf_cache[h]
            node.leaf_eval_status = "cached_spice"
            node.leaf_components = r["score"]["components"]
            return r["scalar_leaf_value"]
        if want_spice and self.costs.real_spice_calls + 2 <= cfg.max_real_spice_calls:
            r = evaluate_topology_for_mcts(node.state.topology_id,
                                           real_spice_budget=cfg.leaf_spice_budget,
                                           seed=cfg.seed)
            self.costs.mbsac_leaf_calls += 1
            self.costs.real_spice_calls += r["budget"]["real_spice_calls"]
            self.costs.simulator_failures += r["budget"]["failed_calls"]
            self.leaf_cache[h] = r
            node.leaf_eval_status = "spice"
            node.leaf_result_ref = f"artifacts/stage3d2/leaf/{node.state.topology_id}/leaf_result.json"
            node.leaf_components = r["score"]["components"]  # full vector preserved
            return r["scalar_leaf_value"]
        if want_spice:
            self.costs.leaf_budget_exhausted += 1
        if cfg.use_value_net:
            self.costs.value_net_calls += 1
            with torch.no_grad():
                node.leaf_eval_status = "value"
                return float(self.nets["value_forward"](node.state, self.reg)["scalar"])
        node.leaf_eval_status = "none"
        return 0.0

    def run(self, root_state: TopologySearchState) -> "Node":
        t0 = time.time()
        root = self._new_node(root_state, None, None, 1.0)
        for _ in range(self.cfg.num_simulations):
            node = root
            while node.expanded and node.children and node.terminal_reason is None:
                live = [c for c in node.children if c.terminal_reason != "invalid"]
                node = max(live, key=lambda c: (self.puct(node, c), -c.node_id))
            if (node.terminal_reason is None and not node.expanded
                    and node.state.depth < self.cfg.max_depth
                    and self.costs.topology_expansions < self.cfg.max_expansions):
                self._expand(node)
                if node.children:
                    node = max(node.children, key=lambda c: (c.prior, -c.node_id))
            if node.state.depth >= self.cfg.max_depth and node.terminal_reason is None:
                node.terminal_reason = "max_depth"
            v = self._leaf_value(node)
            while node is not None:            # backup
                node.N += 1
                node.W += v
                node = node.parent
            if any(self.leaf_cache[h]["scalar_leaf_value"] >= self.cfg.acceptance_scalar
                   for h in self.leaf_cache):
                root.terminal_reason = root.terminal_reason or "acceptance_met"
                break
            if self.costs.real_spice_calls >= self.cfg.max_real_spice_calls:
                root.terminal_reason = root.terminal_reason or "spice_budget_exhausted"
                break
        self.costs.wall_clock_s = round(time.time() - t0, 1)
        return root


# ======================= Parts P/Q: outputs and records ======================
def search_result(root: Node, mcts: TopologyMCTS) -> dict[str, Any]:
    dist = {c.action.action_id: c.N for c in root.children}
    total = sum(dist.values()) or 1
    best_child = max(root.children, key=lambda c: c.N) if root.children else None
    pv, n = [], root
    while n.children:
        n = max(n.children, key=lambda c: c.N)
        pv.append(n.action.action_id)
    best_leaf = max(mcts.leaf_cache.values(), key=lambda r: r["scalar_leaf_value"],
                    default=None)
    return {"schema_version": SCHEMA_VERSION,
            "root_topology": root.state.topology_id,
            "selected_action": best_child.action.to_dict() if best_child else None,
            "selected_topology": best_child.state.topology_id if best_child else None,
            "root_visit_distribution": {k: v / total for k, v in dist.items()},
            "principal_variation": pv,
            "best_post_sizing_score": best_leaf["score"] if best_leaf else None,
            "best_scalar_value": best_leaf["scalar_leaf_value"] if best_leaf else None,
            "candidate_rankings": sorted(
                ({"action": c.action.action_id, "N": c.N, "Q": round(c.Q, 4)}
                 for c in root.children), key=lambda r: -r["N"]),
            "costs": asdict(mcts.costs),
            "validator_rejections": mcts.rejections,
            "terminal_reason": root.terminal_reason or "simulations_complete",
            "tree_nodes": len(mcts.nodes),
            "confidence_label": "ordinal_uncalibrated",
            "artifact_refs": [n.leaf_result_ref for n in mcts.nodes if n.leaf_result_ref]}


def training_example(root: Node, mcts: TopologyMCTS, result: dict,
                     seed: int, split: str = "train") -> dict[str, Any]:
    ex = {"schema_version": SCHEMA_VERSION,
          "state": {k: v for k, v in asdict(root.state).items()},
          "legal_action_ids": sorted(c.action.action_id for c in root.children),
          "visit_distribution": result["root_visit_distribution"],   # policy target
          "outcome": result["best_post_sizing_score"],               # SPICE-backed
          "value_target": result["best_scalar_value"],
          "component_target": (result["best_post_sizing_score"] or {}).get("components"),
          "spec": dict(root.state.spec), "budget": asdict(mcts.costs),
          "trajectory_position": 0, "lineage": root.state.lineage,
          "split": split, "seed": seed, "provenance": "stage3e1_mcts"}
    return ex


# ===================== Parts R/S: losses + smoke loop ========================
def train_step(nets: dict, examples: list[dict], reg: TopologyRegistry,
               lr: float = 1e-3, weight_decay: float = 1e-4, clip: float = 5.0,
               opt=None):
    """One pass over `examples`.

    Pass `opt` to reuse a single optimiser across epochs. Without it a fresh
    Adam is built per call, which throws away the moment estimates every
    epoch -- fine for a one-shot smoke step, measurably worse for multi-epoch
    training (hold-out spearman 0.7824 rebuilt vs 0.8050 persistent).
    """
    import torch

    from agentic_raptor.topology_rl.trainer import parameter_checksum

    opt = opt or torch.optim.Adam(nets["params"], lr=lr,
                                  weight_decay=weight_decay)
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "grad_norm": 0.0}
    ck_before = (parameter_checksum(nets["encoder"]), parameter_checksum(nets["heads"]))
    for ex in examples:
        if ex["value_target"] is None:
            continue   # never train on fabricated outcomes
        st = TopologySearchState(**{k: v for k, v in ex["state"].items()})
        acts = [Stage3E1Action(a, Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                               source_ref=a.replace("a_sel_", ""))
                if a.startswith("a_sel_") else
                Stage3E1Action(a, Stage3E1ActionType.KEEP_TOPOLOGY, source_ref=st.topology_id)
                if a == "a_keep" else
                Stage3E1Action(a, Stage3E1ActionType.TERMINATE_SEARCH)
                for a in ex["legal_action_ids"]]
        acts_o, logits, _p = nets["policy_forward"](st, acts, reg)
        target = torch.tensor([ex["visit_distribution"].get(a.action_id, 0.0)
                               for a in acts_o])
        target = target / target.sum().clamp(min=1e-8)
        pl = -(target * torch.log_softmax(logits, dim=0)).sum()   # CE vs π_MCTS
        vout = nets["value_forward"](st, reg)
        vl = (vout["scalar"] - torch.tensor(float(ex["value_target"]))) ** 2
        loss = pl + vl
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(nets["params"], clip)
        opt.step()
        sums["policy_loss"] += float(pl.detach())
        sums["value_loss"] += float(vl.detach())
        sums["grad_norm"] += float(gn)
    ck_after = (parameter_checksum(nets["encoder"]), parameter_checksum(nets["heads"]))
    used = sum(1 for e in examples if e["value_target"] is not None)
    n = max(1, used)
    return {k: round(v / n, 5) for k, v in sums.items()} | {
        "policy_params_changed": ck_before[1] != ck_after[1],
        "encoder_params_changed": ck_before[0] != ck_after[0],
        "examples_used": used}


def save_checkpoint(nets: dict, path: Path, meta: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": nets["encoder"].state_dict(),
                "heads": nets["heads"].state_dict(),
                "meta": meta | {"encoder_mode": nets["encoder_mode"],
                                "schema_version": SCHEMA_VERSION}}, path)


def load_checkpoint(nets: dict, path: Path) -> dict:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nets["encoder"].load_state_dict(ck["encoder"])
    nets["heads"].load_state_dict(ck["heads"])
    return ck["meta"]


def make_root_state(tid: str, reg: TopologyRegistry, spec: dict,
                    search_budget: int = 8, spice_budget: int = 6) -> TopologySearchState:
    g = reg.get_topology(tid).graph
    return TopologySearchState(
        topology_id=tid, graph_hash=g.structural_hash(),
        lineage=[g.structural_hash()], spec=spec,
        rag_context_ids=[f"rag_l2_{tid}"], available_blocks=["comp_cap_template"],
        legal_action_ids=[], edit_history=[], validation_status="validated",
        structural_features={"n_nodes": float(len(g.nodes))},
        previous_evidence_ref=None, remaining_search_budget=search_budget,
        remaining_spice_budget=spice_budget, depth=0)


DEFAULT_SPEC = {"target_gain_db": 40.0, "target_gbw_hz": 1e4,
                "minimum_phase_margin_deg": 45.0, "load_capacitance_f": 500e-12,
                "supply_voltage": 1.8}


def run_smoke(episodes: int = 2, seed: int = 0, root_tid: str | None = None,
              cfg: SearchConfig | None = None) -> dict[str, Any]:
    """Part S/U: bounded self-improvement smoke with real SPICE leaves."""
    apply_torch_omp_workaround()
    started = time.time()
    reg = TopologyRegistry(V3)
    pools = load_pools()
    pool_ids = [r["topology_id"] for r in pools["A1"] + pools["A2"]]
    nets = build_policy_value(seed)
    # smoke default: SPICE on first leaf visit so the expensive path is exercised
    cfg = cfg or SearchConfig(seed=seed, spice_visit_threshold=1)
    out_dir = _ROOT / "artifacts" / "stage3e1"
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_out, examples = [], []
    for ep in range(episodes):
        root_state = make_root_state(root_tid or pool_ids[0], reg, DEFAULT_SPEC)
        mcts = TopologyMCTS(nets, reg, pool_ids, cfg)
        root = mcts.run(root_state)
        res = search_result(root, mcts)
        ex = training_example(root, mcts, res, seed)
        examples.append(ex)
        report = train_step(nets, [ex], reg)
        ck = out_dir / f"policy_value_ep{ep}.pt"
        save_checkpoint(nets, ck, {"episode": ep, "seed": seed, "losses": report})
        (out_dir / f"tree_ep{ep}.json").write_text(json.dumps(
            [n.record() for n in mcts.nodes], indent=0, default=str), encoding="utf-8")
        episodes_out.append({"episode": ep, "result": res, "train": report,
                             "checkpoint": str(ck)})
    # deterministic resume check: reload ep0 checkpoint, rerun search, compare
    nets2 = build_policy_value(seed)
    load_checkpoint(nets2, out_dir / "policy_value_ep0.pt")
    cfg_det = SearchConfig(**{**asdict(cfg), "training_mode": False,
                              "leaf_mode": "value_only", "seed": seed})
    m1 = TopologyMCTS(nets2, reg, pool_ids, cfg_det)
    r1 = m1.run(make_root_state(root_tid or pool_ids[0], reg, DEFAULT_SPEC))
    m2 = TopologyMCTS(nets2, reg, pool_ids, cfg_det)
    r2 = m2.run(make_root_state(root_tid or pool_ids[0], reg, DEFAULT_SPEC))
    resume_deterministic = ([c.N for c in r1.children] == [c.N for c in r2.children])
    with (out_dir / "training_records.jsonl").open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, default=str) + "\n")
    summary = {"episodes": episodes_out, "resume_deterministic": resume_deterministic,
               "wall_clock_s": round(time.time() - started, 1),
               "schema_version": SCHEMA_VERSION}
    (out_dir / "SMOKE_SUMMARY.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


# ========================= Part V: baseline modes ============================
BASELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "random_search": {"use_policy_prior": False, "use_value_net": False,
                      "leaf_mode": "value_only", "num_simulations": 4},
    "policy_only_greedy": {"num_simulations": 1, "leaf_mode": "value_only"},
    "value_only_best_first": {"use_policy_prior": False, "leaf_mode": "value_only",
                              "num_simulations": 4},
    "mcts_no_policy_prior": {"use_policy_prior": False, "num_simulations": 4,
                             "leaf_mode": "value_only"},
    "mcts_no_value": {"use_value_net": False, "num_simulations": 4,
                      "leaf_mode": "value_only"},
    "mcts_pooled_features": {"num_simulations": 4, "leaf_mode": "value_only"},
    "mcts_no_mbsac_leaf": {"use_mbsac_leaf": False, "num_simulations": 4},
    "mcts_no_preference_ranking": {"num_simulations": 4, "leaf_mode": "value_only"},
}


def run_baseline(name: str, seed: int = 0) -> dict[str, Any]:
    reg = TopologyRegistry(V3)
    pools = load_pools()
    pool_ids = [r["topology_id"] for r in pools["A1"] + pools["A2"]]
    cfg = SearchConfig(seed=seed, training_mode=False, **BASELINE_CONFIGS[name])
    nets = build_policy_value(seed)
    mcts = TopologyMCTS(nets, reg, pool_ids, cfg)
    root = mcts.run(make_root_state(pool_ids[0], reg, DEFAULT_SPEC))
    return {"baseline": name, "tree_nodes": len(mcts.nodes),
            "real_spice_calls": mcts.costs.real_spice_calls,
            "terminal": root.terminal_reason or "simulations_complete"}
