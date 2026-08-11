"""TRUE_ALPHAZERO topology search (2026-08-11): agentic_raptor.topology_rl.
alphazero replaces the retired root-level PUCT selector (run_raptor_v2.
puct_select_two -> stage3e1's one-root wiring, which could only ever
KEEP/TERMINATE/SELECT among complete LLM candidates -- proven via a real,
empirical call: root_action_ids never contained "a_comp" because the
_LLMRegistry it constructed had no derive_edited()).

These tests exercise the REAL engine (agentic_raptor.topology_rl.stage3e1.
TopologyMCTS, unmodified except for the generate_actions_fn/apply_action_fn
injection points) wired to the new LLMSeededEditRegistry, whose
derive_edited() calls the REAL, pre-existing stage3e2_edits.apply_edit() --
not mocked. Kept fast (small simulation counts, leaf_mode="value_only", no
SPICE, no GPU) but behaviorally real: same code path a genuine search runs.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from agentic_raptor.topology_rl import alphazero as az
from agentic_raptor.topology_rl.stage3e1 import (Stage3E1Action,
                                                  Stage3E1ActionType,
                                                  TopologySearchState)
from agentic_raptor.topology_rl.stage3e2_edits import EditRejected

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"


def _real_candidates(n=3):
    if not CORPUS.is_file():
        pytest.skip("corpus_diverse.json not present in this checkout")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    seen = {}
    for r in corpus["records"]:
        h = r["canonical_graph_hash"]
        if h not in seen:
            seen[h] = r
        if len(seen) >= n:
            break
    out = []
    for i, (h, r) in enumerate(seen.items()):
        out.append({"llm_proposal_id": f"p{i:02d}", "canonical_graph_hash": h,
                   "canonical_family": r["topology_signature"],
                   "obj": json.loads(r["response"]), "source": "llm"})
    return out


SPEC = {"spec_id": "az_test", "gain_target_db": 60.0,
       "phase_margin_target_deg": 55.0, "load_capacitance_pf": 100.0,
       "ugbw_target_hz": 1e5}


# ---------------------------------------------------------------------------
# real graph transitions
# ---------------------------------------------------------------------------
def test_registry_derive_edited_produces_real_distinct_graph():
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    seed = cands[0]["llm_proposal_id"]
    parent_hash = reg.get_topology(seed).graph.structural_hash()
    child_tid = reg.derive_edited(seed, "ADD_VERIFIED_STAGE")
    child_hash = reg.get_topology(child_tid).graph.structural_hash()
    assert child_tid != seed
    assert child_hash != parent_hash


def test_registry_derive_edited_raises_editrejected_when_illegal():
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    seed = cands[0]["llm_proposal_id"]
    # REMOVE_OPTIONAL_SUPPORTED_STAGE on a graph with no removable optional
    # stage must be rejected, not silently no-op or crash with something
    # other than EditRejected.
    with pytest.raises(EditRejected):
        reg.derive_edited(seed, "REMOVE_OPTIONAL_SUPPORTED_STAGE")


def test_derive_edited_is_deterministic_and_cached():
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    seed = cands[0]["llm_proposal_id"]
    a = reg.derive_edited(seed, "ADD_VERIFIED_STAGE")
    b = reg.derive_edited(seed, "ADD_VERIFIED_STAGE")
    assert a == b


# ---------------------------------------------------------------------------
# legal action generation / application
# ---------------------------------------------------------------------------
def test_super_root_offers_only_select_actions():
    """Section 1 (Campaign 01B): the virtual super-root's ONLY legal
    actions are SELECT_EXISTING_TOPOLOGY -- no edits, no a_keep, no
    a_term. Postmortem on CAMPAIGN_01_ATTEMPT_1: the previous version
    offered edits AND selects AND a_keep simultaneously at the root."""
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    state = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    legal, _rejections = az.generate_alphazero_actions(state, reg, reg.seed_ids)
    assert legal, "no legal action at the super-root"
    assert all(a.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY
              for a in legal)
    assert all(a.action_id.startswith("a_sel_") for a in legal)
    assert not any(a.action_id == "a_keep" for a in legal)
    assert not any(a.action_id == "a_term" for a in legal)
    assert {a.source_ref for a in legal} <= set(reg.seed_ids)


def test_post_selection_offers_edits_and_terminate_never_select():
    """Section 1: once committed to a seed, SELECT_EXISTING_TOPOLOGY must
    NEVER be offered again -- only real structural edits and TERMINATE."""
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    root_state = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    seed_id = reg.seed_ids[0]
    select_action = Stage3E1Action(f"a_sel_{seed_id}",
                                   Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                                   source_ref=seed_id)
    committed_state = az.apply_alphazero_action(root_state, select_action, reg)
    assert committed_state.topology_id == seed_id

    legal, _rejections = az.generate_alphazero_actions(committed_state, reg, reg.seed_ids)
    edit_actions = [a for a in legal if a.action_type in az.AZ_EDIT_ACTION_TYPES]
    assert edit_actions, "no real structural-edit action was legal post-selection"
    assert any(a.action_id == "a_term" for a in legal)
    assert not any(a.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY
                  for a in legal), "SELECT re-offered after seed commitment"
    assert not any(a.action_id == "a_keep" for a in legal)

    # depth > super-root, still no SELECT, even after an edit is applied
    edit_action = next(a for a in edit_actions)
    edited_state = az.apply_alphazero_action(committed_state, edit_action, reg)
    legal2, _ = az.generate_alphazero_actions(edited_state, reg, reg.seed_ids)
    assert not any(a.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY
                  for a in legal2), "SELECT re-offered at depth > 1"


def test_apply_alphazero_action_edit_changes_topology_and_hash():
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    state = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    action = Stage3E1Action("a_edit_ADD_VERIFIED_STAGE",
                            Stage3E1ActionType.ADD_VERIFIED_STAGE,
                            source_ref=state.topology_id)
    child = az.apply_alphazero_action(state, action, reg)
    assert child.topology_id != state.topology_id
    assert child.graph_hash != "proposal_root"
    assert child.depth == state.depth + 1
    assert child.lineage == state.lineage + [child.graph_hash]


def test_apply_alphazero_action_select_does_not_edit():
    """SELECT_EXISTING_TOPOLOGY must still be a candidate swap, not an
    edit -- the two mechanisms coexist, they must not be conflated."""
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    state = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    other = [s for s in reg.seed_ids if s != reg.seed_ids[0]][0]
    action = Stage3E1Action(f"a_sel_{other}",
                            Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                            source_ref=other)
    child = az.apply_alphazero_action(state, action, reg)
    assert child.topology_id == other
    assert child.graph_hash == reg.get_topology(other).graph.structural_hash()


# ---------------------------------------------------------------------------
# mechanical search properties (Section 29)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_search_result():
    cands = _real_candidates(3)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=48,
                             alphazero_max_edit_depth=4, seed=0)
    return az.run_alphazero_search(cands, SPEC, "az_test", seed=0, config=cfg)


def test_search_reaches_depth_greater_than_one(small_search_result):
    assert small_search_result["max_depth_reached"] > 1


def test_search_tree_has_multiple_nodes(small_search_result):
    assert small_search_result["tree_nodes"] > len(small_search_result["seed_ids"]) + 2


def test_search_can_find_topologies_absent_from_seed_pool(small_search_result):
    assert small_search_result["novel_vs_seed_count"] > 0
    for h in small_search_result["novel_vs_seed_hashes"]:
        assert h not in small_search_result["seed_topology_hashes"]


def test_root_visit_distribution_normalised(small_search_result):
    total = sum(small_search_result["root_visit_distribution"].values())
    assert abs(total - 1.0) < 1e-6


def test_node_N_W_Q_consistency(small_search_result):
    """Q must equal W/N for every visited node (Node.Q's own invariant).
    Node.record() independently rounds Q and W to 4dp for the serialized
    trace, so the reconstructed check tolerates that rounding rather than
    demanding exact equality on already-rounded numbers."""
    for rec in small_search_result["nodes"]:
        if rec["N"] > 0:
            assert abs(rec["Q"] - rec["W"] / rec["N"]) < 2e-4


def test_selected_topology_is_a_registered_real_topology(small_search_result):
    """"proposal_root" is never returned directly (run_alphazero_search
    resolves it to seed_ids[0], since it's always an alias for that seed's
    own borrowed graph). Since Section 1 (Campaign 01B), the super-root's
    ONLY legal actions are SELECT_EXISTING_TOPOLOGY, so run_alphazero_
    search()'s root-level `selected_topology_id` (best_child of the ROOT
    specifically, see stage3e1.search_result) is always one of the
    ORIGINAL seed ids -- an edited descendant can still win a slot in
    alphazero_select_two()'s whole-tree ranking (see
    test_alphazero_select_two_can_pick_an_edited_descendant), just never
    at this single root-level readout."""
    sel = small_search_result["selected_topology_id"]
    assert sel is not None
    assert sel != "proposal_root"
    assert sel in small_search_result["seed_ids"]


def test_puct_score_matches_documented_equation():
    """score(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)),
    the exact equation implemented in stage3e1.TopologyMCTS.puct() (reused
    unmodified by the AlphaZero engine)."""
    from agentic_raptor.topology_rl.stage3e1 import Node, SearchConfig, TopologyMCTS
    parent_state = TopologySearchState(
        topology_id="x", graph_hash="hx", lineage=[], spec={},
        rag_context_ids=[], available_blocks=[], legal_action_ids=[],
        edit_history=[], validation_status="validated", structural_features={},
        previous_evidence_ref=None, remaining_search_budget=8,
        remaining_spice_budget=0, depth=0)
    parent = Node(0, parent_state, None, None, 1.0)
    parent.N = 9
    child = Node(1, parent_state, parent, None, 0.4)
    child.N, child.W = 3, 1.5
    cfg = SearchConfig(c_puct=1.5)
    mcts = TopologyMCTS.__new__(TopologyMCTS)   # skip __init__'s torch/net setup
    mcts.cfg = cfg
    import math
    expected_u = cfg.c_puct * child.prior * math.sqrt(max(parent.N, 1)) / (1 + child.N)
    expected = child.Q + expected_u
    assert abs(mcts.puct(parent, child) - expected) < 1e-9
    assert abs(expected - (0.5 + 1.5 * 0.4 * 3 / 4)) < 1e-9


def test_no_sign_reversal_in_backup():
    """Section 20: circuit design has no opponent -- the backup loop must
    never negate the value climbing toward the root. Source-inspected
    (not just black-box) so a future edit can't silently reintroduce
    alternating-sign backup without this test catching it."""
    from agentic_raptor.topology_rl.stage3e1 import TopologyMCTS
    src = inspect.getsource(TopologyMCTS.run)
    assert "-v" not in src.replace(" ", "")
    assert "node.W += v" in src or "node.W +=v" in src.replace(" ", "")


def test_terminate_action_sets_terminal_reason():
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    state = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    action = Stage3E1Action("a_term", Stage3E1ActionType.TERMINATE_SEARCH)
    child = az.apply_alphazero_action(state, action, reg)
    assert child.terminal_reason == "terminate_action"


# ---------------------------------------------------------------------------
# Section 2/4/5 (Campaign 01B): termination, root exploration, sampling --
# a controlled/fixed policy_forward lets these be proven mechanically
# rather than hoping a real untrained net happens to behave as needed.
# ---------------------------------------------------------------------------
def _fixed_policy_forward(weight_fn):
    """A drop-in nets["policy_forward"] replacement returning a FIXED
    distribution (weight_fn(action) -> relative weight) instead of a real
    network's output -- used to force specific, reproducible search
    behavior for mechanical proofs."""
    import torch

    def _fwd(state, actions, reg):
        actions = sorted(actions, key=lambda a: a.action_id)
        weights = torch.tensor([max(float(weight_fn(a)), 1e-6) for a in actions],
                               dtype=torch.float32)
        probs = weights / weights.sum()
        lg = torch.log(probs)
        return actions, lg, probs
    return _fwd


def test_terminate_can_occur_before_max_edit_depth():
    """Section 2 (Campaign 01B): a controlled policy that overwhelmingly
    prefers TERMINATE_SEARCH once it is offered (i.e. post-selection) must
    make the episode end well before max_episode_depth -- proving
    TERMINATE is genuinely reachable at any post-selection step, not
    artificially forced to run the full depth (as every one of
    CAMPAIGN_01_ATTEMPT_1's 8 real episodes in fact did)."""
    cands = _real_candidates(2)
    nets = az.load_alphazero_nets(seed=0)
    biased = dict(nets)
    biased["policy_forward"] = _fixed_policy_forward(
        lambda a: 40.0 if a.action_type == Stage3E1ActionType.TERMINATE_SEARCH else 1.0)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=32,
                             alphazero_max_edit_depth=4, seed=0)
    ep = az.run_alphazero_episode(cands, SPEC, "term_test", "term_test_hash",
                                  seed=0, config=cfg, max_episode_depth=4,
                                  deterministic=True, nets=biased)
    assert len(ep["steps"]) < 4, "episode ran to full depth despite a policy that heavily favors TERMINATE"
    assert ep["steps"][-1]["selected_action_type"] == "TERMINATE_SEARCH"


def test_dirichlet_noise_enabled_only_when_training_mode_true():
    """Section 4 (Campaign 01B): AlphaZeroConfig.dirichlet_epsilon/alpha
    must actually reach stage3e1's existing root-noise machinery, and only
    take effect when training_mode=True. Compares root-child PRIORS (not
    N, which also depends on simulation order) across two different
    cfg.seed values: identical priors regardless of seed when
    training_mode=False (pure policy_forward, deterministic); DIFFERENT
    priors when training_mode=True with dirichlet_epsilon > 0 (numpy
    Dirichlet noise keyed by seed)."""
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    root_state = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    nets = az.load_alphazero_nets(seed=0)

    def _priors(training_mode, seed):
        cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=1,
                                 alphazero_max_edit_depth=2,
                                 training_mode=training_mode,
                                 dirichlet_epsilon=0.25, dirichlet_alpha=0.3, seed=seed)
        root, _mcts = az._search_from_state(root_state, reg, reg.seed_ids, nets, cfg)
        return {c.action.action_id: c.prior for c in root.children}

    off_a, off_b = _priors(False, 0), _priors(False, 1)
    assert off_a == pytest.approx(off_b, abs=1e-9), \
        "priors differed across seeds with training_mode=False -- noise leaked outside training"

    on_a, on_b = _priors(True, 0), _priors(True, 1)
    assert any(abs(on_a[k] - on_b[k]) > 1e-6 for k in on_a), \
        "priors were identical across seeds with training_mode=True -- Dirichlet noise never applied"


def test_alphazero_select_two_never_gets_training_mode_noise_by_default():
    """Section 4's IMPORTANT note: FULL selection (live inference) must
    never see root exploration noise. alphazero_select_two()'s default
    AlphaZeroConfig() has training_mode=False -- verified directly on the
    dataclass default, not just by convention."""
    assert az.AlphaZeroConfig().training_mode is False


def test_episode_deterministic_mode_is_reproducible():
    """Section 5: deterministic=True (validation/paper-inference shape)
    must be exactly reproducible -- no RNG draw ever influences which
    action is taken."""
    cands = _real_candidates(2)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=16,
                             alphazero_max_edit_depth=2, seed=0)
    a = az.run_alphazero_episode(cands, SPEC, "det_test", "det_test_hash",
                                 seed=0, config=cfg, max_episode_depth=2,
                                 deterministic=True)
    b = az.run_alphazero_episode(cands, SPEC, "det_test", "det_test_hash",
                                 seed=0, config=cfg, max_episode_depth=2,
                                 deterministic=True)
    assert [s["selected_action_id"] for s in a["steps"]] == \
        [s["selected_action_id"] for s in b["steps"]]


def test_deterministic_and_stochastic_modes_can_diverge():
    """Section 5: deterministic=False (TRAIN collection) must be CAPABLE
    of sampling a different action than the deterministic max-visit
    choice when pi assigns real probability elsewhere -- proven with a
    near-uniform fixed policy (both SELECT actions at the super-root get
    comparable visit counts) and a small search over rollout seeds."""
    cands = _real_candidates(2)
    nets = az.load_alphazero_nets(seed=0)
    uniform = dict(nets)
    uniform["policy_forward"] = _fixed_policy_forward(lambda a: 1.0)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=16,
                             alphazero_max_edit_depth=1,
                             alphazero_temperature=1.0, seed=0)
    det = az.run_alphazero_episode(cands, SPEC, "temp_test", "temp_test_hash",
                                   seed=0, config=cfg, max_episode_depth=1,
                                   deterministic=True, nets=uniform)
    det_action = det["steps"][0]["selected_action_id"]
    diverged = False
    for seed in range(8):
        st = az.run_alphazero_episode(cands, SPEC, "temp_test", "temp_test_hash",
                                      seed=seed, config=cfg, max_episode_depth=1,
                                      deterministic=False, nets=uniform)
        if st["steps"][0]["selected_action_id"] != det_action:
            diverged = True
            break
    assert diverged, "stochastic sampling never diverged from the deterministic choice across 8 seeds"


# ---------------------------------------------------------------------------
# Section 3 (Campaign 01B): per-episode RNG derivation
# ---------------------------------------------------------------------------
def test_stable_episode_rng_seed_deterministic_across_reruns():
    a = az.stable_episode_rng_seed(0, "AZ_G1", "spechash123", 0)
    b = az.stable_episode_rng_seed(0, "AZ_G1", "spechash123", 0)
    assert a == b


def test_stable_episode_rng_seed_differs_across_specs():
    a = az.stable_episode_rng_seed(0, "AZ_G1", "spec_a", 0)
    b = az.stable_episode_rng_seed(0, "AZ_G1", "spec_b", 0)
    assert a != b


def test_stable_episode_rng_seed_differs_across_rollout_seeds():
    a = az.stable_episode_rng_seed(0, "AZ_G1", "spechash123", 0)
    b = az.stable_episode_rng_seed(0, "AZ_G1", "spechash123", 1)
    assert a != b


def test_stable_episode_rng_seed_differs_across_campaign_seeds():
    a = az.stable_episode_rng_seed(0, "AZ_G1", "spechash123", 0)
    b = az.stable_episode_rng_seed(7, "AZ_G1", "spechash123", 0)
    assert a != b


def test_episode_uses_stable_rng_seed_not_rollout_seed_alone():
    """The exact bug found via CAMPAIGN_01_ATTEMPT_1: every episode
    previously called random.Random(seed) with `seed` being ONLY the
    rollout seed. run_alphazero_episode's returned episode_rng_seed must
    match stable_episode_rng_seed()'s own derivation, and two episodes
    that differ only in spec_hash must get DIFFERENT episode_rng_seed."""
    cands = _real_candidates(2)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=8,
                             alphazero_max_edit_depth=1, seed=0)
    ep_a = az.run_alphazero_episode(cands, SPEC, "rng_test", "hash_a", seed=0,
                                    campaign_seed=3, config=cfg, max_episode_depth=1,
                                    generation_id="AZ_G1")
    ep_b = az.run_alphazero_episode(cands, SPEC, "rng_test", "hash_b", seed=0,
                                    campaign_seed=3, config=cfg, max_episode_depth=1,
                                    generation_id="AZ_G1")
    assert ep_a["episode_rng_seed"] == az.stable_episode_rng_seed(3, "AZ_G1", "hash_a", 0)
    assert ep_a["episode_rng_seed"] != ep_b["episode_rng_seed"]


# ---------------------------------------------------------------------------
# Section 6 (Campaign 01B): spec-conditioning audit
# ---------------------------------------------------------------------------
def test_spec_conditioning_changes_policy_and_value_outputs():
    """Section 6: feeding the SAME topology state with two meaningfully
    different specs must change the policy logits and/or the value
    scalar -- an untrained net is not required to make GOOD decisions,
    but its outputs must not be spec-invariant (which would indicate a
    disconnected/missing conditioning path)."""
    import torch

    nets = az.load_alphazero_nets(seed=0)
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    state_a = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    alt_spec = {**SPEC, "gain_target_db": SPEC["gain_target_db"] + 40.0,
               "phase_margin_target_deg": SPEC["phase_margin_target_deg"] - 20.0}
    state_b = az.build_root_state(alt_spec, "az_test", reg.seed_ids)
    legal, _ = az.generate_alphazero_actions(state_a, reg, reg.seed_ids)

    _, logits_a, _probs_a = nets["policy_forward"](state_a, legal, reg)
    v_a = nets["value_forward"](state_a, reg)["scalar"]
    _, logits_b, _probs_b = nets["policy_forward"](state_b, legal, reg)
    v_b = nets["value_forward"](state_b, reg)["scalar"]

    assert not torch.allclose(logits_a, logits_b, atol=1e-6), \
        "policy logits identical across distinct specs -- spec conditioning may be disconnected"
    assert abs(v_a.item() - v_b.item()) > 1e-6, \
        "value scalar identical across distinct specs -- spec conditioning may be disconnected"


def test_spec_conditioning_pathway_receives_gradient():
    """Section 6: gradients must actually propagate from BOTH the policy
    and value outputs back through the spec-conditioning pathway, and the
    resulting parameter gradients must genuinely DEPEND on which spec was
    fed in (not just be nonzero from some spec-independent shortcut)."""
    nets = az.load_alphazero_nets(seed=0)
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    state_a = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    alt_spec = {**SPEC, "gain_target_db": SPEC["gain_target_db"] + 40.0}
    state_b = az.build_root_state(alt_spec, "az_test", reg.seed_ids)
    legal, _ = az.generate_alphazero_actions(state_a, reg, reg.seed_ids)

    def _grads_for(state):
        for p in nets["params"]:
            p.grad = None
        _, _lg, probs = nets["policy_forward"](state, legal, reg)
        v = nets["value_forward"](state, reg)["scalar"]
        (probs.sum() + v).backward()
        return [p.grad.clone() for p in nets["params"] if p.grad is not None]

    grads_a = _grads_for(state_a)
    grads_b = _grads_for(state_b)
    assert grads_a, "backward() produced no gradients at all -- pathway is dead"
    assert any(g.abs().sum() > 0 for g in grads_a), "all gradients were exactly zero"
    assert any((ga - gb).abs().sum() > 1e-8 for ga, gb in zip(grads_a, grads_b)), \
        "gradients identical across distinct specs -- spec pathway carries no real signal"


# ---------------------------------------------------------------------------
# episode / replay / training / generations (Section 25 categories)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_episode():
    cands = _real_candidates(3)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=24,
                             alphazero_max_edit_depth=3, seed=0)
    return az.run_alphazero_episode(cands, SPEC, "ep_test", "ep_test_hash",
                                    seed=0, config=cfg, max_episode_depth=3)


FAKE_TERMINAL = {"electrical_environment_version": "POST_CLOAD_FIX_V1",
                 "requested_c_load_f": 1e-10, "simulated_c_load_f": 1e-10,
                 "call_id": "fake:call:1", "z": 0.42}


def test_episode_produces_real_multistep_trajectory(small_episode):
    assert len(small_episode["steps"]) >= 1
    for s in small_episode["steps"]:
        assert abs(sum(s["pi"].values()) - 1.0) < 1e-6
        assert s["selected_action_id"] in s["legal_action_ids"]


def test_episode_state_topology_changes_across_steps(small_episode):
    ids = [s["state_topology_id"] for s in small_episode["steps"]]
    # at least the FIRST step must be the neutral root; subsequent steps
    # (when present) must reflect wherever the real transition landed
    assert ids[0] == "proposal_root"
    if len(ids) > 1:
        assert len(set(ids)) > 1, "episode never actually moved states"


def test_terminal_topology_hash_reflects_the_true_final_state(small_episode):
    """Bug found via a real campaign dry-run: `steps` only ever records a
    state BEFORE its own action is applied, so the truly terminal state
    (after the LAST action) is never itself a step -- episode[
    "terminal_topology_hash"] must be computed from the real final state,
    never derived as steps[-1]["state_graph_hash"] (that's the SECOND-TO-
    LAST state, off by one)."""
    reg = small_episode["registry"]
    expected = reg.get_topology(small_episode["terminal_topology_id"]).graph.structural_hash()
    assert small_episode["terminal_topology_hash"] == expected
    if small_episode["steps"]:
        # only coincidentally equal to the last step's PRE-action hash when
        # the last action was the no-op-shaped TERMINATE_SEARCH (KEEP_
        # TOPOLOGY no longer exists in the action space, Section 1); for a
        # real SELECT/edit step (the common case) they must differ.
        last_step = small_episode["steps"][-1]
        if last_step["selected_action_type"] != "TERMINATE_SEARCH":
            assert small_episode["terminal_topology_hash"] != last_step["state_graph_hash"]


def test_build_replay_rows_terminal_hash_matches_episode_terminal_hash(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    for r in rows:
        assert r["terminal_topology_hash"] == small_episode["terminal_topology_hash"]


def test_build_replay_rows_schema_and_z_propagation(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL,
                                checkpoint_hash="fakehash")
    assert len(rows) == len(small_episode["steps"])
    required = {"generation", "spec_hash", "context_id", "step", "state",
               "state_topology_id", "state_graph_hash", "edit_depth",
               "edit_history", "legal_action_ids", "raw_visit_counts", "pi",
               "selected_action_id", "z", "terminal_topology_hash",
               "terminal_authoritative_call_id", "requested_c_load_f",
               "simulated_c_load_f", "electrical_environment_version",
               "policy_value_checkpoint_hash", "seed", "split"}
    for r in rows:
        assert required <= set(r.keys())
        assert r["z"] == pytest.approx(0.42)   # SAME z, every step, no sign flips


def test_build_replay_rows_rejects_protected_spec(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL,
                                protected_context_ids=frozenset({"ep_test"}))
    assert rows == []


def test_build_replay_rows_rejects_pre_cload_fix(small_episode):
    bad = {**FAKE_TERMINAL, "electrical_environment_version": "PRE_CLOAD_FIX"}
    assert az.build_replay_rows(small_episode, bad) == []


def test_build_replay_rows_rejects_cload_mismatch(small_episode):
    bad = {**FAKE_TERMINAL, "simulated_c_load_f": 2e-10}
    assert az.build_replay_rows(small_episode, bad) == []


def test_build_replay_rows_rejects_missing_z(small_episode):
    bad = {**FAKE_TERMINAL, "z": None}
    assert az.build_replay_rows(small_episode, bad) == []


def test_training_step_actually_updates_parameters(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    rep = az.train_az_generation(rows, small_episode["registry"], epochs=1, seed=0)
    assert rep["examples_used"] == len(rows)
    r0 = rep["epoch_reports"][0]
    assert r0["policy_params_changed"] is True
    assert r0["encoder_params_changed"] is True
    assert r0["policy_loss"] >= 0
    assert r0["value_loss"] >= 0


def test_training_reproducible_with_fixed_seed(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    r1 = az.train_az_generation(rows, small_episode["registry"], epochs=1, seed=0)
    r2 = az.train_az_generation(rows, small_episode["registry"], epochs=1, seed=0)
    assert r1["epoch_reports"][0]["policy_loss"] == pytest.approx(
        r2["epoch_reports"][0]["policy_loss"], abs=1e-6)


def test_g0_manifest_is_random_untrained_and_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(az, "AZ_GENERATIONS_ROOT", tmp_path / "az_generations")
    g0 = az.ensure_az_g0_manifest()
    assert g0["generation_id"] == "AZ_G0"
    assert g0["initialization"] == "random_untrained"
    assert g0["immutable"] is True
    with pytest.raises(RuntimeError, match="immutable"):
        az.write_az_generation_manifest("AZ_G0", {"x": 1})
    g0_again = az.ensure_az_g0_manifest()
    assert g0_again == g0


def test_next_generation_id():
    assert az.next_az_generation_id("AZ_G0") == "AZ_G1"
    assert az.next_az_generation_id("AZ_G1") == "AZ_G2"


def test_validate_az_candidate_passes_mechanical_checks():
    cands = _real_candidates(3)
    nets = az.load_alphazero_nets(value_ckpt=None, seed=0)
    result = az.validate_az_candidate(nets, cands, SPEC, "val_test", seed=0)
    assert result["passed"], result["failures"]
    assert result["max_depth_reached"] > 1


# ---------------------------------------------------------------------------
# live FULL contract: alphazero_select_two / direct_prior_select_two
# ---------------------------------------------------------------------------
def test_alphazero_select_two_returns_distinct_canonical_hashes():
    cands = _real_candidates(4)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=48,
                             alphazero_max_edit_depth=3, seed=0)
    sel = az.alphazero_select_two(cands, SPEC, "sel2_test", seed=0, config=cfg)
    assert len(sel["selected"]) == 2
    a, b = sel["selected"]
    assert a["canonical_graph_hash"] != b["canonical_graph_hash"]
    assert a["device_graph"] is not None and b["device_graph"] is not None
    assert a["rank"] == 0 and b["rank"] == 1
    assert a["visit_count"] >= b["visit_count"]


def test_alphazero_select_two_can_pick_an_edited_descendant():
    """Must be POSSIBLE for an edited descendant to win a top-2 slot, not
    guaranteed for a real, untrained net on an arbitrary seed. Since
    Section 1 (Campaign 01B)'s two-phase action space made edited nodes
    structurally one level deeper than the seed they descend from (N(seed)
    = sum of N over ALL its children, so a seed's own N necessarily
    upper-bounds any single edit beneath it) -- an untrained G0's fairly
    flat, weakly-differentiated value estimates no longer reliably produce
    this outcome within a small simulation budget purely by chance (this
    was tried up to 512 simulations x 10 seeds x {2, 4} candidates with
    the real net and never once won). This is an EXPECTED, correct
    consequence of the fix (edits are no longer direct root children),
    not a regression -- proven here instead with a controlled value net
    that strongly and specifically prefers one real, legally-derived
    edited state over everything else (including the alternate seed),
    which is exactly the shape a genuinely TRAINED value net's confident
    preference would take."""
    import torch

    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    target_tid = reg.derive_edited(reg.seed_ids[0], "ADD_VERIFIED_STAGE")
    base_nets = az.load_alphazero_nets(seed=0)
    real_value_forward = base_nets["value_forward"]

    def _targeted_value_forward(state, r):
        out = dict(real_value_forward(state, r))
        out["scalar"] = (torch.tensor(1.0) if state.topology_id == target_tid
                         else torch.tensor(-1.0))
        return out

    nets = {**base_nets, "value_forward": _targeted_value_forward}
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=128,
                             alphazero_max_edit_depth=4, seed=0)
    sel = az.alphazero_select_two(cands, SPEC, "sel2_edit_test", seed=0,
                                  config=cfg, nets=nets)
    assert any(c["is_edited_descendant"] and c["llm_proposal_id"] == target_tid
              for c in sel["selected"]), (
        f"targeted edited descendant {target_tid!r} never won a top-2 slot: "
        f"{sel['ranked_all'][:5]}")


def test_direct_prior_select_two_never_uses_search():
    cands = _real_candidates(3)
    sel = az.direct_prior_select_two(list(cands), SPEC)
    assert len(sel["selected"]) == 2
    for c in sel["selected"]:
        assert c["device_graph"] is None    # A5 never edits
        assert c["visit_count"] == 0         # zero search involvement
    assert sel["selected"][0]["canonical_graph_hash"] != sel["selected"][1]["canonical_graph_hash"]


def test_select_two_functions_hard_fail_on_too_few_candidates():
    with pytest.raises(az.AlphaZeroSelectionError):
        az.alphazero_select_two(_real_candidates(1), SPEC, "x")
    with pytest.raises(az.AlphaZeroSelectionError):
        az.direct_prior_select_two(_real_candidates(1), SPEC)


def test_require_promoted_checkpoint_hard_fails_without_one(tmp_path, monkeypatch):
    monkeypatch.setattr(az, "AZ_GENERATIONS_ROOT", tmp_path / "empty_generations")
    with pytest.raises(az.AlphaZeroSelectionError, match="PROMOTED"):
        az.require_promoted_az_checkpoint()


def test_require_promoted_checkpoint_finds_a_promoted_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(az, "AZ_GENERATIONS_ROOT", tmp_path / "gens")
    az.write_az_generation_manifest("AZ_G0", {
        "generation_id": "AZ_G0", "checkpoint_status": "SMOKE",
        "checkpoint_path": "fake_g0.pt"})
    az.write_az_generation_manifest("AZ_G1", {
        "generation_id": "AZ_G1", "checkpoint_status": "PROMOTED",
        "checkpoint_path": "fake_g1.pt"})
    result = az.require_promoted_az_checkpoint()
    assert str(result) == "fake_g1.pt"


def test_edited_graph_identity_preserved_through_size_and_predict():
    """Section 6: an AlphaZero-selected edited descendant's realised graph
    must survive INTACT into MB-SAC -- never silently re-derived back to
    its parent family template. Real size_and_predict() call (small
    budget, real SPICE)."""
    from agentic_raptor.electrical import discover_ngspice
    from agentic_raptor.topology_rl.value_refresh import \
        device_graph_to_circuit_graph
    from run_raptor_v2 import size_and_predict

    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    seed_id = cands[0]["llm_proposal_id"]
    edited_tid = reg.derive_edited(seed_id, "ADD_VERIFIED_STAGE")
    edited_dg = reg._device_graphs[edited_tid]
    edited_hash = device_graph_to_circuit_graph(edited_dg, edited_tid).structural_hash()
    assert edited_hash != cands[0]["canonical_graph_hash"]

    sel_edited = {"llm_proposal_id": edited_tid, "canonical_graph_hash": edited_hash,
                 "canonical_family": "edited", "obj": None,
                 "device_graph": edited_dg, "rank": 0, "source": "alphazero"}
    sel_seed = {**cands[0], "device_graph": None, "rank": 1}

    exe = discover_ngspice()
    branches = size_and_predict([sel_edited, sel_seed], SPEC, exe, budget=3)
    (da, *_rest_a), (db, *_rest_b) = branches
    assert da.canonical_graph_hash == edited_hash
    assert da.canonical_graph_hash != cands[0]["canonical_graph_hash"]


def test_full_pipeline_new_search_key_present_not_old_key():
    """run_pipeline's trace uses "stage5_alphazero" now, never
    "stage5_puct" for a NEW run -- structural check on the source (not a
    full pipeline execution, which needs real SPICE/RAG/GPU)."""
    import inspect

    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    assert '"stage5_alphazero"' in src or "trace[\"stage5_alphazero\"]" in src
    # the retired function's DEFINITION still exists elsewhere in the
    # module (Section 12/20: historical, non-executable-from-FULL) and a
    # comment here explains the migration by name -- neither counts. What
    # must be absent is an actual CALL to it.
    assert "= puct_select_two(" not in src
    assert "alphazero_select_two(" in src
    assert "direct_prior_select_two(" in src


def test_illegal_edit_is_excluded_from_legal_actions_not_just_apply():
    """The legality PROBE (generate_alphazero_actions, via validate_
    alphazero_candidate) and the actual APPLICATION (apply_alphazero_
    action) must agree -- both go through registry.derive_edited(), so an
    edit rejected at generation time can never somehow succeed if applied
    anyway. Uses a single-stage-free graph state directly via a family
    template edge case: REMOVE_OPTIONAL_SUPPORTED_STAGE with no optional
    stage present."""
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    root_state = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    seed_id = reg.seed_ids[0]
    select_action = Stage3E1Action(f"a_sel_{seed_id}",
                                   Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                                   source_ref=seed_id)
    # edits are only ever offered post-selection (Section 1) -- the
    # legality probe under test must be exercised on such a state, not
    # the super-root (which no longer offers edits at all).
    state = az.apply_alphazero_action(root_state, select_action, reg)
    legal, rejections = az.generate_alphazero_actions(state, reg, reg.seed_ids)
    rejected_ids = {r["action_id"] for r in rejections}
    legal_ids = {a.action_id for a in legal}
    assert not (rejected_ids & legal_ids), "an action was both rejected and legal"
