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
def test_generate_alphazero_actions_offers_real_edit_types():
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    state = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    legal, _rejections = az.generate_alphazero_actions(state, reg, reg.seed_ids)
    edit_actions = [a for a in legal if a.action_type in az.AZ_EDIT_ACTION_TYPES]
    assert edit_actions, "no real structural-edit action was ever legal at the root"
    assert any(a.action_id == "a_keep" for a in legal)
    assert any(a.action_id == "a_term" for a in legal)
    assert any(a.action_id.startswith("a_sel_") for a in legal)


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
    own borrowed graph) -- but landing on an UNEDITED seed is still a
    legitimate outcome: the search can rationally decide a_term beats
    every edit/reseed option it explored within budget."""
    sel = small_search_result["selected_topology_id"]
    assert sel is not None
    assert sel != "proposal_root"
    assert sel in small_search_result["seed_ids"] or "~" in sel


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
    """Not guaranteed on every spec/seed (depends on the value net's own
    scoring), but must be POSSIBLE -- ran with a wide simulation budget
    across a few seeds until one produces an edited top-2 pick, proving
    the mechanism, not asserting it always happens."""
    cands = _real_candidates(4)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=64,
                             alphazero_max_edit_depth=4, seed=0)
    found = False
    for seed in range(4):
        sel = az.alphazero_select_two(cands, SPEC, "sel2_edit_test", seed=seed,
                                      config=az.AlphaZeroConfig(
                                          alphazero_simulations_per_move=64,
                                          alphazero_max_edit_depth=4, seed=seed))
        if any(c["is_edited_descendant"] for c in sel["selected"]):
            found = True
            break
    assert found, "no edited descendant ever won a top-2 slot across 4 seeds"


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
    state = az.build_root_state(SPEC, "az_test", reg.seed_ids)
    legal, rejections = az.generate_alphazero_actions(state, reg, reg.seed_ids)
    rejected_ids = {r["action_id"] for r in rejections}
    legal_ids = {a.action_id for a in legal}
    assert not (rejected_ids & legal_ids), "an action was both rejected and legal"
