"""AlphaZero FINAL PERFORMANCE REPAIR: tests for the Phase 1 selection
audit and the Phase 2 principal-variation repaired selector
(run_per_seed_alphazero_search_pv / extract_principal_variation).
Real engine, real edits, no SPICE/LLM (frozen corpus fixture).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.topology_rl import alphazero as az
from agentic_raptor.topology_rl.stage3e1 import Stage3E1ActionType

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"
AUDIT = ROOT / "artifacts/publication_v3/azsel_phase1_selection_audit/PHASE1_SELECTION_AUDIT.json"


def _real_candidates(n=4):
    if not CORPUS.is_file():
        pytest.skip("corpus_diverse.json not present")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    seen = {}
    for r in corpus["records"]:
        h = r["canonical_graph_hash"]
        if h not in seen:
            seen[h] = r
        if len(seen) >= n:
            break
    return [{"llm_proposal_id": f"p{i:02d}", "canonical_graph_hash": h,
            "canonical_family": r["topology_signature"],
            "obj": json.loads(r["response"]), "source": "llm"}
           for i, (h, r) in enumerate(seen.items())]


SPEC = {"spec_id": "az_test", "gain_target_db": 60.0,
       "phase_margin_target_deg": 55.0, "load_capacitance_pf": 100.0,
       "ugbw_target_hz": 1e5}
CFG = lambda **kw: az.PerSeedAlphaZeroConfig(   # noqa: E731
    total_mcts_simulations=64, alphazero_max_edit_depth=3, **kw)


# ---------------------------------------------------------------------------
# SELECTION AUDIT
# ---------------------------------------------------------------------------
def test_phase1_audit_confirmed_depth_bias():
    if not AUDIT.is_file():
        pytest.skip("phase 1 audit not run in this checkout")
    d = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert d["classification"] == "OUTPUT_SELECTION_DEPTH_BIAS_CONFIRMED"
    ps = d["aggregate"]["per_seed"]
    # the mathematical signature: no descendant's summed N ever reached the
    # weakest selected root's N
    assert ps["max_descendant_N"] < ps["min_selected_N"]
    assert ps["selected_edited_count"] == 0


def test_old_rule_still_reproducible_as_baseline():
    """PER_SEED_OLD_SELECTION must remain byte-for-byte available as the
    audited historical baseline (Phase 9 control)."""
    cands = _real_candidates(4)
    r = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0, config=CFG())
    assert all(not c["is_edited_descendant"] for c in r["selected"]), (
        "the OLD summed-N rule selecting an edited descendant would "
        "contradict the Phase-1 structural proof")


# ---------------------------------------------------------------------------
# PRINCIPAL VARIATION
# ---------------------------------------------------------------------------
def test_pv_traversal_follows_highest_visit_action():
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    nets = az.load_alphazero_nets(None, seed=0)
    from agentic_raptor.topology_rl import stage3e1 as s1
    cfg = s1.SearchConfig(num_simulations=32, max_depth=3, max_children=12,
                          leaf_mode="value_only", training_mode=False, seed=1)
    m = s1.TopologyMCTS(nets, reg, sorted(reg.seed_ids), cfg,
                        generate_actions_fn=az.generate_alphazero_actions,
                        apply_action_fn=az.apply_alphazero_action)
    root = m.run(az.build_seed_root_state(SPEC, "az_test", reg.seed_ids[0], reg))
    pv = az.extract_principal_variation(root)
    # first PV step must be the root's max-N child
    best = max(root.children, key=lambda c: (c.N, -c.node_id))
    assert pv["root_action_id"] == best.action.action_id
    assert pv["pv_root_decision_Q"] == round(best.Q, 6)
    assert 0.0 <= pv["root_action_visit_fraction"] <= 1.0


def test_pv_root_terminate_yields_unchanged_seed():
    """A root whose most-visited action is TERMINATE must output the
    unchanged seed graph -- proven with a synthetic tree, not assumed."""
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    sid = reg.seed_ids[0]
    seed_hash = reg.get_topology(sid).graph.structural_hash()
    from agentic_raptor.topology_rl.stage3e1 import Node, Stage3E1Action
    root_state = az.build_seed_root_state(SPEC, "az_test", sid, reg)
    root = Node(0, root_state, None, None, 1.0)
    root.expanded = True
    term_action = Stage3E1Action("a_term", Stage3E1ActionType.TERMINATE_SEARCH)
    term_state = az.apply_alphazero_action(root_state, term_action, reg)
    term_child = Node(1, term_state, root, term_action, 0.5)
    term_child.N, term_child.W = 20, 2.0
    edit_action = Stage3E1Action("a_edit_ADD_VERIFIED_STAGE",
                                 Stage3E1ActionType.ADD_VERIFIED_STAGE, source_ref=sid)
    edit_state = az.apply_alphazero_action(root_state, edit_action, reg)
    edit_child = Node(2, edit_state, root, edit_action, 0.5)
    edit_child.N, edit_child.W = 5, 1.0
    root.children = [term_child, edit_child]
    root.N = 26
    pv = az.extract_principal_variation(root)
    assert pv["root_action_is_terminate"] is True
    assert pv["output_hash"] == seed_hash
    assert pv["pv_length"] == 1


def test_pv_edited_descendant_extraction_when_edits_dominate():
    """The inverse: when an edit action holds the visit majority, the PV
    output IS the edited descendant."""
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    sid = reg.seed_ids[0]
    from agentic_raptor.topology_rl.stage3e1 import Node, Stage3E1Action
    root_state = az.build_seed_root_state(SPEC, "az_test", sid, reg)
    root = Node(0, root_state, None, None, 1.0)
    root.expanded = True
    edit_action = Stage3E1Action("a_edit_ADD_VERIFIED_STAGE",
                                 Stage3E1ActionType.ADD_VERIFIED_STAGE, source_ref=sid)
    edit_state = az.apply_alphazero_action(root_state, edit_action, reg)
    edit_child = Node(1, edit_state, root, edit_action, 0.5)
    edit_child.N = 20
    term_action = Stage3E1Action("a_term", Stage3E1ActionType.TERMINATE_SEARCH)
    term_state = az.apply_alphazero_action(root_state, term_action, reg)
    term_child = Node(2, term_state, root, term_action, 0.5)
    term_child.N = 3
    root.children = [edit_child, term_child]
    root.N = 24
    pv = az.extract_principal_variation(root)
    assert pv["output_hash"] == edit_state.graph_hash
    assert pv["output_hash"] != root_state.graph_hash
    assert pv["root_action_is_terminate"] is False


def test_pv_deterministic_and_max_depth_bounded():
    cands = _real_candidates(4)
    r1 = az.run_per_seed_alphazero_search_pv(cands, SPEC, "az_test", seed=0, config=CFG())
    r2 = az.run_per_seed_alphazero_search_pv(cands, SPEC, "az_test", seed=0, config=CFG())
    assert ([c["canonical_graph_hash"] for c in r1["selected"]]
           == [c["canonical_graph_hash"] for c in r2["selected"]])
    for p in r1["pv_primaries"]:
        assert p["output_depth"] <= 3   # alphazero_max_edit_depth


# ---------------------------------------------------------------------------
# CROSS-TREE
# ---------------------------------------------------------------------------
def test_cross_tree_ranking_never_compares_raw_visit_counts():
    import inspect
    src = inspect.getsource(az.run_per_seed_alphazero_search_pv)
    # the rank key uses Q + visit FRACTION + hash -- no raw N term
    assert "pv_root_decision_Q" in src
    assert "root_action_visit_fraction" in src
    assert 'key=lambda r: (-r["N"], r["hash"])' not in src


def test_pv_top2_canonical_distinct_and_dedupe():
    cands = _real_candidates(4)
    r = az.run_per_seed_alphazero_search_pv(cands, SPEC, "az_test", seed=0, config=CFG())
    hashes = [c["canonical_graph_hash"] for c in r["selected"]]
    assert len(hashes) == 2 and hashes[0] != hashes[1]


def test_pv_selection_rule_version_recorded():
    cands = _real_candidates(2)
    r = az.run_per_seed_alphazero_search_pv(cands, SPEC, "az_test", seed=0, config=CFG())
    assert r["selection_rule"] == az.AZ_SELECTION_RULE_PRINCIPAL_VARIATION
    assert r["search"] == "true_alphazero_per_seed_pv"
    assert r["action_schema"] == az.AZ_ACTION_SCHEMA_PER_SEED


def test_pv_never_offers_select_actions():
    cands = _real_candidates(4)
    r = az.run_per_seed_alphazero_search_pv(cands, SPEC, "az_test", seed=0, config=CFG())
    for nodes in r["nodes"].values():
        for n in nodes:
            if n["action"] is not None:
                assert n["action"]["action_type"] != "SELECT_EXISTING_TOPOLOGY"


def test_pv_hard_fails_below_two_distinct():
    cands = _real_candidates(1)
    with pytest.raises(az.AlphaZeroSelectionError):
        az.run_per_seed_alphazero_search_pv(cands, SPEC, "az_test", seed=0, config=CFG())


# ---------------------------------------------------------------------------
# run_pipeline wiring
# ---------------------------------------------------------------------------
def test_run_pipeline_supports_per_seed_pv_mode():
    import inspect

    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    assert '"per_seed_pv"' in src
    assert "run_per_seed_alphazero_search_pv" in src
    # live default unchanged
    sig = inspect.signature(v2.run_pipeline)
    assert sig.parameters["search"].default == "bandit_top2"  # 2026-08-15 selector promotion
