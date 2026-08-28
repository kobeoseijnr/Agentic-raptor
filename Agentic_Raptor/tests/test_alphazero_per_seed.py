"""AlphaZero improvement task, Part 2/16: PER-SEED MCTS -- every validated
LLM seed becomes its own independent structural-edit-only MCTS root,
instead of one virtual super-root choosing among seeds via
SELECT_EXISTING_TOPOLOGY. Real engine (agentic_raptor.topology_rl.stage3e1.
TopologyMCTS, unmodified), real LLMSeededEditRegistry.derive_edited() (not
mocked). Kept fast: leaf_mode="value_only", no SPICE, no GPU/LLM (candidates
come from the same frozen corpus_diverse.json fixture test_alphazero.py
uses).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.topology_rl import alphazero as az
from agentic_raptor.topology_rl.stage3e1 import Stage3E1ActionType

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"


def _real_candidates(n=4):
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


def _small_cfg(**overrides):
    base = dict(total_mcts_simulations=64, alphazero_max_edit_depth=3)
    base.update(overrides)
    return az.PerSeedAlphaZeroConfig(**base)


# ---------------------------------------------------------------------------
# no SELECT actions anywhere in the per-seed action schema
# ---------------------------------------------------------------------------
def test_no_select_actions_anywhere_in_per_seed_search():
    cands = _real_candidates(4)
    result = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0,
                                              config=_small_cfg())
    action_types = {n["action"]["action_type"]
                    for nodes in result["nodes"].values()
                    for n in nodes if n["action"] is not None}
    assert Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY.value not in action_types
    assert action_types <= ({t.value for t in az.AZ_EDIT_ACTION_TYPES}
                            | {Stage3E1ActionType.TERMINATE_SEARCH.value})


def test_action_schema_label_is_per_seed():
    cands = _real_candidates(2)
    result = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0,
                                              config=_small_cfg())
    assert result["action_schema"] == az.AZ_ACTION_SCHEMA_PER_SEED
    assert az.AZ_ACTION_SCHEMA_PER_SEED != az.AZ_ACTION_SCHEMA_SUPERROOT


def test_generate_alphazero_actions_never_offers_select_when_rooted_at_a_real_seed():
    """The structural mechanism behind the guarantee above: since
    build_seed_root_state() never uses the "proposal_root" sentinel,
    generate_alphazero_actions()'s at_super_root branch is unreachable
    for a per-seed root -- confirmed directly, one level below the full
    search."""
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    root_state = az.build_seed_root_state(SPEC, "az_test", reg.seed_ids[0], reg)
    legal, _ = az.generate_alphazero_actions(root_state, reg, reg.seed_ids)
    assert legal, "no legal action at the per-seed root"
    assert not any(a.action_type == Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY
                  for a in legal)
    assert any(a.action_id == "a_term" for a in legal)


# ---------------------------------------------------------------------------
# every validated seed becomes its own root
# ---------------------------------------------------------------------------
def test_every_seed_gets_its_own_independent_tree():
    cands = _real_candidates(4)
    result = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0,
                                              config=_small_cfg())
    reg_seed_ids = sorted(c["llm_proposal_id"] for c in cands)
    assert result["seed_ids"] == reg_seed_ids
    assert set(result["per_seed_tree_nodes"]) == set(reg_seed_ids)
    assert set(result["per_seed_simulations_allocated"]) == set(reg_seed_ids)
    # every tree actually ran (root + at least one expansion)
    assert all(n >= 1 for n in result["per_seed_tree_nodes"].values())


def test_per_seed_root_state_topology_id_is_the_real_seed_not_a_sentinel():
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    sid = reg.seed_ids[0]
    root_state = az.build_seed_root_state(SPEC, "az_test", sid, reg)
    assert root_state.topology_id == sid
    assert root_state.topology_id != "proposal_root"
    assert root_state.graph_hash == reg.get_topology(sid).graph.structural_hash()
    assert root_state.lineage == [root_state.graph_hash]


# ---------------------------------------------------------------------------
# matched total simulation budget + deterministic remainder allocation
# ---------------------------------------------------------------------------
def test_allocate_simulations_per_seed_sums_exactly_to_total():
    for total in (0, 1, 3, 64, 100, 128, 130, 257):
        for k in (1, 2, 3, 4, 5):
            seed_ids = [f"p{i:02d}" for i in range(k)]
            alloc = az.allocate_simulations_per_seed(total, seed_ids)
            assert sum(alloc.values()) == total, (total, k, alloc)


def test_allocate_simulations_per_seed_deterministic_remainder_goes_to_sorted_first():
    alloc = az.allocate_simulations_per_seed(130, ["p03", "p01", "p00", "p02"])
    # sorted order is p00,p01,p02,p03 -- remainder=2 -> p00,p01 get +1
    assert alloc == {"p00": 33, "p01": 33, "p02": 32, "p03": 32}


def test_allocate_simulations_per_seed_rejects_unknown_allocation_mode():
    with pytest.raises(ValueError):
        az.allocate_simulations_per_seed(128, ["p00", "p01"], allocation_mode="learned")


def test_total_search_budget_matches_config_exactly():
    cands = _real_candidates(4)
    cfg = _small_cfg(total_mcts_simulations=64)
    result = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0, config=cfg)
    assert sum(result["per_seed_simulations_allocated"].values()) == 64
    assert result["total_mcts_simulations"] == 64


# ---------------------------------------------------------------------------
# independent tree mutable state + independent RNG streams
# ---------------------------------------------------------------------------
def test_independent_trees_have_independent_node_counts():
    """If trees shared mutable state, per-seed tree node counts would be
    identical or nonsensically coupled; independent search dynamics on
    genuinely different starting graphs should differ."""
    cands = _real_candidates(4)
    result = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0,
                                              config=_small_cfg())
    counts = list(result["per_seed_tree_nodes"].values())
    assert len(set(counts)) > 1, f"all trees produced identical node counts: {counts}"


def test_distinct_seed_hashes_get_distinct_rng_seeds():
    cands = _real_candidates(4)
    reg = az.LLMSeededEditRegistry(cands)
    hashes = {sid: reg.get_topology(sid).graph.structural_hash() for sid in reg.seed_ids}
    seeds = {sid: az.stable_per_seed_mcts_search_seed(0, "AZ_G0", "spechash", 0, h)
            for sid, h in hashes.items()}
    # seeds with the SAME underlying structural hash legitimately share an
    # RNG stream (same starting distribution to explore); seeds with
    # DIFFERENT hashes must not collide.
    hash_groups: dict = {}
    for sid, h in hashes.items():
        hash_groups.setdefault(h, []).append(sid)
    distinct_hash_seeds = {seeds[group[0]] for group in hash_groups.values()}
    assert len(distinct_hash_seeds) == len(hash_groups), (
        "distinct structural hashes must map to distinct RNG seeds")
    for group in hash_groups.values():
        assert len({seeds[sid] for sid in group}) == 1, (
            "identical structural hashes should share an RNG stream")


def test_stable_per_seed_mcts_search_seed_is_a_distinct_function_from_the_episode_axis():
    """Guards against reintroducing the exact RNG-conflation bug Stage 5D
    fixed on a different axis -- the per-seed seeding function must not be
    stable_mcts_search_seed with a repurposed parameter."""
    import inspect
    src_per_seed = inspect.getsource(az.stable_per_seed_mcts_search_seed)
    src_step = inspect.getsource(az.stable_mcts_search_seed)
    assert "def stable_per_seed_mcts_search_seed" in src_per_seed
    assert src_per_seed != src_step


def test_per_seed_search_deterministic_given_identical_inputs():
    cands = _real_candidates(4)
    cfg = _small_cfg(training_mode=True)
    r1 = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0, config=cfg)
    r2 = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0, config=cfg)
    assert r1["per_seed_tree_nodes"] == r2["per_seed_tree_nodes"]
    assert ([s["canonical_graph_hash"] for s in r1["selected"]]
           == [s["canonical_graph_hash"] for s in r2["selected"]])


# ---------------------------------------------------------------------------
# structural actions only + TERMINATE semantics
# ---------------------------------------------------------------------------
def test_terminate_is_always_legal_at_a_per_seed_root():
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    root_state = az.build_seed_root_state(SPEC, "az_test", reg.seed_ids[0], reg)
    legal, _ = az.generate_alphazero_actions(root_state, reg, reg.seed_ids)
    assert any(a.action_id == "a_term"
              and a.action_type == Stage3E1ActionType.TERMINATE_SEARCH
              for a in legal)


# ---------------------------------------------------------------------------
# candidate ancestry preserved + canonical top-2 distinctness
# ---------------------------------------------------------------------------
def test_selected_candidates_carry_originating_seed_and_edit_ancestry():
    cands = _real_candidates(4)
    result = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0,
                                              config=_small_cfg())
    for c in result["selected"]:
        assert c["originating_seed_id"] in result["seed_ids"]
        assert "edit_history" in c
        assert isinstance(c["edit_depth"], int)
        assert c["is_edited_descendant"] in (True, False)


def test_top2_are_canonical_distinct():
    cands = _real_candidates(4)
    result = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0,
                                              config=_small_cfg())
    hashes = [c["canonical_graph_hash"] for c in result["selected"]]
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]


def test_top2_rule_is_summed_visit_count_identical_to_superroot_rule():
    """Part 4 Section 10: the frozen top-2 rule must be the SAME rule
    alphazero_select_two() uses (rank by summed visit count N per
    canonical hash, ties by hash string) -- verified by source inspection
    so architecture comparisons never accidentally compare a different
    selection rule too."""
    import inspect
    src = inspect.getsource(az.run_per_seed_alphazero_search)
    assert 'sorted(by_hash.values(), key=lambda r: (-r["N"], r["hash"]))' in src


def test_raises_when_fewer_than_two_distinct_topologies_found():
    cands = _real_candidates(2)
    # force a trivial search (0 simulations at each seed beyond the root
    # itself) unlikely to discover a second distinct hash reliably is
    # fragile; instead directly test the guard with a single candidate
    with pytest.raises(az.AlphaZeroSelectionError):
        az.run_per_seed_alphazero_search(cands[:1], SPEC, "az_test", seed=0,
                                         config=_small_cfg())


# ---------------------------------------------------------------------------
# no dormant SELECT ids retained (Section 3 explicit requirement)
# ---------------------------------------------------------------------------
def test_per_seed_policy_head_never_scores_a_select_action():
    """generate_alphazero_actions() is what feeds actions to policy_
    forward() -- confirming it structurally excludes SELECT for a
    per-seed root (already tested above) is sufficient to guarantee the
    policy head is never asked to score one during per-seed search; this
    test additionally confirms no a_sel_* action id appears in ANY
    per-seed tree's recorded nodes."""
    cands = _real_candidates(4)
    result = az.run_per_seed_alphazero_search(cands, SPEC, "az_test", seed=0,
                                              config=_small_cfg())
    for nodes in result["nodes"].values():
        for n in nodes:
            if n["action"] is not None:
                assert not n["action"]["action_id"].startswith("a_sel_")
