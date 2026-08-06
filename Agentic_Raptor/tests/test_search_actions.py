"""Search legality tests: the PUCT layer must be able to actually choose.

Regression guard for a silent failure mode -- when the candidate validator
only recognised the extractor's "gain_stage" block role, every
mapping-built (family-id) candidate was rejected as gainless, leaving
TERMINATE_SEARCH the sole legal action. The search then "agreed" with
every proposal because it could never do anything else.
"""

from __future__ import annotations

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

from agentic_raptor.topology_rl import stage3e1 as s1
from agentic_raptor.topology_rl.value_refresh import FamilyRegistry

FAMILIES = ["3s_miller", "3s_none", "3s_rc"]


def _state(topology_id, graph):
    return s1.TopologySearchState(
        topology_id=topology_id, graph_hash=graph.structural_hash(),
        lineage=[graph.structural_hash()],
        spec={"target_gain_db": 89.0, "target_gbw_hz": 1e6,
              "minimum_phase_margin_deg": 45.0,
              "load_capacitance_f": 100e-12, "supply_voltage": 1.8},
        rag_context_ids=[], available_blocks=[], legal_action_ids=[],
        edit_history=[], validation_status="validated",
        structural_features={"n_nodes": float(len(graph.nodes))},
        previous_evidence_ref=None, remaining_search_budget=6,
        remaining_spice_budget=0, depth=0)


def test_mapping_built_families_carry_a_gain_device():
    """The mapping vocabulary must satisfy the validator's gain check."""
    reg = FamilyRegistry(set(FAMILIES))
    for fam in FAMILIES:
        nodes = list(reg.get_topology(fam).graph.nodes.values())
        assert s1._has_gain_device(nodes), f"{fam} read as gainless"


def test_family_root_offers_keep_and_switch_actions():
    """A family-id root must get real alternatives, not only terminate."""
    reg = FamilyRegistry(set(FAMILIES))
    root = "3s_miller"
    pool = [f for f in FAMILIES if f != root]
    legal, rejections = s1.generate_actions(
        _state(root, reg.get_topology(root).graph), reg, pool)
    ids = {a.action_id for a in legal}
    assert "a_keep" in ids, f"keep rejected; rejections={rejections}"
    switches = {i for i in ids if i.startswith("a_sel_")}
    assert switches == {f"a_sel_{f}" for f in pool}, (
        f"switch actions missing: legal={sorted(ids)} "
        f"rejections={rejections}")


def test_terminate_is_not_the_only_option():
    """Guards the exact regression: visits collapsing onto a_term."""
    reg = FamilyRegistry(set(FAMILIES))
    root = "3s_none"
    legal, _ = s1.generate_actions(
        _state(root, reg.get_topology(root).graph), reg,
        [f for f in FAMILIES if f != root])
    assert {a.action_id for a in legal} != {"a_term"}
