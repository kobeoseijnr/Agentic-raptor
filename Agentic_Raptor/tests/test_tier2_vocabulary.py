"""TIER-2 VOCABULARY (2026-08-17): cascode input, class-AB output, 4-stage
cascade -- structures the stock template library cannot express, added so
the evaluation can discriminate again (heldout saturated at 9/9 for 9/10
arms). Zero LLM; ngspice not required."""
from __future__ import annotations

import json

import pytest

from agentic_raptor.llm_dpo import proposal_dict_valid
from agentic_raptor.llm_dpo.integrity import candidate_identity
from agentic_raptor.llm_dpo.stage3e4 import tier2_flags, variant_hash
from agentic_raptor.mapping import emit_netlist, map_family, static_validate
from agentic_raptor.publication.tier2 import (STOCK_CLASSES, TIER2_CLASSES,
                                              class_to_proposal,
                                              generate_tier2_specs)


class _E:
    topology_id = "t"


def _row(stages, blocks, **flags):
    return {"topology_id": "t", "gain_stages": stages, "functional_blocks": blocks,
            "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
            "graph_hash": None, **flags}


# ---------------------------------------------------------------------------
# mapper realizes the new blocks; old shapes byte-identical
# ---------------------------------------------------------------------------
def test_cascode_input_adds_four_cascode_devices_and_validates():
    g, note = map_family(_E(), _row(2, ["RC_series"], cascode_input=True))
    ids = {d.device_id for d in g.devices}
    assert {"MC1", "MC2", "MC3", "MC4"} <= ids
    assert all(d.provenance.get("tier2_block") == "cascode_input"
               for d in g.devices if d.device_id.startswith("MC"))
    net = emit_netlist(g, "p")
    assert static_validate(g, net)["status"] == "mapped_static_valid"


def test_class_ab_output_replaces_fixed_load_with_driven_pmos():
    g, _ = map_family(_E(), _row(3, ["RC_series"], class_ab_output=True))
    last_p = next(d for d in g.devices if d.device_id == "M11")   # 5+2*3
    assert last_p.role == "second_stage_gain_device"            # signal-driven
    assert last_p.nets["g"] != "nmir"                             # not a fixed load
    assert last_p.provenance.get("tier2_block") == "class_ab_output"


def test_four_stage_cascade_now_realizes():
    g, _ = map_family(_E(), _row(4, ["RC_series"]))
    assert g.stage_count == 4
    assert [d.kind for d in g.devices].count("res") == 3          # nulling per stage
    assert static_validate(g, emit_netlist(g, "p"))["status"] == "mapped_static_valid"


def test_stock_shapes_unchanged_without_flags():
    g, _ = map_family(_E(), _row(2, ["RC_series"]))
    ids = {d.device_id for d in g.devices}
    assert not any(i.startswith("MC") for i in ids)
    assert len(g.devices) == 11


# ---------------------------------------------------------------------------
# schema: validate, hash distinctly, name families, preserve history
# ---------------------------------------------------------------------------
def test_tier2_proposals_validate_and_hash_distinctly():
    plain = class_to_proposal("2s_rc")
    cas = class_to_proposal("2s_rc_cas")
    ab4 = class_to_proposal("4s_rc_ab")
    for o in (plain, cas, ab4):
        ok, why = proposal_dict_valid(o)
        assert ok, why
    assert variant_hash(plain) == "14de1f1a714ea91a"       # historical, unchanged
    assert variant_hash(cas) != variant_hash(plain)
    assert candidate_identity(cas)["topology_family_id"] == "2s_rc_cas"
    assert candidate_identity(ab4)["topology_family_id"] == "4s_rc_ab"
    assert tier2_flags(ab4) == (False, True)


def test_every_declared_class_round_trips():
    for cls in STOCK_CLASSES + TIER2_CLASSES:
        o = class_to_proposal(cls)
        ok, why = proposal_dict_valid(o)
        assert ok, (cls, why)
        assert candidate_identity(o)["topology_family_id"] == cls


# ---------------------------------------------------------------------------
# tier-2 spec set: sealed split, deterministic
# ---------------------------------------------------------------------------
def test_tier2_specs_deterministic_and_split():
    a = generate_tier2_specs()
    b = generate_tier2_specs()
    assert [s["spec_hash"] for s in a] == [s["spec_hash"] for s in b]
    splits = {s["split"] for s in a}
    assert splits == {"tier2_train", "tier2_heldout"}
    assert all(s["gain_target_db"] >= 140 for s in a)      # beyond stock reach
    assert len(a) == 72
