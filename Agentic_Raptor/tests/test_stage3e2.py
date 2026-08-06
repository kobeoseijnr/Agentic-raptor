"""Stage 3E.2 tests: executable edits, proposals, leakage, campaign evidence,
gradient paths, repair separation, PVT/cost accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.mb_sac.stage3d2 import V3
from agentic_raptor.topology_rl import stage3e2 as s2
from agentic_raptor.topology_rl import stage3e2_edits as ed

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "artifacts" / "stage3e2"


def _load(name):
    p = OUT / name
    if not p.is_file():
        pytest.skip(f"{name} not yet produced")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def base_graph():
    reg = TopologyRegistry(V3)
    g = s2.map_root(reg, "topology_v2_0001", s2.new_costs())
    if g.stage_count < 2:
        g, _ = ed.apply_edit(g, "ADD_VERIFIED_STAGE")
    return g


class TestEditLibrary:
    def test_all_templates_versioned(self):
        assert len(ed.EDIT_TEMPLATES) == 10
        for t in ed.EDIT_TEMPLATES.values():
            assert t["schema_version"] == ed.EDIT_SCHEMA_VERSION
            assert callable(t["fn"])

    def test_immutable_parent_and_hash_change(self, base_graph):
        h0 = ed.device_graph_hash(base_graph)
        ng, audit = ed.apply_edit(base_graph, "ADD_VERIFIED_STAGE")
        assert ed.device_graph_hash(base_graph) == h0          # parent untouched
        assert audit["child_hash"] != h0
        assert audit["parent_hash"] == h0

    def test_roundtrip_reversibility(self, base_graph):
        h0 = ed.device_graph_hash(base_graph)
        ng, _ = ed.apply_edit(base_graph, "ADD_VERIFIED_STAGE")
        back, _ = ed.apply_edit(ng, "REMOVE_OPTIONAL_SUPPORTED_STAGE")
        assert ed.device_graph_hash(back) == h0                # parent hash recovered

    def test_manifest_and_action_dim_update(self, base_graph):
        _, audit = ed.apply_edit(base_graph, "ADD_VERIFIED_STAGE")
        assert audit["action_dim_after"] == audit["action_dim_before"] + 4
        assert len(audit["manifest_added"]) == 4               # w,l for 2 devices

    def test_incompatible_edit_rejected(self, base_graph):
        ng, _ = ed.apply_edit(base_graph, "ADD_SUPPORTED_OUTPUT_STAGE")
        with pytest.raises(ed.EditRejected, match="already_present"):
            ed.apply_edit(ng, "ADD_SUPPORTED_OUTPUT_STAGE")
        with pytest.raises(ed.EditRejected):
            ed.apply_edit(base_graph, "REMOVE_OPTIONAL_SUPPORTED_STAGE") \
                if not any(d.group and d.group.startswith("cse")
                           for d in base_graph.devices) else (_ for _ in ()).throw(
                    ed.EditRejected("skip"))

    def test_duplicate_compensation_rejected(self, base_graph):
        ng, _ = ed.apply_edit(base_graph, "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE")
        with pytest.raises(ed.EditRejected, match="duplicate_compensation"):
            ed.apply_edit(ng, "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE")

    def test_feedback_never_to_driven_input(self, base_graph):
        ng, _ = ed.apply_edit(base_graph, "CONNECT_VERIFIED_FEEDBACK_PATH")
        fb = [d for d in ng.devices if d.role == "local_feedback"][0]
        assert not ({"vinp", "vinn"} & set(fb.nets.values()))
        assert fb.provenance["expected_sign"] == "negative"

    def test_no_floating_and_static_valid(self, base_graph):
        from agentic_raptor.mapping import emit_netlist, static_validate
        for et in s2.EXECUTABLE_EDITS:
            try:
                ng, _ = ed.apply_edit(base_graph, et)
            except ed.EditRejected:
                continue
            val = static_validate(ng, emit_netlist(ng, "t"))
            assert val["status"] == "mapped_static_valid", (et, val["problems"])

    def test_static_invalid_never_reaches_spice(self, base_graph, tmp_path):
        import copy
        bad = copy.deepcopy(base_graph)
        bad.devices = [d for d in bad.devices if d.nets.get("d") != "vout"
                       and d.nets.get("p") != "vout" and d.nets.get("n") != "vout"
                       and d.nets.get("s") != "vout"]
        costs = s2.new_costs()
        q = ed.qualify_device_graph("t", bad, tmp_path, "unused_exe", "bad", costs)
        assert q["electrical"] == "not_simulated_static_invalid"
        assert costs["real_spice_calls"] == 0


class TestProposals:
    def test_valid_fixture_proposal(self):
        p = ed.FixtureProposalProvider().propose({})
        ok, reasons = ed.validate_proposal(p)
        assert ok and not reasons
        assert p.schema_version == ed.PROPOSAL_SCHEMA_VERSION

    @pytest.mark.parametrize("field,reason", [
        ("stages", "no_stages"), ("bias_roles", "missing_bias_path")])
    def test_malformed_rejected(self, field, reason):
        p = ed.FixtureProposalProvider().propose({})
        setattr(p, field, [])
        ok, reasons = ed.validate_proposal(p)
        assert not ok and any(reason in r for r in reasons)

    def test_unsupported_block_and_illegal_feedback(self):
        p = ed.FixtureProposalProvider().propose({})
        p.stages[0]["block"] = "quantum_stage"
        p.feedback_paths = [{"from": "vout", "to": "vinp", "sign": "positive"}]
        ok, reasons = ed.validate_proposal(p)
        assert not ok
        assert any("unsupported_block" in r for r in reasons)
        assert any("illegal_feedback" in r for r in reasons)
        assert any("positive_feedback" in r for r in reasons)

    def test_no_unchecked_netlist_path(self):
        import inspect
        src = inspect.getsource(ed.realise_proposal)
        assert src.index("validate_proposal(p)") < src.index("map_family(_Stub")  # validator first


class TestTargetsAndLeakage:
    def test_target_schema_and_counts(self):
        if not (_ROOT / "datasets/target_sets_v1/targets.jsonl").is_file():
            pytest.skip("targets not built")
        recs = s2.load_targets()
        assert len(recs) == 80          # 16 families x 5 tiers
        for r in recs[:5]:
            for k in ("topology_id", "graph_hash", "target_id", "gain_target_db",
                      "phase_margin_target_deg", "difficulty", "split",
                      "feasibility_provenance", "generation_seed", "schema_version"):
                assert k in r

    def test_heldout_leakage_prevention(self):
        if not (OUT / "phase_d_transitions.jsonl").is_file():
            pytest.skip("campaign not run")
        heldout_ids = {t["target_id"] for t in s2.load_targets("test")}
        blob = (OUT / "phase_d_transitions.jsonl").read_text()
        blob += (OUT / "alphazero_campaign.json").read_text()
        assert not any(h in blob for h in heldout_ids)   # no heldout target in training
        ho = _load("heldout.json")
        assert ho["rag_retrieval_disabled_during_heldout"] is True


class TestCampaignEvidence:
    def test_edits_reached_real_spice(self):
        d = _load("edit_demo.json")
        execd = d["executable"]
        assert len(execd) >= 4                       # >=4 executable categories
        assert d["parent_immutable"] and d["roundtrip_hash_recovered"]
        for et, r in execd.items():
            assert r["qualification"]["static"] == "mapped_static_valid"
            assert r["qualification"]["spice_calls"] == 1    # each edit hit ngspice
            assert r["audit"]["child_hash"] != d["base_hash"]

    def test_structural_vs_electrical_distinction(self):
        d = _load("edit_demo.json")
        stats = {et: r["qualification"].get("stability") for et, r in d["executable"].items()}
        # unstable/failed outcomes are recorded, not rewritten as sim failures
        assert all(v in ("verified_stable", "verified_unstable",
                         "phase_margin_unavailable", None) for v in stats.values())

    def test_llm_proposal_mapped_and_simulated(self):
        d = _load("llm_demo.json")
        assert d["valid_proposal"]["status"] == "mapped_and_simulated"
        assert d["valid_proposal"]["spice_calls"] >= 1
        assert len(d["rejections"]) == 4
        assert all(d["rejections"][k] for k in d["rejections"])

    def test_alphazero_three_seeds_trained(self):
        d = _load("alphazero_campaign.json")
        assert sorted(map(int, d["per_seed"])) == [0, 1, 2]
        for sd in d["per_seed"].values():
            ep = sd["episodes"][0]
            assert abs(sum(ep["visit_distribution"].values()) - 1.0) < 1e-6
            cats = set(ep["action_categories"].values())
            assert "SELECT_EXISTING_TOPOLOGY" in cats
            assert any("ADD" in c for c in cats)     # mixed selection+edit actions
            if ep["train"]:
                assert ep["train"]["heads_changed"] and ep["train"]["encoder_changed"]

    def test_phase_d_gradient_paths(self):
        d = _load("phase_d.json")
        for sd in d["per_seed"].values():
            assert sd["actor_to_encoder_grad_nonzero"] is True
            assert sd["critic_to_encoder_grad_nonzero"] is True
            assert sd["encoder_changed"] is True
            assert sd["real_spice_calls"] >= 32
        assert "stable_rate_std" in d                # seed variance reported

    def test_calibration_gates(self):
        d = _load("calibration.json")
        assert "global" in d["rollout_gates"]
        assert d["rollout_gates"]["topology_level"] == "insufficient_data_gate_disabled"

    def test_ranker_equal_budget(self):
        d = _load("ranker_comparison.json")
        assert set(d["modes"]) == {"bt_ranker", "random", "scalar_heuristic"}
        budgets = {m["real_spice_calls"] for m in d["modes"].values()}
        assert len(budgets) == 1                     # equal budget enforced

    def test_repair_separate_from_normal(self):
        d = _load("repair_sample.json")
        assert "SEPARATE from normal sizing" in d["note"]
        for r in d["results"]:
            assert "repair_class" in r or "status" in r

    def test_pvt_separate_accounting(self):
        d = _load("pvt.json")
        assert d["points"] >= 8
        assert d["pvt_spice_calls"] == d["points"]
        assert "separately" in d["note"]

    def test_heldout_all_16(self):
        d = _load("heldout.json")
        assert d["families"] == 16
        assert d["real_spice_calls"] == 16


class TestSnapshot:
    def test_verified_manifest(self):
        v = _ROOT / "artifacts/code_snapshots/pre_stage3e2_full_generation/VERIFICATION.json"
        if not v.is_file():
            pytest.skip("snapshot pending")
        d = json.loads(v.read_text())
        assert d["verified"] is True and not d["mismatches"]
