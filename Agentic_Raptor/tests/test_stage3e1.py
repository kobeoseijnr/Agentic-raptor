"""Stage 3E.1 tests: schemas, legal actions, validator, policy/value, PUCT,
MCTS mechanics, accounting, training targets, checkpoint/resume, baselines."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.mb_sac import load_pools
from agentic_raptor.mb_sac.stage3d2 import V3
from agentic_raptor.topology_rl import stage3e1 as s

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reg():
    return TopologyRegistry(V3)


@pytest.fixture(scope="module")
def pool_ids():
    pools = load_pools()
    return [r["topology_id"] for r in pools["A1"] + pools["A2"]]


@pytest.fixture(scope="module")
def root_state(reg, pool_ids):
    return s.make_root_state(pool_ids[0], reg, s.DEFAULT_SPEC)


class TestSchemas:
    def test_state_schema(self, root_state):
        d = asdict(root_state)
        for key in ("topology_id", "graph_hash", "lineage", "spec", "rag_context_ids",
                    "available_blocks", "legal_action_ids", "edit_history",
                    "validation_status", "structural_features", "previous_evidence_ref",
                    "remaining_search_budget", "remaining_spice_budget", "depth",
                    "terminal_reason", "schema_version"):
            assert key in d, key
        assert d["schema_version"] == "3e1.1"
        # no future SPICE measurement outcomes in state (budget fields are fine)
        blob = json.dumps(d).lower()
        # (spec *targets* like minimum_phase_margin_deg are inputs, not outcomes)
        for leak in ('"phase_margin_deg"', '"dc_gain_db"', '"metrics"',
                     '"scalar_value"', '"measured"'):
            assert leak not in blob

    def test_action_schema(self):
        a = s.Stage3E1Action("a_x", s.Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                             source_ref="topology_0003", target_location="root",
                             port_mapping=(("p", "n"),), preconditions=("c",),
                             provenance="registry")
        d = a.to_dict()
        for key in ("action_id", "action_type", "source_ref", "target_location",
                    "port_mapping", "preconditions", "compatibility", "provenance",
                    "schema_version"):
            assert key in d
        assert len(s.Stage3E1ActionType) == 11

    def test_no_unchecked_netlist_action(self):
        """No action category permits raw netlist text emission."""
        assert not any("NETLIST" in t.value or "TEXT" in t.value
                       for t in s.Stage3E1ActionType)


class TestActions:
    def test_deterministic_generation(self, root_state, reg, pool_ids):
        a1, r1 = s.generate_actions(root_state, reg, pool_ids)
        a2, r2 = s.generate_actions(root_state, reg, pool_ids)
        assert [a.action_id for a in a1] == [a.action_id for a in a2]
        assert r1 == r2

    def test_structural_edit_rejected_with_reason(self, root_state, reg, pool_ids):
        _, rej = s.generate_actions(root_state, reg, pool_ids)
        comp = [r for r in rej if r["action_id"] == "a_comp"]
        assert comp and "structural_edit_not_yet_mapping_supported" in comp[0]["reason"]

    def test_immutable_application_and_lineage(self, root_state, reg):
        act = s.Stage3E1Action("a_sel_topology_0003",
                               s.Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                               source_ref="topology_0003")
        before = json.dumps(asdict(root_state), default=str)
        child = s.apply_topology_action(root_state, act, reg)
        assert json.dumps(asdict(root_state), default=str) == before  # parent untouched
        assert child.depth == root_state.depth + 1
        assert child.lineage[:-1] == root_state.lineage
        assert child.graph_hash == reg.get_topology("topology_0003").graph.structural_hash()
        assert child.edit_history[-1]["parent_hash"] == root_state.graph_hash

    def test_cycle_prevention(self, reg, pool_ids, root_state):
        act = s.Stage3E1Action("a_sel_topology_0003",
                               s.Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                               source_ref="topology_0003")
        child = s.apply_topology_action(root_state, act, reg)
        back = s.Stage3E1Action(f"a_sel_{root_state.topology_id}",
                                s.Stage3E1ActionType.SELECT_EXISTING_TOPOLOGY,
                                source_ref=root_state.topology_id)
        with pytest.raises(ValueError, match="cycle"):
            s.apply_topology_action(child, back, reg)
        # generator also masks ancestor recreation
        legal, rej = s.generate_actions(child, reg, pool_ids, max_alternatives=16)
        assert root_state.topology_id not in [a.source_ref for a in legal
                                              if a.action_type.value == "SELECT_EXISTING_TOPOLOGY"]

    def test_budget_exhaustion_masks_actions(self, reg, pool_ids, root_state):
        st = s.make_root_state(root_state.topology_id, reg, s.DEFAULT_SPEC,
                               search_budget=0)
        legal, rej = s.generate_actions(st, reg, pool_ids)
        assert [a.action_type for a in legal] == [s.Stage3E1ActionType.TERMINATE_SEARCH]
        assert all(r["reason"] == "search_budget_exhausted" for r in rej
                   if not r["action_id"].startswith("a_comp"))


class TestValidator:
    def test_taxonomy_fields(self, reg):
        v = s.validate_candidate(reg, "topology_0002", s.Stage3E1Action(
            "a_keep", s.Stage3E1ActionType.KEEP_TOPOLOGY, source_ref="topology_0002"), set())
        d = asdict(v)
        for key in ("structurally_valid", "semantically_valid", "mapping_supported",
                    "bias_complete", "supply_complete", "io_complete",
                    "no_floating_nodes", "no_illegal_cycles", "feedback_status",
                    "compensation_status", "transistor_realisation_supported",
                    "reasons", "warnings", "validation_version"):
            assert key in d
        assert v.ok

    def test_rejections_tracked_separately_from_sim_failures(self, reg, pool_ids):
        nets = s.build_policy_value(0)
        cfg = s.SearchConfig(num_simulations=2, leaf_mode="value_only",
                             training_mode=False)
        m = s.TopologyMCTS(nets, reg, pool_ids, cfg)
        m.run(s.make_root_state(pool_ids[0], reg, s.DEFAULT_SPEC))
        assert len(m.rejections) > 0
        assert m.costs.simulator_failures == 0


class TestPolicyValue:
    def test_masked_distribution_normalised_variable_size(self, reg, pool_ids, root_state):
        import torch
        nets = s.build_policy_value(0)
        for n_alt in (2, 4):
            legal, _ = s.generate_actions(root_state, reg, pool_ids, max_alternatives=n_alt)
            acts, logits, probs = nets["policy_forward"](root_state, legal, reg)
            assert len(probs) == len(legal)          # variable action-set size
            assert abs(float(probs.sum()) - 1.0) < 1e-5
            assert [a.action_id for a in acts] == sorted(a.action_id for a in acts)
        # illegal actions never enter the distribution: only legal set is scored
        assert all(p > 0 for p in probs)

    def test_value_output_shape_and_aux(self, reg, root_state):
        nets = s.build_policy_value(0)
        v = nets["value_forward"](root_state, reg)
        assert v["scalar"].dim() == 0
        assert -1.0 <= float(v["scalar"]) <= 1.0
        assert "feasibility_logit_uncalibrated" in v   # never called calibrated
        assert "stability_logit_uncalibrated" in v

    def test_gradient_flow_shared_encoder(self, reg, pool_ids, root_state):
        import torch
        nets = s.build_policy_value(0)
        legal, _ = s.generate_actions(root_state, reg, pool_ids)
        _, logits, _ = nets["policy_forward"](root_state, legal, reg)
        loss = logits.sum() + nets["value_forward"](root_state, reg)["scalar"]
        loss.backward()
        enc_grads = sum(float(p.grad.abs().sum()) for p in nets["encoder"].parameters()
                        if p.grad is not None)
        head_grads = sum(float(p.grad.abs().sum()) for p in nets["heads"].parameters()
                         if p.grad is not None)
        assert enc_grads > 0 and head_grads > 0      # shared encoder gets gradients


class TestMCTS:
    def _run(self, reg, pool_ids, **kw):
        nets = s.build_policy_value(kw.pop("seed", 0))
        cfg = s.SearchConfig(leaf_mode="value_only", training_mode=False, **kw)
        m = s.TopologyMCTS(nets, reg, pool_ids, cfg)
        root = m.run(s.make_root_state(pool_ids[0], reg, s.DEFAULT_SPEC))
        return root, m

    def test_puct_selection_and_expansion(self, reg, pool_ids):
        root, m = self._run(reg, pool_ids, num_simulations=10)
        assert len(m.nodes) > 1                       # traversed more than one node
        assert root.N == 10
        assert sum(c.N for c in root.children) >= root.N - 1
        # PUCT influenced selection: after every child has one visit, PUCT's
        # Q+U ordering concentrates further visits — non-uniform distribution
        visits = [c.N for c in root.children]
        assert max(visits) > min(visits)

    def test_deterministic_tie_breaking(self, reg, pool_ids):
        r1, m1 = self._run(reg, pool_ids, num_simulations=4, seed=3)
        r2, m2 = self._run(reg, pool_ids, num_simulations=4, seed=3)
        assert [c.N for c in r1.children] == [c.N for c in r2.children]
        assert [n.record()["graph_hash"] for n in m1.nodes] == \
               [n.record()["graph_hash"] for n in m2.nodes]

    def test_node_record_schema(self, reg, pool_ids):
        root, m = self._run(reg, pool_ids, num_simulations=2)
        rec = root.record()
        for key in ("node_id", "graph_hash", "parent", "action", "depth", "N", "W",
                    "Q", "prior", "expanded", "terminal", "leaf_eval", "leaf_ref",
                    "components", "children", "schema_version"):
            assert key in rec

    def test_value_only_evaluation_no_spice(self, reg, pool_ids):
        _, m = self._run(reg, pool_ids, num_simulations=4)
        assert m.costs.real_spice_calls == 0
        assert m.costs.value_net_calls > 0

    def test_backup_statistics(self, reg, pool_ids):
        root, m = self._run(reg, pool_ids, num_simulations=5)
        for c in root.children:
            if c.N:
                assert abs(c.Q - c.W / c.N) < 1e-9
        assert root.N == 5

    def test_depth_termination(self, reg, pool_ids):
        root, m = self._run(reg, pool_ids, num_simulations=6, max_depth=1)
        assert all(n.state.depth <= 1 for n in m.nodes)

    def test_root_dirichlet_noise_changes_priors(self, reg, pool_ids):
        nets = s.build_policy_value(0)
        cfgA = s.SearchConfig(num_simulations=1, leaf_mode="value_only",
                              training_mode=True, root_noise_eps=0.5, seed=1)
        cfgB = s.SearchConfig(num_simulations=1, leaf_mode="value_only",
                              training_mode=False, seed=1)
        mA = s.TopologyMCTS(nets, reg, pool_ids, cfgA)
        rA = mA.run(s.make_root_state(pool_ids[0], reg, s.DEFAULT_SPEC))
        mB = s.TopologyMCTS(nets, reg, pool_ids, cfgB)
        rB = mB.run(s.make_root_state(pool_ids[0], reg, s.DEFAULT_SPEC))
        pa = sorted(round(c.prior, 6) for c in rA.children)
        pb = sorted(round(c.prior, 6) for c in rB.children)
        assert pa != pb                    # noise applied only in training mode
        # deterministic evaluation without noise reproduces exactly
        mC = s.TopologyMCTS(nets, reg, pool_ids, cfgB)
        rC = mC.run(s.make_root_state(pool_ids[0], reg, s.DEFAULT_SPEC))
        assert pb == sorted(round(c.prior, 6) for c in rC.children)


class TestSmokeArtifacts:
    """Assertions over the real-SPICE smoke artifacts (Part U evidence)."""

    @pytest.fixture(scope="class")
    def summary(self):
        p = _ROOT / "artifacts" / "stage3e1" / "SMOKE_SUMMARY.json"
        if not p.is_file():
            pytest.skip("smoke not yet run")
        return json.loads(p.read_text(encoding="utf-8"))

    def test_spice_leaf_and_backup(self, summary):
        ep = summary["episodes"][0]["result"]
        assert ep["costs"]["mbsac_leaf_calls"] >= 1     # expensive leaf reached
        assert ep["costs"]["real_spice_calls"] >= 1     # real SPICE returned
        assert ep["best_post_sizing_score"] is not None
        assert ep["best_post_sizing_score"]["components"]  # component vector kept
        assert ep["confidence_label"] == "ordinal_uncalibrated"
        assert sum(ep["root_visit_distribution"].values()) == pytest.approx(1.0)

    def test_exact_spice_and_cache_accounting(self, summary):
        for e in summary["episodes"]:
            c = e["result"]["costs"]
            assert c["real_spice_calls"] <= 6
            assert c["cache_hits"] >= 0
            for key in ("topology_expansions", "validator_calls", "value_net_calls",
                        "mbsac_leaf_calls", "real_spice_calls", "cache_hits",
                        "simulator_failures", "leaf_budget_exhausted", "wall_clock_s"):
                assert key in c

    def test_policy_and_value_updated(self, summary):
        for e in summary["episodes"]:
            assert e["train"]["policy_params_changed"] is True
            assert e["train"]["encoder_params_changed"] is True
            assert e["train"]["policy_loss"] >= 0.0
            assert e["train"]["examples_used"] >= 1

    def test_training_records_targets(self):
        p = _ROOT / "artifacts" / "stage3e1" / "training_records.jsonl"
        if not p.is_file():
            pytest.skip("smoke not yet run")
        recs = [json.loads(x) for x in p.read_text().splitlines()]
        for r in recs:
            assert abs(sum(r["visit_distribution"].values()) - 1.0) < 1e-6  # policy target
            assert r["value_target"] is not None          # SPICE-backed value target
            assert r["outcome"]["real_spice_calls"] >= 1  # not fabricated
            assert r["provenance"] == "stage3e1_mcts"

    def test_checkpoint_resume_deterministic(self, summary):
        assert summary["resume_deterministic"] is True

    def test_checkpoint_metadata(self):
        import torch
        p = _ROOT / "artifacts" / "stage3e1" / "policy_value_ep0.pt"
        if not p.is_file():
            pytest.skip("smoke not yet run")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert ck["meta"]["encoder_mode"] == "shared_trainable_copy_of_stage3d2_mp"
        assert "encoder" in ck and "heads" in ck


class TestBaselines:
    @pytest.mark.parametrize("name", list(s.BASELINE_CONFIGS))
    def test_baseline_config_constructs(self, name):
        cfg = s.SearchConfig(training_mode=False, **s.BASELINE_CONFIGS[name])
        assert cfg.num_simulations >= 1

    def test_random_and_no_policy_execute(self, reg, pool_ids):
        for name in ("random_search", "mcts_no_policy_prior", "mcts_no_value",
                     "mcts_no_mbsac_leaf", "mcts_no_preference_ranking"):
            cfg = s.SearchConfig(training_mode=False, **s.BASELINE_CONFIGS[name])
            if name == "mcts_no_mbsac_leaf":
                cfg = s.SearchConfig(training_mode=False, use_mbsac_leaf=False,
                                     leaf_mode="value_only", num_simulations=2)
            nets = s.build_policy_value(0)
            m = s.TopologyMCTS(nets, reg, pool_ids, cfg)
            root = m.run(s.make_root_state(pool_ids[0], reg, s.DEFAULT_SPEC))
            assert len(m.nodes) >= 1
            assert m.costs.real_spice_calls == 0   # baselines validated without SPICE
