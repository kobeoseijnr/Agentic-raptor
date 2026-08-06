"""Stage 3D.2 tests: MP-encoder evidence, DPO classification honesty,
leaf-evaluator record schema, and campaign-summary invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

_ROOT = Path(__file__).resolve().parents[1]
_SUM = _ROOT / "artifacts" / "stage3d2" / "SUMMARY.json"


def _summary():
    if not _SUM.is_file():
        pytest.skip("stage3d2 campaign not yet run")
    return json.loads(_SUM.read_text(encoding="utf-8"))


class TestMPConditioner:
    def test_shapes_and_gradients(self):
        """Part B evidence: per-device + global embeds, nonzero grads, checksum change."""
        import torch

        from agentic_raptor.corpus import TopologyRegistry
        from agentic_raptor.mb_sac.stage3d2 import V3, build_mp_conditioner
        from agentic_raptor.topology_rl.trainer import parameter_checksum

        torch.manual_seed(0)
        module, embed = build_mp_conditioner()
        reg = TopologyRegistry(V3)
        tid = reg.list_topologies()[0]
        per_dev, glob = embed(reg.get_topology(tid).graph)
        n_nodes = len(reg.get_topology(tid).graph.nodes)
        # graph_to_tensors emits device rows + net rows (bipartite device–net graph)
        assert per_dev.shape[0] >= n_nodes and per_dev.shape[1] == 16
        assert glob.shape == (16,)                  # global graph embedding
        before = parameter_checksum(module)
        loss = (torch.tanh(glob.mean()) - 0.5) ** 2
        loss.backward()
        gnorm = sum(float(p.grad.abs().sum()) for p in module.parameters()
                    if p.grad is not None)
        assert gnorm > 0.0                          # gradients flow through MP rounds
        opt = torch.optim.SGD(module.parameters(), lr=0.1)
        opt.step()
        assert parameter_checksum(module) != before  # params actually change

    def test_distinct_graphs_distinct_embeddings(self):
        import torch

        from agentic_raptor.corpus import TopologyRegistry
        from agentic_raptor.mb_sac.stage3d2 import V3, build_mp_conditioner

        torch.manual_seed(0)
        _, embed = build_mp_conditioner()
        reg = TopologyRegistry(V3)
        ids = reg.list_topologies()[:2]
        g0 = embed(reg.get_topology(ids[0]).graph)[1]
        g1 = embed(reg.get_topology(ids[1]).graph)[1]
        assert not torch.allclose(g0, g1)


class TestDPOClassification:
    def test_docstring_states_pairwise_preference_not_policy_dpo(self):
        """Part I: honest classification is recorded at module level."""
        import agentic_raptor.mb_sac.stage3d2 as m

        doc = m.__doc__ or ""
        assert "PAIRWISE PREFERENCE RANKER" in doc
        assert "NOT policy-based LLM DPO" in doc

    def test_ranker_is_bradley_terry_pairwise(self):
        from agentic_raptor.dpo import ranker

        src = (Path(ranker.__file__)).read_text(encoding="utf-8")
        assert "logsigmoid" in src or "log_sigmoid" in src or "sigmoid" in src


class TestCampaignSummary:
    def test_mp_encoder_evidence(self):
        s = _summary()
        assert s["mp_encoder_active"] is True
        assert s["mp_params_changed"] is True
        assert s["nonzero_grad_norms"] is True
        assert s["grad_samples"] >= 32              # >= 2 steps x 16 families
        for ph in s["phases"].values():
            assert ph["per_device_embed_shape"][1] == 16
            assert ph["global_embed_shape"] == [16]

    def test_multi_step_not_single_transition(self):
        """'Do not call one transition per family a completed training phase.'"""
        s = _summary()
        fams = sum(ph["families"] for ph in s["phases"].values())
        assert fams == 16
        assert s["grad_samples"] >= 2 * fams        # >=2 real transitions/family
        assert s["budget"]["real_spice_calls"] >= 2 * fams

    def test_checkpoint_exists_with_encoder_metadata(self):
        s = _summary()
        import torch
        ck = torch.load(s["checkpoint"], map_location="cpu", weights_only=False)
        assert ck["encoder"] == "MPConditioner-2round-residual"
        assert any(k.startswith("rounds") for k in ck["mp"])


class TestLeafEvaluator:
    def test_leaf_record_schema(self):
        s = _summary()
        assert len(s["leaf"]) == 2                  # one A1 + one A2
        for tid, r in s["leaf"].items():
            full = json.loads((_ROOT / "artifacts" / "stage3d2" / "leaf" / tid /
                               "leaf_result.json").read_text(encoding="utf-8"))
            sc = full["score"]
            for key in ("status", "verified_stable", "feasible", "metrics", "margins",
                        "real_spice_calls", "simulator_failures", "components"):
                assert key in sc, key
            assert full["ranking_confidence"] == "ordinal_uncalibrated"
            assert full["budget"]["real_spice_calls"] <= 3   # budget respected
            assert isinstance(full["scalar_leaf_value"], float)

    def test_stability_dominates_scalar(self):
        """'An unstable high-FoM candidate must never outrank a stable feasible one.'"""
        from agentic_raptor.mb_sac.stage3d1 import PostSizingTopologyScore

        stable = PostSizingTopologyScore("t", "g", "successful_within_budget", True, True,
                                         {}, {"gain": 0.1, "pm": 0.1}, 3, 1, 0)
        unstable = PostSizingTopologyScore("t", "g", "unstable", False, False,
                                           {}, {"gain": 5.0, "pm": -0.5}, 3, None, 0)
        assert stable.compute_scalar() > unstable.compute_scalar()


class TestSnapshot:
    def test_pre_stage3d2_freeze_manifest(self):
        man = _ROOT / "artifacts" / "code_snapshots" / "pre_stage3d2_training" / "MANIFEST.json"
        if not man.is_file():
            pytest.skip("snapshot still copying")
        m = json.loads(man.read_text(encoding="utf-8"))
        assert m["reason"].startswith("Part A")
        assert len(m["files"]) > 30
        assert all(len(h) == 64 for h in m["files"].values())
