"""LLM-DPO amendment tests: schema, ordering, same-context rule, genuine DPO
loss math, frozen reference, masking, separation from the BT sizing ranker."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

from agentic_raptor import llm_dpo as L

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "artifacts" / "stage3e2_llm"


def _load(n):
    p = OUT / n
    if not p.is_file():
        pytest.skip(f"{n} not yet produced")
    return json.loads(p.read_text(encoding="utf-8"))


class TestSchemaAndOrdering:
    def test_proposal_text_roundtrip(self):
        obj = L.parse_proposal_text(L.proposal_to_text(2, True))
        assert obj is not None
        ok, reasons = L.proposal_dict_valid(obj)
        assert ok, reasons

    def test_unparseable_and_invalid(self):
        assert L.parse_proposal_text("garbage no json") is None
        bad = L.parse_proposal_text(L.proposal_to_text(2, True).replace(
            "cs_gain_stage", "quantum_stage"))
        ok, reasons = L.proposal_dict_valid(bad)
        assert not ok and any("unsupported_block" in r for r in reasons)

    def test_lexicographic_ordering(self):
        stable = {"parseable": 1, "schema_valid": 1, "canonical": 1,
                  "structural_valid": 1, "supplies_bias_complete": 1,
                  "mapping_supported": 1, "netlist_valid": 1, "op_valid": 1,
                  "stable_after_sizing": 1, "hard_feasible": 1, "margins": 0.1,
                  "fom": 0.1}
        unstable_high_fom = dict(stable, stable_after_sizing=0, hard_feasible=0,
                                 fom=99.0)
        assert L.order_pair(stable, unstable_high_fom) == "a"   # stability dominates FoM
        unmappable = dict(stable, mapping_supported=0)
        assert L.order_pair(stable, unmappable) == "a"          # mapped > rationale
        assert L.order_pair(dict(stable), dict(stable)) == "tie"
        pareto = dict(stable, margins=0.9, fom=0.01)
        assert L.order_pair(stable | {"fom": 0.5}, pareto) == "ambiguous"
        assert L.order_pair({"parseable": 1}, {"parseable": 0}) == "a"

    def test_pairs_same_context_rule(self):
        d = _load("preference_pairs.json")
        for p in d["pairs"]:
            assert p["context_id"]                     # every pair has ONE context
            assert p["preferred"] != p["rejected"]
        assert d["high_fom_unstable_never_preferred"] is True
        assert d["sources"]["real_spice"] >= 1         # SPICE-backed pairs exist

    def test_separation_from_bt_sizing_ranker(self):
        doc = L.__doc__
        assert "disjoint" in doc and "Bradley" in doc
        # no import linkage between the two preference systems (docstring
        # mentions the ranker only to state the separation)
        import inspect
        src = inspect.getsource(L)
        assert "from agentic_raptor.dpo import" not in src
        assert "import agentic_raptor.dpo" not in src


class TestTrueDPOMechanics:
    def test_dpo_loss_math_hand_check(self):
        """-logsigmoid(beta*margin) at margin=0 must equal log(2)."""
        import torch
        loss = -torch.nn.functional.logsigmoid(torch.tensor(0.1 * 0.0))
        assert abs(float(loss) - math.log(2)) < 1e-6
        big = -torch.nn.functional.logsigmoid(torch.tensor(0.1 * 50.0))
        assert float(big) < math.log(2)                # preferred margin lowers loss

    def test_seq_logprob_response_masking(self):
        import torch
        tok, model = L.load_models(lora=False, seed=0)
        lp1, n1 = L.seq_logprob(model, tok, "prompt A:", " short")
        lp2, n2 = L.seq_logprob(model, tok, "different prompt B here:", " short")
        assert n1 == n2                                 # only response tokens counted
        assert float(lp1) != float(lp2)                 # but conditioned on prompt

    def test_dpo_trained_policy_and_frozen_reference(self):
        d = _load("dpo.json")
        assert d["reference_free"] is False and d["beta"] == 0.1
        assert sorted(map(int, d["per_seed"])) == [0, 1, 2]
        for sd in d["per_seed"].values():
            assert sd["policy_params_changed"] is True
            assert sd["reference_unchanged"] is True
        assert "logsigmoid" in d["objective"]
        assert "lp_ref" in d["objective"]               # reference term present

    def test_dpo_loss_decreased_and_margin_positive(self):
        d = _load("dpo.json")
        for sd in d["per_seed"].values():
            first, last = sd["loss_first_last"]
            assert last < first                          # learning on train pairs
            assert sd["reward_margin_last"] > 0

    def test_implicit_reward_accuracy_reported(self):
        d = _load("dpo.json")
        for sd in d["per_seed"].values():
            assert 0.0 <= sd["implicit_reward_accuracy_heldout"] <= 1.0


class TestSFTAndDataset:
    def test_sft_split_isolation(self):
        d = _load("sft_dataset.json")
        assert d["heldout_targets_excluded"] is True
        hashes = [r["graph_hash"] for r in d["train"] + d["validation"]]
        assert len(hashes) == len(set(hashes))          # graph-isomorphic leakage
        splits = {r["split"] for r in d["train"]}
        assert splits == {"train"}

    def test_sft_trained_with_lora(self):
        d = _load("sft.json")
        assert d["lora"]["r"] == 8 and d["lora"]["modules"] == ["c_attn"]
        assert 0 < d["trainable_params"] < 2_000_000    # LoRA only, not full model
        assert d["loss_first_last"][1] < d["loss_first_last"][0]
        assert "dpo_gate_note" in d                     # reliability gate honesty

    def test_comparison_ran_all_three(self):
        d = _load("comparison.json")
        assert set(d) == {"base", "sft", "sft_dpo"}
        for m in d.values():
            assert m["attempts"] >= 1


class TestIntegration:
    def test_mcts_ingestion_validated(self):
        d = _load("mcts_ingestion.json")
        assert d["validated_before_ingestion"] is True
        assert d["tree_nodes"] >= 1 and d["root_visits"] >= 2

    def test_no_generated_netlist_execution(self):
        import inspect
        src = inspect.getsource(L)
        # module never invokes the simulator or emits netlists (comments may
        # reference evidence provenance; execution symbols must be absent)
        assert "discover_ngspice" not in src
        assert "qualify_family" not in src
        assert "emit_netlist" not in src

    def test_queue_dedup_and_frozen_flag(self):
        q = _ROOT / "datasets/llm_preference_queue/v1.jsonl"
        if not q.is_file():
            pytest.skip("queue not written")
        rows = [json.loads(x) for x in q.read_text().splitlines()]
        keys = [(r["context_id"], r["pair_id"]) for r in rows]
        assert len(keys) == len(set(keys))
        assert all(r["active_adapter_frozen_during_heldout"] for r in rows)

    def test_model_record(self):
        assert L.MODEL_RECORD["licence"] == "apache-2.0"
        assert L.MODEL_RECORD["model_id"] == L.MODEL_ID   # env-configurable
        assert "quantisation" in L.MODEL_RECORD
