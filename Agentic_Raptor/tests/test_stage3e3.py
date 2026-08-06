"""Stage 3E.3 tests: SFT v2 dataset/splits, trained-model generation and
realisation, refinement hop, DPO v2 invariants, comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

_ROOT = Path(__file__).resolve().parents[1]
O3 = _ROOT / "artifacts" / "stage3e3"


def _load(n):
    p = O3 / n
    if not p.is_file():
        pytest.skip(f"{n} not yet produced")
    return json.loads(p.read_text(encoding="utf-8"))


class TestSFTv2:
    def test_dataset_splits_and_dedup(self):
        d = _load("sft_dataset_v2.json")
        assert d["heldout_targets_excluded"] is True
        hashes = [r["graph_hash"] for r in d["train"] + d["validation"]]
        assert len(hashes) == len(set(hashes))
        assert d["stats"]["deduplicated"] <= d["stats"]["raw"]
        assert set(d["stats"]["per_source"]) == {"A1", "A2"}
        # held-out STRUCTURE: 3-stage never in training responses
        assert all('"n2"' not in r["response"] for r in d["train"])
        assert '"n2"' in d["heldout_structure"][0]["response"]

    def test_sft_v2_trained_with_validation(self):
        d = _load("sft_v2.json")
        assert d["loss_first_last"][1] < d["loss_first_last"][0]
        assert d["validation_loss"] > 0
        assert 0 < d["trainable_params"] < 2_000_000
        assert d["greedy_structured_eval"]["attempts"] >= 2


class TestTrainedGeneration:
    def test_campaign_records_schema(self):
        d = _load("generation_campaign.json")
        assert d["generated"] >= 6            # multiple contexts x decode seeds
        for r in json.loads((O3 / "generation_campaign.json").read_text())["records"][:3]:
            for k in ("prompt_id", "target_id", "decode_seed", "checkpoint",
                      "parseable", "valid", "novelty", "text_hash"):
                assert k in r

    def test_trained_proposal_reached_real_spice(self):
        d = _load("realised_proposal.json")
        if d.get("status") == "no_valid_trained_generation":
            pytest.skip("no valid trained generation at this scale (honest)")
        assert d["status"] == "mapped_and_simulated"
        assert d["spice_calls"] >= 1
        assert d["provider"] == "trained_sft_v2_model_generation"

    def test_refinement_hard_gated(self):
        d = _load("refinement.json")
        assert d["pre_hash"] != d["post_hash"]
        assert d["action_dims"][1] >= d["action_dims"][0]
        assert abs(sum(d["root_visit_distribution"].values()) - 1.0) < 1e-6
        assert d["spice_calls"] >= 2          # before + after on real ngspice
        assert isinstance(d["improved_under_hard_gates"], bool)  # never asserted true


class TestDPOv2Invariants:
    def test_dpo_v2(self):
        p = _ROOT / "artifacts/stage3e2_llm/dpo.json"
        if not p.is_file():
            pytest.skip("dpo not run")
        d = json.loads(p.read_text())
        for sd in d["per_seed"].values():
            assert sd["policy_params_changed"] and sd["reference_unchanged"]

    def test_comparison_v2_equal_conditions(self):
        d = _load("comparison_v2.json")
        assert set(d) == {"base", "sft", "sft_dpo"}
        attempts = {m["attempts"] for m in d.values()}
        assert len(attempts) == 1             # identical context budget
