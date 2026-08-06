"""Part M tests for the data-integrity layer of the repaired self-improvement
campaign: prompt propagation, quarantine, same-context pairing, dedup and
contradiction controls, spec-conditioned preferences, dominance guards,
frozen-exam leakage, campaign hygiene, lineage and pre-training blockers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.llm_dpo.stage3e4 import build_corpus, variant_text

_ROOT = Path(__file__).resolve().parents[1]
O4 = _ROOT / "artifacts" / "stage3e4"


# ------------------------------ helpers --------------------------------------
def prompt_for(gain=75.0, pm=45.0, cl=500, ugbw=1e5, tech="sky130"):
    return (f"### SPEC gain>={gain}dB pm>={pm}deg cl={cl}pF "
            f"ugbw>={ugbw:.0e}Hz tech={tech}\n### RAG rag_l2_topology_0001\n"
            f"### BLOCKS five_transistor_first_stage,cs_gain_stage,"
            f"miller_cap,bias_mirror\n"
            f"### FORBIDDEN raw_netlist,feedback_to_input\n### PROPOSAL\n")


def cand(stages, comp):
    obj = json.loads(variant_text(stages, comp, False, False))
    ident = ig.candidate_identity(obj)
    return {"valid": True, "parseable": True, "obj": obj,
            "graph_hash": ident["canonical_graph_hash"]}, ident


def meas(stability="verified_stable", pm=60.0, electrical="electrically_functional",
         testbench_hash=None):
    m = {"stability": stability, "pm": pm, "electrical": electrical,
         "variant": "x", "device_hash": "d"}
    if testbench_hash:
        m["testbench_hash"] = testbench_hash
    return m


def row_for(stages, comp, cands, prompt=None, **extra):
    tgt = json.loads(variant_text(stages, comp, False, False))
    return dict({"context_id": "t_test",
                 "prompt": prompt_for() if prompt is None else prompt,
                 "topology_id": "topology_test",
                 "target_variant": ig.candidate_identity(tgt)["canonical_graph_hash"],
                 "target_stages": stages, "candidates": cands}, **extra)


def build(rows, meas_map):
    return ig.build_context_pairs(rows, meas_map, {"campaign_id": "test",
                                                   "generation_id": 0})


@pytest.fixture(scope="module")
def corpus():
    build_corpus()
    return json.loads((O4 / "corpus.json").read_text())


@pytest.fixture(scope="module")
def exam_manifest(corpus):
    return json.loads((O4 / "frozen_exam.json").read_text())


# --------------------- M1-M5: prompt propagation + quarantine -----------------
class TestPromptPropagation:
    def test_real_prompt_propagates_from_corpus_to_pair(self, corpus):
        r = next(x for x in corpus["records"] if x["split"] == "train")
        c3, _ = cand(r["stages"], r["comp"])
        c2, _ = cand(2 if r["stages"] == 3 else 3, "miller")
        mm = {c3["graph_hash"]: meas(), c2["graph_hash"]: meas("verified_unstable", -10)}
        rows = [{"context_id": r["context_id"], "prompt": r["prompt"],
                 "topology_id": r["topology_id"],
                 "target_variant": r["variant_hash"],
                 "target_stages": r["stages"], "candidates": [c3, c2]}]
        out = build(rows, mm)
        assert out["pairs"], "no pair built from a real corpus record"
        p = out["pairs"][0]
        assert p["prompt"] == r["prompt"] == p["original_prompt"]
        assert p["structured_spec"]["gain_target_db"] > 0
        assert p["evaluation_context_id"] == \
            ig.evaluation_context_id(p["structured_spec"])

    def test_missing_prompt_causes_quarantine_and_no_pair(self):
        c1, _ = cand(2, "miller")
        quarantined = []
        out = ig.build_context_pairs(
            [row_for(2, "miller", [c1], prompt="")],
            {c1["graph_hash"]: meas()}, {}, quarantine_fn=quarantined.append)
        assert not out["pairs"]
        assert out["drops"]["dropped_missing_prompt"] == 1
        assert quarantined and "missing_real_prompt" in quarantined[0]["reason"]

    def test_unparseable_prompt_is_not_a_spec(self):
        assert ig.parse_spec("Design an amplifier") is None
        assert ig.parse_spec("") is None

    def test_synthetic_prompt_cannot_enter_training(self):
        c1, _ = cand(2, "miller")
        c2, _ = cand(3, "miller")
        mm = {c1["graph_hash"]: meas(), c2["graph_hash"]: meas()}
        out = build([row_for(2, "miller", [c1, c2],
                             synthetic_prompt=True, training_eligible=False)],
                    mm)
        assert not out["pairs"]


# ------------------- M6-M10: evaluation-context discipline --------------------
class TestEvaluationContext:
    def test_same_context_candidates_form_pair(self):
        ca, _ = cand(2, "miller")
        cb, _ = cand(3, "miller")
        mm = {ca["graph_hash"]: meas(), cb["graph_hash"]: meas("verified_unstable", -6)}
        out = build([row_for(2, "miller", [ca, cb])], mm)
        assert len(out["pairs"]) == 1

    def test_cross_spec_candidates_cannot_pair(self):
        # candidates proposed under DIFFERENT specs live in different rows and
        # different evaluation contexts — no cross-row pair may exist
        ca, _ = cand(2, "miller")
        cb, _ = cand(3, "miller")
        mm = {ca["graph_hash"]: meas(), cb["graph_hash"]: meas()}
        out = build([row_for(2, "miller", [ca], prompt=prompt_for(gain=50)),
                     row_for(3, "miller", [cb], prompt=prompt_for(gain=90))],
                    mm)
        assert not out["pairs"]

    def test_different_loads_are_different_contexts(self):
        a = ig.parse_spec(prompt_for(cl=100))
        b = ig.parse_spec(prompt_for(cl=500))
        assert ig.evaluation_context_id(a) != ig.evaluation_context_id(b)

    def test_different_pdk_is_a_different_context(self):
        a = ig.parse_spec(prompt_for())
        b = dict(a, pdk_version="sky130B")
        assert ig.evaluation_context_id(a) != ig.evaluation_context_id(b)

    def test_different_testbench_measurement_is_dropped(self):
        ca, _ = cand(2, "miller")
        cb, _ = cand(3, "miller")
        mm = {ca["graph_hash"]: meas(),
              cb["graph_hash"]: meas(testbench_hash="othertb0000000")}
        out = build([row_for(2, "miller", [ca, cb])], mm)
        assert not out["pairs"]
        assert out["drops"]["dropped_cross_context"] == 1


# ---------------- M11-M15: dedup, contradiction, tie controls -----------------
class TestDedupAndContradiction:
    def _pair(self, chosen_h, rejected_h, conf=0.9, ectx="ctx1"):
        return {"pair_id": f"p_{chosen_h}_{rejected_h}_{conf}",
                "evaluation_context_id": ectx,
                "chosen_topology_hash": chosen_h,
                "rejected_topology_hash": rejected_h, "confidence": conf}

    def test_duplicate_pairs_are_removed(self):
        d = ig.dedupe_pairs([self._pair("A", "B"), self._pair("A", "B")])
        assert len(d["pairs"]) == 1
        assert d["report"]["dropped_exact_duplicates"] == 1

    def test_reverse_duplicate_resolved_by_confidence_gap(self):
        d = ig.dedupe_pairs([self._pair("A", "B", 0.9),
                             self._pair("B", "A", 0.6)])
        assert len(d["pairs"]) == 1
        assert d["pairs"][0]["chosen_topology_hash"] == "A"
        assert d["report"]["dropped_reverse_duplicates"] == 1

    def test_contradictory_equal_confidence_pairs_all_dropped(self):
        d = ig.dedupe_pairs([self._pair("A", "B", 0.9),
                             self._pair("B", "A", 0.9)])
        assert not d["pairs"]
        assert d["report"]["dropped_contradictory"] == 2
        assert d["report"]["dropped_unresolved_conflict"] == 1

    def test_tie_within_pm_noise_produces_no_pair(self):
        ca, _ = cand(3, "miller")
        cb, _ = cand(3, "rc")
        # both wrong-exact, same tier, both stable, pm within noise
        mm = {ca["graph_hash"]: meas(pm=61.0),
              cb["graph_hash"]: meas(pm=59.0)}
        out = build([row_for(3, "none", [ca, cb])], mm)
        assert not out["pairs"]
        assert out["drops"]["dropped_ties"] == 1

    def test_ambiguous_stability_cannot_decide(self):
        ca, _ = cand(3, "miller")
        cb, _ = cand(3, "rc")
        mm = {ca["graph_hash"]: meas("verified_stable", 60),
              cb["graph_hash"]: meas("phase_margin_unavailable", None)}
        out = build([row_for(3, "none", [ca, cb])], mm)
        assert not out["pairs"]


# --------------- M16-M17: spec-conditioned preference policy ------------------
class TestSpecConditionedPreference:
    def test_raw_capability_alone_cannot_win(self):
        # under a 2-stage spec the exact 2-stage design must beat the
        # higher-raw-gain 3-stage design even when both are stable
        c2, _ = cand(2, "miller")
        c3, _ = cand(3, "miller")
        mm = {c2["graph_hash"]: meas(pm=50),
              c3["graph_hash"]: meas(pm=70)}     # better raw pm, wrong tier
        out = build([row_for(2, "miller", [c2, c3])], mm)
        assert len(out["pairs"]) == 1
        assert out["pairs"][0]["chosen_topology_hash"] == c2["graph_hash"]
        assert out["pairs"][0]["preference_rule"] == "gain_tier_feasibility"

    def test_same_candidates_flip_with_the_target(self):
        c2, _ = cand(2, "miller")
        c3, _ = cand(3, "miller")
        mm = {c2["graph_hash"]: meas(pm=50), c3["graph_hash"]: meas(pm=50)}
        low = build([row_for(2, "miller", [c2, c3],
                             prompt=prompt_for(gain=50))], mm)
        high = build([row_for(3, "miller", [c2, c3],
                              prompt=prompt_for(gain=90))], mm)
        assert low["pairs"][0]["chosen_topology_hash"] == c2["graph_hash"]
        assert high["pairs"][0]["chosen_topology_hash"] == c3["graph_hash"]

    def test_stable_wrong_structure_does_not_beat_right_structure(self):
        # A5 regression: 'stable but cannot reach the gain tier' must lose
        c3, _ = cand(3, "miller")               # right tier, unstable
        c2, _ = cand(2, "miller")               # wrong tier, stable
        mm = {c3["graph_hash"]: meas("verified_unstable", -6),
              c2["graph_hash"]: meas("verified_stable", 34)}
        out = build([row_for(3, "miller", [c3, c2],
                             prompt=prompt_for(gain=90))], mm)
        assert out["pairs"][0]["chosen_topology_hash"] == c3["graph_hash"]


# ------------------- M18-M19: dominance / collapse guards ---------------------
class TestDominanceGuards:
    def _pairs(self, n, chosen_stages=2):
        out = []
        for i in range(n):
            c, ci = cand(chosen_stages, "miller")
            r, ri = cand(3, "rc")
            out.append({"pair_id": f"p{i}", "confidence": 0.9,
                        "spec_id": f"spec{i}", "prompt_hash": f"ph{i}",
                        "evaluation_context_id": f"ctx{i}",
                        "chosen_topology_hash": ci["canonical_graph_hash"],
                        "rejected_topology_hash": ri["canonical_graph_hash"],
                        "chosen_identity": ci, "rejected_identity": ri})
        return out

    def test_candidate_dominance_is_capped(self):
        limits = dict(ig.DEFAULT_LIMITS,
                      max_chosen_appearances_per_candidate=3)
        b = ig.balance_pairs(self._pairs(10), limits)
        assert len(b["pairs"]) == 3
        assert b["distribution"]["dropped_balance_cap"] == 7

    def test_single_response_dominance_raises_collapse_flag(self):
        b = ig.balance_pairs(self._pairs(10))
        assert any("dominance" in f or "family" in f or "chosen circuits" in f
                   for f in b["collapse_flags"])

    def test_family_dominance_raises_collapse_flag(self):
        b = ig.balance_pairs(self._pairs(10))
        assert b["distribution"]["maximum_family_fraction"] == 1.0
        assert b["collapse_flags"]


# ------------------- M20-M22: frozen-exam leakage guard -----------------------
class TestFrozenExamLeakage:
    def test_no_family_or_spec_overlap_between_splits(self, corpus):
        sm = corpus["split_manifest"]
        # validation_* = the frozen held-out acceptance exam;
        # blind_test_* = the untouched final test set
        assert not set(sm["train_family_ids"]) & set(sm["validation_family_ids"])
        assert not set(sm["train_spec_ids"]) & set(sm["validation_spec_ids"])
        assert not set(sm["blind_test_family_ids"]) & \
            set(sm["validation_family_ids"])
        assert not set(sm["train_family_ids"]) & \
            set(sm["blind_test_family_ids"])

    def test_heldout_family_cannot_enter_training(self, corpus, exam_manifest):
        fam = exam_manifest["frozen_exam_family_ids"][0]
        offenders = ig.leakage_check(
            [{"prompt": prompt_for(), "topology_id": fam}], exam_manifest)
        assert offenders and "heldout_topology_family" in offenders[0]["reasons"]

    def test_frozen_exam_prompt_cannot_enter_training(self, corpus,
                                                      exam_manifest):
        held = next(r for r in corpus["records"] if r["split"] == "heldout")
        offenders = ig.leakage_check([{"prompt": held["prompt"]}],
                                     exam_manifest)
        assert offenders and "exact_exam_prompt" in offenders[0]["reasons"]

    def test_exam_spec_with_extra_evidence_lines_still_caught(self, corpus,
                                                              exam_manifest):
        # graph-equivalent / reformatted derivation of an exam prompt: same
        # SPEC line with extra KNOWN evidence appended must still be blocked
        held = next(r for r in corpus["records"] if r["split"] == "heldout")
        mutated = held["prompt"].replace(
            "### PROPOSAL", "### KNOWN 3stage verified_stable\n### PROPOSAL")
        offenders = ig.leakage_check([{"prompt": mutated}], exam_manifest)
        assert offenders and "exam_spec_line" in offenders[0]["reasons"]

    def test_exam_is_frozen_across_rebuilds(self, corpus, exam_manifest):
        build_corpus()
        again = json.loads((O4 / "frozen_exam.json").read_text())
        assert again["frozen_exam_hash"] == exam_manifest["frozen_exam_hash"]


# ------------- M23-M26: lineage, acceptance, rollback --------------------------
class TestLineageAndAcceptance:
    def test_checkpoint_hash_is_content_addressed(self, tmp_path):
        a = tmp_path / "ck_a"; a.mkdir()
        (a / "adapter_config.json").write_text('{"r": 8}')
        b = tmp_path / "ck_b"; b.mkdir()
        (b / "adapter_config.json").write_text('{"r": 16}')
        assert ig.sha_checkpoint(a) != ig.sha_checkpoint(b)
        assert ig.sha_checkpoint(a) == ig.sha_checkpoint(a)

    def test_generation_with_wrong_parent_hash_is_blocked(self, corpus,
                                                          tmp_path):
        import run_self_improvement as rsi
        fake_parent = tmp_path / "gen0_sft"; fake_parent.mkdir()
        (fake_parent / "adapter_config.json").write_text("{}")
        camp = ig.new_campaign(tmp_path / "si", [], "test")
        exam_manifest = json.loads((O4 / "frozen_exam.json").read_text())
        with pytest.raises(AssertionError, match="lineage"):
            rsi.run_generation(1, corpus, camp, fake_parent,
                               "not_the_accepted_hash", exam_manifest,
                               rsi.CFG_VALIDATE)

    def test_dpo_collapse_causes_rejection(self):
        sft = {"valid_rate": 1.0, "spec_match_rate": 0.4,
               "unique_structures": 4, "most_common_response_fraction": 0.3}
        collapsed = {"valid_rate": 1.0, "spec_match_rate": 0.0,
                     "unique_structures": 1,
                     "most_common_response_fraction": 1.0}
        accept, reasons = ig.decide_acceptance(sft, collapsed)
        assert not accept
        assert any("unique" in r for r in reasons)
        assert any("spec_match" in r for r in reasons)

    def test_healthy_dpo_is_accepted(self):
        sft = {"valid_rate": 1.0, "spec_match_rate": 0.4,
               "unique_structures": 4, "most_common_response_fraction": 0.3}
        better = {"valid_rate": 1.0, "spec_match_rate": 0.5,
                  "unique_structures": 4,
                  "most_common_response_fraction": 0.3}
        accept, reasons = ig.decide_acceptance(sft, better)
        assert accept and not reasons


# ---------------- M27-M30: campaign hygiene + hard blockers -------------------
class TestCampaignHygiene:
    def test_stale_artifacts_are_archived_with_manifest(self, tmp_path):
        stale = tmp_path / "old_pairs.jsonl"
        stale.write_text('{"promptless": true}')
        camp = ig.new_campaign(tmp_path / "si", [stale], "poisoned")
        assert not stale.exists()
        assert camp["archived"][0]["file_hash"]
        assert camp["archived"][0]["archive_reason"] == "poisoned"
        assert Path(camp["archived"][0]["archived_path"]).is_file()
        m = json.loads((camp["dirs"]["manifests"] /
                        "archive_manifest.json").read_text())
        assert m[0]["original_path"] == str(stale)

    def test_new_campaign_does_not_load_old_pair_caches(self, corpus,
                                                        tmp_path):
        import run_self_improvement as rsi
        camp = ig.new_campaign(tmp_path / "si", [], "test")
        exam_manifest = json.loads((O4 / "frozen_exam.json").read_text())
        pairs, ready, _ = rsi.prepare_dpo_dataset(camp, corpus, exam_manifest,
                                                  True, "ok")
        assert pairs == []              # only campaign-local pair files count
        assert not ready["training_ready"]

    def test_training_blocks_on_any_integrity_failure(self, exam_manifest):
        c2, ci = cand(2, "miller")
        bad_pair = {"pair_id": "p0", "prompt": "Design an amplifier",
                    "structured_spec": None, "context_id": "x",
                    "evaluation_context_id": "e", "confidence": 0.9,
                    "spec_id": "s", "prompt_hash": "h",
                    "chosen_topology_hash": "A", "rejected_topology_hash": "B",
                    "chosen_identity": ci, "rejected_identity": ci}
        ready = ig.readiness_report([bad_pair], exam_manifest,
                                    {"unique_specs": 1,
                                     "unique_chosen_candidates": 1},
                                    ["collapse"], False, "broken parent", False)
        assert not ready["training_ready"]
        for blocker in ("all_pairs_have_real_prompts",
                        "generation_parent_checkpoint_correct",
                        "frozen_exam_hash_unchanged"):
            assert blocker in ready["blockers"]

    def test_readiness_passes_on_clean_pairs(self, corpus, exam_manifest):
        # one train record per structure class so no family dominates
        by_class = {}
        for r in corpus["records"]:
            if r["split"] == "train":
                by_class.setdefault((r["stages"], r["comp"]), r)
        train = list(by_class.values())
        pairs = []
        for r in train:
            ca, _ = cand(r["stages"], r["comp"])
            cb, _ = cand(2 if r["stages"] == 3 else 3, "miller")
            mm = {ca["graph_hash"]: meas(),
                  cb["graph_hash"]: meas("verified_unstable", -10)}
            rows = [{"context_id": r["context_id"], "prompt": r["prompt"],
                     "topology_id": r["topology_id"],
                     "target_variant": r["variant_hash"],
                     "target_stages": r["stages"], "candidates": [ca, cb]}]
            pairs += ig.build_context_pairs(rows, mm, {})["pairs"]
        deduped = ig.dedupe_pairs(pairs)
        balanced = ig.balance_pairs(deduped["pairs"])
        ready = ig.readiness_report(balanced["pairs"], exam_manifest,
                                    balanced["distribution"],
                                    balanced["collapse_flags"], True, "ok",
                                    True)
        assert ready["training_ready"], ready["blockers"]
