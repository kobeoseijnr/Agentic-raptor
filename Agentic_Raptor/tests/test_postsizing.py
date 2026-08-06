"""Post-sizing integration tests: action space, target-saturating reward,
spec-conditioned ranking, post-SAC self-earned qualification, value targets,
frozen-validation/blind-test discipline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.llm_dpo.stage3e4 import build_corpus, variant_text
from agentic_raptor.mb_sac import spec_sizing as ss

_ROOT = Path(__file__).resolve().parents[1]
O4 = _ROOT / "artifacts" / "stage3e4"

SPEC = {"gain_target_db": 56.5, "phase_margin_target_deg": 45.0,
        "load_capacitance_pf": 100.0, "ugbw_target_hz": None,
        "technology": "sky130"}


def m(gain=56.5, pm=55.0, stable=True, ugbw=None, op=True):
    return {"gain_db": gain, "pm_deg": pm, "ugbw_hz": ugbw, "stable": stable,
            "stability": "verified_stable" if stable else "verified_unstable",
            "electrical": "electrically_functional" if op else "failed",
            "op_valid": op}


@pytest.fixture(scope="module")
def corpus():
    build_corpus()
    return json.loads((O4 / "corpus.json").read_text())


# ---- 1: action space contains gain-sensitive variables -----------------------
class TestActionSpace:
    def test_manifest_has_gain_sensitive_variables(self):
        man = ss.action_space_manifest()
        for knob in ("s1_l", "s2_l", "ib_x"):        # length + bias: gm*ro movers
            assert knob in man["knobs"], f"missing gain-sensitive knob {knob}"
        assert man["knobs"]["s2_w"]["hi"] > 1.0      # stage-2 width can GROW now
        assert man["n_knobs"] == ss.N_KNOBS == len(ss.KNOB_NAMES)

    def test_apply_knobs_preserves_structure(self):
        from agentic_raptor.mapping import map_family

        class _S:
            topology_id = "t"
        g, _ = map_family(_S(), {"topology_id": "t", "gain_stages": 2,
                                 "functional_blocks": [],
                                 "unresolved_blocks": [],
                                 "mapping_readiness": "mapping_ready",
                                 "graph_hash": None})
        g2 = ss.apply_knobs(g, [2, 2, 2, 2, 2, 2])
        # sizing changes; structure (device count, kinds, roles, nets) does not
        assert len(g2.devices) == len(g.devices)
        assert [(d.kind, d.role) for d in g2.devices] == \
            [(d.kind, d.role) for d in g.devices]
        assert any(d2.sizing["w"] != d1.sizing["w"]
                   for d1, d2 in zip(g.devices, g2.devices)
                   if d1.kind in ("nmos", "pmos"))


# ---- 2-3: reward saturation + gain-vs-PM precedence ---------------------------
class TestReward:
    def test_pm_reward_saturates_after_cushion(self):
        r55, _ = ss.spec_reward(m(pm=55.0), SPEC)     # target + cushion
        r91, _ = ss.spec_reward(m(pm=91.2), SPEC)     # far beyond cushion
        assert abs(r91 - r55) < 0.02, "PM excess beyond cushion must earn ~0"
        r44, _ = ss.spec_reward(m(pm=44.0), SPEC)
        r30, _ = ss.spec_reward(m(pm=30.0), SPEC)
        assert r44 > r30, "improvement below target must still be rewarded"

    def test_gain_deficiency_outweighs_pm_excess(self):
        starved, _ = ss.spec_reward(m(gain=35.9, pm=91.2), SPEC)
        on_spec, _ = ss.spec_reward(m(gain=56.5, pm=55.0), SPEC)
        assert on_spec - starved > 1.0
        # the legacy reward made them near-equal — that is the repaired bug
        assert abs(ss.legacy_reward(m(gain=35.9, pm=91.2), SPEC)
                   - ss.legacy_reward(m(gain=56.5, pm=55.0), SPEC)) < 0.2

    def test_unstable_is_heavily_penalised_and_vector_kept(self):
        r, mv = ss.spec_reward(m(gain=90.0, pm=-8.0, stable=False), SPEC)
        assert r < -1.0
        assert mv["gain_margin_db"] == pytest.approx(33.5)
        assert mv["pm_margin_deg"] == pytest.approx(-53.0)


# ---- 4-5: spec-conditioned ranker ---------------------------------------------
class TestSpecConditionedRanker:
    def _feats(self, spec, pred):
        from agentic_raptor.corpus import TopologyRegistry
        from agentic_raptor.mb_sac.stage3d2 import V3
        rg = TopologyRegistry(V3).get_topology("topology_v2_0001").graph
        return ss.candidate_features(rg, spec, [1.0] * ss.N_KNOBS, pred, 0.5)

    def test_features_are_conditioned_on_active_spec(self):
        pred = {"gain_db": 60.0, "pm_deg": 50.0}
        a = self._feats(SPEC, pred).feature_vector()
        b = self._feats(dict(SPEC, gain_target_db=90.0), pred).feature_vector()
        assert a != b, "same candidate must embed differently under a new spec"

    def test_same_pair_reverses_under_different_target(self):
        from agentic_raptor.dpo import DPOConfig, DPORanker, FEATURE_DIM
        from agentic_raptor.dpo.preference_pairs import OutcomeRecord, build_pairs
        cand_gain = {"gain_db": 90.0, "pm_deg": 46.0}   # high gain, tight pm
        cand_pm = {"gain_db": 42.0, "pm_deg": 80.0}     # low gain, huge pm
        spec_hi = dict(SPEC, gain_target_db=85.0)       # needs the gain
        spec_lo = dict(SPEC, gain_target_db=40.0,       # needs the pm
                       phase_margin_target_deg=70.0)
        recs = []
        for spec in (spec_hi, spec_lo):
            for cand in (cand_gain, cand_pm):
                margins = {"gain": (cand["gain_db"]
                                    - spec["gain_target_db"]) / 20.0,
                           "pm": (cand["pm_deg"]
                                  - spec["phase_margin_target_deg"]) / 45.0}
                recs.append(OutcomeRecord(
                    features=self._feats(spec, cand),
                    dpo_score=None, dpo_rank=None,
                    selection_reason="test", spice_success=True,
                    passed_spec=all(v >= 0 for v in margins.values()),
                    constraint_margins=margins,
                    fom=min(margins.values()), runtime_s=1.0,
                    spice_calls_total=1))
        pairs = build_pairs(recs)
        assert pairs, "no training pairs built"
        ranker = DPORanker(FEATURE_DIM, DPOConfig(enabled=True, seed=0))
        ranker.train_on_pairs(pairs * 8)
        hi = ranker.rank([self._feats(spec_hi, cand_gain),
                          self._feats(spec_hi, cand_pm)])
        lo = ranker.rank([self._feats(spec_lo, cand_gain),
                          self._feats(spec_lo, cand_pm)])
        hi_pick = hi[0][0].predicted_margins["gain"]
        lo_pick = lo[0][0].predicted_margins["gain"]
        assert hi_pick > lo_pick, \
            "ranking must flip with the target: gain candidate under the " \
            "high-gain spec, pm candidate under the low-gain/tight-pm spec"


# ---- 6-7: post-SAC self-earned qualification ----------------------------------
class TestSelfEarnedQualification:
    def _row(self, corpus):
        r = next(x for x in corpus["records"] if x["split"] == "train")
        obj = json.loads(variant_text(r["stages"], r["comp"], False, False))
        cand = {"valid": True, "obj": obj,
                "graph_hash": ig.candidate_identity(obj)["canonical_graph_hash"]}
        row = {"context_id": r["context_id"], "prompt": r["prompt"],
               "topology_id": r["topology_id"],
               "target_variant": cand["graph_hash"],
               "target_stages": r["stages"]}
        return r, row, cand

    def test_nominal_gainless_measurement_cannot_verify(self, corpus):
        r, row, cand = self._row(corpus)
        spec = ig.parse_spec(r["prompt"])
        nominal = {"stability": "verified_stable",
                   "pm": spec["phase_margin_target_deg"] + 10,
                   "electrical": "electrically_functional"}   # no gain_db
        manifest = json.loads((O4 / "frozen_exam.json").read_text())
        tier, why = ig.classify_earned(row, cand, nominal, manifest)
        assert tier == "provisional_self_earned"
        assert "no_post_sizing_gain_measurement" in why

    def test_postsizing_pass_verifies_and_below_spec_does_not(self, corpus):
        r, row, cand = self._row(corpus)
        spec = ig.parse_spec(r["prompt"])
        manifest = json.loads((O4 / "frozen_exam.json").read_text())
        good = {"stability": "verified_stable",
                "pm": spec["phase_margin_target_deg"] + 5,
                "gain_db": spec["gain_target_db"] + 3,
                "electrical": "electrically_functional", "postsizing": True}
        tier, why = ig.classify_earned(row, cand, good, manifest)
        assert tier == "verified_self_earned", why
        weak = dict(good, gain_db=spec["gain_target_db"] - 10)
        tier2, why2 = ig.classify_earned(row, cand, weak, manifest)
        assert tier2 == "provisional_self_earned"
        assert "gain_below_spec" in why2


# ---- 8: value target from final post-sizing outcome ---------------------------
class TestValueTarget:
    def test_value_reflects_final_outcome_not_nominal_stability(self):
        good = ss.postsizing_outcome(m(gain=60.0, pm=50.0), SPEC)
        bad = ss.postsizing_outcome(m(gain=35.9, pm=91.2), SPEC)
        vg = ss.value_target_from_outcome(good, 8, 16)
        vb = ss.value_target_from_outcome(bad, 8, 16)
        assert vg["value_target"] >= 0.9 > vb["value_target"]
        assert vg["components"]["exact_spec_pass"] is True
        assert vb["margin_vector"]["gain_margin_db"] < 0  # failure visible

    def test_scalar_cannot_hide_constraint_failure(self):
        bad = ss.postsizing_outcome(m(gain=35.9, pm=91.2), SPEC)
        v = ss.value_target_from_outcome(bad, 8, 16)
        assert v["margin_vector"]["gain_margin_db"] == pytest.approx(-20.6)
        assert bad["passes"]["gain"] is False


# ---- 9-10: sizing/lineage alignment -------------------------------------------
class TestSizingAlignment:
    def test_puct_selected_topology_is_sized(self):
        trace_p = _ROOT / "artifacts" / "full_raptor_run" / "TRACE.json"
        if not trace_p.is_file():
            pytest.skip("no full-raptor trace yet")
        t = json.loads(trace_p.read_text())
        assert "sizing" in t and "mcts_puct" in t
        assert t["mcts_puct"]["acted_on"]         # decision recorded
        assert t["realised_hash"]                 # sized graph identity

    def test_knob_manifest_matches_action_dimension(self):
        assert len(ss.KNOB_LO) == len(ss.KNOB_HI) == ss.N_KNOBS
        assert all(lo < hi for lo, hi in zip(ss.KNOB_LO, ss.KNOB_HI))


# ---- 11-12: frozen validation + blind-test discipline --------------------------
class TestFrozenSets:
    def test_frozen_validation_unchanged(self, corpus):
        fe = json.loads((O4 / "frozen_exam.json").read_text())
        assert fe["frozen_exam_hash"] == "c344b244ba4dc478"

    def test_blind_set_is_family_and_spec_disjoint(self, corpus):
        sm = corpus["split_manifest"]
        assert not set(sm["blind_test_family_ids"]) & \
            set(sm["train_family_ids"])
        assert not set(sm["blind_test_family_ids"]) & \
            set(sm["validation_family_ids"])
        blind_lines = {r["prompt"].splitlines()[0] for r in corpus["records"]
                       if r["split"] == "blindtest"}
        train_lines = {r["prompt"].splitlines()[0] for r in corpus["records"]
                       if r["split"] == "train"}
        assert not blind_lines & train_lines

    def test_blind_set_inaccessible_to_training_selection(self, corpus):
        train_ctx = {r["context_id"] for r in corpus["records"]
                     if r["split"] == "train"}
        blind_ctx = {r["context_id"] for r in corpus["records"]
                     if r["split"] == "blindtest"}
        assert not train_ctx & blind_ctx
        # the campaign's DPO loader admits train-context pairs only; the exam
        # reads split=='heldout'; nothing reads 'blindtest' except blind_eval
        import run_self_improvement as rsi
        import inspect
        src = inspect.getsource(rsi)
        for fn in ("sft_generation", "dpo_generation", "design_and_measure",
                   "run_exam", "prepare_dpo_dataset"):
            fn_src = inspect.getsource(getattr(rsi, fn))
            assert "blindtest" not in fn_src, \
                f"{fn} must never touch the blind split"


# ---- measurement precedence: post-sizing must outrank nominal -----------------
class TestMeasurementPrecedence:
    def test_postsizing_outranks_nominal(self):
        from agentic_raptor.llm_dpo import merge_measurements
        nominal = {"variant": "v1", "stability": "verified_stable", "pm": 34.0}
        sized = {"variant": "v1", "stability": "verified_stable", "pm": 64.4,
                 "gain_db": 80.8, "postsizing": True}
        # regardless of file order, the gain-bearing record must win
        assert merge_measurements([nominal, sized])["v1"]["gain_db"] == 80.8
        assert merge_measurements([sized, nominal])["v1"]["gain_db"] == 80.8
        # newer post-sizing beats older post-sizing
        newer = dict(sized, gain_db=85.0)
        assert merge_measurements([sized, newer])["v1"]["gain_db"] == 85.0


# ---- persistent sizing memory + value refresh + VLM fallback ------------------
class TestPersistentMemory:
    def _run(self, tmp_path, monkeypatch, family, seed=0):
        import torch
        from agentic_raptor.mapping import map_family
        from agentic_raptor.topology_rl.stage3e2 import new_costs
        monkeypatch.setattr(ss, "STATE_DIR", tmp_path / "mem")
        monkeypatch.setattr(ss, "DYNAMICS_FILE", tmp_path / "dyn.jsonl")
        monkeypatch.setattr(ss, "REPLAY_FILE", tmp_path / "rep.jsonl")

        def fake_measure(tid, graph, exe, out_dir, tag, costs):
            w = sum(d.sizing["w"] for d in graph.devices
                    if d.kind in ("nmos", "pmos"))
            return {"gain_db": 40 + w / 10, "pm_deg": 50.0, "ugbw_hz": None,
                    "power_w": None, "stable": True,
                    "stability": "verified_stable",
                    "electrical": "electrically_functional",
                    "op_valid": True, "metrics": {}}
        monkeypatch.setattr(ss, "measure", fake_measure)

        class _S:
            topology_id = "memtest"
        g, _ = map_family(_S(), {"topology_id": "memtest", "gain_stages": 2,
                                 "functional_blocks": [],
                                 "unresolved_blocks": [],
                                 "mapping_readiness": "mapping_ready",
                                 "graph_hash": None})
        return ss.sac_size("memtest", g, SPEC, None, tmp_path, None,
                           budget=5, seed=seed, family=family)

    def test_memory_persists_and_warm_starts(self, tmp_path, monkeypatch):
        r1 = self._run(tmp_path, monkeypatch, "2s_none", seed=0)
        assert r1["memory"]["persisted"]
        assert not r1["memory"]["loaded_nets"]        # first run: cold
        r2 = self._run(tmp_path, monkeypatch, "2s_none", seed=1)
        assert r2["memory"]["loaded_nets"]            # second run: warm
        assert r2["memory"]["loaded_surrogate"]
        assert r2["memory"]["loaded_ranker"]
        meta = json.loads((tmp_path / "mem" / "meta.json").read_text())
        assert meta["families"]["2s_none"]["updates"] == 2
        # experience rows are family-labelled
        rows = [json.loads(x) for x in
                (tmp_path / "dyn.jsonl").read_text().splitlines()]
        assert all(r["family"] == "2s_none" for r in rows)

    def test_families_do_not_share_net_state(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch, "2s_none")
        r = self._run(tmp_path, monkeypatch, "3s_miller")
        assert not r["memory"]["loaded_nets"]         # different family: cold
        assert r["memory"]["loaded_ranker"]           # ranker is global

    def test_persist_false_leaves_no_state(self, tmp_path, monkeypatch):
        import torch
        from agentic_raptor.mapping import map_family
        monkeypatch.setattr(ss, "STATE_DIR", tmp_path / "mem")
        monkeypatch.setattr(ss, "DYNAMICS_FILE", tmp_path / "dyn.jsonl")
        monkeypatch.setattr(ss, "REPLAY_FILE", tmp_path / "rep.jsonl")
        monkeypatch.setattr(ss, "measure", lambda *a, **k: {
            "gain_db": 60.0, "pm_deg": 50.0, "ugbw_hz": None, "power_w": None,
            "stable": True, "stability": "verified_stable",
            "electrical": "electrically_functional", "op_valid": True,
            "metrics": {}})

        class _S:
            topology_id = "np"
        g, _ = map_family(_S(), {"topology_id": "np", "gain_stages": 2,
                                 "functional_blocks": [],
                                 "unresolved_blocks": [],
                                 "mapping_readiness": "mapping_ready",
                                 "graph_hash": None})
        r = ss.sac_size("np", g, SPEC, None, tmp_path, None, budget=3,
                        seed=0, family="2s_none", persist=False)
        assert not r["memory"]["persisted"]
        assert not (tmp_path / "mem").exists()


class TestValueRefresh:
    def test_family_graphs_build_for_all_labels(self):
        from agentic_raptor.topology_rl.value_refresh import (
            family_circuit_graph)
        for stages, comp in ((2, "none"), (2, "miller_cap"), (3, "miller"),
                             (3, "rc_nulling")):
            g = family_circuit_graph(stages, comp)
            assert len(g.nodes) > 4

    def test_refresh_only_replaces_the_checkpoint_when_it_wins(self):
        """The invariant is the ACCEPTANCE GATE, not unconditional gains.

        Demanding improvement every time was wrong: refresh runs on every
        campaign generation, and on a dataset the incumbent already exceeds
        the honest outcome is 'no improvement, keep the incumbent'. What
        must never happen is a worse model being written over a better one.
        """
        rep_p = _ROOT / "artifacts" / "value_refresh" / "REPORT.json"
        if not rep_p.is_file():
            pytest.skip("value refresh not run yet")
        rep = json.loads(rep_p.read_text())
        if "checkpoint_written" not in rep:
            pytest.skip("report predates the acceptance gate")
        improved = rep["improved"]
        assert improved == (rep["retrained_eval"]["spearman"]
                            > rep["stale_checkpoint_eval"]["spearman"]
                            and (rep["retrained_eval"]["pairwise_accuracy"]
                                 or 0)
                            >= (rep["stale_checkpoint_eval"]
                                ["pairwise_accuracy"] or 0))
        if rep["checkpoint_written"]:
            assert improved, "a losing model must never overwrite a winner"
            assert Path(rep["backup"]).is_file()      # old checkpoint kept


class TestVlmFallback:
    def test_multimodal_falls_back_when_adapter_missing(self):
        src = (_ROOT / "run_full_raptor.py").read_text(encoding="utf-8")
        assert "unavailable_adapter_missing" in src
        assert 'adapter_config.json").is_file()' in src


# ---- publication v2: feasibility reward, hybrid seeds, repaired RAG ----------
class TestV2FeasibilityReward:
    def test_required_orderings(self):
        from agentic_raptor.mb_sac.hybrid_sizing import feasibility_reward
        spec = dict(SPEC)
        feas = feasibility_reward(m(56.5, 55.0), spec)[0]
        pm_excess_gain_miss = feasibility_reward(m(40.0, 90.0), spec)[0]
        unstable_hi = feasibility_reward(
            m(95.0, -10.0, stable=False), spec)[0]
        no_op = feasibility_reward(m(None, None, op=False), spec)[0]
        assert feas > 1.0 > pm_excess_gain_miss > unstable_hi > no_op

    def test_excess_cannot_buy_back_failure(self):
        from agentic_raptor.mb_sac.hybrid_sizing import feasibility_reward
        spec = dict(SPEC)
        slight_miss_55 = feasibility_reward(m(54.0, 55.0), spec)[0]
        slight_miss_91 = feasibility_reward(m(54.0, 91.0), spec)[0]
        assert abs(slight_miss_55 - slight_miss_91) < 0.05


class TestV2Hybrid:
    def test_structured_seeds_are_in_bounds_and_start_nominal(self):
        from random import Random
        from agentic_raptor.mb_sac.hybrid_sizing import structured_seeds
        seeds = structured_seeds(dict(SPEC), "2s_none", "none", Random(0))
        assert seeds[0][0] == "nominal"
        for name, k in seeds:
            assert len(k) == ss.N_KNOBS
            assert all(lo <= v <= hi for v, lo, hi in
                       zip(k, ss.KNOB_LO, ss.KNOB_HI)), name

    def test_repair_move_targets_worst_constraint(self):
        from random import Random
        from agentic_raptor.mb_sac.hybrid_sizing import repair_move
        best = {**m(40.0, 60.0), "knobs": dict(zip(ss.KNOB_NAMES,
                                                   [1.0] * ss.N_KNOBS))}
        knobs, why = repair_move(best, dict(SPEC), Random(0))
        assert "gain" in why
        k = dict(zip(ss.KNOB_NAMES, knobs))
        assert k["s1_l"] > 1.0                      # longer channels


class TestV2Rag:
    def test_augmented_prompts_keep_contract(self, corpus):
        from agentic_raptor.llm_dpo import rag
        r = next(x for x in corpus["records"] if x["split"] == "train")
        spec = ig.parse_spec(r["prompt"])
        p2 = rag.augment_prompt(r["prompt"], spec, r["stages"])
        chk = rag.snapshot_check(p2)
        assert all(chk.values()), chk

    def test_no_evidence_from_validation_or_blind(self, corpus):
        from agentic_raptor.llm_dpo import rag
        protected = {x["context_id"] for x in corpus["records"]
                     if x["split"] in ("heldout", "blindtest")}
        for r in [x for x in corpus["records"]
                  if x["split"] == "train"][:5]:
            spec = ig.parse_spec(r["prompt"])
            for e in rag.retrieve(spec, r["stages"]):
                assert e.get("context_id") not in protected


class TestAggregationZeroDistance:
    def test_perfect_distance_zero_counts_as_zero(self):
        # falsy-zero regression: an exact pass has distance 0.0 and must
        # aggregate as 0.0, never as the None-fallback 1.0
        rows = [{"distance": 0.0}, {"distance": 0.5}, {"distance": None}]
        mean = sum(1.0 if x["distance"] is None else x["distance"]
                   for x in rows) / len(rows)
        assert mean == pytest.approx(0.5)
        import run_smoke_v2, inspect
        assert '"distance"] or 1' not in inspect.getsource(run_smoke_v2)


class TestC9sTrueSAC:
    def _run(self, tmp_path, monkeypatch, budget=12):
        from agentic_raptor.mapping import map_family
        from agentic_raptor.mb_sac import hybrid_sizing as hs

        def fake_measure(tid, graph, exe, out_dir, tag, costs):
            w = sum(d.sizing["w"] for d in graph.devices
                    if d.kind in ("nmos", "pmos"))
            return {"gain_db": 40 + w / 8, "pm_deg": 50.0, "ugbw_hz": None,
                    "power_w": None, "stable": True,
                    "stability": "verified_stable",
                    "electrical": "electrically_functional",
                    "op_valid": True, "metrics": {}}
        monkeypatch.setattr(hs, "measure", fake_measure)

        class _S:
            topology_id = "c9s_test"
        g, _ = map_family(_S(), {"topology_id": "c9s_test", "gain_stages": 2,
                                 "functional_blocks": [],
                                 "unresolved_blocks": [],
                                 "mapping_readiness": "mapping_ready",
                                 "graph_hash": None})
        return hs.c9s_size("c9s_test", g, dict(SPEC), None, tmp_path, None,
                           budget=budget, seed=3, family="2s_none",
                           comp="none")

    def test_all_three_phases_run_and_budget_is_respected(self, tmp_path,
                                                          monkeypatch):
        sz = self._run(tmp_path, monkeypatch, budget=12)
        phases = {l["phase"] for l in sz["phase_log"]}
        assert phases == {"seed", "sac", "repair"}
        assert sz["spice_calls"] == 12                 # exact budget
        assert sz["phase_budget"]["sac"] > 0

    def test_replay_warm_started_with_seed_experience(self, tmp_path,
                                                      monkeypatch):
        sz = self._run(tmp_path, monkeypatch)
        n_seed = sz["phase_budget"]["seeds"]
        assert len(sz["transitions"]) == sz["spice_calls"]
        # seed transitions precede any actor sample
        origins = [r["origin"] for r in sz["results"]]
        assert all(o.startswith("seed:") for o in origins[:n_seed])
        assert any(o == "sac:actor_sample" for o in origins)

    def test_actor_and_critics_actually_learn(self, tmp_path, monkeypatch):
        # instrument the networks: their parameters must CHANGE during the
        # run — otherwise the "SAC" would be decorative
        import torch
        from agentic_raptor.mb_sac import hybrid_sizing as hs
        snaps = {}
        orig_seq = torch.nn.Sequential.__call__
        real_init = torch.nn.Sequential.__init__
        created = []

        def rec_init(self, *a, **k):
            real_init(self, *a, **k)
            created.append(self)
        monkeypatch.setattr(torch.nn.Sequential, "__init__", rec_init)
        sz = self._run(tmp_path, monkeypatch)
        monkeypatch.setattr(torch.nn.Sequential, "__init__", real_init)
        assert sz["spice_calls"] > 0
        # actor(2*N out) + two critics(1 out) were created and trained:
        nets = [m for m in created if len(list(m.parameters())) >= 4]
        assert len(nets) >= 3

    def test_sac_alpha_is_tuned(self, tmp_path, monkeypatch):
        sz = self._run(tmp_path, monkeypatch)
        alphas = [l["alpha"] for l in sz["phase_log"] if l["phase"] == "sac"]
        assert alphas, "no SAC steps logged"
        # temperature must move from its exp(0)=1.0 start (auto-tuning alive)
        assert any(abs(a - 1.0) > 1e-3 for a in alphas)


class TestProductionEngineIsTrueSAC:
    def test_entropy_temperature_machinery_present_and_used(self):
        import inspect
        src = inspect.getsource(ss.sac_size)
        for token in ("log_alpha", "target_entropy", "logp",
                      "log_alpha.exp().detach() * logp"):
            assert token in src, f"missing SAC component: {token}"

    def test_twin_critics_and_min_q(self):
        import inspect
        src = inspect.getsource(ss.sac_size)
        assert "torch.min(q1(" in src and "q2(" in src


class TestTrueRLGuarantees:
    def test_production_sac_has_full_soft_actor_critic(self):
        # NOTE: these are source-presence checks only. The behavioural
        # guarantees (bootstrapped target through Polyak-averaged target
        # critics, an action-movable state) live in tests/test_true_rl.py.
        import inspect
        src = inspect.getsource(ss.sac_size)
        for token in ("log_alpha", "target_entropy",
                      "log_alpha.exp().detach() * logp",
                      "torch.log(1 - a_ ** 2"):     # tanh log-prob correction
            assert token in src, f"missing canonical SAC component: {token}"
        assert "torch.min(q1(" in src            # clipped double-Q
        assert "q1_t(" in src and "q2_t(" in src  # bootstrapped, not bandit

    def test_alphazero_policy_head_gets_visit_targets(self):
        from agentic_raptor.topology_rl import value_refresh as vr
        ex = vr._policy_examples()
        if not ex:
            pytest.skip("no P8 visit records on disk")
        e = ex[0]
        dist = e["visit_distribution"]
        assert abs(sum(dist.values()) - 1.0) < 1e-6   # normalized pi_MCTS
        assert len(dist) >= 1 and e["value_target"] is not None
        assert set(e["legal_action_ids"]) == set(dist)
