"""RAPTOR (2026-08-16): Design Planner, Topology Critic,
Optimization Supervisor, Recovery Agent + DesignState/BudgetLedger +
pipeline wiring. Zero LLM, zero ngspice -- every agent decision is
deterministic and testable from synthetic inputs."""
from __future__ import annotations

import inspect

import pytest

from agentic_raptor import agents as ag
from agentic_raptor.agents.state import BudgetExceeded

EASY = {"gain_target_db": 56.5, "phase_margin_target_deg": 45.0,
        "ugbw_target_hz": 1e4, "load_capacitance_pf": 100.0}
HARD_GAIN = {"gain_target_db": 108.29, "phase_margin_target_deg": 60.0,
             "ugbw_target_hz": 1e5, "load_capacitance_pf": 100.0}
HARD_BW = {"gain_target_db": 89.45, "phase_margin_target_deg": 45.0,
           "ugbw_target_hz": 1e6, "load_capacitance_pf": 200.0}


# ---------------------------------------------------------------------------
# Design Planner
# ---------------------------------------------------------------------------
def test_planner_prefers_shallow_on_easy_and_deep_on_high_gain():
    easy = ag.plan(EASY)
    hard = ag.plan(HARD_GAIN)
    assert 2 in easy.preferred_stages
    assert easy.difficulty == "easy"
    assert 3 in hard.preferred_stages and 2 in hard.discouraged_stages
    # 2026-08-17 ceiling refit: 108 dB now has TWO feasible cascades (3, 4),
    # so it is "medium" (a fallback family exists), no longer "hard"
    assert hard.difficulty in ("medium", "hard")
    assert any("SOFT prior" in r for r in hard.rationale)
    # genuinely hard: beyond every measured ceiling
    extreme = ag.plan({**HARD_GAIN, "gain_target_db": 180.0})
    assert extreme.difficulty == "hard" and extreme.sizing_budget_class == "high"
    assert extreme.preferred_stages == (4,)


def test_planner_prefers_rc_compensation_on_bandwidth_heavy_specs():
    p = ag.plan(HARD_BW)
    assert p.compensation_preferences[0] == "rc"
    assert p.difficulty == "hard"
    assert p.rationale                       # every decision explains itself


# ---------------------------------------------------------------------------
# soft strategy screen: discouraged is a prior, NEVER a ban
# ---------------------------------------------------------------------------
def _cand(fam):
    return {"canonical_family": fam, "canonical_graph_hash": f"h_{fam}"}


def test_screen_drops_discouraged_only_with_enough_preferred():
    st = ag.make_state(HARD_GAIN, ("planner",), spice_cap=32, llm_attempt_cap=20)
    pool = [_cand("2s_miller"), _cand("2s_rc"), _cand("3s_miller"), _cand("3s_rc")]
    kept = ag.apply_strategy_screen(pool, st.plan, st)
    assert {c["canonical_family"] for c in kept} == {"3s_miller", "3s_rc"}
    # the exception path: only ONE preferred candidate -> nothing is dropped
    st2 = ag.make_state(HARD_GAIN, ("planner",), spice_cap=32, llm_attempt_cap=20)
    pool2 = [_cand("2s_miller"), _cand("2s_rc"), _cand("3s_rc")]
    assert ag.apply_strategy_screen(pool2, st2.plan, st2) == pool2


# ---------------------------------------------------------------------------
# Budget ledger: the fairness invariant is ENFORCED, not advisory
# ---------------------------------------------------------------------------
def test_ledger_hard_caps_spice_and_llm():
    st = ag.make_state(EASY, (), spice_cap=10, llm_attempt_cap=5)
    st.ledger.spend_spice(8, "t", "x")
    with pytest.raises(BudgetExceeded):
        st.ledger.spend_spice(3, "t", "over")
    st.ledger.spend_llm(5, "t", "x")
    with pytest.raises(BudgetExceeded):
        st.ledger.spend_llm(1, "t", "over")
    st.ledger.bank(2, "t", "savings")
    assert st.ledger.banked == 2             # banked never raises the cap


# ---------------------------------------------------------------------------
# Topology Critic
# ---------------------------------------------------------------------------
def test_critic_flags_missing_preferred_stage_and_comp():
    plan = ag.plan(HARD_GAIN)
    v = ag.critique([_cand("2s_miller"), _cand("2s_rc")], plan)
    assert not v["satisfied"] and v["n_preferred_stage"] == 0
    v2 = ag.critique([_cand("3s_miller"), _cand("3s_rc")], plan)
    assert v2["satisfied"]


def test_critic_loop_feeds_feedback_and_respects_attempt_cap():
    plan = ag.plan(HARD_GAIN)
    st = ag.make_state(HARD_GAIN, ("planner", "critic"),
                       spice_cap=32, llm_attempt_cap=20)
    calls = []

    def fake_propose(prompt, target_k, max_attempts):
        calls.append({"fb": "CRITIC FEEDBACK" in prompt, "n": max_attempts})
        cands = ([_cand("2s_miller")] if len(calls) == 1
                 else [_cand("3s_miller"), _cand("3s_rc")])
        return {"candidates": cands, "attempts": max_attempts}
    out = ag.run_critic_loop(fake_propose, "P", plan, st.ledger, st)
    assert len(calls) >= 2
    assert calls[0]["fb"] is False and calls[1]["fb"] is True   # re-prompted
    assert st.ledger.llm_attempts_spent <= st.ledger.llm_attempt_cap
    fams = {c["canonical_family"] for c in out["candidates"]}
    assert {"3s_miller", "3s_rc"} <= fams
    assert out["final_verdict"]["satisfied"]


# ---------------------------------------------------------------------------
# Optimization Supervisor
# ---------------------------------------------------------------------------
def _probe_result(dists, capx=1.0):
    return {"results": [{"gain_db": 60, "pm_deg": 60, "ugbw_hz": 1e5,
                         "stable": True, "op_valid": True,
                         "spice_converged": True,
                         "knobs": {"cap_x": capx},
                         # postsizing distance derives from margins; encode
                         # the intended distance via ugbw shortfall is
                         # complex -- tests use allocate() directly instead
                         } for _ in dists]}


def test_supervisor_allocates_to_the_stronger_branch():
    a = {"verdict": "improving", "best_distance": 0.03}
    b = {"verdict": "hopeless", "best_distance": 1.7}
    out = ag.allocate(a, b, remaining=20)
    assert out["A"] == 20 and out["B"] == 0
    # v4.1 (2026-08-19): on a tie the Supervisor COMMITS the remainder to
    # one branch (cross-tier validation: splitting starved both branches on
    # circuits needing 24-31 calls; A0 passes them with a full budget)
    tie = ag.allocate({"verdict": "improving", "best_distance": 0.1},
                      {"verdict": "improving", "best_distance": 0.11}, 20)
    assert tie["A"] + tie["B"] == 20 and max(tie["A"], tie["B"]) == 20   # v4.2: full commit


def test_supervisor_banks_when_both_probe_pass():
    out = ag.allocate({"verdict": "passed", "best_distance": 0.0},
                      {"verdict": "passed", "best_distance": 0.0}, 20)
    assert out["A"] == 0 and out["B"] == 0


def test_supervisor_never_touches_global_safety_bounds():
    src = inspect.getsource(ag.supervise) + inspect.getsource(ag.probe_verdict)
    # actual bound identifiers / mutation calls only ("clamp" appears in
    # comments explaining exactly why the Supervisor must never do this)
    for forbidden in ("KNOB_LO", "KNOB_HI", "RZ_MIN", "RZ_MAX", ".clamp(",
                      "clamp("):
        assert forbidden not in src          # bounds are global, every arm
    assert "RESTART_SEEDS" in inspect.getsource(
        __import__("agentic_raptor.agents.supervisor",
                   fromlist=["supervisor"]))  # predefined recovery schedule


# ---------------------------------------------------------------------------
# Recovery Agent
# ---------------------------------------------------------------------------
def test_recovery_is_bounded_once_and_bank_funded():
    st = ag.make_state(EASY, ("recovery",), spice_cap=32, llm_attempt_cap=20)
    d = ag.recovery_decide(st, st.ledger, 0.1)
    assert d["action"] == "NONE"             # nothing banked yet
    st.ledger.bank(8, "t", "probe savings")
    d2 = ag.recovery_decide(st, st.ledger, 0.1)
    assert d2["action"] == "RESIZE_BACKUP" and d2["budget"] == 8
    log = ag.recovery_execute(st, st.ledger, d2,
                              lambda b: {"distance": 0.0, "pass": True,
                                         "spice_calls": 5})
    assert log["executed"] and st.recovery_already_used
    assert st.ledger.banked == 3             # bank drained by actual spend
    # second attempt in the same run is refused
    assert ag.recovery_decide(st, st.ledger, 0.1)["action"] == "NONE"


def test_recovery_diagnose_names_the_failing_constraint():
    d = ag.recovery_diagnose({"gain_db": 89.9, "pm_deg": 55.5,
                              "ugbw_hz": 2.1e4},
                             {"gain_target_db": 89.45,
                              "phase_margin_target_deg": 45.0,
                              "ugbw_target_hz": 1e6})
    assert d["failing"] == ["ugbw_ratio"]
    assert d["gaps"]["ugbw_ratio"] < 0.05


# ---------------------------------------------------------------------------
# pipeline wiring: default byte-identical; every hook present
# ---------------------------------------------------------------------------
def test_run_pipeline_agents_default_empty_and_hooks_wired():
    import run_raptor_v2 as v2
    sig = inspect.signature(v2.run_pipeline)
    assert sig.parameters["agents"].default == ()
    src = inspect.getsource(v2.run_pipeline)
    for hook in ("make_state", "run_critic_loop", "apply_strategy_screen",
                 "supervise", "recovery_execute"):
        assert hook in src
    # fairness: agentic ledger cap is the baseline envelope
    assert "spice_cap=2 * budget" in src
    assert "llm_attempt_cap=20" in src


def test_agentic_arm_table_covers_all_minus_one_configs():
    import run_agentic_ablation as agb
    assert agb.ARMS["BASELINE"] == ()
    assert set(agb.ARMS["AG_FULL"]) == set(ag.VALID_AGENTS)
    for name, missing in (("AG_NO_PLAN", "planner"), ("AG_NO_CRITIC", "critic"),
                          ("AG_NO_SUPER", "supervisor"),
                          ("AG_NO_RECOV", "recovery")):
        assert missing not in agb.ARMS[name]
        assert len(agb.ARMS[name]) == 3


def test_component_driver_supports_ag_full_arm():
    import run_ablation_v3 as drv
    assert set(drv.AGENTIC_ARMS["AG_FULL"]) == set(ag.VALID_AGENTS)
    s = inspect.getsource(drv.main)
    assert 'kwargs["agents"] = AGENTIC_ARMS[aid]' in s
    assert 'row["ablation_id"] = aid' in s        # never mislabeled as A0


# ---------------------------------------------------------------------------
# QUALITY POLISH + ADAPTIVE ATTEMPTS (2026-08-17)
# ---------------------------------------------------------------------------
def test_sac_size_select_by_fom_keeps_best_passing_design(tmp_path, monkeypatch):
    from agentic_raptor.mb_sac import spec_sizing as ss
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    import json
    from pathlib import Path
    corpus = Path(__file__).resolve().parents[1] / \
        "artifacts/publication_v2/proposer_repair/corpus_diverse.json"
    if not corpus.is_file():
        pytest.skip("corpus not present")
    from run_puct_ablation import _realise
    g = _realise(json.loads(json.loads(corpus.read_text(encoding="utf-8"))
                            ["records"][0]["response"]))
    spec = {"spec_id": "s", "gain_target_db": 60.0,
            "phase_margin_target_deg": 55.0, "load_capacitance_pf": 100.0,
            "ugbw_target_hz": 1e5}
    step = {"n": 0}

    def fake_measure(tid, graph, exe, out_dir, tag, costs, c_load_f=None, **kw):
        step["n"] += 1
        # every step passes; IDD falls each step -> FoM rises each step
        return {"gain_db": 90.0, "pm_deg": 70.0, "ugbw_hz": 1e6,
                "stable": True, "op_valid": True,
                "idd_a": 1e-3 / step["n"], "power_w": 1e-4,
                "spice_converged": True, "c_load_f": 100e-12}
    monkeypatch.setattr(ss, "measure", fake_measure)
    r_rew = ss.sac_size("t", g, spec, None, tmp_path / "a", new_costs(),
                        budget=5, seed=17, persist=False)
    step["n"] = 0
    r_fom = ss.sac_size("t", g, spec, None, tmp_path / "b", new_costs(),
                        budget=5, seed=17, persist=False, select_by="fom")
    # fom mode returns the LAST (lowest-IDD) step; both are passes
    assert r_fom["best"]["idd_a"] <= r_rew["best"]["idd_a"]
    assert r_fom["outcome"]["exact_spec_pass"]


def test_measured_fom_gate_breaks_both_passed_ties():
    from agentic_raptor.ranking import PostSACDesign, SurrogatePrediction, compare

    def design(label, h):
        return PostSACDesign(label=label, spec_id="S1", llm_proposal_id=f"p_{label}",
                             canonical_graph_hash=h, topology_signature="2s_rc",
                             topology_family="2s_rc", sizing_vector={"s1_w": 1.0},
                             sizing_manifest_hash=f"m_{label}", sizing_spice_calls=8,
                             sizing_spice_call_ids=[f"sz_{label}_1"])

    def pred(h):
        return SurrogatePrediction(topology_hash=h, sizing_manifest_hash=f"m_{h}",
                                   gain_db=85.0, pm_deg=65.0,
                                   normalized_margins={"gain": 0.25, "pm": 0.1},
                                   operating_point_probability=None,
                                   stability_probability=0.9,
                                   predictive_uncertainty=0.2,
                                   surrogate_checkpoint_hash="s")
    a, b = design("A", "hA"), design("B", "hB")
    ev = lambda f: {"n_measured": 8, "exact_spec_pass": True,
                    "best_distance": 0.0, "best_pass_fom": f}
    out = compare(a, b, pred("hA"), pred("hB"),
                  {"spec_id": "S1", "gain_target_db": 80.0,
                   "phase_margin_target_deg": 60.0},
                  model=None, ranker_arm="explicit_baseline",
                  measured_a=ev(300.0), measured_b=ev(9000.0),
                  gate_mode="measured_first")
    assert out["selected_design"] == "B"
    assert out["decision_basis"] == "measured_fom_gate"


def test_supervisor_v3_single_run_polish_no_early_stop():
    # v3 (2026-08-18): passed branch -> ONE run without early stop (polish is
    # the run); no probe/final fragmentation; pathology never zeroes a branch
    src = inspect.getsource(ag.supervise)
    assert "QUALITY_POLISH" in src
    assert "early_stop=False" in src
    assert "committed full run" in src   # v4.2: one full-length trajectory
    assert "min([probes[label], final]" not in src     # fragmentation gone
    from agentic_raptor.agents import supervisor as sv
    assert sv.PROBE_BUDGET == 3
    a = ag.allocate({"verdict": "pathological", "best_distance": 0.2},
                    {"verdict": "improving", "best_distance": 0.4}, 26)
    assert a["A"] > 0 and a["B"] > a["A"]              # floor, not zero


def test_adaptive_attempts_wired_opt_in_only():
    import run_raptor_v2 as v2
    import run_ablation_v3 as drv
    assert inspect.signature(v2.run_pipeline).parameters[
        "proposal_stall_stop"].default is None
    assert 'kwargs["proposal_stall_stop"] = None' in inspect.getsource(drv.main)
    from agentic_raptor.llm_dpo.stage3e4 import propose_diverse_excl
    p = inspect.signature(propose_diverse_excl).parameters
    assert p["stall_stop"].default is None and p["max_attempts"].default is None


def test_supervisor_v45_reports_physical_spice_calls():
    """v4.5 keeps the better of baseline/continuation; the discarded run's
    simulations must still be counted (fairness reporting)."""
    import inspect
    src = inspect.getsource(ag.supervise)
    assert "PHYSICAL ACCOUNTING" in src
    assert 'summ["spice_calls"] = phys' in src
    assert "spice_calls_components" in src
