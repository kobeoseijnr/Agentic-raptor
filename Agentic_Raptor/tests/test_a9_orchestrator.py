"""A9 orchestrator (run_a9_generations.py): completion tests for the
self-improvement pipeline. No LLM/SPICE -- state machine, gap-fix guards,
stream gating, lineage semantics."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import run_a9_generations as a9

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# THE GAP FIX: self-improvement never injects value checkpoints into live search
# ---------------------------------------------------------------------------
def test_pipeline_runs_always_use_enforced_promoted_checkpoint():
    src = inspect.getsource(a9.run_generation)
    assert "value_ckpt=None" in src        # -> Stage-8 enforced promoted loader
    # the orchestrator must never thread its own checkpoint into run_pipeline
    assert "value_ckpt=acc" not in src and "value_ckpt=str(" not in src


def test_value_candidates_never_written_to_az_generations():
    src = inspect.getsource(a9)
    assert "write_az_generation_manifest" not in src
    assert "PROMOTED" not in inspect.getsource(a9.value_stream_update).replace(
        "promoted checkpoint (never touched", "")


def test_value_stream_admits_only_graph_complete_rows():
    rows = ([{"replay_schema_version": "az_replay.2", "state_graph": None}]
            + [{"replay_schema_version": "legacy_x"}] * 5)
    r = a9.value_stream_update(rows)
    assert r["admitted_rows"] == 0          # graph missing => not admitted
    assert r["legacy_rows_excluded"] == 6 - 0 - r["admitted_rows"]
    assert r["status"] == "ACCUMULATING"


def test_value_stream_gate_is_the_frozen_value_probe_gate():
    src = inspect.getsource(a9.value_stream_update)
    assert "value_probe_gate" in src and "evaluate_candidate" in src


# ---------------------------------------------------------------------------
# DPO V2 stream
# ---------------------------------------------------------------------------
def test_dpo_stream_gates_against_promoted_v2_and_never_silent_swaps():
    d = a9.dpo_v2_stream_update([{"spec_hash": f"s{i}"} for i in range(80)])
    assert d["status"] == "READY_TO_TRAIN"
    assert d["train_pairs"] + d["dev_pairs"] == 80
    assert "never silently replaced" in d["gate_rule"]
    # spec-disjoint: no dev spec in train
    assert d["dev_specs"] >= 1


# ---------------------------------------------------------------------------
# bandit stream (Stage 9B promotion follow-up)
# ---------------------------------------------------------------------------
def _pair(spec_hash, ha, hb, da, db, pa=False, pb=False):
    return {"spec_hash": spec_hash,
            "design_A": {"canonical_graph_hash": ha},
            "design_B": {"canonical_graph_hash": hb},
            "outcome_A": {"normalized_distance_to_feasibility": da,
                          "exact_spec_pass": pa},
            "outcome_B": {"normalized_distance_to_feasibility": db,
                          "exact_spec_pass": pb}}


def test_bandit_stream_accumulates_below_threshold():
    r = a9.bandit_stream_update([_pair("s1", "h1", "h2", 0.1, 0.2)])
    assert r["status"] == "ACCUMULATING"
    assert r["new_records"] == 2            # one pair -> two outcome records
    assert "never in place" in r["note"]


def test_bandit_stream_ready_is_spec_disjoint_and_gated_on_incumbent():
    pairs = [_pair(f"s{i}", f"h{i}a", f"h{i}b", 0.1, 0.3) for i in range(25)]
    r = a9.bandit_stream_update(pairs)
    assert r["status"] == "READY_TO_TRAIN"
    assert r["train_records"] + r["dev_records"] == 50
    assert r["dev_specs"] >= 1
    from agentic_raptor.topology_rl.bandit_selector import \
        PROMOTED_BANDIT_SHA256
    assert r["incumbent_sha256"] == PROMOTED_BANDIT_SHA256
    assert "STAGED" in r["gate_rule"] and "never silently" in r["gate_rule"]


def test_bandit_stream_z_matches_frozen_outcome_scale():
    r = a9.bandit_stream_update([_pair("s1", "h1", "h2", 0.25, None,
                                       pa=False, pb=True)])
    assert r["new_records"] == 2            # pass side admitted without dist
    src = inspect.getsource(a9.bandit_stream_update)
    assert "1.0 - 2.0 * dist" in src and "max(-1.0" in src


def test_bandit_stream_never_writes_weights_or_pin():
    src = inspect.getsource(a9.bandit_stream_update)
    assert "BANDIT_TOP2_V1.json" not in src      # never touches the artifact
    assert "write" not in src.lower() or "written as a NEW versioned" in src


def test_static_lineage_has_no_learning_in_any_stream():
    src = inspect.getsource(a9.run_generation)
    assert '"value_stream", "dpo_stream", "bandit_stream"' in src
    assert '"rag_stream", "sft_stream"' in src
    assert '{"status": "STATIC_NO_LEARNING"}' in src


# ---------------------------------------------------------------------------
# RAG memory growth stream (binding-component learning)
# ---------------------------------------------------------------------------
def _rag_row(cid, spec="s1", pm=60.0, gain=70.0):
    return {"call_id": cid, "generation_spec_id": spec, "pm": pm,
            "gain_db": gain, "exact_spec_pass": True}


def test_rag_memory_grows_dedupes_and_never_touches_frozen(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text(json.dumps(_rag_row("c0")) + "\n", encoding="utf-8")
    monkeypatch.setattr(a9, "RAG_MEMORY", frozen)
    mem = a9.lineage_rag_memory(tmp_path / "data", "adaptive")
    assert mem != frozen and mem.is_file()          # copied, not shared
    r = a9.rag_stream_update([_rag_row("c1"), _rag_row("c1"),   # dup in-batch
                              _rag_row("c0"),                    # dup vs memory
                              {"pm": 1.0}], mem)                 # no call_id
    assert r["status"] == "APPENDED"
    assert r["admitted"] == 1 and r["rejected_or_duplicate"] == 3
    assert r["memory_rows"] == 2
    # idempotent on resume
    r2 = a9.rag_stream_update([_rag_row("c1")], mem)
    assert r2["status"] == "NO_NEW_ROWS" and r2["memory_rows"] == 2
    # the frozen file is byte-identical
    assert frozen.read_text(encoding="utf-8") == json.dumps(_rag_row("c0")) + "\n"


def test_static_lineage_rag_memory_is_the_frozen_file(tmp_path):
    assert a9.lineage_rag_memory(tmp_path, "static") == a9.RAG_MEMORY


# ---------------------------------------------------------------------------
# SFT refresh stream (essential-component learning, explicit-command only)
# ---------------------------------------------------------------------------
def test_sft_stream_accumulates_then_hands_explicit_command(tmp_path):
    r = a9.sft_stream_update([{"x": 1}] * 5, tmp_path)
    assert r["status"] == "ACCUMULATING"
    r2 = a9.sft_stream_update([{"x": 1}] * a9.MIN_NEW_SFT_ROWS, tmp_path)
    assert r2["status"] == "READY_TO_TRAIN"
    assert "train_sft_self_improvement.py" in r2["run_command"]
    assert "never from a harvest event" in r2["gate_rule"]


# ---------------------------------------------------------------------------
# shared pipeline settings: early stop for BOTH lineages, lineage RAG memory
# ---------------------------------------------------------------------------
def test_generation_runs_use_early_stop_and_lineage_memory_for_both_lineages():
    src = inspect.getsource(a9.run_generation)
    assert "sizing_early_stop=True" in src
    assert "rag_memory=str(rag_mem)" in src
    assert "lineage_rag_memory(data_root, lineage)" in src


def test_generation_runs_take_harvest_custody():
    # CUSTODY FIX (2026-08-16): without harvest=True, run_pipeline returns no
    # payload (G0's first attempt harvested {} on every run) AND appends the
    # rows to global shared pools instead -- silently growing RAG_MEMORY_V2
    # 182->222 while the orchestrator learned nothing.
    src = inspect.getsource(a9.run_generation)
    assert "harvest=True" in src


def test_run_pipeline_external_custody_suppresses_global_pool_writes():
    import inspect as _i

    import run_raptor_v2 as v2
    src = _i.getsource(v2.run_pipeline)
    # harvest=True -> rows go ONLY to the external harvester: no LIVE-pool
    # append, no trusted-pairs persistence
    assert '"reason": "external_harvest_custody"' in src
    assert 'persist=(learning_mode == "adaptive" and not harvest)' in src


def test_stream_thresholds_accumulate_across_generations(tmp_path):
    # gen_000 and gen_001 each hold 1 row; the lineage view must see both
    for g in (0, 1):
        d = tmp_path / "adaptive" / f"gen_{g:03d}" / "streams"
        d.mkdir(parents=True)
        (d / "ranker_pairs.jsonl").write_text(
            json.dumps({"spec_hash": f"s{g}"}) + "\n", encoding="utf-8")
    rows = a9.lineage_stream_rows(tmp_path, "adaptive", "ranker_pairs")
    assert len(rows) == 2
    assert a9.lineage_stream_rows(tmp_path, "adaptive", "missing") == []
    src = inspect.getsource(a9.run_generation)
    assert "lineage_stream_rows" in src         # updates use the lineage view


def test_bandit_refit_never_touches_v1_or_pin():
    src = inspect.getsource(a9._bandit_refit_and_gate)
    assert "BANDIT_TOP2_V1.json" not in src
    assert "A9_CANDIDATES" in src
    assert "HUMAN DECISION REQUIRED" in src


# ---------------------------------------------------------------------------
# GenerationState wiring
# ---------------------------------------------------------------------------
def test_orchestrator_uses_atomic_state_machine_with_idempotent_resume():
    src = inspect.getsource(a9.run_generation)
    for required in ("resume_or_start", "mark_run_processed",
                    "checkpoint_progress", "publish_atomic"):
        assert required in src


def test_static_lineage_never_retrains():
    src = inspect.getsource(a9.run_generation)
    assert 'learning_mode = "adaptive" if lineage == "adaptive" else "static"' in src
    assert '"STATIC_NO_LEARNING"' in src


def test_g0_hashes_identical_for_both_lineages():
    h = a9.g0_component_hashes()
    assert all(v is not None for v in h.values())
    from agentic_raptor.publication.generation_state import REQUIRED_FOR_COMPLETE
    assert set(REQUIRED_FOR_COMPLETE) <= set(h)


def test_dry_run_root_is_isolated_from_real_root():
    assert a9.A9_DRY != a9.A9_REAL
    assert "dryrun" in str(a9.A9_DRY)


def test_generation_state_machine_roundtrip(tmp_path):
    from agentic_raptor.publication.generation_state import (
        GenerationState, checkpoint_progress, publish_atomic, resume_or_start)
    st = resume_or_start(tmp_path, "adaptive")
    assert st.generation_id == 0
    assert st.mark_run_processed("r1") is True
    assert st.mark_run_processed("r1") is False      # idempotent
    checkpoint_progress(st, tmp_path)
    st2 = resume_or_start(tmp_path, "adaptive")      # resumes in-progress
    assert st2.generation_id == 0 and "r1" in st2.processed_run_ids
    with pytest.raises(ValueError):
        publish_atomic(st2, tmp_path)                # incomplete: refuses
    for f in ("rag_snapshot_hash", "sft_checkpoint_hash", "puct_policy_hash",
             "puct_value_hash", "dpo_checkpoint_hash", "evaluation_set_hash"):
        setattr(st2, f, "x")
    st2.status = "COMPLETE"
    publish_atomic(st2, tmp_path)
    st3 = resume_or_start(tmp_path, "adaptive")      # successor starts fresh
    assert st3.generation_id == 1 and st3.parent_generation_id == 0


# ---------------------------------------------------------------------------
# AGENTIC SELF-IMPROVEMENT (2026-08-17): loop runs the production agents
# ---------------------------------------------------------------------------
def test_generation_runs_the_agentic_production_system():
    src = inspect.getsource(a9.run_generation)
    assert "agents=A9_AGENTS" in src
    assert "proposal_stall_stop=A9_STALL_STOP" in src
    assert set(a9.A9_AGENTS) == {"planner", "critic", "supervisor", "recovery"}
    # both lineages get the SAME agents (deterministic rules) -- the call is
    # unconditional, so adaptive-vs-static differences are learned-only
    assert src.count("agents=A9_AGENTS") == 1


def test_agent_episode_extracts_decisions_and_outcome():
    tr = {"agents": {"ledger": {"spice_spent": 20, "spice_cap": 32, "banked": 4},
                     "interventions": [{"what": "QUALITY_POLISH"}]},
          "nominal": {"complete_pass": True},
          "stage9_verification": {"distance_to_feasibility": 0.0},
          "agent_planner": {"difficulty": "hard", "preferred_stages": [3]},
          "agent_supervisor": {"probe": {"A": {"verdict": "passed"},
                                         "B": {"verdict": "stalled"}},
                               "allocation": {"A": 0, "B": 0}},
          "agent_recovery": {"executed": False},
          "agent_critic": {"rounds": 2, "final_verdict": {"satisfied": True}},
          "fom": {"fom_value": 5000.0}, "pvt": {"robust_complete_pass": True},
          "stage1_spec": {"spec": {"gain_target_db": 100.0}, "spec_hash": "h"},
          "stage8_ranker": {"selected_topology_hash": "t"}}
    ep = a9.agent_episode(tr, "r1")
    assert ep["plan_difficulty"] == "hard" and ep["critic_rounds"] == 2
    assert ep["probe_verdicts"] == {"A": "passed", "B": "stalled"}
    assert ep["interventions"] == ["QUALITY_POLISH"]
    assert ep["pass"] and ep["fom"] == 5000.0 and ep["banked"] == 4
    assert a9.agent_episode({"nominal": {}}, "r2") is None   # non-agentic run


def test_agent_stream_audits_but_never_mutates_agent_constants():
    ep = {"plan_difficulty": "easy", "pass": True, "spice_spent": 12,
          "banked": 6, "interventions": ["QUALITY_POLISH"],
          "recovery_executed": False, "critic_rounds": 1}
    r = a9.agent_stream_update([ep] * 3)
    assert r["status"] == "AUDITED" and r["episodes"] == 3
    assert r["by_difficulty"]["easy"]["pass"] == 3
    assert r["quality_polish_runs"] == 3
    src = inspect.getsource(a9.agent_stream_update)
    assert "TRAIN_GAIN_CEILING_DB" not in src      # never rewrites planner code
    assert "never automatic" in r["note"]
    assert a9.agent_stream_update([])["status"] == "NO_EPISODES"


def test_static_lineage_has_no_agent_learning():
    src = inspect.getsource(a9.run_generation)
    assert '"rag_stream", "sft_stream", "agent_stream"' in src


def test_sft_trainer_resolves_lineage_root_and_refuses_empty(tmp_path):
    """2026-08-22: the A9 data root is lineage-scoped. A --si-root at the
    data root must resolve to <root>/<lineage>; a root with no generations
    must be a hard error, never a silent 'nothing eligible'."""
    import inspect, train_sft_self_improvement as tr
    src = inspect.getsource(tr.main)
    assert "si_root / args.lineage" in src
    assert "no gen_* directories under" in src
    # run_command in the harvest summary points at the lineage directory
    import run_a9_generations as a9
    assert "data_root / 'adaptive'" in inspect.getsource(a9)
