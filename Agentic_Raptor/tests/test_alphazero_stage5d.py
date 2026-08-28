"""Stage 5D: fix AlphaZero training-target generation (Dirichlet RNG
reuse), make replay graph-complete, and validate deterministic shuffled
minibatch training. Reuses the same real-candidate fixture pattern as
test_alphazero.py (real graphs from corpus_diverse.json, no LLM calls).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.topology_rl import alphazero as az
from agentic_raptor.topology_rl.stage3e1 import Stage3E1ActionType

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"


def _real_candidates(n=2):
    if not CORPUS.is_file():
        pytest.skip("corpus_diverse.json not present in this checkout")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    seen = {}
    for r in corpus["records"]:
        h = r["canonical_graph_hash"]
        if h not in seen:
            seen[h] = r
        if len(seen) >= n:
            break
    out = []
    for i, (h, r) in enumerate(seen.items()):
        out.append({"llm_proposal_id": f"p{i:02d}", "canonical_graph_hash": h,
                   "canonical_family": r["topology_signature"],
                   "obj": json.loads(r["response"]), "source": "llm"})
    return out


SPEC = {"spec_id": "az_test", "gain_target_db": 60.0,
       "phase_margin_target_deg": 55.0, "load_capacitance_pf": 100.0,
       "ugbw_target_hz": 1e5}


# ---------------------------------------------------------------------------
# Section 1/3: stable_mcts_search_seed
# ---------------------------------------------------------------------------
def test_mcts_search_seed_deterministic_across_reruns():
    a = az.stable_mcts_search_seed(0, "AZ_G1", "spechash", 0, 0)
    b = az.stable_mcts_search_seed(0, "AZ_G1", "spechash", 0, 0)
    assert a == b


def test_mcts_search_seed_differs_by_decision_step():
    a = az.stable_mcts_search_seed(0, "AZ_G1", "spechash", 0, 0)
    b = az.stable_mcts_search_seed(0, "AZ_G1", "spechash", 0, 1)
    c = az.stable_mcts_search_seed(0, "AZ_G1", "spechash", 0, 2)
    assert len({a, b, c}) == 3


def test_mcts_search_seed_differs_by_spec_hash():
    a = az.stable_mcts_search_seed(0, "AZ_G1", "spec_a", 0, 0)
    b = az.stable_mcts_search_seed(0, "AZ_G1", "spec_b", 0, 0)
    assert a != b


def test_mcts_search_seed_differs_by_rollout_seed():
    a = az.stable_mcts_search_seed(0, "AZ_G1", "spechash", 0, 0)
    b = az.stable_mcts_search_seed(0, "AZ_G1", "spechash", 1, 0)
    assert a != b


def test_episode_receives_distinct_mcts_search_seed_per_step():
    """The actual bug Stage 5C found: every step of a real episode
    previously shared the SAME cfg.seed. run_alphazero_episode's steps
    must now each carry a DIFFERENT mcts_search_seed."""
    cands = _real_candidates(2)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=16,
                             alphazero_max_edit_depth=3, seed=0,
                             training_mode=True, dirichlet_epsilon=0.25,
                             dirichlet_alpha=0.3)
    ep = az.run_alphazero_episode(cands, SPEC, "mcts_seed_test", "hash1",
                                  seed=0, campaign_seed=0, config=cfg,
                                  max_episode_depth=3, deterministic=False,
                                  generation_id="AZ_G1")
    seeds = ep["mcts_search_seeds"]
    assert len(seeds) == len(ep["steps"])
    if len(seeds) > 1:
        assert len(set(seeds)) == len(seeds)
    assert all(s["mcts_search_seed"] == seeds[i] for i, s in enumerate(ep["steps"]))


# ---------------------------------------------------------------------------
# Section 3: training-noise reproducibility, validation-noise no-op
# ---------------------------------------------------------------------------
def test_training_dirichlet_noise_reproducible_given_same_inputs():
    """Same (campaign, generation, spec, rollout) -> identical root prior
    (Dirichlet-noised) across two independent runs."""
    cands = _real_candidates(2)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=8,
                             alphazero_max_edit_depth=1, seed=0,
                             training_mode=True, dirichlet_epsilon=0.25,
                             dirichlet_alpha=0.3)
    a = az.run_alphazero_episode(cands, SPEC, "repro_test", "hashA", seed=0,
                                 campaign_seed=1, config=cfg, max_episode_depth=1,
                                 deterministic=True, generation_id="AZ_G1")
    b = az.run_alphazero_episode(cands, SPEC, "repro_test", "hashA", seed=0,
                                 campaign_seed=1, config=cfg, max_episode_depth=1,
                                 deterministic=True, generation_id="AZ_G1")
    assert a["mcts_search_seeds"] == b["mcts_search_seeds"]
    assert a["steps"][0]["pi"] == b["steps"][0]["pi"]


def test_validation_search_unaffected_by_per_step_seed_variation():
    """training_mode=False must make the per-step seed a no-op: two
    configs differing ONLY in cfg.seed must produce IDENTICAL search
    results when training_mode=False (Dirichlet noise branch never
    executes), proving the fix cannot introduce validation stochasticity."""
    cands = _real_candidates(2)
    reg = az.LLMSeededEditRegistry(cands)
    root_state = az.build_root_state(SPEC, "val_test", reg.seed_ids)
    nets = az.load_alphazero_nets(seed=0)

    def _priors(seed):
        cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=8,
                                 alphazero_max_edit_depth=1, seed=seed,
                                 training_mode=False)
        root, _mcts = az._search_from_state(root_state, reg, reg.seed_ids, nets, cfg)
        return {c.action.action_id: c.prior for c in root.children}

    assert _priors(111) == _priors(222)


# ---------------------------------------------------------------------------
# Section 4/5: graph serialization / deserialization / round-trip
# ---------------------------------------------------------------------------
def test_serialize_deserialize_device_graph_round_trip():
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    seed_id = reg.seed_ids[0]
    dg = reg.get_device_graph(seed_id)
    d = az.serialize_device_graph(dg)
    # JSON round-trip too, not just python-object round-trip
    d2 = json.loads(json.dumps(d))
    reconstructed = az.deserialize_device_graph(d2)
    assert reconstructed.topology_id == dg.topology_id
    assert reconstructed.stage_count == dg.stage_count
    assert len(reconstructed.devices) == len(dg.devices)
    assert reconstructed.ports == dg.ports


def test_graph_round_trip_hash_matches_for_seed_and_edited_graph():
    cands = _real_candidates(1)
    reg = az.LLMSeededEditRegistry(cands)
    seed_id = reg.seed_ids[0]
    ok, orig, recon = az.graph_round_trip_hash_matches(
        reg.get_device_graph(seed_id), seed_id)
    assert ok
    assert orig == recon

    edited_tid = reg.derive_edited(seed_id, "ADD_VERIFIED_STAGE")
    ok2, orig2, recon2 = az.graph_round_trip_hash_matches(
        reg.get_device_graph(edited_tid), edited_tid)
    assert ok2
    assert orig2 == recon2
    assert orig2 != orig   # a real, distinct edited graph


# ---------------------------------------------------------------------------
# Section 4/6: build_replay_rows is now graph-complete
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_episode():
    cands = _real_candidates(2)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=16,
                             alphazero_max_edit_depth=2, seed=0)
    return az.run_alphazero_episode(cands, SPEC, "replay_test", "replay_test_hash",
                                    seed=0, config=cfg, max_episode_depth=2)


FAKE_TERMINAL = {"electrical_environment_version": "POST_CLOAD_FIX_V1",
                 "requested_c_load_f": 1e-10, "simulated_c_load_f": 1e-10,
                 "call_id": "fake:call:1", "z": 0.42}


def test_build_replay_rows_are_graph_complete(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    assert rows
    for r in rows:
        assert r["replay_schema_version"] == az.AZ_REPLAY_SCHEMA_GRAPH_COMPLETE
        assert r["state_graph"] is not None
        assert "state_topology_hash" in r
        assert r["state_topology_hash"] == r["state_graph_hash"]
        assert "mcts_search_seed" in r


def test_build_replay_rows_state_graph_round_trips_to_matching_hash(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    for r in rows:
        reconstructed = az.deserialize_device_graph(r["state_graph"])
        from agentic_raptor.topology_rl.value_refresh import \
            device_graph_to_circuit_graph
        recon_hash = device_graph_to_circuit_graph(
            reconstructed, r["state_topology_id"]).structural_hash()
        assert recon_hash == r["state_graph_hash"]


def test_build_replay_rows_falls_back_to_legacy_schema_without_registry(small_episode):
    """No registry -> no state_graph -> legacy schema, not a silent
    graph-complete claim."""
    ep_no_registry = {**small_episode, "registry": None}
    rows = az.build_replay_rows(ep_no_registry, FAKE_TERMINAL)
    assert rows
    for r in rows:
        assert r["replay_schema_version"] == az.AZ_REPLAY_SCHEMA_LEGACY
        assert r["state_graph"] is None


# ---------------------------------------------------------------------------
# Section 8/9/22: deterministic shuffled minibatch training
# ---------------------------------------------------------------------------
def test_minibatch_trainer_rejects_legacy_rows():
    legacy_row = {"replay_schema_version": az.AZ_REPLAY_SCHEMA_LEGACY,
                 "state_graph": None, "spec_hash": "h", "step": 0, "seed": 0}
    with pytest.raises(az.LegacyReplayRejected):
        az.train_az_generation_minibatch([legacy_row])


def test_minibatch_trainer_rejects_rows_missing_state_graph():
    row = {"replay_schema_version": az.AZ_REPLAY_SCHEMA_GRAPH_COMPLETE,
          "state_graph": None, "spec_hash": "h", "step": 0, "seed": 0}
    with pytest.raises(az.LegacyReplayRejected):
        az.train_az_generation_minibatch([row])


def test_minibatch_trainer_accepts_graph_complete_rows_and_trains(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    rep = az.train_az_generation_minibatch(rows, epochs=1, batch_size=16, shuffle_seed=0)
    assert rep["batch_reports"], "no batches were trained"
    for b in rep["batch_reports"]:
        assert b["policy_loss"] >= 0
        assert b["value_loss"] >= 0
        assert b["batch_size"] <= 16


def test_minibatch_trainer_one_optimizer_step_per_batch_not_per_row(small_episode):
    """With batch_size >= n_rows and epochs=1, exactly ONE batch (hence
    ONE optimizer step) must be recorded, regardless of how many replay
    rows exist -- the bug being fixed had len(rows) individual steps."""
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    rep = az.train_az_generation_minibatch(rows, epochs=1, batch_size=len(rows) + 100,
                                           shuffle_seed=0)
    assert len(rep["batch_reports"]) == 1
    assert rep["batch_reports"][0]["examples_used"] == len(rows)


def test_minibatch_trainer_deterministic_shuffle_given_same_seed(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    r1 = az.train_az_generation_minibatch(rows, epochs=2, batch_size=1, shuffle_seed=42)
    r2 = az.train_az_generation_minibatch(rows, epochs=2, batch_size=1, shuffle_seed=42)
    losses1 = [b["policy_loss"] for b in r1["batch_reports"]]
    losses2 = [b["policy_loss"] for b in r2["batch_reports"]]
    assert losses1 == pytest.approx(losses2, abs=1e-6)


def test_minibatch_trainer_different_shuffle_seed_can_change_order(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    if len(rows) < 2:
        pytest.skip("episode too short to demonstrate order sensitivity")
    r1 = az.train_az_generation_minibatch(rows, epochs=3, batch_size=1, shuffle_seed=1)
    r2 = az.train_az_generation_minibatch(rows, epochs=3, batch_size=1, shuffle_seed=2)
    losses1 = [b["policy_loss"] for b in r1["batch_reports"]]
    losses2 = [b["policy_loss"] for b in r2["batch_reports"]]
    assert losses1 != losses2


def test_minibatch_trainer_only_legal_actions_enter_policy_loss(small_episode):
    """Each row's own legal_action_ids gate exactly which actions the
    policy loss is computed over -- verified by checking every row's
    stored pi keys are a subset of its legal_action_ids (the invariant
    the trainer relies on, already enforced by generate_alphazero_actions/
    visit_policy, exercised here through the real training path)."""
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    for r in rows:
        assert set(r["pi"]) <= set(r["legal_action_ids"])
    # must not raise -- proves the trainer's action reconstruction from
    # legal_action_ids alone is sufficient and consistent with pi's keys
    az.train_az_generation_minibatch(rows, epochs=1, batch_size=16, shuffle_seed=0)


def test_minibatch_trainer_episode_balanced_weighting_changes_loss_scale():
    """A synthetic 2-episode replay where one episode has many more states
    than the other -- episode_balanced=True must change the recorded
    (weighted) policy_loss relative to episode_balanced=False, proving
    the weighting is actually applied, not a no-op flag."""
    cands = _real_candidates(2)
    cfg = az.AlphaZeroConfig(alphazero_simulations_per_move=8,
                             alphazero_max_edit_depth=3, seed=0)
    ep_short = az.run_alphazero_episode(cands, SPEC, "short", "short_hash", seed=0,
                                        config=cfg, max_episode_depth=1)
    ep_long = az.run_alphazero_episode(cands, SPEC, "long", "long_hash", seed=1,
                                       config=cfg, max_episode_depth=3)
    rows = (az.build_replay_rows(ep_short, FAKE_TERMINAL)
           + az.build_replay_rows(ep_long, FAKE_TERMINAL))
    if len({(r["spec_hash"], len([x for x in rows if x["spec_hash"] == r["spec_hash"]]))
           for r in rows}) < 2:
        pytest.skip("episodes ended up the same length -- weighting has nothing to prove")
    balanced = az.train_az_generation_minibatch(rows, epochs=1, batch_size=len(rows),
                                                shuffle_seed=0, episode_balanced=True)
    unbalanced = az.train_az_generation_minibatch(rows, epochs=1, batch_size=len(rows),
                                                  shuffle_seed=0, episode_balanced=False)
    assert balanced["batch_reports"][0]["policy_loss"] != pytest.approx(
        unbalanced["batch_reports"][0]["policy_loss"], abs=1e-9)


def test_minibatch_trainer_gradient_clipping_retained(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    rep = az.train_az_generation_minibatch(rows, epochs=1, batch_size=16,
                                           shuffle_seed=0, clip=5.0)
    for b in rep["batch_reports"]:
        assert b["grad_norm"] <= 5.0 + 1e-4


def test_minibatch_trainer_config_recorded(small_episode):
    rows = az.build_replay_rows(small_episode, FAKE_TERMINAL)
    rep = az.train_az_generation_minibatch(
        rows, epochs=2, batch_size=4, lr=3e-4, shuffle_seed=7, episode_balanced=True)
    cfg = rep["config"]
    assert cfg["batch_size"] == 4
    assert cfg["epochs"] == 2
    assert cfg["lr"] == 3e-4
    assert cfg["shuffle_seed"] == 7
    assert cfg["episode_balanced"] is True
