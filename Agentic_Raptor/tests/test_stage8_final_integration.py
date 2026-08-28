"""Stage 8 FINAL integration: checkpoint-loading enforcement tests + FULL
live-configuration guards."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "artifacts/publication_v3/stage8_final_smoke/STAGE8_FINAL_SMOKE.json"


# ---------------------------------------------------------------------------
# enforced checkpoint loading
# ---------------------------------------------------------------------------
def test_promoted_loader_verifies_sha_and_differs_from_random():
    from agentic_raptor.topology_rl.alphazero import load_promoted_alphazero_nets
    _nets, prov = load_promoted_alphazero_nets()
    assert prov["checkpoint_loaded"] is True
    assert prov["checkpoint_sha256"] == ("822a305c81e856f8d5d56d29e62fedaeb6"
                                         "a57614afdde4bba1ad3346fd6a5125")
    assert prov["differs_from_random_init"] is True
    assert prov["parameter_fingerprint"]["encoder"] != 0


def test_promoted_loader_hard_fails_on_tampered_checkpoint(monkeypatch, tmp_path):
    from agentic_raptor.topology_rl import alphazero as az
    fake = tmp_path / "policy_value.pt"
    fake.write_bytes(b"tampered")
    # fake generation dir shape: <root>/<gen>/checkpoint/policy_value.pt
    gen = tmp_path / "AZ_FAKE" / "checkpoint"
    gen.mkdir(parents=True)
    fake2 = gen / "policy_value.pt"
    fake2.write_bytes(b"tampered")
    monkeypatch.setattr(az, "require_promoted_az_checkpoint", lambda: fake2)
    monkeypatch.setattr(az, "read_az_generation_manifest",
                        lambda gid: {"checkpoint_hash": "deadbeef" * 8})
    with pytest.raises(az.AlphaZeroCheckpointError):
        az.load_promoted_alphazero_nets()


def test_promoted_loader_hard_fails_when_load_has_no_effect(monkeypatch):
    from agentic_raptor.topology_rl import alphazero as az
    real_load = az.load_alphazero_nets
    # simulate a load that silently does nothing (returns fresh random)
    monkeypatch.setattr(az, "load_alphazero_nets",
                        lambda value_ckpt=None, seed=0: real_load(None, seed=0))
    with pytest.raises(az.AlphaZeroCheckpointError):
        az.load_promoted_alphazero_nets()


def test_run_pipeline_live_branch_uses_enforced_loader_no_silent_random():
    import inspect

    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    assert "load_promoted_alphazero_nets" in src
    assert "AlphaZeroCheckpointError" in src
    # the old silent path must be gone: NO alphazero_select_two call site
    # anywhere in run_pipeline is handed value_ckpt directly anymore --
    # every call goes through nets built by the enforced loader. (Checked
    # at every call site, not a fixed char window: the dispatch legitimately
    # grows new opt-in branches.)
    parts = src.split("alphazero_select_two(")[1:]
    assert parts, "live branch must still call alphazero_select_two"
    for call_args in parts:
        assert "value_ckpt=value_ckpt" not in call_args[:250]
        assert "nets=" in call_args[:250]


def test_run_pipeline_explicit_ckpt_must_exist():
    import inspect

    import run_raptor_v2 as v2
    src = inspect.getsource(v2.run_pipeline)
    assert "explicit value_ckpt does not exist" in src


# ---------------------------------------------------------------------------
# experimental modes not live
# ---------------------------------------------------------------------------
def test_experimental_alphazero_modes_are_opt_in_only():
    import inspect

    import run_raptor_v2 as v2
    sig = inspect.signature(v2.run_pipeline)
    assert sig.parameters["search"].default == "bandit_top2"  # 2026-08-15 selector promotion
    assert sig.parameters["ranker_mode"].default == "dpo"


def test_preflight_has_critical_checkpoint_loading_check():
    from agentic_raptor.publication.preflight import (
        ALL_CHECKS, check_alphazero_checkpoint_loading_enforced)
    assert check_alphazero_checkpoint_loading_enforced in ALL_CHECKS
    r = check_alphazero_checkpoint_loading_enforced()
    assert r["critical"] is True
    assert r["ok"] is True, r["detail"]


# ---------------------------------------------------------------------------
# smoke-derived checks (skip until the smoke has run)
# ---------------------------------------------------------------------------
def test_final_smoke_passed_all_checks():
    if not SMOKE.is_file():
        pytest.skip("final smoke not run in this checkout")
    d = json.loads(SMOKE.read_text(encoding="utf-8"))
    assert d["smoke_ok"] is True, d["checks"]
    assert d["az_checkpoint_loaded"] is True
    assert d["az_checkpoint_sha256"].startswith("822a305c")
    assert d["az_search_mode"] == "true_alphazero_multi_depth_edit_search"
    assert d["topology_identity_ok"] is True
    assert d["dpo_selector"] == "learned_dpo"
