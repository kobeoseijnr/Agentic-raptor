"""Stage 3D.1 tests."""
import json
from pathlib import Path
from agentic_raptor.mb_sac.stage3d1 import PostSizingTopologyScore, parse_params, write_params

_ROOT = Path(__file__).resolve().parents[1]
SUM = json.loads((_ROOT / "artifacts/stage3d1/SUMMARY.json").read_text())

def test_a1_manifests_and_roundtrip():
    assert len(SUM["a1_manifests"]) == 9
    assert SUM["a1_roundtrip_all_identical"] is True
    for t, r in SUM["a1_manifests"].items():
        assert r["params"] > 5 and (_ROOT / f"artifacts/stage3d1/design_variables/a1/{t}.json").is_file()

def test_param_writer_deterministic():
    p = {"MOSFET_1_1_W_gm1_NMOS": "2", "MOSFET_1_1_L_gm1_NMOS": "1", "X": "3"}
    assert parse_params(write_params(p)) == p
    assert write_params(p) == write_params(dict(p))

def test_all_16_environments_pass():
    assert SUM["env_pass"] == 16
    assert all(r["spice_success"] for r in SUM["env_validation"].values())

def test_phases_ran_all_families():
    assert SUM["phase_results"]["B_a1"]["families"] == 9
    assert SUM["phase_results"]["C_a2"]["families"] == 7

def test_exact_spice_accounting():
    b = SUM["budget"]
    # env 16 + phase (9+7) + scores 3x2 = 38 exactly
    assert b["real_spice_calls"] == 38 and b["failed_calls"] == 0
    assert b["model_transitions"] == 0  # none counted as SPICE

def test_score_schema_and_scalar_components():
    s = PostSizingTopologyScore("t", "x", "unstable", False, False, {}, {"pm": -0.5},
                                3, None, 0)
    v = s.compute_scalar()
    assert s.components["stable"] == 0.0 and v < 0.5
    assert set(s.components) == {"valid", "stable", "feasible", "margin", "cost"}

def test_score_repeatability_zero_spread():
    for t, r in SUM["score_repeatability"].items():
        assert r["spread"] == 0.0
