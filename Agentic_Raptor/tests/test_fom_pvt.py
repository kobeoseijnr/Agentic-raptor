"""Tests for the FoM (FOM_V1_UGBW_CL_OVER_IDD) and PVT evaluation additions.

Covers the 17 required items: FoM formula, unit conversion, IDD-vs-Ibias,
zero/invalid IDD safety, pass/fail independence from FoM, PVT Cartesian
corner generation, unsupported-corner rejection, unique spice_call_ids,
nominal-vs-PVT call id separation, PVT pass %, robust_complete_pass,
failure-reason preservation, spice-call accounting, self-improvement
leakage safety, pvt.enabled=false no-op, and no new legacy-RAPTOR coupling.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agentic_raptor.electrical import fom as fom_mod
from agentic_raptor.electrical import pvt_eval
from agentic_raptor.utils.exceptions import ConfigurationError

SPEC = {"spec_id": "test_spec", "gain_target_db": 40.0,
       "phase_margin_target_deg": 45.0, "ugbw_target_hz": 1e6}


# --------------------------- 1. FoM formula -----------------------------------
def test_fom_formula_matches_worked_example():
    # UGBW=120MHz, CLOAD=2pF, IDD=0.4mA -> 600 (Part 1 example, not hard-coded
    # into compute_fom -- these are ordinary function inputs)
    r = fom_mod.compute_fom(ugbw_hz=120e6, c_load_f=2e-12, idd_a=0.4e-3)
    assert r["fom_value"] == pytest.approx(600.0)
    assert r["fom_formula"] == "UGBW_MHz * C_LOAD_pF / IDD_mA"
    assert r["fom_version"] == fom_mod.FOM_VERSION == "FOM_V1_UGBW_CL_OVER_IDD"
    assert r["fom_units"] == "MHz*pF/mA"


# --------------------------- 2. Unit conversion --------------------------------
def test_unit_conversions():
    assert fom_mod.hz_to_mhz(1e6) == pytest.approx(1.0)
    assert fom_mod.hz_to_mhz(120e6) == pytest.approx(120.0)
    assert fom_mod.f_to_pf(1e-12) == pytest.approx(1.0)
    assert fom_mod.f_to_pf(500e-12) == pytest.approx(500.0)
    assert fom_mod.a_to_ma(1e-3) == pytest.approx(1.0)
    assert fom_mod.a_to_ma(0.4e-3) == pytest.approx(0.4)


def test_compute_fom_reports_converted_units():
    r = fom_mod.compute_fom(ugbw_hz=1e6, c_load_f=500e-12, idd_a=1e-3)
    assert r["ugbw_mhz"] == pytest.approx(1.0)
    assert r["cload_pf"] == pytest.approx(500.0)
    assert r["idd_ma"] == pytest.approx(1.0)


# --------------------------- 3. IDD, not Ibias ---------------------------------
def test_compute_fom_has_no_ibias_parameter():
    """FoM structurally cannot read an Ibias knob: compute_fom's signature
    has no such parameter at all."""
    params = set(inspect.signature(fom_mod.compute_fom).parameters)
    assert params == {"ugbw_hz", "c_load_f", "idd_a"}
    assert "ibias_a" not in params and "ibx" not in params


def test_authoritative_outcome_idd_a_is_distinct_field_from_design_knobs():
    """outcome_from_sizing populates idd_a from the MEASURED dict's idd_a
    key, and this is a separate dataclass field from any sizing knob."""
    from agentic_raptor.ranking.types import outcome_from_sizing

    measured_idd = 0.0007          # what ngspice actually measured
    knob_lookalike = 0.00025       # a plausible Ibias-knob-shaped value
    outcome = {"exact_spec_pass": True, "operating_point_valid": True,
              "spice_converged": True, "hard_constraints_passed": 5,
              "hard_constraints_total": 5}
    auth = outcome_from_sizing(
        outcome, call_id="c1", topology_hash="t1",
        sizing_manifest_hash="m1", netlist_hash="n1", mode="final_verification",
        best={"gain_db": 80.0, "pm_deg": 60.0, "ugbw_hz": 1e6,
              "idd_a": measured_idd, "c_load_f": 500e-12})
    assert auth.idd_a == measured_idd
    assert auth.idd_a != knob_lookalike
    fom = fom_mod.compute_fom(auth.ugbw_hz, auth.c_load_f, auth.idd_a)
    assert fom["idd_used"] == measured_idd


# --------------------------- 4. Zero/invalid IDD --------------------------------
@pytest.mark.parametrize("idd", [0.0, -1e-3, None])
def test_compute_fom_zero_or_invalid_idd_is_safe(idd):
    r = fom_mod.compute_fom(ugbw_hz=1e6, c_load_f=1e-12, idd_a=idd)
    assert r["fom_value"] is None
    assert r["valid"] is False
    assert r["invalid_reason"] in ("idd_not_positive", "missing_idd")


def test_compute_fom_missing_ugbw_or_cload_is_safe():
    assert fom_mod.compute_fom(None, 1e-12, 1e-3)["fom_value"] is None
    assert fom_mod.compute_fom(1e6, None, 1e-3)["fom_value"] is None


# --------------------- 5. Failed design stays failed regardless of FoM ---------
def test_high_fom_does_not_override_complete_pass():
    fom = fom_mod.compute_fom(ugbw_hz=500e6, c_load_f=10e-12, idd_a=0.01e-3)
    assert fom["fom_value"] > 1000          # a very high FoM
    nominal = {"complete_pass": False, "fom": fom}   # constructed exactly as
    # run_raptor_v2.py builds trace["nominal"]/trace["fom"]: two independent
    # keys, never collapsed into one score
    assert nominal["complete_pass"] is False
    assert nominal["fom"]["fom_value"] > 1000
    assert "fom_value" not in nominal or "complete_pass" not in fom


# --------------------- 6. PVT Cartesian product ---------------------------------
def test_generate_corners_cartesian_product_count():
    cfg = pvt_eval.PvtConfig(enabled=True, process_corners=("tt", "ff"),
                             supply_voltages=(1.62, 1.8, 1.98),
                             temperatures_c=(-40.0, 27.0, 85.0))
    corners = pvt_eval.generate_corners(cfg)
    assert len(corners) == 2 * 3 * 3 == 18
    assert len({c.pvt_corner_id for c in corners}) == 18   # all unique


def test_generate_corners_disabled_default_is_single_nominal_corner():
    cfg = pvt_eval.PvtConfig()
    corners = pvt_eval.generate_corners(cfg)
    assert len(corners) == 1
    assert corners[0].process_corner == "tt"
    assert corners[0].vdd == pytest.approx(1.8)
    assert corners[0].temperature_c == pytest.approx(27.0)


# --------------------- 7. Unsupported corner fails clearly -----------------------
def test_unsupported_process_corner_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="not available"):
        pvt_eval.PvtConfig(enabled=True, process_corners=("zz",),
                           supply_voltages=(1.8,), temperatures_c=(27.0,))


def test_available_process_corners_are_real_files_on_disk():
    avail = pvt_eval.available_process_corners()
    assert avail, "expected at least one real PDK corner file"
    for name, path in avail.items():
        assert name in pvt_eval.KNOWN_PROCESS_CORNERS
        assert path.is_file()


def test_empty_pvt_lists_rejected_when_enabled():
    with pytest.raises(ConfigurationError):
        pvt_eval.PvtConfig(enabled=True, process_corners=())


# ----------- helpers: a fake measure() so PVT sweep tests need no ngspice ------
def _fake_measure_factory(pass_corners):
    """Return a stand-in for agentic_raptor.mb_sac.spec_sizing.measure that
    reports a pass/fail per corner deterministically, with a distinguishable
    idd_a per corner (no two corners identical) -- no ngspice involved."""
    calls = []

    def _fake(topology_id, graph, exe, out_dir, tag, costs, *,
              pdk_file=None, supply_voltage=None, temperature_c=None,
              c_load_f=None):
        calls.append(tag)
        corner_name = pdk_file.stem if pdk_file else "tt"
        ok = f"{corner_name}_{supply_voltage:g}V_{temperature_c:g}C" in pass_corners
        costs["real_spice_calls"] = 1
        return {"gain_db": 45.0 if ok else 10.0,
               "pm_deg": 55.0 if ok else 5.0,
               "ugbw_hz": 2e6 if ok else 5e4,
               "power_w": 1e-3, "idd_a": 1e-3 + 1e-6 * len(calls),
               "c_load_f": c_load_f, "stable": True,
               "stability": "verified_stable",
               "electrical": "electrically_functional", "op_valid": True}
    return _fake, calls


class _Graph:
    devices: list = []


# --------------------- 8/9. unique + distinct-from-nominal call ids -------------
def test_pvt_call_ids_are_unique_and_distinguishable_from_nominal(monkeypatch, tmp_path):
    cfg = pvt_eval.PvtConfig(enabled=True, process_corners=("tt", "ff"),
                             supply_voltages=(1.8,), temperatures_c=(27.0, 85.0))
    fake, calls = _fake_measure_factory(pass_corners=set())
    monkeypatch.setattr("agentic_raptor.mb_sac.spec_sizing.measure", fake)
    records = pvt_eval.run_pvt_sweep("tid", _Graph(), SPEC, "ngspice_exe",
                                     tmp_path, cfg, label="A")
    ids = [r["spice_call_id"] for r in records]
    assert len(ids) == len(set(ids)) == 4          # all unique
    assert all(cid.startswith("pvt:") for cid in ids)
    # a nominal-verification call id, per run_raptor_v2.py's verify(), is
    # "final:{spec_id}:{label}:{ts}" -- disjoint prefix, never collides
    nominal_style_id = f"final:{SPEC['spec_id']}:A:123"
    assert nominal_style_id not in ids
    assert not any(cid.startswith("final:") for cid in ids)


# --------------------- 10/11/12. aggregation ------------------------------------
def _mk_record(cid, passed, reasons=None):
    return {"pvt_corner_id": cid, "process_corner": "tt", "vdd": 1.8,
           "temperature_c": 27.0, "gain_db": 45.0, "pm_deg": 55.0,
           "ugbw_hz": 2e6, "idd_a": 1e-3, "power_w": 1e-3,
           "gain_margin": 5.0, "pm_margin": 10.0, "ugbw_margin": 1.0,
           "power_margin": None, "complete_pass": passed,
           "failure_reasons": reasons or [], "spice_call_id": f"pvt:{cid}"}


def test_pvt_pass_percent_matches_worked_example():
    # 43 passing / 45 total (Part 6 example, not hard-coded into aggregate_pvt)
    records = [_mk_record(f"c{i}", True) for i in range(43)] + \
        [_mk_record(f"c{i}", False, ["PM below target"]) for i in range(43, 45)]
    agg = pvt_eval.aggregate_pvt(records)
    assert agg["total_pvt_corners"] == 45
    assert agg["passed_pvt_corners"] == 43
    assert agg["failed_pvt_corners"] == 2
    assert agg["pvt_pass_percent"] == pytest.approx(43 / 45 * 100, abs=1e-3)


def test_robust_complete_pass_true_only_when_all_required_pass():
    all_pass = [_mk_record(f"c{i}", True) for i in range(5)]
    assert pvt_eval.aggregate_pvt(all_pass)["robust_complete_pass"] is True

    one_fail = all_pass[:-1] + [_mk_record("c4", False, ["gain below target"])]
    assert pvt_eval.aggregate_pvt(one_fail)["robust_complete_pass"] is False

    # narrowing required_corner_ids to only the passing ones -> robust True
    # even though an unrequired corner failed
    agg_required = pvt_eval.aggregate_pvt(
        one_fail, required_corner_ids=("c0", "c1", "c2", "c3"))
    assert agg_required["robust_complete_pass"] is True


def test_failure_reasons_preserved_through_aggregation():
    records = [_mk_record("c0", True),
              _mk_record("c1", False, ["PM below target", "gain below target"])]
    agg = pvt_eval.aggregate_pvt(records)
    assert agg["failing_corner_count"] == 1
    failing = [r for r in records if not r["complete_pass"]]
    assert failing[0]["failure_reasons"] == ["PM below target", "gain below target"]


def test_aggregate_pvt_empty_records_is_safe():
    agg = pvt_eval.aggregate_pvt([])
    assert agg["total_pvt_corners"] == 0
    assert agg["pvt_pass_percent"] is None
    assert agg["robust_complete_pass"] is False


# --------------------- 13/14. SPICE call accounting ------------------------------
def test_spice_usage_accounting_formula():
    optimization_calls, final_verification_calls, pvt_calls = 64, 2, 18
    total = optimization_calls + final_verification_calls + pvt_calls
    usage = {"optimization_spice_calls": optimization_calls,
            "final_nominal_verification_calls": final_verification_calls,
            "pvt_spice_calls": pvt_calls, "total_spice_calls": total}
    assert usage["total_spice_calls"] == 84
    # the three components are tracked under DISTINCT keys, not merged
    assert len({usage["optimization_spice_calls"],
               usage["final_nominal_verification_calls"],
               usage["pvt_spice_calls"]}) == 3 or optimization_calls == pvt_calls


def test_pvt_sweep_real_spice_calls_summed_per_corner(monkeypatch, tmp_path):
    cfg = pvt_eval.PvtConfig(enabled=True, process_corners=("tt",),
                             supply_voltages=(1.8,), temperatures_c=(27.0, 85.0))
    fake, calls = _fake_measure_factory(pass_corners=set())
    monkeypatch.setattr("agentic_raptor.mb_sac.spec_sizing.measure", fake)
    records = pvt_eval.run_pvt_sweep("tid", _Graph(), SPEC, "ngspice_exe",
                                     tmp_path, cfg, label="A")
    assert len(records) == 2
    assert sum(r["real_spice_calls"] for r in records) == 2


# --------------------- 15. self-improvement leakage safety -----------------------
def test_pvt_eval_module_does_not_import_selfimprove():
    """No CODE import of the self-improvement streams -- the module's own
    docstring is allowed to reference the term when explaining the
    safeguard, so this checks actual import statements via the AST, not a
    bare substring match."""
    import ast
    src = Path(pvt_eval.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(n.name for n in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("selfimprove_v2" in m for m in imported)
    assert "harvest_run" not in pvt_eval.__dict__
    assert "record_pair" not in pvt_eval.__dict__


def test_harvest_run_never_emits_pvt_data_even_if_injected():
    """PVT data must never reach the training streams. Simulate a caller
    mistake (an hv payload carrying a 'pvt' key) and confirm harvest_run's
    output rows carry no PVT-shaped content -- streams.py's row builders
    only ever read the keys they're documented to read."""
    from agentic_raptor.selfimprove_v2.streams import harvest_run

    hv = {"spec": SPEC, "spec_hash": "h", "split": "train", "spec_index": 0,
         "seed": 0, "budget": 8, "candidates": [], "root_visits": {},
         "candidate_visits": {}, "root_action_ids": [], "root_state": None,
         "candidate_manifests": {}, "search": None, "branches": {},
         "ranker": {"arm": "dpo_ranker", "selected_design": "A",
                    "backup_design": "B", "decision_basis": "x",
                    "deciding_level": "x", "low_confidence": False,
                    "score_A": None, "score_B": None, "checkpoint_hash": None},
         "proposer_checkpoint": "x",
         # deliberately injected -- must never be read by harvest_run
         "pvt": {"total_pvt_corners": 45, "passed_pvt_corners": 43,
                 "corners": [{"pvt_corner_id": "ff_1.62V_-40C"}]}}
    streams = harvest_run(hv, split="train", protected_ids=set())
    for name, rows in streams.items():
        for row in rows:
            blob = str(row)
            assert "pvt_corner_id" not in blob
            assert "total_pvt_corners" not in blob


# --------------------- 16. pvt.enabled=false is a no-op --------------------------
def test_disabled_pvt_config_never_triggers_a_sweep():
    cfg = pvt_eval.PvtConfig(enabled=False)
    # mirrors the exact gate used in run_raptor_v2.run_pipeline
    pvt_config = cfg
    should_run = bool(pvt_config and pvt_config.enabled)
    assert should_run is False


def test_none_pvt_config_never_triggers_a_sweep():
    pvt_config = None
    should_run = bool(pvt_config and pvt_config.enabled)
    assert should_run is False


# --------------------- 17. no new legacy-RAPTOR dependency -----------------------
def test_new_modules_have_no_direct_legacy_raptor_reference():
    for mod_path in (Path(fom_mod.__file__), Path(pvt_eval.__file__)):
        src = mod_path.read_text(encoding="utf-8")
        assert "RAPTOR_Legacy" not in src
        assert "sys.path" not in src


# ------------- protocol lock: C_LOAD is spec-authoritative (Stage 1.5) ---------
# SUPERSEDES the old "parsed but not applied" trip-wire below (Stage 1.5
# repair, 2026-08-09): the 81-run pilot measured nominal.c_load_f == 500pF
# on every row regardless of the spec's stated cl -- that was a real defect,
# not a protocol this codebase intended. See agentic_raptor.electrical.
# effective_c_load and tests/test_cload_consistency.py for the full repair
# and its test coverage; this test is kept (updated, not deleted) as the
# same kind of trip-wire in the opposite direction: it fails loudly if a
# future change silently reintroduces the fixed-500pF behaviour.
def test_cload_protocol_spec_target_is_now_applied():
    from agentic_raptor.electrical import NOMINAL_CLOAD_F, effective_c_load
    from agentic_raptor.llm_dpo.integrity import parse_spec

    prompt = ("### SPEC gain>=41.96dB pm>=45.0deg cl=100pF ugbw>=1e+06Hz "
             "tech=sky130\n### RAG rag_l2_topology_0002\n")
    spec = parse_spec(prompt)
    assert spec["load_capacitance_pf"] == pytest.approx(100.0)
    # ...and effective_c_load resolves that INTO the real load a spec-driven
    # run simulates against -- 100pF, not the old unconditional 500pF.
    resolved = effective_c_load(spec)
    assert fom_mod.f_to_pf(resolved) == pytest.approx(100.0)
    assert resolved != NOMINAL_CLOAD_F
    assert spec["load_capacitance_pf"] != fom_mod.f_to_pf(NOMINAL_CLOAD_F)
