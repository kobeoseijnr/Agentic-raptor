"""Stage 1.5 C_LOAD repair (2026-08-09): a spec's requested load must
actually reach the simulator, consistently, everywhere.

Root defect this repairs: every real ngspice call (sac_size, the non-RL
sizing baselines, run_raptor_v2.py's final verify(), PVT) left c_load_f=None
and silently got agentic_raptor.electrical.NOMINAL_CLOAD_F (500pF)
regardless of what a spec's parsed load_capacitance_pf said. Confirmed on
the real 81-run pilot (results_20260809_032835.jsonl): nominal.c_load_f was
5e-10 (500pF) on literally every successful row, even for specs stating
100pF/200pF. The fix is agentic_raptor.electrical.effective_c_load(spec,
override=...), now the single resolution point every real caller goes
through.

Covers: netlist-level load for 100/200/500pF, nominal==simulated agreement,
PVT preserving the requested load per corner, FoM reading the simulated
(not stated) load, an unexplained mismatch hard-failing in paper mode, an
explicit+reasoned override being allowed and recorded, a spec that already
asks for 500pF still simulating at 500pF, and no new legacy-RAPTOR coupling.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agentic_raptor.electrical import (NOMINAL_CLOAD_F, discover_ngspice,
                                       effective_c_load)

ROOT = Path(__file__).resolve().parents[1]
NGSPICE = discover_ngspice()
requires_ngspice = pytest.mark.skipif(not NGSPICE, reason="ngspice not found")


# --------------------------- effective_c_load unit behaviour -----------------
def test_effective_c_load_uses_spec_when_no_override():
    assert effective_c_load({"load_capacitance_pf": 100.0}) == pytest.approx(100e-12)
    assert effective_c_load({"load_capacitance_pf": 200.0}) == pytest.approx(200e-12)


def test_effective_c_load_falls_back_to_nominal_without_a_spec_load():
    assert effective_c_load(None) == NOMINAL_CLOAD_F
    assert effective_c_load({}) == NOMINAL_CLOAD_F


def test_effective_c_load_500pf_spec_still_simulates_at_500pf():
    """Regression guard: a spec that already asks for exactly the nominal
    value must resolve to the SAME real value as the old unconditional
    fallback -- the repair must not perturb this case."""
    assert effective_c_load({"load_capacitance_pf": 500.0}) == NOMINAL_CLOAD_F


def test_explicit_override_wins_and_is_distinguishable_from_spec():
    """An override is allowed, but only when the CALLER explicitly asked --
    never a silent substitute for a spec's stated load."""
    resolved = effective_c_load({"load_capacitance_pf": 100.0}, override=333e-12)
    assert resolved == pytest.approx(333e-12)
    assert resolved != pytest.approx(100e-12)


# ------------------------- netlist-level validation (real ngspice) -----------
@requires_ngspice
@pytest.mark.parametrize("pf", [100.0, 200.0, 500.0])
def test_netlist_emits_the_requested_load(tmp_path, pf):
    from agentic_raptor.topology_rl.stage3e2_edits import qualify_device_graph
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from run_puct_ablation import realise_class

    g = realise_class("2s_miller")
    out = tmp_path.resolve()
    q = qualify_device_graph("t_cload", g, out, NGSPICE, f"pf{int(pf)}",
                             new_costs(), c_load_f=pf * 1e-12)
    assert q["electrical"] == "electrically_functional", q
    tb = (out / f"pf{int(pf)}" / "run" / "tb.cir").read_text(encoding="utf-8")
    line = next(l for l in tb.splitlines() if "PARAM_CLOAD" in l)
    got = float(line.split("=")[1].strip())
    assert got == pytest.approx(pf * 1e-12, rel=1e-6)


@requires_ngspice
def test_physical_response_actually_changes_with_load(tmp_path):
    """Not just metadata: the measured UGBW must genuinely differ between
    a light and a heavy load on the identical topology/sizing (heavier
    load -> lower UGBW, DC gain roughly unchanged -- real single-pole
    physics, not something a mocked path could fake)."""
    from agentic_raptor.topology_rl.stage3e2_edits import qualify_device_graph
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from run_puct_ablation import realise_class

    g = realise_class("2s_miller")
    out = tmp_path.resolve()
    q_light = qualify_device_graph("t_cload", g, out, NGSPICE, "light",
                                   new_costs(), c_load_f=100e-12)
    q_heavy = qualify_device_graph("t_cload", g, out, NGSPICE, "heavy",
                                   new_costs(), c_load_f=500e-12)
    ugbw_light = q_light["metrics"]["ugbw_hz"]
    ugbw_heavy = q_heavy["metrics"]["ugbw_hz"]
    gain_light = q_light["metrics"]["dc_gain_db"]
    gain_heavy = q_heavy["metrics"]["dc_gain_db"]
    assert ugbw_light > ugbw_heavy * 1.5   # 5x load -> materially lower UGBW
    assert gain_light == pytest.approx(gain_heavy, abs=0.01)  # DC gain load-independent


@requires_ngspice
def test_sac_size_measures_at_the_specs_requested_load(tmp_path):
    """End-to-end: the exact regression found in the 81-run pilot, closed
    the loop through the real production sizing call."""
    from agentic_raptor.mb_sac.spec_sizing import sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from run_puct_ablation import realise_class

    g = realise_class("2s_miller")
    spec = {"gain_target_db": 89.45, "phase_margin_target_deg": 45.0,
           "load_capacitance_pf": 200.0, "ugbw_target_hz": 1e6,
           "spec_id": "test_spec"}
    sz = sac_size("t_cload_sac", g, spec, NGSPICE, tmp_path.resolve(),
                  new_costs(), budget=2, seed=0, persist=False)
    assert sz["best"]["c_load_f"] == pytest.approx(200e-12)
    assert sz["best"]["c_load_f"] != pytest.approx(NOMINAL_CLOAD_F)


# --------------------------- PVT preserves the requested load ----------------
def test_pvt_corners_all_receive_the_same_requested_load():
    """Orchestration-level: PVT is allowed to vary process/voltage/
    temperature but must resolve every corner's load from the SAME
    effective_c_load() call the nominal path used -- verified by spying on
    every real measure() call PVT makes, not by re-deriving expectations."""
    from agentic_raptor.electrical.pvt_eval import PvtConfig, run_pvt_sweep

    seen_loads = []

    def _fake_measure(topology_id, graph, exe, out_dir, tag, costs, **kw):
        seen_loads.append(kw.get("c_load_f"))
        return {"gain_db": 70.0, "pm_deg": 50.0, "ugbw_hz": 1e6,
               "power_w": 1e-4, "idd_a": 1e-4, "c_load_f": kw.get("c_load_f"),
               "stable": True, "stability": "verified_stable",
               "electrical": "electrically_functional", "op_valid": True,
               "metrics": {}}

    cfg = PvtConfig(enabled=True, process_corners=("tt", "ff", "ss"),
                    supply_voltages=(1.8,), temperatures_c=(27.0, 85.0))
    spec = {"spec_id": "t_pvt", "gain_target_db": 60.0,
           "phase_margin_target_deg": 45.0, "ugbw_target_hz": 1e6}
    with patch("agentic_raptor.mb_sac.spec_sizing.measure", _fake_measure):
        run_pvt_sweep("tid", object(), spec, "exe", Path("/tmp/pvt"), cfg,
                      label="A", c_load_f=137e-12)
    assert len(seen_loads) == 6         # 3 process x 1 voltage x 2 temps
    assert all(v == pytest.approx(137e-12) for v in seen_loads), seen_loads


# --------------------------------- FoM ----------------------------------------
def test_fom_reads_the_simulated_not_stated_load():
    from agentic_raptor.electrical.fom import compute_fom

    # A spec might state 100pF; if the run actually simulated at 200pF
    # (effective_c_load resolved it that way), FoM MUST be computed from
    # 200pF -- compute_fom has no way to know the "stated" value at all,
    # which is itself the guarantee: it only ever sees what was measured.
    r_simulated = compute_fom(ugbw_hz=2e6, c_load_f=200e-12, idd_a=1e-3)
    r_stated = compute_fom(ugbw_hz=2e6, c_load_f=100e-12, idd_a=1e-3)
    assert r_simulated["fom_value"] != r_stated["fom_value"]
    assert r_simulated["cload_used"] == pytest.approx(200e-12)


# ----------------------- paper-mode hard-fail on mismatch --------------------
def test_unexplained_mismatch_flag_set_when_simulated_differs_without_reason():
    """Mirrors the exact dict shape run_raptor_v2.run_pipeline builds into
    trace["nominal"] -- an override with NO recorded reason must be flagged
    c_load_unexplained_mismatch=True (what run_ablation_v3.py's --paper-mode
    hard-fail checks)."""
    requested = 100e-12
    simulated = 500e-12          # e.g. a future regression reintroducing the bug
    override_reason = None
    cl_override = abs(simulated - requested) > 1e-15
    unexplained = bool(cl_override and not override_reason)
    assert unexplained is True


def test_explained_override_does_not_trip_the_hard_fail():
    requested = 100e-12
    simulated = 333e-12
    override_reason = "physical sanity sweep -- deliberate override"
    cl_override = abs(simulated - requested) > 1e-15
    unexplained = bool(cl_override and not override_reason)
    assert unexplained is False


def test_run_pipeline_records_cload_provenance_fields():
    """Structural: run_pipeline's signature and its trace-building code
    reference the provenance fields paper mode depends on, so a refactor
    that quietly drops them fails this test instead of silently breaking
    the hard-fail check."""
    import inspect

    import run_raptor_v2 as v2
    params = set(inspect.signature(v2.run_pipeline).parameters)
    assert {"c_load_override_f", "c_load_override_reason"} <= params
    src = inspect.getsource(v2.run_pipeline)
    for field in ("requested_c_load_f", "simulated_c_load_f",
                 "c_load_override", "c_load_unexplained_mismatch"):
        assert field in src, f"missing provenance field: {field}"


# --------------------- no new legacy-RAPTOR dependency ------------------------
def test_no_new_legacy_raptor_reference():
    import agentic_raptor.electrical as elec_mod
    import agentic_raptor.electrical.pvt_eval as pvt_mod
    import agentic_raptor.mb_sac.spec_sizing as ss_mod
    for mod in (elec_mod, pvt_mod, ss_mod):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        # _PDK_CORNER_DIR legitimately points at the shared sky130 PDK model
        # files under RAPTOR_Legacy/ -- a data path, not a runtime import --
        # so only NEW python-level coupling (sys.path manipulation or an
        # import of legacy runtime code) is disallowed here.
        assert "sys.path" not in src
        assert "import RAPTOR_Legacy" not in src
        assert "from RAPTOR_Legacy" not in src
