"""Real-SPICE backend: discovery, unavailability, output parsing, real runs."""

from __future__ import annotations

import pytest

from agentic_raptor.core.candidate import CircuitCandidate
from agentic_raptor.core.types import GenerationSource
from agentic_raptor.sizing.parameter_space import SizingParameterSpace
from agentic_raptor.spice.ngspice_simulator import NgspiceSimulator, discover_ngspice
from agentic_raptor.spice.result_parser import metrics_from_ngspice, parse_ngspice_stdout

_NGSPICE = discover_ngspice()


def test_discovery_rejects_missing_configured_path():
    assert discover_ngspice(configured="Z:/does/not/exist/ngspice.exe") is None


def test_unavailable_simulator_returns_structured_error(ota_graph, spec, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # hide any real ngspice
    sim = NgspiceSimulator(ngspice_exe="Z:/missing/ngspice.exe")
    # discover() saw a configured-but-missing path → exe None regardless of PATH
    candidate = CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)
    result = sim.simulate(candidate, ["op", "ac"])
    assert not result.success
    assert result.error_type == "simulator_unavailable"
    assert "ngspice" in (result.error_message or "")


def test_stdout_measurement_parsing():
    stdout = """
Note: some banner
v(n_out) = 9.00907e-01
vvdd#branch = -5.55e-05
dc_gain_db = 5.158633e+01
ugf_hz = 2.049488e+08
phase_at_ugf = -9.646081e+01
"""
    parsed = parse_ngspice_stdout(stdout)
    assert parsed.failure_type is None
    metrics = metrics_from_ngspice(parsed, supply_voltage=1.8, output_node="n_out")
    assert metrics["gain_db"] == pytest.approx(51.586, abs=1e-2)
    assert metrics["gbw_hz"] == pytest.approx(2.049e8, rel=1e-3)
    assert metrics["phase_margin_deg"] == pytest.approx(180 - 96.46, abs=0.1)
    assert metrics["power_w"] == pytest.approx(1.8 * 5.55e-05, rel=1e-3)
    assert metrics["output_dc_v"] == pytest.approx(0.909, abs=1e-2)


def test_radians_phase_fallback():
    parsed = parse_ngspice_stdout("phase_at_ugf = -1.683\nugf_hz = 1e6\ndc_gain_db = 40\n")
    metrics = metrics_from_ngspice(parsed, 1.8, "out")
    assert metrics["phase_margin_deg"] == pytest.approx(180 - 96.43, abs=0.2)


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        ("doAnalyses: TRAN:  Timestep too small", "timestep_failure"),
        ("No convergence in dc analysis", "convergence_failure"),
        ("singular matrix:  check nodes", "singular_matrix"),
        ("Error: unable to find definition of model nch", "missing_model"),
        ("Error on line 12 : mx 1 2 3", "malformed_netlist"),
        ("Warning: Output overflow detected", "numerical_overflow"),
    ],
)
def test_failure_signature_classification(snippet, expected):
    parsed = parse_ngspice_stdout(snippet)
    assert parsed.failure_type == expected
    assert parsed.failure_message


def test_failed_measurement_detection():
    parsed = parse_ngspice_stdout("ugf_hz = failed\ndc_gain_db = 12.0\n")
    assert "ugf_hz" in parsed.failed_measurements
    assert parsed.values["dc_gain_db"] == 12.0


@pytest.mark.requires_ngspice
@pytest.mark.skipif(_NGSPICE is None, reason="ngspice not installed")
def test_real_ota_simulation(ota_graph, spec):
    space = SizingParameterSpace.from_graph(ota_graph)
    candidate = CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)
    candidate.sizing_state = space.denormalize(space.default_vector())
    sim = NgspiceSimulator(seed=1)
    result = sim.simulate(candidate, ["op", "ac"], timeout_s=60.0)
    assert result.success, f"{result.error_type}: {result.error_message}"
    for key in ("gain_db", "gbw_hz", "phase_margin_deg", "power_w"):
        assert key in result.metrics
    assert result.constraint_margins
    assert result.raw_output_path
    assert result.runtime_s > 0


@pytest.mark.requires_ngspice
@pytest.mark.skipif(_NGSPICE is None, reason="ngspice not installed")
def test_real_simulation_deterministic_and_sizing_sensitive(ota_graph, spec):
    space = SizingParameterSpace.from_graph(ota_graph)
    sim = NgspiceSimulator(seed=1)
    c1 = CircuitCandidate.create(ota_graph, spec, GenerationSource.MOCK)
    c1.sizing_state = space.denormalize(space.default_vector())
    a = sim.simulate(c1, ["op", "ac"])
    b = sim.simulate(c1, ["op", "ac"])
    assert a.metrics == b.metrics, "identical inputs must give identical real results"
    c2 = CircuitCandidate.create(ota_graph.copy(), spec, GenerationSource.MOCK)
    c2.sizing_state = space.denormalize([0.4] * space.dim)
    changed = sim.simulate(c2, ["op", "ac"])
    assert changed.success and changed.metrics != a.metrics
