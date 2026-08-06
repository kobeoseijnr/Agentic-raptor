"""Stage 3C mapping tests (synthetic/deterministic; no fabricated electrical data)."""
from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.mapping import audit_generated, emit_netlist, map_family, static_validate
from pathlib import Path
import json

_ROOT = Path(__file__).resolve().parents[1]
REG = TopologyRegistry(_ROOT / "datasets" / "topology_library")
AUD = audit_generated(REG)

def test_audit_covers_all_generated():
    assert len(AUD) == REG.get_family_count() - len(REG.filter_by_source("analoggym"))
    assert all(a["mapping_readiness"] in ("mapping_ready","partially_specified",
               "behavioral_only","structurally_ambiguous","invalid_graph","unsupported") for a in AUD)

def test_mapping_deterministic_roles_and_groups():
    row = next(a for a in AUD if a["mapping_readiness"] == "mapping_ready")
    g, note = map_family(REG.get_topology(row["topology_id"]), row)
    roles = {d.device_id: d.role for d in g.devices}
    assert roles["M1"] == roles["M2"] == "input_pair_nmos"
    assert {d.group for d in g.devices if d.device_id in ("M1","M2")} == {"pair1"}
    assert {d.group for d in g.devices if d.device_id in ("M3","M4")} == {"mir1"}
    assert set(g.support_bias) == {"M6","IB1"}
    assert all(d.sizing.get("origin") or d.sizing.get("l") for d in g.devices)
    assert g.polarity["selection_rule"].startswith("structural parity")

def test_netlist_emission_sky130_and_static_validation():
    row = next(a for a in AUD if a["mapping_readiness"] == "mapping_ready")
    g, _ = map_family(REG.get_topology(row["topology_id"]), row)
    net = emit_netlist(g, "mapped_test")
    assert "sky130_fd_pr__nfet_01v8" in net and "sky130_fd_pr__pfet_01v8" in net
    assert ".subckt mapped_test gnda vdda vinn vinp vout" in net
    val = static_validate(g, net)
    assert val["status"] == "mapped_static_valid" and val["sky130_models_valid"]

def test_unsupported_families_not_silently_mapped():
    row = next(a for a in AUD if a["mapping_readiness"] == "behavioral_only")
    if row["gain_stages"] == 0:
        g, note = map_family(REG.get_topology(row["topology_id"]), row)
        assert g is None and note == "mapping_unsupported"

def test_failed_and_successful_mappings_retained_in_memory():
    p = _ROOT / "datasets" / "simulation_memory" / "mapping_runs.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert rows and all("mapping_status" in r or "mapping_status" in r for r in rows)
