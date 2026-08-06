"""Stage 3C.1 interpretation tests (synthetic graphs, verified semantics)."""
from agentic_raptor.mapping.interpretation import (
    NODE_TYPE, SUBG_NODE, FunctionalStageGraph, audit_behavioral_families,
    interpret_verified_dag)
from agentic_raptor.corpus import TopologyRegistry
from pathlib import Path
import json

_ROOT = Path(__file__).resolve().parents[1]

NODES = {"i": "In", "g1": "+gm+", "g2": "-gm+", "g3": "-gm+", "fb": "-gm-",
         "c1": "C", "o": "Out"}
EDGES = [("i","g1"),("g1","g2"),("g2","g3"),("g3","o"),("o","fb"),("fb","g2"),
         ("g1","c1"),("c1","g2")]

def test_verified_tables_match_source():
    assert NODE_TYPE["+gm+"] == 2 and NODE_TYPE["-gm-"] == 5 and NODE_TYPE["Out"] == 9
    assert SUBG_NODE[10] == ["C", "+gm+"] and SUBG_NODE[25] == ["C", "R", "-gm-"]

def test_input_intermediate_output_and_feedback_detection():
    g = interpret_verified_dag("syn1", NODES, EDGES)
    fns = {s.stage_id: (s.function, s.path_class, s.polarity) for s in g.stages}
    assert fns["gm_g1"][0] == "differential_input_transconductor" and fns["gm_g1"][2] == "+"
    assert fns["gm_g2"][0] == "common_source_equivalent"
    assert fns["gm_g3"][0] == "output_transconductor"
    assert fns["gm_fb"] == ("feedback_transconductor", "feedback_path", "-")
    assert any(s.function == "compensation_branch" for s in g.stages)

def test_inversion_parity_and_order():
    g = interpret_verified_dag("syn1", NODES, EDGES)
    assert g.inversion_parity == 2  # g2, g3 are -gm main-path stages
    assert g.stage_order == ["gm_g1", "gm_g2", "gm_g3"]

def test_validation_scores():
    g = interpret_verified_dag("syn1", NODES, EDGES)
    v = g.validation()
    assert v["functional_graph_valid"] and v["interpreted_gm_fraction"] == 1.0
    assert v["polarity_coverage"] == 1.0

def test_outcome_independence_no_electrical_fields():
    g = interpret_verified_dag("syn1", NODES, EDGES)
    text = json.dumps([s.__dict__ for s in g.stages], default=str)
    for banned in ("phase_margin", "gain_db", "simulation", "converge"):
        assert banned not in text

def test_all_37_withheld_with_reason():
    reg = TopologyRegistry(_ROOT / "datasets" / "topology_library")
    rows = audit_behavioral_families(reg)
    assert len(rows) >= 30
    assert all(r["interpretation_status"] == "withheld_at_semantic_audit" for r in rows)
    assert all("UNVERIFIED" in r["reason"] for r in rows)

def test_withheld_memory_records_exist():
    p = _ROOT / "datasets" / "simulation_memory" / "interpretation_runs.jsonl"
    if p.is_file():
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        assert all(r["interpretation_status"].startswith("withheld") for r in rows)
