"""NULLING-BRANCH REPAIR (2026-08-16): rc-type compensation must realise
the series nulling resistor it always promised.

The measured defect: map_family() flattened C / RC_parallel / RC_series to
one lone Miller cap and _realise() flattened every compensation type to
"C", so 2s_rc and 2s_miller were byte-identical netlists. Consequences
(all measured): pipeline branches sized the same circuit twice (identical
A/B verification netlists on the GATE3 boundary spec), the selection
layer had no real choices, the rz_x knob scaled a nonexistent device, and
plain-Miller compensation forced 2.8 nF caps that buried UGBW 47x below
the boundary spec's target. Zero LLM; ngspice not required."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.mapping import emit_netlist, map_family, static_validate

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"


def _row(blocks, stages=2):
    return {"topology_id": "t", "gain_stages": stages,
            "functional_blocks": blocks, "unresolved_blocks": [],
            "mapping_readiness": "mapping_ready", "graph_hash": None}


class _E:
    topology_id = "t"


def _kinds(g):
    return [d.kind for d in g.devices]


# ---------------------------------------------------------------------------
# mapper: RC-type blocks realise R in series with the Miller cap
# ---------------------------------------------------------------------------
def test_rc_series_block_maps_to_nulling_resistor_plus_cap():
    g, note = map_family(_E(), _row(["RC_series"]))
    assert note == "mapped"
    assert _kinds(g).count("res") == 1 and _kinds(g).count("cap") == 1
    rz = next(d for d in g.devices if d.kind == "res")
    cc = next(d for d in g.devices if d.kind == "cap")
    assert rz.role == "nulling_resistor"
    # series chain: prev -> RZ -> mid -> CC -> out
    assert rz.nets["n"] == cc.nets["p"] == "nz2"
    assert cc.provenance["compensation_topology"] == "miller_rz_series"
    net = emit_netlist(g, "probe")
    assert "rRZ2" in net
    assert static_validate(g, net)["status"] == "mapped_static_valid"


def test_pure_c_block_still_maps_to_plain_miller_cap():
    g, _ = map_family(_E(), _row(["C"]))
    assert _kinds(g).count("res") == 0 and _kinds(g).count("cap") == 1
    cc = next(d for d in g.devices if d.kind == "cap")
    assert cc.provenance["compensation_topology"] == "miller_per_stage"


def test_no_compensation_block_maps_no_comp_devices():
    g, _ = map_family(_E(), _row([]))
    assert _kinds(g).count("res") == 0 and _kinds(g).count("cap") == 0


def test_three_stage_rc_gets_one_nulling_branch_per_stage():
    g, _ = map_family(_E(), _row(["RC_series"], stages=3))
    assert _kinds(g).count("res") == 2 and _kinds(g).count("cap") == 2


# ---------------------------------------------------------------------------
# _realise: the LLM's compensation TYPE survives translation
# ---------------------------------------------------------------------------
def test_realise_translates_compensation_types_faithfully():
    from run_puct_ablation import _realise
    base = {"stages": [{}, {}]}
    g_rc = _realise({**base, "compensation": [{"type": "rc_nulling"}]})
    g_mi = _realise({**base, "compensation": [{"type": "miller_cap"}]})
    g_no = _realise({**base, "compensation": []})
    assert _kinds(g_rc).count("res") == 1
    assert _kinds(g_mi).count("res") == 0 and _kinds(g_mi).count("cap") == 1
    assert _kinds(g_no).count("cap") == 0


def test_corpus_rc_and_miller_families_are_physically_distinct():
    if not CORPUS.is_file():
        pytest.skip("corpus_diverse.json not present in this checkout")
    from run_puct_ablation import _realise
    by_fam = {}
    for r in json.loads(CORPUS.read_text(encoding="utf-8"))["records"]:
        by_fam.setdefault(r["topology_signature"], r)
    core = lambda n: "\n".join(n.splitlines()[1:])   # drop uuid comment
    for a, b in (("2s_rc", "2s_miller"), ("3s_rc", "3s_miller")):
        na = core(emit_netlist(_realise(json.loads(by_fam[a]["response"])), "p"))
        nb = core(emit_netlist(_realise(json.loads(by_fam[b]["response"])), "p"))
        assert na != nb, f"{a} and {b} still collapse to one netlist"


# ---------------------------------------------------------------------------
# the rz_x knob now scales a real device
# ---------------------------------------------------------------------------
def test_rz_knob_scales_the_realised_resistor():
    from agentic_raptor.mb_sac.spec_sizing import apply_knobs
    g, _ = map_family(_E(), _row(["RC_series"]))
    g2 = apply_knobs(g, [1, 1, 1, 1, 1, 1, 3.0])
    rz = next(d for d in g2.devices if d.kind == "res")
    assert rz.sizing["value"] == pytest.approx(6000.0)
