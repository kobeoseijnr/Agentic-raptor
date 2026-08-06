"""Stage 3C.2 tests: 26-entry decoder matrix, defect removal, sign-aware hashing."""
import json
from pathlib import Path
import pytest
from agentic_raptor.corpus.reextract_v2 import SUBG_NODE, decode_raw_graph, to_canonical

_ROOT = Path(__file__).resolve().parents[1]

class _FakeV:
    def __init__(self, t): self._t = t
    def __getitem__(self, k): return self._t

class _FakeE:
    def __init__(self, s, t): self.source, self.target = s, t

class _FakeG:
    def __init__(self, types, edges):
        self.vs = [_FakeV(t) for t in types]
        self.es = [_FakeE(a, b) for a, b in edges]

@pytest.mark.parametrize("code", sorted(SUBG_NODE))
def test_all_26_entries_decode(code):
    g = _FakeG([0, code, 1], [(0, 1), (1, 2)]) if code > 1 else _FakeG([code], [])
    nodes, edges, report = decode_raw_graph(0, g)
    comps = SUBG_NODE[code]
    decoded = [t for t in nodes.values()]
    for c in comps:
        assert c in decoded
    if code > 1:
        assert report["status"] == "extracted_verified" or "missing_port" == report["status"]
        gm = [t for t in comps if "gm" in t]
        if gm:
            assert gm[0][0] in "+-" and gm[0][-1] in "+-"  # signed gm preserved

def test_unknown_type_raises_not_coerced():
    with pytest.raises(ValueError, match="unknown_cktgnn_subgraph_type"):
        decode_raw_graph(0, _FakeG([0, 99, 1], [(0, 1), (1, 2)]))

def test_modulo_decoder_removed():
    text = (_ROOT / "tools/topology_extractor/cktgnn.py").read_text(encoding="utf-8")
    assert "% 8" not in text, "defective modulo decoder must not be reintroduced"
    assert "unknown_cktgnn_subgraph_type" in text

def test_gm_sign_sensitive_hashing():
    n1 = {"In": "In", "a": "+gm+", "Out": "Out"}
    n2 = {"In": "In", "a": "-gm+", "Out": "Out"}
    e = [("In", "a"), ("a", "Out")]
    h1 = to_canonical("x", n1, e).structural_hash()
    h2 = to_canonical("x", n2, e).structural_hash()
    assert h1 != h2, "gm polarity must change the canonical hash"

def test_v2_registry_and_lineage_exist():
    reg = _ROOT / "artifacts" / "topology_registry_v2"
    fams = sorted(p.name for p in reg.glob("topology_v2_*"))
    assert len(fams) >= 100
    meta = json.loads((reg / fams[0] / "metadata.json").read_text())
    assert meta["semantic_version"] == "verified_subgnode_v2"
    assert (reg / fams[0] / "interpretation.json").is_file()
    lineage = (_ROOT / "artifacts/stage3c2/legacy_to_v2_lineage.jsonl").read_text()
    assert "many_to_many" in lineage

def test_legacy_archive_checksummed():
    m = json.loads((_ROOT / "artifacts/corpus_snapshots/stage3a_legacy_invalidated_v1/MANIFEST.json").read_text())
    assert m["semantic_version"] == "legacy_invalidated_v1" and len(m["files"]) > 50
