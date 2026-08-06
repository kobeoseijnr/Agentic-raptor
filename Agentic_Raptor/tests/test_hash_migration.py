"""Stage 3C.2c migration tests."""
import json
from pathlib import Path
from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.corpus.hash_migration_v2 import HASH_VERSION

_ROOT = Path(__file__).resolve().parents[1]
REG = TopologyRegistry(_ROOT / "datasets" / "topology_library")

def test_all_v1_records_migrated_with_legacy_preserved():
    for tid in REG.list_topologies():
        m = REG.get_metadata(tid)
        assert m["hash_version"] == HASH_VERSION
        assert "legacy_graph_hash" in m
        assert m["graph_hash"] == REG.get_graph(tid).structural_hash()

def test_analoggym_ids_and_evidence_preserved():
    rows = [json.loads(x) for x in
            (_ROOT / "artifacts/stage3c2c/v1_hash_migration.jsonl").read_text().splitlines()]
    ag = [r for r in rows if r["source"] == "analoggym"]
    assert len(ag) == 17
    assert all(r["electrical_evidence"] == "electrical_evidence_preserved" for r in ag)
    assert all(r["netlist_hash"] for r in ag)  # netlists untouched, hashed

def test_snapshot_immutable_manifest():
    m = json.loads((_ROOT / "artifacts/corpus_snapshots/pre_stage3c2c_v1_hash_migration/MANIFEST.json").read_text())
    assert len(m["files"]) >= 200 and m["hasher_before"] == "wl_role_blind_v1"

def test_opamp_corrected_families_and_lineage():
    reg2 = _ROOT / "artifacts/topology_registry_v1_hash_v2"
    fams = sorted(p.name for p in reg2.glob("topology_og2_*"))
    assert len(fams) >= 10
    meta = json.loads((reg2 / fams[0] / "metadata.json").read_text())
    assert meta["hash_version"] == HASH_VERSION and meta["relationship"] == "legacy_family_split"
    lineage = [json.loads(x) for x in
               (_ROOT / "artifacts/stage3c2c/v1_to_hash_v2_lineage.jsonl").read_text().splitlines()]
    kinds = {r["relationship"] for r in lineage}
    assert {"legacy_family_split", "hash_only_migration"} <= kinds
    # one source row never in two active families
    seen = set()
    for r in lineage:
        for s in r.get("source_rows", []):
            assert s not in seen
            seen.add(s)

def test_role_aware_hash_distinguishes_semantics():
    from agentic_raptor.corpus.reextract_v2 import to_canonical
    e = [("In","a"),("a","Out")]
    h = lambda t: to_canonical("x", {"In":"In","a":t,"Out":"Out"}, e).structural_hash()
    assert len({h("+gm+"), h("-gm+"), h("+gm-"), h("-gm-")}) == 4

def test_no_hash_version1_active():
    for tid in REG.list_topologies():
        assert REG.get_metadata(tid).get("hash_version") == 2
