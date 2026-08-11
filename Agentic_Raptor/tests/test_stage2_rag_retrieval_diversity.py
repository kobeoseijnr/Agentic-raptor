"""Stage 2 diagnostic finding (2026-08-10): retrieve() returned k
near-duplicate measurements of ONE topology family instead of k diverse
examples -- confirmed on real post-repair data: 36/36 retrieved records
across 6 real diagnostic pairs came from a single variant, and WITH_RAG /
NO_RAG produced byte-identical candidate sets in 5/6 pairs as a direct
consequence. Root cause: the old dedup keyed on rendered TEXT (stage count
+ stability + rounded pm), which many different real measurements of the
SAME graph can each satisfy distinctly, while family was never used to cap
anything. Fixed with a per-family cap (default 2) enforced on a first pass,
lifted only if that under-fills k.
"""
from __future__ import annotations

import json

import pytest


def _write_memory(tmp_path, records):
    p = tmp_path / "mem.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def test_retrieval_caps_records_per_family_when_pool_is_dominated(tmp_path):
    from run_raptor_v2 import retrieve
    # 20 measurements of ONE family, 1 measurement each of three others --
    # exactly the real, observed pool shape (one family vastly overrepresented)
    records = [{"stability": "verified_stable", "stages": 2, "pm": 45 + i,
               "gain_db": 72.0, "family": "2s_none", "variant": "dom"}
              for i in range(20)]
    records += [{"stability": "verified_stable", "stages": 2, "pm": 55,
                "gain_db": 89.0, "family": "2s_rc", "variant": "alt1"},
               {"stability": "verified_stable", "stages": 3, "pm": 60,
                "gain_db": 95.0, "family": "3s_rc", "variant": "alt2"},
               {"stability": "verified_stable", "stages": 3, "pm": 58,
                "gain_db": 92.0, "family": "3s_miller", "variant": "alt3"}]
    p = _write_memory(tmp_path, records)
    spec = {"gain_target_db": 89.45, "phase_margin_target_deg": 45.0}
    r = retrieve(spec, "### SPEC\n### BLOCKS", memory_path=str(p), max_per_family=2)
    families = [rec["family"] for rec in r["records"]]
    assert families.count("2s_none") <= 2, (
        f"dominant family exceeded its cap: {families}")
    assert len(set(families)) >= 3, (
        f"retrieval should surface multiple families when they exist: {families}")


def test_retrieval_still_fills_k_when_diversity_is_scarce(tmp_path):
    """The cap must not starve retrieval -- if only one family truly exists
    in the pool, k slots still get filled (pass 2 lifts the cap)."""
    from run_raptor_v2 import retrieve
    records = [{"stability": "verified_stable", "stages": 2, "pm": 45 + i,
               "gain_db": 72.0, "family": "2s_none", "variant": f"v{i}"}
              for i in range(10)]
    p = _write_memory(tmp_path, records)
    spec = {"gain_target_db": 70.0, "phase_margin_target_deg": 45.0}
    r = retrieve(spec, "### SPEC\n### BLOCKS", memory_path=str(p), k=6)
    assert len(r["records"]) == 6


def test_retrieval_never_returns_duplicate_variants():
    """Real end-to-end check against the actual clean POST_CLOAD_FIX_V1
    memory -- confirms the fix landed on the real data, not just synthetic
    fixtures."""
    from run_raptor_v2 import retrieve
    from agentic_raptor.publication.preflight import CLEAN_RAG_PATH
    if not CLEAN_RAG_PATH.is_file():
        pytest.skip("clean RAG snapshot not present in this checkout")
    spec = {"gain_target_db": 89.45, "phase_margin_target_deg": 45.0}
    r = retrieve(spec, "### SPEC\n### BLOCKS", memory_path=str(CLEAN_RAG_PATH))
    assert len(r["retrieval_ids"]) == len(set(r["retrieval_ids"]))
    families = [rec.get("family") for rec in r["records"]]
    assert len(set(families)) > 1, (
        f"real memory retrieval still collapsed to one family: {families}")
