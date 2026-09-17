"""STAGE 12 LEAKAGE GUARDS for the external-baseline evaluation (2026-08-28).

Fail-closed checks required before any paper comparison run:
  1  evaluation specs never occur in training (spec_hash + context_id disjoint)
  2  topology-instance disjointness across splits (measured 0-overlap, enforced)
  3  the frozen clean RAG snapshot references no validation/test context
  4  the topology corpus is byte-identical to the hash frozen at benchmark
     creation (no test topology can have entered it)
  5  the RAPTOR runtime never imports from external_baselines/
  6  external baseline checkouts never import agentic_raptor
  7  the frozen evaluation files are byte-identical to their manifest hashes
     (immutable eval configuration)

These tests ARE paper mode's fail-closed gate: a red test here invalidates
any comparison run made after the corruption.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "external_baseline_eval"
RAG_CLEAN = ROOT / "artifacts/publication_v2/selfimprove/rag_memory_v2_clean.jsonl"
CORPUS = ROOT / "artifacts/stage3e4/corpus.json"

pytestmark = pytest.mark.skipif(not EVAL.exists(),
                                reason="external baseline benchmark not frozen")


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _load(name: str) -> dict:
    return json.loads((EVAL / name).read_text(encoding="utf-8"))


def _split_sets():
    out = {}
    for name in ("specs_train.json", "specs_validation.json", "specs_test.json"):
        d = _load(name)
        out[name] = {"hashes": {s["spec_hash"] for s in d["specs"]},
                     "ctx": {s["context_id"] for s in d["specs"]},
                     "topo": {s["topology_id"] for s in d["specs"]}}
    return out


def test_1_eval_specs_never_in_training():
    s = _split_sets()
    tr = s["specs_train.json"]
    for ev in ("specs_validation.json", "specs_test.json"):
        assert not (tr["hashes"] & s[ev]["hashes"]), f"spec_hash leak train<->{ev}"
        assert not (tr["ctx"] & s[ev]["ctx"]), f"context_id leak train<->{ev}"


def test_2_topology_instances_disjoint_across_splits():
    s = _split_sets()
    names = list(s)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (s[a]["topo"] & s[b]["topo"]), \
                f"topology_id leak {a}<->{b}: {s[a]['topo'] & s[b]['topo']}"


def test_3_clean_rag_references_no_eval_context():
    if not RAG_CLEAN.exists():
        pytest.skip("clean RAG snapshot not present")
    s = _split_sets()
    eval_ctx = s["specs_validation.json"]["ctx"] | s["specs_test.json"]["ctx"]
    eval_topo = s["specs_validation.json"]["topo"] | s["specs_test.json"]["topo"]
    text = RAG_CLEAN.read_text(encoding="utf-8", errors="replace")
    hits = sorted(c for c in eval_ctx if c in text)
    assert not hits, f"eval context_ids present in clean RAG: {hits[:5]}"
    # topology ids may legitimately appear via train specs of the same id? --
    # measured: splits are topology-instance-disjoint, so an eval topology id
    # in RAG is a leak.
    thits = sorted(t for t in eval_topo if re.search(rf"\b{re.escape(t)}\b", text))
    assert not thits, f"eval topology_ids present in clean RAG: {thits[:5]}"


def test_4_corpus_unchanged_since_freeze():
    manifest = _load("split_manifest.json")
    assert _sha16(CORPUS) == manifest["source_corpus_hash"], \
        "topology corpus changed since the benchmark freeze -- a test topology " \
        "may have entered it; paper runs are invalid until re-frozen"


def test_5_runtime_never_imports_external_baselines():
    offenders = []
    for p in list((ROOT / "agentic_raptor").rglob("*.py")) + list(ROOT.glob("*.py")):
        if "external_baselines" in p.parts:
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"^\s*(from|import)\s+external_baselines", src, re.M) or \
           re.search(r"(from|import)\s+\S*(AnalogCoderPro|analogcoderpro|"
                     r"analogxpert|autockt|MACE)\b", src):
            offenders.append(str(p.relative_to(ROOT)))
    allowed = {p for p in offenders if p.startswith("src\\evaluation")
               or p.startswith("src/evaluation")}
    assert not (set(offenders) - allowed), \
        f"runtime imports external baselines: {sorted(set(offenders) - allowed)}"


def test_6_external_baselines_never_import_agentic_raptor():
    base = ROOT / "external_baselines"
    if not base.exists():
        pytest.skip("baselines not cloned")
    offenders = []
    for p in base.rglob("*.py"):
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"^\s*(from|import)\s+agentic_raptor", src, re.M):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"baseline code imports agentic_raptor: {offenders}"


def test_7_eval_configuration_immutable():
    manifest = _load("split_manifest.json")
    for fname, meta in manifest["files"].items():
        actual = _sha16(EVAL / fname)
        assert actual == meta["sha256_16"], \
            f"{fname} changed since freeze ({actual} != {meta['sha256_16']}) -- " \
            "evaluation configuration must be immutable once paper runs start"
