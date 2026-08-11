"""Shared topology-generation diagnostics for A0-A4 (Part: TOPOLOGY-GENERATION
DIAGNOSTICS).

Used by both A1 (retrieval-only candidates) and A4 (exclusion-conditioned
diversity) so novelty/diversity accounting is computed identically for
LLM-generated and retrieval-generated candidate pools -- a shared
implementation, not two ad-hoc counters that could quietly disagree.

Novelty is NOT "the hash differs" (the task is explicit that this is not a
valid novelty claim on its own). Two hash granularities are compared:
  - family hash  (`variant_hash`): stage count + compensation type +
    output_buffer + local_feedback -- coarse, this IS `canonical_graph_hash`
    throughout the v2 pipeline.
  - content hash (`_content_hash`): sha256 of the full canonicalised `obj`
    -- fine, distinguishes structurally-identical-family candidates that
    still differ in role wiring / port structure / feedback paths.

    exact_corpus_match   : content hash already in the known corpus
    near_duplicate        : family hash known, content hash is new
    structurally_distinct : family hash itself is new
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def content_hash(obj: dict) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def classify_novelty(obj: dict, family_hash: str,
                     known_family_hashes: set[str],
                     known_content_hashes: set[str]) -> str:
    """One of exact_corpus_match | near_duplicate | structurally_distinct."""
    if content_hash(obj) in known_content_hashes:
        return "exact_corpus_match"
    if family_hash in known_family_hashes:
        return "near_duplicate"
    return "structurally_distinct"


def candidate_pool_diagnostics(candidates: list[dict[str, Any]], *,
                               target_k: int,
                               known_family_hashes: set[str] | None = None,
                               known_content_hashes: set[str] | None = None,
                               valid_count: int | None = None,
                               raw_proposal_count: int | None = None,
                               downstream_pass_by_id: dict[str, bool] | None = None
                               ) -> dict[str, Any]:
    """Unique@K, duplicate rate, family coverage, graph diversity, novelty
    breakdown, Valid@K, Pass@K -- computed the SAME way regardless of
    whether `candidates` came from the LLM or from retrieval.

    `candidates` entries need at minimum: canonical_graph_hash,
    canonical_family, obj. `downstream_pass_by_id` (llm_proposal_id ->
    exact_spec_pass), when given, yields Pass@K.
    """
    known_family_hashes = known_family_hashes or set()
    known_content_hashes = known_content_hashes or set()
    n = len(candidates)
    family_hashes = [c["canonical_graph_hash"] for c in candidates]
    families = [c["canonical_family"] for c in candidates]
    content_hashes = [content_hash(c["obj"]) for c in candidates]
    unique_family = len(set(family_hashes))
    unique_content = len(set(content_hashes))
    novelty = [classify_novelty(c["obj"], c["canonical_graph_hash"],
                                known_family_hashes, known_content_hashes)
              for c in candidates]
    novelty_counts = {cat: novelty.count(cat) for cat in
                      ("exact_corpus_match", "near_duplicate",
                       "structurally_distinct")}
    out = {
        "k_requested": target_k,
        "k_returned": n,
        "unique_at_k": unique_family,                  # family-hash unique
        "unique_content_at_k": unique_content,          # finer, structural
        "duplicate_rate": round(1.0 - unique_family / n, 4) if n else None,
        "canonical_family_coverage": sorted(set(families)),
        "canonical_family_count": len(set(families)),
        "graph_diversity": round(unique_content / n, 4) if n else None,
        "novelty_breakdown": novelty_counts,
        "novelty_by_candidate": dict(zip(
            [c.get("llm_proposal_id") for c in candidates], novelty)),
    }
    if raw_proposal_count is not None:
        out["raw_proposal_count"] = raw_proposal_count
        out["valid_at_k"] = (round(n / raw_proposal_count, 4)
                             if raw_proposal_count else None)
    if valid_count is not None:
        out["valid_count"] = valid_count
    if downstream_pass_by_id:
        passed = sum(1 for v in downstream_pass_by_id.values() if v)
        out["pass_at_k"] = round(passed / n, 4) if n else None
        out["pass_count"] = passed
    return out


def known_corpus_hashes(corpus_records: list[dict], *,
                        exclude_topology_ids: set[str] | None = None
                        ) -> tuple[set[str], set[str]]:
    """(family_hashes, content_hashes) for every corpus record's structural
    `response`, so a retrieval or LLM candidate can be checked for novelty
    against the corpus it was drawn from / trained on."""
    from agentic_raptor.llm_dpo.stage3e4 import variant_hash

    exclude_topology_ids = exclude_topology_ids or set()
    fam, cont = set(), set()
    for r in corpus_records:
        if r.get("topology_id") in exclude_topology_ids:
            continue
        try:
            obj = json.loads(r["response"])
        except (KeyError, ValueError):
            continue
        fam.add(variant_hash(obj))
        cont.add(content_hash(obj))
    return fam, cont
