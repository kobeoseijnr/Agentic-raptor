"""Stage 4 diagnostic script (2026-08-11 rewrite): per-attempt instrumented
ladder comparing exclusion vs temperature conditioning, classifying every
attempt into RUN_REHIT / CORPUS_OR_MEMORY_REHIT / NOVEL_CANONICAL_GRAPH
rather than only inspecting the final (already-deduplicated-by-construction)
candidate set. This file locks in the manual verifications done while
building it -- no GPU/model calls, matching this repo's convention.
"""
from __future__ import annotations

import run_stage4_diversity_diagnostic as s4


def test_a4_config_audit_is_clean():
    """Section 2: A4_NO_DIVERSITY must differ from A0_FULL in exactly
    `conditioning`, nothing else -- verified against the real
    agentic_raptor.publication.ablation_v3 definitions, not a fixture."""
    audit = s4.audit_a4_config()
    assert audit["clean"], audit["unexpected_diffs"]
    assert set(audit["kwarg_diffs"]) == {"conditioning"}
    assert audit["kwarg_diffs"]["conditioning"] == ("exclusion", "temperature")
    assert audit["budget_diffs"] == {}


def test_graph_distance_counts_every_dimension():
    a = {"stages": [1, 2], "compensation": [], "output_buffer": False,
        "local_feedback": False}
    b = {"stages": [1, 2, 3], "compensation": [{"type": "miller_cap"}],
        "output_buffer": True, "local_feedback": False}
    assert s4.graph_distance(a, a) == 0
    assert s4.graph_distance(a, b) == 3   # stages + compensation + buffer


def test_attempts_to_k_finds_the_correct_attempt():
    log = [
        {"attempt": 1, "valid": False, "graph_hash": None},
        {"attempt": 2, "valid": True, "graph_hash": "hA"},
        {"attempt": 3, "valid": False, "graph_hash": "hA"},   # run-rehit, invalid
        {"attempt": 4, "valid": True, "graph_hash": "hB"},
        {"attempt": 5, "valid": True, "graph_hash": "hC"},
    ]
    assert s4._attempts_to_k(log, target_k=3) == 5
    assert s4._attempts_to_k(log, target_k=2) == 4
    assert s4._attempts_to_k(log, target_k=10) is None


def _obj(stages, comp=None, buf=False, fb=False):
    return {"stages": list(range(stages)), "compensation": comp or [],
           "output_buffer": buf, "local_feedback": fb}


def test_measure_pool_arithmetic_matches_hand_computation():
    """Hand-verified: attempt1=failure(A), attempt2=novel valid hA
    (2s_none), attempt3=RUN_REHIT of hA (excluded, violated), attempt4=
    CORPUS_OR_MEMORY_REHIT hKNOWN (2s_rc, switched family), attempt5=novel
    valid hB (3s_miller, switched family)."""
    objA, objB = _obj(2), _obj(3, comp=[{"type": "miller_cap"}])
    log = [
        {"attempt": 1, "temperature": 0.8, "top_p": 0.92,
         "prompt_word_count": 50, "exclusion_applied": False,
         "excluded_hashes_before": [], "excluded_families_before": [],
         "raw_response": "", "parsed": False, "valid": False,
         "validator_reasons": [], "graph_hash": None, "family": None,
         "taxonomy_code": "A", "taxonomy_label": "empty_refusal_irrelevant_text",
         "run_rehit": False, "corpus_or_memory_rehit": False,
         "novel_canonical_graph": False, "violated_exclusion": False,
         "switched_family": False},
        {"attempt": 2, "temperature": 0.8, "top_p": 0.92,
         "prompt_word_count": 55, "exclusion_applied": True,
         "excluded_hashes_before": [], "excluded_families_before": [],
         "raw_response": "", "parsed": True, "valid": True,
         "validator_reasons": [], "graph_hash": "hA", "family": "2s_none",
         "taxonomy_code": "I", "taxonomy_label": "valid_graph",
         "run_rehit": False, "corpus_or_memory_rehit": False,
         "novel_canonical_graph": True, "violated_exclusion": False,
         "switched_family": False},
        {"attempt": 3, "temperature": 0.8, "top_p": 0.92,
         "prompt_word_count": 60, "exclusion_applied": True,
         "excluded_hashes_before": ["hA"], "excluded_families_before": ["2s_none"],
         "raw_response": "", "parsed": True, "valid": False,
         "validator_reasons": [], "graph_hash": "hA", "family": "2s_none",
         "taxonomy_code": "H", "taxonomy_label": "canonical_duplicate",
         "run_rehit": True, "corpus_or_memory_rehit": False,
         "novel_canonical_graph": False, "violated_exclusion": True,
         "switched_family": False},
        {"attempt": 4, "temperature": 1.0, "top_p": 0.95,
         "prompt_word_count": 62, "exclusion_applied": True,
         "excluded_hashes_before": ["hA"], "excluded_families_before": ["2s_none"],
         "raw_response": "", "parsed": True, "valid": True,
         "validator_reasons": [], "graph_hash": "hKNOWN", "family": "2s_rc",
         "taxonomy_code": "I", "taxonomy_label": "valid_graph",
         "run_rehit": False, "corpus_or_memory_rehit": True,
         "novel_canonical_graph": False, "violated_exclusion": False,
         "switched_family": True},
        {"attempt": 5, "temperature": 1.0, "top_p": 0.95,
         "prompt_word_count": 66, "exclusion_applied": True,
         "excluded_hashes_before": ["hA", "hKNOWN"],
         "excluded_families_before": ["2s_none", "2s_rc"],
         "raw_response": "", "parsed": True, "valid": True,
         "validator_reasons": [], "graph_hash": "hB", "family": "3s_miller",
         "taxonomy_code": "I", "taxonomy_label": "valid_graph",
         "run_rehit": False, "corpus_or_memory_rehit": False,
         "novel_canonical_graph": True, "violated_exclusion": False,
         "switched_family": True},
    ]
    ladder_out = {
        "target_k": 3, "distinct": 3, "attempts": 5,
        "rungs_used": [{"temperature": 1.0}], "max_temperature": 1.0,
        "candidates": [
            {"canonical_graph_hash": "hA", "obj": objA, "family": "2s_none"},
            {"canonical_graph_hash": "hKNOWN", "obj": objA, "family": "2s_rc"},
            {"canonical_graph_hash": "hB", "obj": objB, "family": "3s_miller"}],
        "attempts_log": log}
    m = s4.measure_pool(ladder_out, target_k=3)

    assert m["attempts"] == 5
    assert m["raw_successful_generations"] == 3
    assert m["schema_success_rate"] == 4 / 5
    assert m["graph_construction_success_rate"] == 4 / 5
    assert m["validator_success_rate"] == 4 / 5
    assert m["valid_at_k"] == 1.0
    assert m["unique_at_k"] == 1.0
    assert m["run_rehit_count"] == 1 and m["run_rehit_rate"] == 1 / 5
    assert m["corpus_or_memory_rehit_count"] == 1
    assert m["corpus_or_memory_rehit_rate"] == 1 / 5
    assert m["novel_canonical_graph_count"] == 2
    assert m["novel_valid_yield"] == 2 / 5
    assert m["unique_yield"] == 3 / 5
    assert m["attempts_to_k_unique_valid"] == 5
    assert m["family_diversity_count"] == 3
    assert m["canonical_duplicate_rate"] == 1 / 4        # 1 / (1 run-rehit + 3 candidates)
    assert m["exclusion_applicable_attempts"] == 4
    assert m["exclusion_compliance_rate"] == 3 / 4         # 3 of 4 did not violate
    assert m["family_switch_rate"] == 2 / 4
    assert m["taxonomy_counts"] == {"empty_refusal_irrelevant_text": 1,
                                    "valid_graph": 3, "canonical_duplicate": 1}
    assert m["max_prompt_word_count"] == 66


def test_load_known_hashes_real_data_matches_stage3_finding():
    """Real end-to-end check: Stage 3 found the SFT corpus/RAG memory
    together realise exactly 5 canonical topology families (Stage3 report:
    'SFT generated all 5 realizable topology families'). This confirms
    load_known_hashes() sees the SAME 5, not an under/over-count from a
    key-name mismatch between corpus_diverse.json's `canonical_graph_hash`
    and the RAG memory's `variant` field."""
    import json

    from agentic_raptor.publication.preflight import CLEAN_RAG_PATH
    if not s4.BASE_CORPUS.is_file() or not CLEAN_RAG_PATH.is_file():
        import pytest
        pytest.skip("base corpus / clean RAG snapshot not present in this checkout")
    base_corpus = json.loads(s4.BASE_CORPUS.read_text(encoding="utf-8"))
    known = s4.load_known_hashes(base_corpus, CLEAN_RAG_PATH)
    assert len(known) == 5
