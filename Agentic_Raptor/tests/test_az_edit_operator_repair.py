"""EDIT-OPERATOR REPAIR task: tests for the causal-pair records, the
experimental repaired templates (and their isolation from live), and
NovelFeasible-path preservation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "artifacts/publication_v3/az_edit_causal_pairs/CAUSAL_PAIR_RECORDS.json"
REPAIRED = ROOT / "artifacts/publication_v3/az_edit_causal_pairs/REPAIRED_PAIR_RECORDS.json"


def _pairs():
    if not PAIRS.is_file():
        pytest.skip("causal pairs not run")
    return json.loads(PAIRS.read_text(encoding="utf-8"))


def test_causal_pairs_are_single_edit_same_spec_post_cload():
    recs = _pairs()
    for r in recs:
        if r["kind"] != "pair":
            continue
        assert r["child_tid"] == f"{r['seed_id']}~{r['edit']}"   # exactly ONE edit
        assert r["child_term"]["electrical_environment_version"] == "POST_CLOAD_FIX_V1"
        assert (r["child_term"]["requested_c_load_f"]
                == r["child_term"]["simulated_c_load_f"])
        assert abs(r["delta_z"] - (r["child_term"]["z"] - r["parent_term"]["z"])) < 1e-3
        assert r["parent_hash"] != r["child_hash"]


def test_spice_budget_respected():
    recs = _pairs()
    n_new = sum(1 for r in recs if r["kind"] in ("parent", "pair"))
    n_rep = (len(json.loads(REPAIRED.read_text(encoding="utf-8")))
             if REPAIRED.is_file() else 0)
    assert n_new + n_rep <= 20


def test_replace_compensation_measured_helpful():
    recs = _pairs()
    rc = [r for r in recs if r["kind"] == "pair"
         and r["edit"] == "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE"]
    assert len(rc) >= 2
    assert all(r["delta_z"] > 0.097 for r in rc)      # beyond 3x noise floor
    assert any(r["fail_to_pass"] for r in rc)          # the causal fail->pass


def test_live_edit_templates_untouched_by_repair_module():
    from agentic_raptor.topology_rl.stage3e2_edits import EDIT_TEMPLATES
    from agentic_raptor.topology_rl.stage3e2_edits_repaired import (
        REPAIRED_EDIT_TEMPLATES, add_output_stage_repaired,
        add_verified_stage_repaired)
    # importing the repaired module must not mutate the live dict
    assert EDIT_TEMPLATES["ADD_VERIFIED_STAGE"]["fn"] is not add_verified_stage_repaired
    assert EDIT_TEMPLATES["ADD_SUPPORTED_OUTPUT_STAGE"]["fn"] is not add_output_stage_repaired
    assert REPAIRED_EDIT_TEMPLATES["ADD_VERIFIED_STAGE"]["fn"] is add_verified_stage_repaired
    # untouched operators are byte-identical entries
    for k in ("REPLACE_SUPPORTED_COMPENSATION_STRUCTURE", "CONNECT_VERIFIED_FEEDBACK_PATH",
             "REPLACE_LOAD_WITH_COMPATIBLE_BLOCK", "REPLACE_STAGE_WITH_COMPATIBLE_BLOCK"):
        assert REPAIRED_EDIT_TEMPLATES[k] is EDIT_TEMPLATES[k]


def test_apply_edit_default_templates_unchanged():
    import inspect

    from agentic_raptor.topology_rl.stage3e2_edits import apply_edit
    sig = inspect.signature(apply_edit)
    assert sig.parameters["templates"].default is None   # live default = EDIT_TEMPLATES


def test_registry_default_has_no_template_override():
    import json as _json

    from agentic_raptor.topology_rl import alphazero as az
    corpus = _json.loads((ROOT / "artifacts/publication_v2/proposer_repair/"
                         "corpus_diverse.json").read_text(encoding="utf-8"))
    seen = {}
    for r in corpus["records"]:
        if r["canonical_graph_hash"] not in seen:
            seen[r["canonical_graph_hash"]] = r
        if len(seen) >= 1:
            break
    cands = [{"llm_proposal_id": "p00", "canonical_graph_hash": h,
             "canonical_family": r["topology_signature"],
             "obj": _json.loads(r["response"]), "source": "llm"}
            for h, r in seen.items()]
    reg = az.LLMSeededEditRegistry(cands)
    assert reg._edit_templates is None


def test_repaired_pairs_recorded_honestly_as_failures():
    if not REPAIRED.is_file():
        pytest.skip("repaired pairs not run")
    recs = json.loads(REPAIRED.read_text(encoding="utf-8"))
    assert len(recs) == 4
    # the honest negative result: every repaired append-stage child failed
    assert all(r["child_term"]["z"] == -1.0 for r in recs)


def test_novel_feasible_paths_still_recorded():
    p = ROOT / "artifacts/publication_v3/az_edit_operator_audit/EDIT_OPERATOR_AUDIT.json"
    if not p.is_file():
        pytest.skip("audit not present")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["n_novel_feasible_paths"] >= 3
