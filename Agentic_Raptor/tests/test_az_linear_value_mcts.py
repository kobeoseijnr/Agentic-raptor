"""LINEAR VALUE PROBE AS MCTS LEAF VALUE (quick test): mechanics tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/publication_v3/az_value_probe_gate/FROZEN_LINEAR_VALUE_V1.json"


def test_frozen_probe_reproduces_part_d_reference_exactly():
    if not FROZEN.is_file():
        pytest.skip("frozen probe not built")
    f = json.loads(FROZEN.read_text(encoding="utf-8"))
    v = f["verified_dev_metrics"]
    assert abs(v["dev_spearman"] - 0.3985) < 1e-3
    assert abs(v["dev_pairwise_ranking"] - 0.7128) < 1e-3
    assert len(f["weights"]) == len(f["feature_names"]) == 24


def test_hybrid_nets_replace_only_value_forward():
    from agentic_raptor.topology_rl.alphazero import (load_alphazero_nets,
                                                       require_promoted_az_checkpoint)
    from agentic_raptor.topology_rl.linear_value import (FrozenLinearValue,
                                                          hybrid_linear_value_nets)
    base = load_alphazero_nets(require_promoted_az_checkpoint(), seed=0)
    hybrid = hybrid_linear_value_nets(base, FrozenLinearValue())
    assert hybrid["policy_forward"] is base["policy_forward"]   # P(s,a) untouched
    assert hybrid["value_forward"] is not base["value_forward"]
    assert hybrid["value_mode"] == "FROZEN_LINEAR_VALUE_V1"


def test_linear_value_scores_are_clamped_and_finite():
    import json as _json

    from agentic_raptor.topology_rl import alphazero as az
    from agentic_raptor.topology_rl.linear_value import FrozenLinearValue
    corpus = _json.loads((ROOT / "artifacts/publication_v2/proposer_repair/"
                         "corpus_diverse.json").read_text(encoding="utf-8"))
    seen = {}
    for r in corpus["records"]:
        if r["canonical_graph_hash"] not in seen:
            seen[r["canonical_graph_hash"]] = r
        if len(seen) >= 2:
            break
    cands = [{"llm_proposal_id": f"p{i:02d}", "canonical_graph_hash": h,
             "canonical_family": r["topology_signature"],
             "obj": _json.loads(r["response"]), "source": "llm"}
            for i, (h, r) in enumerate(seen.items())]
    reg = az.LLMSeededEditRegistry(cands)
    spec = {"spec_id": "t", "gain_target_db": 60.0, "phase_margin_target_deg": 55.0,
           "load_capacitance_pf": 100.0, "ugbw_target_hz": 1e5}
    state = az.build_seed_root_state(spec, "t", reg.seed_ids[0], reg)
    probe = FrozenLinearValue()
    v = probe.score_state(state, reg)
    assert -1.0 <= v <= 1.0


def test_live_full_default_still_neural_one_root():
    import inspect

    import run_raptor_v2 as v2
    sig = inspect.signature(v2.run_pipeline)
    assert sig.parameters["search"].default == "bandit_top2"  # 2026-08-15 selector promotion
    src = inspect.getsource(v2.run_pipeline)
    # the linear value is reachable ONLY through the explicit opt-in
    # "combined_package" search mode (COMBINED REPAIR PACKAGE TEST) --
    # never on the "one_root" default path
    idx = src.index("linear_value")
    branch = src.index('search in ("combined_package", "combined_portfolio")')
    assert branch < idx, "linear_value must be confined to the opt-in branch"
