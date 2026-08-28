"""SIZING DETERMINISM REPAIR (2026-08-15).

Root cause of the long-documented sac_size irreproducibility ("three
identical calls, three different UGBWs"): CandidateFeatures.
pool_candidate_id was uuid4, and DPORanker.rank() breaks score ties on
that id -- with tied scores the measured candidate was a per-object
lottery, which chaotically diverged whole sizing trajectories and put an
arm-order-dependent luck floor under every ablation comparison (the
GATE2 "RAG negative" signal replayed as exactly this: identical proposal
sets and topology pairs, divergent sizing).

Zero ngspice here: measurement is faked deterministically; the real-
ngspice reproducibility check was run manually 2026-08-15 (identical)."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"

SPEC = {"spec_id": "det_test", "gain_target_db": 60.0,
       "phase_margin_target_deg": 55.0, "load_capacitance_pf": 100.0,
       "ugbw_target_hz": 1e5}


def _graph():
    if not CORPUS.is_file():
        pytest.skip("corpus_diverse.json not present in this checkout")
    from run_puct_ablation import _realise
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    return _realise(json.loads(corpus["records"][0]["response"]))


def _rank_graph():
    # the exact graph object type sac_size hands to candidate_features
    from agentic_raptor.corpus import TopologyRegistry
    from agentic_raptor.mb_sac.stage3d2 import V3
    return TopologyRegistry(V3).get_topology("topology_v2_0001").graph


# ---------------------------------------------------------------------------
# the id itself
# ---------------------------------------------------------------------------
def test_pool_candidate_id_is_content_hash_not_uuid():
    from agentic_raptor.dpo import schemas
    src = inspect.getsource(schemas)
    assert "import uuid" not in src      # the lottery is gone at the source
    assert "uuid.uuid4(" not in src
    assert "sha256" in src


def test_same_content_same_id_different_content_different_id():
    from agentic_raptor.mb_sac.spec_sizing import candidate_features
    g = _rank_graph()
    a = candidate_features(g, SPEC, [1.0, 2.0], None, 0.5)
    b = candidate_features(g, SPEC, [1.0, 2.0], None, 0.5)
    c = candidate_features(g, SPEC, [1.0, 2.1], None, 0.5)
    assert a.pool_candidate_id == b.pool_candidate_id
    assert a.pool_candidate_id != c.pool_candidate_id
    assert a.pool_candidate_id.startswith("pool-")


# ---------------------------------------------------------------------------
# end-to-end: the exact configuration that was non-reproducible
# ---------------------------------------------------------------------------
def _fake_measure(tid, graph, exe, out_dir, tag, costs, c_load_f=None, **kw):
    h = int(hashlib.sha256(repr(graph).encode()).hexdigest()[:8], 16) / 2**32
    return {"gain_db": 40 + 40*h, "pm_deg": 20 + 60*h, "ugbw_hz": 1e4*(1+6*h),
            "stable": True, "operating_point_valid": True, "idd_a": 1e-5,
            "power_w": 1e-4, "spice_converged": True, "c_load_f": c_load_f}


def test_sac_size_reproducible_with_ranker(tmp_path, monkeypatch):
    from agentic_raptor.mb_sac import spec_sizing as ss
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    monkeypatch.setattr(ss, "measure", _fake_measure)
    g = _graph()

    def run(tag):
        r = ss.sac_size("det_t", g, SPEC, None, tmp_path / tag, new_costs(),
                        budget=6, seed=17, persist=False,
                        use_surrogate=True, use_ranker=True)
        return [(round(x["reward"], 6),
                 tuple(round(v, 6) for v in x["knobs"].values()))
                for x in r["results"]]
    assert run("r1") == run("r2")


def test_early_stop_on_pass_banks_remaining_budget(tmp_path, monkeypatch):
    from agentic_raptor.mb_sac import spec_sizing as ss
    from agentic_raptor.topology_rl.stage3e2 import new_costs

    def passing_measure(tid, graph, exe, out_dir, tag, costs, c_load_f=None, **kw):
        return {"gain_db": 90.0, "pm_deg": 70.0, "ugbw_hz": 1e6,
                "stable": True, "op_valid": True, "idd_a": 1e-5,
                "power_w": 1e-4, "spice_converged": True, "c_load_f": c_load_f}
    monkeypatch.setattr(ss, "measure", passing_measure)
    g = _graph()
    r = ss.sac_size("es_t", g, SPEC, None, tmp_path / "a", new_costs(),
                    budget=8, seed=17, persist=False, early_stop_on_pass=True)
    assert r["spice_calls"] == 1                     # stopped at first pass
    assert r["calls_to_first_exact_pass"] == 1
    r2 = ss.sac_size("es_t", g, SPEC, None, tmp_path / "b", new_costs(),
                     budget=8, seed=17, persist=False)   # default: unchanged
    assert r2["spice_calls"] == 8


def test_run_pipeline_early_stop_is_opt_in():
    import inspect

    import run_raptor_v2 as v2
    sig = inspect.signature(v2.run_pipeline)
    assert sig.parameters["sizing_early_stop"].default is False
    # the pass-through now lives in the extracted single-branch worker
    # (_size_one_branch, 2026-08-16 agentic refactor)
    assert "early_stop_on_pass=sizing_early_stop" in inspect.getsource(
        v2._size_one_branch)
    assert "sizing_early_stop=sizing_early_stop" in inspect.getsource(
        v2.size_and_predict)


def test_ranker_rank_is_deterministic_for_identical_pools():
    from agentic_raptor.dpo import DPOConfig, DPORanker, FEATURE_DIM
    from agentic_raptor.mb_sac.spec_sizing import candidate_features
    g = _rank_graph()
    def pool():
        return [candidate_features(g, SPEC, [1.0 + i, 2.0], None, 0.5)
                for i in range(4)]
    rk = DPORanker(FEATURE_DIM, DPOConfig(enabled=True, seed=17))
    order1 = [f.pool_candidate_id for f, _, _ in rk.rank(pool())]
    order2 = [f.pool_candidate_id for f, _, _ in rk.rank(pool())]
    assert order1 == order2
