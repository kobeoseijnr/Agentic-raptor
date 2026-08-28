"""A9 tier-2 profile (2026-08-22): a separate lineage pair on a MIXED
distribution (stock TRAIN + tier2_train) running the production tier-2
proposer, with tier2_heldout protected from harvest and the SFT generation
chain re-pointed at its own G0."""
import inspect


def test_profiles_are_separate_roots_and_adapters():
    import run_a9_generations as a9
    st, t2 = a9.PROFILES["stock"], a9.PROFILES["tier2"]
    assert t2["root"] != st["root"] and str(t2["root"]).endswith("tier2")
    assert "tier2_mixed" in str(t2["adapter"])
    assert t2["sft_generations_root"] != st["sft_generations_root"]


def test_tier2_heldout_is_protected_from_harvest():
    import run_a9_generations as a9
    from agentic_raptor.publication.tier2 import load_tier2_specs
    ids = a9.tier2_protected_ids()
    assert ids == {s["spec_id"] for s in load_tier2_specs("tier2_heldout")}
    assert len(ids) >= 18
    assert "tier2_protected_ids()" in inspect.getsource(a9.run_generation)


def test_mixed_job_list_and_run_ids():
    import run_a9_generations as a9
    src = inspect.getsource(a9.run_generation)
    assert 'jobs += [("tier2_train", tier2_start + i)' in src
    assert 'run_id = f"g{st.generation_id}_{lineage}_{split}_{idx}"' in src


def test_trainer_profile_repoints_chain_and_downstream():
    import train_sft_self_improvement as tr
    src = inspect.getsource(tr.main)
    assert "si.configure_profile(" in src
    assert 'downstream_split = "tier2_train"' in src
    assert "harvested_keys" in src          # (split, idx) keys, no cross-split collision


def test_configure_profile_repoints_generation_dir(tmp_path):
    from agentic_raptor.selfimprove_v2 import sft_self_improvement as si
    keep = (si.GENERATIONS_ROOT, si.BASE_ADAPTER, si.BASE_CORPUS)
    try:
        si.configure_profile(tmp_path / "gens", tmp_path / "adapter", tmp_path / "corpus.json")
        assert si.generation_dir("G1") == tmp_path / "gens" / "G1"
    finally:
        si.configure_profile(*keep)


def test_sft_admission_accepts_tier2_train_and_rejects_eval_splits():
    """2026-08-23: tier2_train is a TRAINING split for the SFT queue; the
    sealed evaluation splits stay rejected."""
    from agentic_raptor.selfimprove_v2 import streams
    assert streams.TRAIN_SPLITS == frozenset({"train", "tier2_train"})
    hv = {"branches": {"A": {"authoritative": None, "design": {}}}, "spec": {"spec_id": "x"}}
    for sp in ("train", "tier2_train"):
        r = streams.sft_admission_reasons(hv, "A", split=sp, protected_ids=set())
        assert not any(x.startswith("not_train_split") for x in r), sp
    for sp in ("heldout", "blindtest", "tier2", "tier2_heldout"):
        r = streams.sft_admission_reasons(hv, "A", split=sp, protected_ids=set())
        assert any(x.startswith("not_train_split") for x in r), sp


def test_bandit_refit_indexes_tier2_corpus_and_pair_objs():
    import inspect, run_a9_generations as a9
    src = inspect.getsource(a9._bandit_refit_and_gate)
    assert "corpus_tier2_mixed.json" in src
    assert 'graphs.setdefault(rec["topology_hash"], ("obj", rec["obj"]))' in src
    assert '"obj": d.get("obj")' in inspect.getsource(a9.bandit_stream_update)


def test_g0_tier2_pair_designs_resolve_by_canonical_hash():
    """The 24 tier-2 designs harvested in the tier-2 profile's G0 must all be
    realizable from the tier-2 corpus index (backfill without obj)."""
    import json
    from pathlib import Path
    pairs = Path("artifacts/publication_v3/a9_generations/tier2/data/adaptive/gen_000/streams/ranker_pairs.jsonl")
    if not pairs.is_file():
        return
    m = json.loads(Path("artifacts/publication_v3/tier2/corpus_tier2_mixed.json").read_text(encoding="utf-8"))
    hs = {r["canonical_graph_hash"] for r in m["records"]}
    rows = [json.loads(l) for l in pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    t2 = [r for r in rows if str((r.get("spec") or {}).get("spec_id", "")).startswith("t2_")]
    for r in t2:
        for side in "AB":
            assert r[f"design_{side}"]["canonical_graph_hash"] in hs


def test_trace_prefix_includes_profile():
    """tier2-profile traces must not collide with stock-profile traces."""
    import inspect, run_a9_generations as a9
    assert 'f"A9{profile}_{lineage}_g{st.generation_id}"' in inspect.getsource(a9.run_generation)
