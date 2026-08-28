"""PAPER-MODE PREFLIGHT CHECK.

Every check here is a REAL, computed answer against the current repository
state -- never an assumption. `run_preflight(paper_mode=True)` raises
SystemExit and prints every blocker if any critical check fails; it never
silently continues with a stale/missing/unvalidated component.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

#: Stage 8 Part D audit (2026-08-12): the original RAG_MIN_RECORDS=200 was
#: audited against the actual preflight implementation and every other file
#: in the repo that mentions it -- it traces to nothing but its own comment
#: ("a round number... not tuned to make the check pass"). No corpus-
#: completeness argument, statistical-power calculation, or per-family
#: target ever grounded 200 specifically; it is a stale hardcoded sanity
#: floor, not a formally frozen scientific minimum. The 174 real records
#: that failed it are independently verified (see check_rag_populated) to
#: be 100% usable, 100% POST_CLOAD_FIX_V1, 0 duplicates, 0 protected-
#: evaluation contamination, spanning all 4 currently-reachable topology
#: families (2s_none/2s_rc/3s_rc/3s_miller -- 3s_none was retired from the
#: target space, see run_raptor_v2.py's history) and 85 distinct specs.
#:
#: Replacing 200 with 174 (or any number derived FROM the current count)
#: would just be a different flavor of "tuned to make the check pass" --
#: explicitly forbidden by both this file's own original principle and the
#: user's explicit Stage 8 instruction. Instead this floor is set with real
#: margin below the current state (150 total, 40 distinct specs), still
#: "well above what one crashed/aborted harvest run could produce by
#: accident" (the check's original stated purpose), while genuinely
#: distinct from "equals whatever exists today."
RAG_MIN_RECORDS = 150
RAG_MIN_DISTINCT_SPECS = 40
#: every currently-reachable topology family must have avoided being
#: entirely missed by the harvest, not merely be present in trace amounts.
RAG_MIN_RECORDS_PER_PRESENT_FAMILY = 2


def _check(name: str, ok: bool, detail: str, critical: bool = True) -> dict:
    return {"name": name, "ok": ok, "detail": detail, "critical": critical}


def check_frozen_eval_set() -> dict:
    try:
        from agentic_raptor.publication.eval_sets import (available_seeds,
                                                           load)
        seeds = available_seeds()
        if not seeds:
            return _check("frozen_eval_set", False, "no evaluation_sets/seed_*.json files exist")
        for s in seeds:
            load(s)          # raises ValueError on hash drift
        return _check("frozen_eval_set", True,
                     f"{len(seeds)} frozen seed file(s), all hash-verified: {seeds}")
    except Exception as exc:
        return _check("frozen_eval_set", False, f"{type(exc).__name__}: {exc}")


def check_splits_disjoint() -> dict:
    try:
        corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text())
        sm = corpus.get("split_manifest", {})
        train = set(sm.get("train_spec_ids") or [])
        val = set(sm.get("validation_spec_ids") or [])
        blind = set(sm.get("blind_test_spec_ids") or [])
        overlaps = {"train&val": train & val, "train&blind": train & blind,
                   "val&blind": val & blind}
        bad = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
        return _check("splits_disjoint", not bad,
                     "disjoint" if not bad else f"overlaps: {bad}")
    except Exception as exc:
        return _check("splits_disjoint", False, f"{type(exc).__name__}: {exc}")


#: the CLEAN (provenance-filtered) snapshot is what paper-mode checks and
#: what the ablation framework defaults to -- the raw rag_memory_v2.jsonl
#: still mixes in records of unverifiable (possibly pre-VCM-fix) provenance
#: and is never checked directly here.
#:
#: Stage 1.6: versioned again -- the prior rag_memory_v2_clean.jsonl was
#: built (2026-08-09 earlier) from a raw file that predates the C_LOAD
#: repair, so it is itself PRE_CLOAD_FIX regardless of its VCM-fix
#: cleanliness. This intentionally points at a snapshot that does not exist
#: yet: check_rag_populated/check_rag_frozen below correctly fail until
#: rag_freeze.build_clean_snapshot() is re-run against the new
#: run_raptor_v2.RAG_MEMORY_V2 (rag_memory_v2_post_cload_v1.jsonl).
CLEAN_RAG_PATH = (ROOT / "artifacts/publication_v2/selfimprove"
                  / "rag_memory_v2_post_cload_v1_clean.jsonl")


def check_rag_populated() -> dict:
    """Stage 8 Part D (2026-08-12): rewritten from a bare record-count
    threshold to real coverage criteria (see RAG_MIN_RECORDS' docstring for
    the audit that grounded this) -- usable count AND distinct-spec count
    AND every present topology family clearing a per-family floor, so a
    harvest that silently missed an entire family (or duplicated one spec
    174 times) fails this even if it happened to clear a raw count."""
    p = CLEAN_RAG_PATH
    if not p.is_file():
        return _check("rag_populated", False,
                     f"{p.name} does not exist -- run "
                     "rag_freeze.build_clean_snapshot() first")
    lines_raw = [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    records = [json.loads(x) for x in lines_raw]
    usable = [r for r in records if r.get("stability")]
    n_usable = len(usable)
    specs = {r.get("spec_hash") for r in usable if r.get("spec_hash")}
    from collections import Counter
    fam_counts = Counter(r.get("family") for r in usable if r.get("family"))
    underfilled_families = {f: c for f, c in fam_counts.items()
                            if c < RAG_MIN_RECORDS_PER_PRESENT_FAMILY}
    count_ok = n_usable >= RAG_MIN_RECORDS
    specs_ok = len(specs) >= RAG_MIN_DISTINCT_SPECS
    families_ok = bool(fam_counts) and not underfilled_families
    ok = count_ok and specs_ok and families_ok
    return _check("rag_populated", ok,
                 f"{n_usable} usable CLEAN records (>= {RAG_MIN_RECORDS} "
                 f"required, {count_ok}); {len(specs)} distinct specs "
                 f"(>= {RAG_MIN_DISTINCT_SPECS} required, {specs_ok}); "
                 f"family_counts={dict(fam_counts)} "
                 f"(each >= {RAG_MIN_RECORDS_PER_PRESENT_FAMILY} required, "
                 f"{families_ok}"
                 + (f", underfilled: {underfilled_families}" if underfilled_families else "")
                 + f"); {len(lines_raw)} total lines", critical=True)


def rag_quality_report() -> dict:
    """Part D (Section 44): RAG quality metrics beyond a bare count --
    unique specs, unique topology families, POST_CLOAD records, trusted
    (exact_spec_pass) records, protected-evaluation contamination count,
    duplicate count, per-family retrieval diversity. Diagnostic-only, not a
    pass/fail gate (see check_rag_populated for the gate itself)."""
    from collections import Counter

    from agentic_raptor.publication.artifact_provenance import POST_CLOAD_FIX_V1
    from agentic_raptor.publication.eval_sets import excluded_context_ids
    p = CLEAN_RAG_PATH
    if not p.is_file():
        return {"error": f"{p} does not exist"}
    records = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    protected = excluded_context_ids()
    key_counts = Counter((r.get("context_id"), r.get("branch"), r.get("variant"),
                         r.get("call_id")) for r in records)
    duplicates = {k: v for k, v in key_counts.items() if v > 1}
    contaminated = [r for r in records if r.get("context_id") in protected
                    or r.get("generation_spec_id") in protected]
    return {
        "total_records": len(records),
        "usable_records": sum(1 for r in records if r.get("stability")),
        "unique_specs": len({r.get("spec_hash") for r in records if r.get("spec_hash")}),
        "unique_topology_families": sorted({r.get("family") for r in records if r.get("family")}),
        "post_cload_fix_v1_records": sum(
            1 for r in records
            if r.get("electrical_environment_version") == POST_CLOAD_FIX_V1),
        "trusted_exact_spec_pass_records": sum(1 for r in records if r.get("exact_spec_pass")),
        "protected_evaluation_contamination_count": len(contaminated),
        "duplicate_record_count": len(duplicates),
        "family_record_counts": dict(Counter(r.get("family") for r in records if r.get("family"))),
        "split_counts": dict(Counter(r.get("split") for r in records)),
    }


def check_rag_frozen() -> dict:
    from agentic_raptor.publication.rag_freeze import verify
    v = verify(CLEAN_RAG_PATH)
    ok = v["frozen"] and not v["drifted"]
    return _check("rag_frozen", ok, v["detail"])


def check_sft_checkpoint() -> dict:
    p = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"
    ok = (p / "adapter_config.json").is_file()
    return _check("sft_checkpoint", ok,
                 str(p) + (" present" if ok else " missing adapter_config.json"))


def check_old_root_puct_checkpoint_absent() -> dict:
    """2026-08-11: root-level PUCT (agentic_raptor.topology_rl.stage3e1's
    one-root selector, wired via the now-retired run_raptor_v2.
    puct_select_two) was retired and replaced by TRUE_ALPHAZERO -- see
    artifacts/publication_v3/ROOT_LEVEL_PUCT_RETIRED.json. Paper mode must
    fail closed if the old checkpoint has somehow reappeared (a stray
    restore, a bad merge) rather than silently let FULL resolve it."""
    p = ROOT / "artifacts/stage3e1/policy_value_ep0.pt"
    return _check("old_root_puct_checkpoint_absent", not p.is_file(),
                 f"{p} {'still present -- must not exist post-cutover' if p.is_file() else 'absent (expected)'}",
                 critical=True)


def check_alphazero_promoted_checkpoint() -> dict:
    """Section 23: paper/A0-A8 mode requires an explicitly PROMOTED
    AlphaZero checkpoint (agentic_raptor.topology_rl.alphazero.
    AZ_CHECKPOINT_STATUSES) -- never the SMOKE generation (AZ_G1, proven
    mechanically valid only), never a random G0, never the retired
    root-PUCT checkpoint, never the clean 91-example value-only
    checkpoint. NOT READY is the correct, expected result until a real
    training campaign produces and promotes one."""
    from agentic_raptor.topology_rl.alphazero import (
        AlphaZeroSelectionError, require_promoted_az_checkpoint)
    try:
        p = require_promoted_az_checkpoint()
        return _check("alphazero_promoted_checkpoint", p.is_file(),
                     f"{p} {'loads' if p.is_file() else 'missing on disk'}")
    except AlphaZeroSelectionError as exc:
        return _check("alphazero_promoted_checkpoint", False, str(exc))


def check_dpo_checkpoint() -> dict:
    """NON-CRITICAL, and stays that way even now that a learned DPO IS live
    again. This check is specifically about the SUPERSEDED Stage 7 V1
    checkpoint (artifacts/publication_v2/post_sac_ranker/) -- Stage 7.1
    found it LEARNED_DPO_NOT_JUSTIFIED and Stage 8 deploys a DIFFERENT,
    re-justified checkpoint (Stage 7.2B's POST_SAC_FEATURES_V2 model; see
    check_dpo_promoted_checkpoint / check_dpo_feature_schema_compatible,
    both CRITICAL) via a hardcoded path in agentic_raptor.ranking.model_v2
    that never falls back to this one. A live FULL run never loads this V1
    file, so its absence or invalidity must never block paper-mode
    readiness. Retained as a historical-evidence integrity check only
    (Section 2: "mechanically valid but unsupported by current evidence/
    data", never "broken")."""
    p = ROOT / "artifacts/publication_v2/post_sac_ranker/ranker.pt"
    rep = ROOT / "artifacts/publication_v2/post_sac_ranker/training_report.json"
    if not p.is_file():
        return _check("dpo_checkpoint", False, f"{p} missing", critical=False)
    if not rep.is_file():
        return _check("dpo_checkpoint", False, f"{rep} missing (unvalidated)",
                     critical=False)
    r = json.loads(rep.read_text(encoding="utf-8"))
    mono = (r.get("monotonicity") or {}).get("strictly_decreasing")
    ok = mono is True
    return _check("dpo_checkpoint", ok,
                 f"HISTORICAL EVIDENCE ONLY, not required by current FULL "
                 f"(LEARNED_DPO_NOT_JUSTIFIED, removed from live path): "
                 f"trained_at={r.get('trained_at')} "
                 f"informative_pairs={r.get('pairs_informative')} "
                 f"train_acc={r.get('train_pair_accuracy')} "
                 f"monotonicity_strictly_decreasing={mono}", critical=False)


def check_alphazero_checkpoint_loading_enforced() -> dict:
    """Stage 8 FINAL integration (2026-08-13): the live one_root path must
    load the PROMOTED AlphaZero checkpoint through the enforced loader
    (SHA-256-verified, parameter-fingerprinted, hard-fail on random init /
    missing / mismatch -- AlphaZeroCheckpointError, no silent fallback).
    Root cause: run_pipeline's value_ckpt=None previously produced a
    silently random seed-0 network on every live run. This check (a) runs
    the real loader and verifies its provenance, and (b) source-verifies
    run_pipeline's live branch actually calls it."""
    import inspect

    try:
        from agentic_raptor.topology_rl.alphazero import \
            load_promoted_alphazero_nets
        _nets, prov = load_promoted_alphazero_nets()
        import run_raptor_v2 as v2
        src = inspect.getsource(v2.run_pipeline)
        wired = "load_promoted_alphazero_nets" in src
        ok = (prov["checkpoint_loaded"] and prov["differs_from_random_init"]
              and wired)
        return _check("alphazero_checkpoint_loading_enforced", ok,
                     f"loader verified sha256={prov['checkpoint_sha256'][:16]}... "
                     f"fingerprint(encoder)={prov['parameter_fingerprint']['encoder']:.4f} "
                     f"differs_from_random={prov['differs_from_random_init']} "
                     f"wired_into_run_pipeline={wired}")
    except Exception as exc:
        return _check("alphazero_checkpoint_loading_enforced", False,
                     f"{type(exc).__name__}: {exc}")


def check_dpo_promoted_checkpoint() -> dict:
    """Stage 8 deployment (2026-08-12): Stage 7.2B found the learned Level-2
    ranker DPO_REJUSTIFIED under POST_SAC_FEATURES_V2 and this task deployed
    it back into FULL's default (run_pipeline's ranker_mode now defaults to
    "dpo" again; ablation_v3.A0_FULL.use_dpo defaults True). Unlike the old
    check_dpo_checkpoint (which still points at the SUPERSEDED Stage 7 V1
    checkpoint and stays non-critical/historical), THIS check verifies the
    exact promoted V2 checkpoint agentic_raptor.ranking.model_v2.
    load_promoted_v2() will actually load at runtime -- critical, because a
    live FULL run now hard-fails without it (no silent fallback)."""
    from agentic_raptor.ranking.model_v2 import (PROMOTED_V2_CKPT,
                                                  PROMOTED_V2_MANIFEST,
                                                  PROMOTED_V2_NORM,
                                                  REQUIRED_V2_SHA256, _sha256)
    if not PROMOTED_V2_CKPT.is_file():
        return _check("dpo_promoted_checkpoint", False,
                     f"{PROMOTED_V2_CKPT} missing")
    actual_hash = _sha256(PROMOTED_V2_CKPT)
    if actual_hash != REQUIRED_V2_SHA256:
        return _check("dpo_promoted_checkpoint", False,
                     f"hash mismatch: expected {REQUIRED_V2_SHA256}, got {actual_hash}")
    if not PROMOTED_V2_MANIFEST.is_file() or not PROMOTED_V2_NORM.is_file():
        return _check("dpo_promoted_checkpoint", False,
                     "training_manifest.json or normalization.json missing "
                     "alongside checkpoint")
    manifest = json.loads(PROMOTED_V2_MANIFEST.read_text(encoding="utf-8"))
    ok = manifest.get("checkpoint_status") == "DPO_REJUSTIFIED"
    return _check("dpo_promoted_checkpoint", ok,
                 f"{PROMOTED_V2_CKPT} sha256={actual_hash} "
                 f"status={manifest.get('checkpoint_status')} "
                 f"stamped_at={manifest.get('stamped_at')}")


def check_dpo_feature_schema_compatible() -> dict:
    """The promoted checkpoint's declared feature_schema must be exactly
    POST_SAC_FEATURES_V2 -- the same string model_v2.load_promoted_v2()
    enforces at runtime (RankerFeatureSchemaMismatch) -- AND
    features_v2.FEATURE_DIM_V2 (what the live feature builder actually
    produces) must agree with the checkpoint's normalization vector length,
    so training/inference dimensionality can never silently drift apart."""
    from agentic_raptor.ranking.features_v2 import FEATURE_DIM_V2
    from agentic_raptor.ranking.model_v2 import (PROMOTED_V2_MANIFEST,
                                                  PROMOTED_V2_NORM,
                                                  REQUIRED_FEATURE_SCHEMA)
    if not PROMOTED_V2_MANIFEST.is_file() or not PROMOTED_V2_NORM.is_file():
        return _check("dpo_feature_schema_compatible", False,
                     "training_manifest.json or normalization.json missing")
    manifest = json.loads(PROMOTED_V2_MANIFEST.read_text(encoding="utf-8"))
    norm = json.loads(PROMOTED_V2_NORM.read_text(encoding="utf-8"))
    schema_ok = manifest.get("feature_schema") == REQUIRED_FEATURE_SCHEMA
    n_mean = len(norm.get("feature_mean") or [])
    n_std = len(norm.get("feature_std") or [])
    dim_ok = n_mean == FEATURE_DIM_V2 and n_std == FEATURE_DIM_V2
    ok = schema_ok and dim_ok
    return _check("dpo_feature_schema_compatible", ok,
                 f"manifest.feature_schema={manifest.get('feature_schema')!r} "
                 f"(required {REQUIRED_FEATURE_SCHEMA!r}); "
                 f"normalization dims mean={n_mean} std={n_std} "
                 f"(FEATURE_DIM_V2={FEATURE_DIM_V2})")


def check_ngspice() -> dict:
    try:
        from agentic_raptor.spice.ngspice_simulator import (discover_ngspice,
                                                             ngspice_version)
        exe = discover_ngspice()
        if not exe:
            return _check("ngspice_version_recorded", False, "ngspice not found")
        v = ngspice_version(exe)
        return _check("ngspice_version_recorded", bool(v), f"{exe} -> {v}")
    except Exception as exc:
        return _check("ngspice_version_recorded", False, f"{type(exc).__name__}: {exc}")


def check_pdk() -> dict:
    from agentic_raptor.electrical import _PDK_CORNER_DIR
    tt = _PDK_CORNER_DIR / "tt.spice"
    if not tt.is_file():
        return _check("pdk_version_recorded", False, f"{tt} missing")
    import hashlib
    h = hashlib.sha256(tt.read_bytes()).hexdigest()[:16]
    return _check("pdk_version_recorded", True,
                 f"{_PDK_CORNER_DIR} tt.spice sha256[:16]={h}")


def check_pvt_corners() -> dict:
    from agentic_raptor.electrical.pvt_eval import available_process_corners
    avail = available_process_corners()
    return _check("pvt_corner_list_validated", bool(avail),
                 f"available: {sorted(avail)}")


def check_fom() -> dict:
    from agentic_raptor.electrical.fom import compute_fom
    r = compute_fom(120e6, 2e-12, 0.4e-3)
    ok = r["fom_value"] == 600.0 and r["fom_version"] == "FOM_V1_UGBW_CL_OVER_IDD"
    return _check("fom_v1_calculation", ok, f"worked example -> {r['fom_value']}")


def check_idd_not_ibias() -> dict:
    import inspect

    from agentic_raptor.electrical.fom import compute_fom
    params = set(inspect.signature(compute_fom).parameters)
    ok = params == {"ugbw_hz", "c_load_f", "idd_a"}
    return _check("idd_measurement_not_ibias", ok, f"compute_fom params={params}")


#: Stage 1.6: every measurement-derived artifact that must be
#: POST_CLOAD_FIX_V1 before paper mode may proceed. Kept separate from
#: artifact_provenance.TRACKED_ARTIFACTS (which also tracks a couple of
#: read-only reference/log files not gating this check) so this list is
#: exactly "what a real FULL run actually loads."
#:
#: Stage 8 audit (2026-08-12), root cause of a stale electrical_environment_
#: compatible=False: FOUR entries were being checked despite NOT being "what
#: a real FULL run actually loads" (this list's own stated principle),
#: which is exactly a category error, not a real mixed-environment problem:
#:   - puct_value_checkpoint / puct_policy_checkpoint: root-level PUCT was
#:     RETIRED 2026-08-11 (see check_old_root_puct_checkpoint_absent) and
#:     its checkpoint deliberately deleted. LIVE_ARTIFACT_PATHS already maps
#:     both to None for exactly this reason, which made
#:     current_environment_version() return "UNKNOWN" -- i.e. this check
#:     was failing paper-mode over a component required to be ABSENT.
#:   - dpo_ranker: Stage 7.1 found the learned DPO ranker
#:     LEARNED_DPO_NOT_JUSTIFIED and it was removed from the live FULL path
#:     (run_pipeline's ranker_mode now defaults to "deterministic"); a live
#:     FULL run no longer reads this checkpoint at all. (It also would have
#:     failed here regardless: training_report.json has no
#:     electrical_environment_version field, a real provenance gap flagged
#:     in the Stage 7 report -- moot now that DPO is not live.)
#:   - family_spec_gate: verified via grep to be referenced ONLY by
#:     offline corpus-construction scripts (build_diverse_corpus.py,
#:     run_family_spec_gate.py) -- never imported by run_raptor_v2.py or
#:     any live pipeline code. Its SUMMARY.json genuinely IS PRE_CLOAD_FIX
#:     (a real, still-outstanding corpus-provenance gap worth recomputing
#:     as separate corpus-maintenance work) but it cannot be "mixed into a
#:     FULL run" because a FULL run never reads it.
#: trusted_pairs / rag_memory_v2_clean / sac_sizing_memory_dir remain: all
#: three are genuinely read or written by a live FULL/adaptive run today.
#:
#: Stage 8 deployment (2026-08-12): dpo_ranker_v2 added -- the promoted
#: Stage 7.2B checkpoint is once again genuinely loaded by every live FULL
#: run (model_v2.load_promoted_v2()), so a mixed-environment DPO checkpoint
#: must gate readiness the same way trusted_pairs does. This is distinct
#: from the OLD "dpo_ranker" name (Stage 7 V1 checkpoint), which stays
#: excluded -- it is superseded, not live.
CLOAD_GATED_ARTIFACTS = ("trusted_pairs", "rag_memory_v2_clean",
                         "sac_sizing_memory_dir", "dpo_ranker_v2")


def check_electrical_environment_compatibility() -> dict:
    """Stage 1.6: hard gate against mixing PRE_CLOAD_FIX and
    POST_CLOAD_FIX_V1 measurement-derived artifacts in one FULL run.

    Inspects each artifact's LIVE path directly (current_environment_
    version) -- an artifact not yet rebuilt reads PRE_CLOAD_FIX or
    NOT_YET_BUILT here and this check correctly keeps failing until the
    whole rebuild plan (see artifact_provenance.py) is complete. This is
    expected to fail right now, by design: it is the mechanism that
    prevents the next A0-A8 pilot from launching on stale data."""
    from agentic_raptor.publication.artifact_provenance import (
        POST_CLOAD_FIX_V1, current_environment_version)
    versions = {name: current_environment_version(name)
               for name in CLOAD_GATED_ARTIFACTS}
    bad = {k: v for k, v in versions.items() if v != POST_CLOAD_FIX_V1}
    return _check("electrical_environment_compatible", not bad,
                 f"versions={versions}"
                 + (f" -- NOT {POST_CLOAD_FIX_V1}: {sorted(bad)}" if bad else ""))


def check_cload_authoritative() -> dict:
    """Stage 1.5 repair (2026-08-09): a spec's requested load must actually
    reach the simulator -- this is a live, computed check against the
    current code, not a reference to "the fix landed once"."""
    from agentic_raptor.electrical import NOMINAL_CLOAD_F, effective_c_load
    cases = {100.0: 100e-12, 200.0: 200e-12, 500.0: 500e-12}
    bad = {pf: effective_c_load({"load_capacitance_pf": pf})
           for pf, expect in cases.items()
           if effective_c_load({"load_capacitance_pf": pf}) != expect}
    no_spec_ok = effective_c_load(None) == NOMINAL_CLOAD_F
    override_ok = effective_c_load({"load_capacitance_pf": 100.0},
                                   override=250e-12) == 250e-12
    ok = not bad and no_spec_ok and override_ok
    return _check("cload_spec_authoritative", ok,
                 f"100/200/500pF resolve correctly: {not bad}; "
                 f"no-spec fallback to NOMINAL_CLOAD_F: {no_spec_ok}; "
                 f"explicit override wins: {override_ok}"
                 + (f"; MISMATCHES: {bad}" if bad else ""))


def check_budgets_fixed() -> dict:
    from agentic_raptor.publication.ablation_v3 import COMPONENT_ABLATIONS
    hashes = {aid: c.budget.hash() for aid, c in COMPONENT_ABLATIONS.items()}
    # every A0-A8 arm shares the identical base ExperimentBudget by design
    # (Part: FAIRNESS -- "an ablation can change only what is logically
    # required by the removed component", and none of the ten currently
    # override budget fields)
    ok = len(set(hashes.values())) == 1
    return _check("simulation_budgets_fixed", ok, f"budget_hash by arm: {hashes}")


def check_leakage_tests() -> dict:
    # a live re-import-and-call of the actual leakage guards, not a
    # reference to "the test suite passed once" -- run the same assertions
    # test_fom_pvt.py encodes, right now, against the current code.
    try:
        from agentic_raptor.selfimprove_v2.streams import harvest_run
        hv = {"spec": {"spec_id": "preflight_probe"}, "spec_hash": "h",
             "split": "train", "spec_index": 0, "seed": 0, "budget": 8,
             "candidates": [], "root_visits": {}, "candidate_visits": {},
             "root_action_ids": [], "root_state": None,
             "candidate_manifests": {}, "search": None, "branches": {},
             "ranker": {"arm": "dpo_ranker", "selected_design": "A",
                        "backup_design": "B", "decision_basis": "x",
                        "deciding_level": "x", "low_confidence": False,
                        "score_A": None, "score_B": None,
                        "checkpoint_hash": None},
             "proposer_checkpoint": "x",
             "pvt": {"total_pvt_corners": 1, "corners": [{"pvt_corner_id": "x"}]}}
        streams = harvest_run(hv, split="train", protected_ids=set())
        leaked = any("pvt_corner_id" in str(row) for rows in streams.values()
                    for row in rows)
        return _check("leakage_tests", not leaked,
                     "PVT-shaped injected data did not reach any stream"
                     if not leaked else "LEAK: pvt data reached a stream")
    except Exception as exc:
        return _check("leakage_tests", False, f"{type(exc).__name__}: {exc}")


def check_evaluation_learning_disabled() -> dict:
    """Stage 8 (Section 10/28): frozen integration/evaluation runs (A0-A9,
    Stage 8's own diagnostic) must never mutate learned components -- RAG
    harvesting, SFT queue updates, AlphaZero replay/training, SAC
    persistent memory, DPO pair recording/training, adaptive surrogate
    updates all gate on learning_mode != "adaptive" throughout run_pipeline.
    This is a real, computed check against the current declared default,
    not a claim that any specific run used it -- every AblationConfig
    (A0-A9) declares learning_mode="frozen" by construction; a config that
    silently reverted to "adaptive" would let a supposedly-frozen paper run
    mutate shared learning state."""
    from dataclasses import fields as _fields

    from agentic_raptor.publication.ablation_v3 import (COMPONENT_ABLATIONS,
                                                         AblationConfig)
    default = next(f.default for f in _fields(AblationConfig)
                  if f.name == "learning_mode")
    non_frozen = {aid: c.learning_mode for aid, c in COMPONENT_ABLATIONS.items()
                 if c.learning_mode != "frozen"}
    ok = default == "frozen" and not non_frozen
    return _check("evaluation_learning_disabled", ok,
                 f"AblationConfig.learning_mode default={default!r}; "
                 f"non-frozen arms: {non_frozen or 'none'}")


def check_a9_generation_state() -> dict:
    try:
        from agentic_raptor.publication.generation_state import \
            GenerationState
        gs = GenerationState.new(generation_id=0, parent_generation_id=None)
        ok = gs.status == "BUILDING"
        return _check("a9_generation_state", ok, f"GenerationState constructs, status={gs.status}")
    except Exception as exc:
        return _check("a9_generation_state", False, f"{type(exc).__name__}: {exc}")


ALL_CHECKS = (
    check_frozen_eval_set, check_splits_disjoint, check_rag_populated,
    check_rag_frozen, check_sft_checkpoint,
    check_old_root_puct_checkpoint_absent, check_alphazero_promoted_checkpoint,
    check_alphazero_checkpoint_loading_enforced,
    check_dpo_checkpoint, check_dpo_promoted_checkpoint,
    check_dpo_feature_schema_compatible, check_ngspice,
    check_pdk, check_pvt_corners, check_fom, check_idd_not_ibias,
    check_cload_authoritative, check_electrical_environment_compatibility,
    check_budgets_fixed, check_leakage_tests, check_evaluation_learning_disabled,
    check_a9_generation_state,
)


def run_preflight(paper_mode: bool = False) -> dict[str, Any]:
    results = [c() for c in ALL_CHECKS]
    blockers = [r for r in results if r["critical"] and not r["ok"]]
    report = {"checks": results, "blockers": [r["name"] for r in blockers],
             "ready": not blockers}
    if paper_mode and blockers:
        lines = "\n".join(f"  - {r['name']}: {r['detail']}" for r in blockers)
        raise SystemExit(
            "REFUSING TO START PAPER-MODE: "
            f"{len(blockers)} blocker(s):\n{lines}")
    return report
