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

#: RAG is "sufficiently populated" above this many usable (stability-tagged)
#: records -- a round number chosen to be well above what one under-run
#: campaign could produce by accident, not tuned to make the check pass.
RAG_MIN_RECORDS = 200


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
    p = CLEAN_RAG_PATH
    if not p.is_file():
        return _check("rag_populated", False,
                     f"{p.name} does not exist -- run "
                     "rag_freeze.build_clean_snapshot() first")
    lines = [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    usable = sum(1 for x in lines if json.loads(x).get("stability"))
    ok = usable >= RAG_MIN_RECORDS
    return _check("rag_populated", ok,
                 f"{usable} usable CLEAN records (>= {RAG_MIN_RECORDS} "
                 f"required); {len(lines)} total lines", critical=True)


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
    p = ROOT / "artifacts/publication_v2/post_sac_ranker/ranker.pt"
    rep = ROOT / "artifacts/publication_v2/post_sac_ranker/training_report.json"
    if not p.is_file():
        return _check("dpo_checkpoint", False, f"{p} missing")
    if not rep.is_file():
        return _check("dpo_checkpoint", False, f"{rep} missing (unvalidated)")
    r = json.loads(rep.read_text(encoding="utf-8"))
    mono = (r.get("monotonicity") or {}).get("strictly_decreasing")
    ok = mono is True
    return _check("dpo_checkpoint", ok,
                 f"trained_at={r.get('trained_at')} "
                 f"informative_pairs={r.get('pairs_informative')} "
                 f"train_acc={r.get('train_pair_accuracy')} "
                 f"monotonicity_strictly_decreasing={mono}")


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
CLOAD_GATED_ARTIFACTS = ("puct_value_checkpoint", "puct_policy_checkpoint",
                         "dpo_ranker", "trusted_pairs", "rag_memory_v2_clean",
                         "family_spec_gate", "sac_sizing_memory_dir")


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
    check_dpo_checkpoint, check_ngspice,
    check_pdk, check_pvt_corners, check_fom, check_idd_not_ibias,
    check_cload_authoritative, check_electrical_environment_compatibility,
    check_budgets_fixed, check_leakage_tests, check_a9_generation_state,
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
