"""Stage 1.6: electrical-environment versioning for measurement-derived
artifacts.

The Stage 1.5 C_LOAD repair changed what "correctly measured" means (a
spec's requested load now actually reaches the simulator, where before it
silently didn't). Anything measured before the repair landed is real data,
honestly measured under a real (if wrong-for-the-spec) environment -- it is
NOT deleted or treated as corrupt. It is simply a DIFFERENT electrical
environment than anything measured after, and the two must never be
silently mixed into one training run.

ELECTRICAL_ENV_CUTOFF is the last-modified time of run_raptor_v2.py at the
moment the Stage 1.5 repair was completed (the final file in the repair
chain to be saved) -- the same "trust file mtimes, not stated version tags"
discipline the earlier VCM-ratio fix used this session, because a stated
version field can be wrong or missing on old data but an mtime-anchored
cutoff cannot be silently spoofed by an artifact that predates the fix.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

PRE_CLOAD_FIX = "PRE_CLOAD_FIX"
POST_CLOAD_FIX_V1 = "POST_CLOAD_FIX_V1"

#: 2026-08-09 17:01:23 local -- run_raptor_v2.py's mtime the moment the
#: Stage 1.5 repair chain finished. Anything measured at or after this
#: instant went through effective_c_load(); anything before did not,
#: regardless of what any stated metadata field claims. Computed (not
#: hand-typed) from the real mtime so the epoch value can't drift from the
#: human-readable timestamp above.
ELECTRICAL_ENV_CUTOFF = time.mktime((2026, 8, 9, 17, 1, 23, 0, 0, -1))

MANIFEST = ROOT / "artifacts/publication_v3/pre_cload_fix_manifest.json"

#: every measurement-derived artifact this repair's impact audit found
#: affected -- the fixed inventory freeze_all_pre_cload_artifacts() walks.
TRACKED_ARTIFACTS = {
    "surrogate_dynamics_data": ROOT / "datasets/simulation_memory/dynamics_surrogate_data.jsonl",
    "sac_replay": ROOT / "datasets/simulation_memory/mbsac_replay.jsonl",
    "sac_sizing_memory_dir": ROOT / "artifacts/sizing_memory",
    "puct_calibration_examples": ROOT / "artifacts/publication_v2/live_streams/puct_examples.jsonl",
    "puct_value_checkpoint": ROOT / "artifacts/publication_v3/puct_value_clean/policy_value_clean.pt",
    "puct_policy_checkpoint": ROOT / "artifacts/stage3e1/policy_value_ep0.pt",
    "trusted_pairs": ROOT / "datasets/ranker_preference_queue/trusted_pairs.jsonl",
    "dpo_ranker": ROOT / "artifacts/publication_v2/post_sac_ranker/ranker.pt",
    "rag_memory_v2": ROOT / "artifacts/publication_v2/selfimprove/rag_memory_v2.jsonl",
    "rag_memory_v2_clean": ROOT / "artifacts/publication_v2/selfimprove/rag_memory_v2_clean.jsonl",
    "family_spec_gate": ROOT / "artifacts/publication_v2/family_spec_gate/SUMMARY.json",
    "family_spec_gate_bufsfb": ROOT / "artifacts/publication_v2/family_spec_gate/SUMMARY_bufsfb.json",
    "puct_value_calibration_log": ROOT / "artifacts/publication_v3/puct_value_calibration/log.jsonl",
}

#: where the LIVE system reads each artifact from AFTER the Stage 1.6
#: rebuild -- distinct from TRACKED_ARTIFACTS (the frozen PRE_CLOAD_FIX
#: snapshot's paths) because SAC's warm-start memory moved to a versioned
#: path (agentic_raptor.mb_sac.spec_sizing.STATE_DIR) rather than being
#: overwritten in place; everything else keeps its old path and is
#: distinguished by the electrical_environment_version field INSIDE it.
LIVE_ARTIFACT_PATHS = {
    **TRACKED_ARTIFACTS,
    "sac_sizing_memory_dir": ROOT / "artifacts/sizing_memory_post_cload_v1",
    "surrogate_dynamics_data": (ROOT / "datasets/simulation_memory"
                                / "dynamics_surrogate_data_post_cload_v1.jsonl"),
    "sac_replay": (ROOT / "datasets/simulation_memory"
                  / "mbsac_replay_post_cload_v1.jsonl"),
    # DPO ranker's version lives in its sidecar report, not the .pt file --
    # ranker.pt is a raw state_dict (loaded weights_only=True elsewhere) and
    # cannot carry a string metadata field without breaking that load path.
    "dpo_ranker": ROOT / "artifacts/publication_v2/post_sac_ranker/training_report.json",
    "puct_calibration_examples": (ROOT / "artifacts/publication_v2"
                                  / "live_streams_post_cload_v1/puct_examples.jsonl"),
    "rag_memory_v2": (ROOT / "artifacts/publication_v2/selfimprove"
                     / "rag_memory_v2_post_cload_v1.jsonl"),
    "rag_memory_v2_clean": (ROOT / "artifacts/publication_v2/selfimprove"
                            / "rag_memory_v2_post_cload_v1_clean.jsonl"),
    "trusted_pairs": (ROOT / "datasets/ranker_preference_queue"
                      / "trusted_pairs_post_cload_v1.jsonl"),
    # 2026-08-11: root-level PUCT retired (artifacts/publication_v3/
    # ROOT_LEVEL_PUCT_RETIRED.json) and its checkpoint deleted -- the LIVE
    # view must not keep resolving to a file that no longer exists.
    # TRACKED_ARTIFACTS (the frozen PRE_CLOAD_FIX historical snapshot)
    # deliberately keeps its old path unchanged above -- only the LIVE
    # view is corrected. AlphaZero's own checkpoints are generation-
    # versioned (agentic_raptor.topology_rl.alphazero.AZ_GENERATIONS_ROOT,
    # require_promoted_az_checkpoint()), not a single fixed path, so no
    # replacement entry is added here.
    "puct_policy_checkpoint": None,
    "puct_value_checkpoint": None,
}

#: puct_value_checkpoint / puct_policy_checkpoint are the SAME file
#: (policy_value_ep0.pt / policy_value_clean.pt were confirmed byte-
#: identical when frozen) and, unlike the SAC per-family nets, this
#: checkpoint IS a rich dict ({"encoder", "heads", "meta", ...}, loaded
#: weights_only=False elsewhere -- see checkpoint_validation.py) so its
#: version can live in ck["meta"] directly.
_RICH_DICT_CHECKPOINTS = {"puct_value_checkpoint", "puct_policy_checkpoint"}


def current_environment_version(name: str) -> str:
    """Inspect the LIVE artifact directly (not the frozen manifest) for its
    electrical_environment_version. A directory or a missing/empty file
    reads as PRE_CLOAD_FIX only if TRACKED_ARTIFACTS' OLD path still has
    content and the new path doesn't exist -- otherwise NOT_YET_BUILT,
    which is deliberately distinct from PRE_CLOAD_FIX (nothing to mix in,
    vs. known-stale content that must not be mixed in)."""
    path = LIVE_ARTIFACT_PATHS.get(name)
    if path is None:
        return "UNKNOWN"
    if path.is_dir():
        if not any(path.rglob("*")):
            return "NOT_YET_BUILT"
        # a populated dir -- every file inside should agree; report the
        # worst (least-fixed) status found, never the best
        versions = set()
        for f in path.rglob("*"):
            if f.is_file() and f.suffix == ".json":
                try:
                    versions.add(json.loads(f.read_text(encoding="utf-8"))
                                .get("electrical_environment_version", PRE_CLOAD_FIX))
                except Exception:
                    versions.add("UNKNOWN")
        return (POST_CLOAD_FIX_V1 if versions == {POST_CLOAD_FIX_V1}
               else PRE_CLOAD_FIX if not versions
               else "MIXED" if len(versions) > 1 else versions.pop())
    if not path.is_file():
        return "NOT_YET_BUILT"
    try:
        if path.suffix == ".json":
            meta = json.loads(path.read_text(encoding="utf-8"))
            return meta.get("electrical_environment_version", PRE_CLOAD_FIX)
        if path.suffix == ".jsonl":
            # JSONL streams: every line should agree; the last line is the
            # most recent write, and disagreement is real and reportable
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                return "NOT_YET_BUILT"
            tail = json.loads(lines[-1]).get("electrical_environment_version", PRE_CLOAD_FIX)
            head = json.loads(lines[0]).get("electrical_environment_version", PRE_CLOAD_FIX)
            return tail if tail == head else "MIXED"
        if path.suffix == ".pt":
            if name not in _RICH_DICT_CHECKPOINTS:
                # a raw state_dict (SAC per-family nets) carries no
                # metadata by construction -- version lives in the sidecar
                # meta.json in the same STATE_DIR, handled by the dir
                # branch above when the whole memory dir is queried.
                return "UNKNOWN"
            import torch
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ck, dict):
                return (ck.get("meta") or {}).get(
                    "electrical_environment_version", PRE_CLOAD_FIX)
            return PRE_CLOAD_FIX
    except Exception:
        return "UNKNOWN"
    return PRE_CLOAD_FIX


def _sha256_of(path: Path) -> str | None:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        h = hashlib.sha256()
        for f in sorted(path.rglob("*")):
            if f.is_file():
                h.update(f.relative_to(path).as_posix().encode())
                h.update(f.read_bytes())
        return h.hexdigest() if any(path.rglob("*")) else None
    return None


def freeze_artifact(name: str, path: Path) -> dict:
    """Read-only fingerprint -- never mutates `path`. Records the fact that
    this artifact, as it exists RIGHT NOW, predates the C_LOAD repair."""
    exists = path.exists()
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = str(path)         # outside ROOT (e.g. a test's tmp_path)
    return {
        "name": name, "path": rel,
        "exists": exists,
        "kind": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
        "sha256": _sha256_of(path) if exists else None,
        "size_bytes": (sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                      if path.is_dir() else path.stat().st_size if exists else None),
        "mtime": path.stat().st_mtime if exists else None,
        "electrical_environment_version": PRE_CLOAD_FIX,
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def freeze_all_pre_cload_artifacts(overwrite: bool = False) -> dict:
    """Freeze the FULL tracked inventory in one manifest. Idempotent: an
    artifact already frozen keeps its original fingerprint unless
    overwrite=True (re-running this after the artifact has since been
    legitimately rebuilt to POST_CLOAD_FIX_V1 must NOT silently re-mark it
    PRE_CLOAD_FIX -- check current version first in that case)."""
    existing = {}
    if MANIFEST.is_file() and not overwrite:
        existing = {e["name"]: e for e in json.loads(
            MANIFEST.read_text(encoding="utf-8")).get("artifacts", [])}
    frozen = []
    for name, path in TRACKED_ARTIFACTS.items():
        if name in existing and not overwrite:
            frozen.append(existing[name])
            continue
        frozen.append(freeze_artifact(name, path))
    doc = {"cutoff": ELECTRICAL_ENV_CUTOFF,
          "cutoff_human": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(ELECTRICAL_ENV_CUTOFF)),
          "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
          "note": ("Read-only fingerprints of every measurement-derived "
                   "artifact as it existed before the Stage 1.5 C_LOAD "
                   "repair. Nothing referenced here was modified or "
                   "deleted by this freeze."),
          "artifacts": frozen}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


def stamp(obj: dict, *, model_type: str | None = None,
         training_data_hash: str | None = None,
         checkpoint_hash: str | None = None,
         validated: bool | None = None, **extra) -> dict:
    """Attach the required POST_CLOAD_FIX_V1 provenance block to a NEW
    artifact's metadata dict. Every measurement-derived artifact rebuilt
    after the repair must carry this -- see check_electrical_environment_
    version() and assert_compatible() below for the paper-mode gate that
    depends on it."""
    block = {"electrical_environment_version": POST_CLOAD_FIX_V1,
            "post_vcm_fix": True, "post_cload_fix": True,
            "stamped_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if model_type is not None:
        block["model_type"] = model_type
    if training_data_hash is not None:
        block["training_data_hash"] = training_data_hash
    if checkpoint_hash is not None:
        block["checkpoint_hash"] = checkpoint_hash
    if validated is not None:
        block["validated"] = validated
    block.update(extra)
    return {**obj, **block}


def environment_version_of(meta: dict | None, *, record_timestamp: float | None = None) -> str:
    """Resolve an artifact's environment version.

    Trust order: an explicit, present `electrical_environment_version`
    field wins -- UNLESS a record_timestamp is given and it predates
    ELECTRICAL_ENV_CUTOFF, in which case the timestamp overrides any
    claimed tag (an old record cannot have been produced by code that
    didn't exist yet, regardless of what a field says)."""
    if record_timestamp is not None and record_timestamp < ELECTRICAL_ENV_CUTOFF:
        return PRE_CLOAD_FIX
    if not meta:
        return "UNKNOWN"
    v = meta.get("electrical_environment_version")
    return v if v else "UNKNOWN"


def assert_compatible(artifacts: dict[str, str], *,
                      required_version: str = POST_CLOAD_FIX_V1) -> dict:
    """artifacts: {artifact_name: resolved_environment_version}.

    Raises AssertionError listing every artifact NOT at required_version.
    This is the hard-fail paper mode calls before loading the FULL
    pipeline (Part 9 / Part 12) -- an artifact whose version is UNKNOWN is
    treated as incompatible, not silently passed."""
    bad = {k: v for k, v in artifacts.items() if v != required_version}
    if bad:
        raise AssertionError(
            "ELECTRICAL ENVIRONMENT MISMATCH: the following artifacts are "
            f"not {required_version}: " +
            "; ".join(f"{k}={v}" for k, v in sorted(bad.items())))
    return {"ok": True, "checked": list(artifacts), "required_version": required_version}
