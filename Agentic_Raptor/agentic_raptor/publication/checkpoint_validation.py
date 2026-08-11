"""PUCT value-checkpoint validation (PUCT VALUE NETWORK BLOCKER).

The existing PUCT value checkpoint (`artifacts/stage3e1/policy_value_ep0.pt`)
predates the input-common-mode (VCM) fix era of this codebase and carries no
metadata proving it was retrained afterward -- confirmed directly: its
on-disk `meta` dict has no `post_vcm_fix`, `validated`, `training_data_hash`,
or `checkpoint_hash` field (only `refresh`/`schema_version`/`timestamp`/a
nested `stale_meta` history). A stale value network silently degrades PUCT's
search to something close to prior-only ranking while still being LABELED
"one_root PUCT" in every trace -- exactly the failure mode this module exists
to make impossible for paper-mode runs.

Nothing here can determine WHETHER a checkpoint is actually post-VCM-fix --
that requires a human to confirm the training run's provenance. What this
module does is refuse to treat silence as an answer: `validate(...)` returns
ok=True only when a human has explicitly attested it via `mark_validated`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_META_FIELDS = ("post_vcm_fix", "validated", "training_data_hash")


@dataclass
class CheckpointValidation:
    ok: bool
    reason: str | None
    meta: dict[str, Any] | None
    checkpoint_hash: str | None
    path: str


def _load_checkpoint(path: Path) -> dict | None:
    import torch
    if not path.is_file():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck if isinstance(ck, dict) else None


def _weights_hash(ck: dict) -> str:
    """Hash of the WEIGHTS only (encoder + heads state dicts) -- never the
    whole file, which would include `meta` itself and make "the hash of
    this checkpoint" change every time metadata (including the hash) is
    written: a self-referential hash can never be made to match its own
    file. This is stable across any number of meta-only re-saves."""
    from agentic_raptor.ranking.types import state_dict_sha256
    parts = [state_dict_sha256(ck[k]) for k in ("encoder", "heads") if k in ck]
    return _sha16("".join(parts))


def _sha16(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def validate(path: str | Path) -> CheckpointValidation:
    """ok=True only if the checkpoint's meta explicitly attests
    post_vcm_fix=True and validated=True. Missing metadata, validated=False,
    or a meta value of "stale"/"pre_vcm_fix"/"unvalidated"/"incompatible"
    for ANY required field are all treated as NOT ok -- there is no default
    that resolves to "probably fine"."""
    p = Path(path)
    if not p.is_file():
        return CheckpointValidation(False, "checkpoint_missing", None,
                                    None, str(p))
    ck = _load_checkpoint(p)
    ckhash = _weights_hash(ck) if ck else None
    meta = ck.get("meta") if ck else None
    if not meta:
        return CheckpointValidation(False, "no_metadata", meta, ckhash, str(p))
    missing = [f for f in REQUIRED_META_FIELDS if f not in meta]
    if missing:
        return CheckpointValidation(
            False, f"missing_fields:{','.join(missing)}", meta, ckhash, str(p))
    bad_values = {"stale", "pre_vcm_fix", "unvalidated", "incompatible", False}
    if meta.get("post_vcm_fix") is not True or meta.get("post_vcm_fix") in bad_values:
        return CheckpointValidation(False, "post_vcm_fix_not_true", meta,
                                    ckhash, str(p))
    if meta.get("validated") is not True or meta.get("validated") in bad_values:
        return CheckpointValidation(False, "validated_not_true", meta,
                                    ckhash, str(p))
    if meta.get("checkpoint_hash") not in (None, ckhash):
        return CheckpointValidation(False, "checkpoint_hash_mismatch", meta,
                                    ckhash, str(p))
    return CheckpointValidation(True, None, meta, ckhash, str(p))


def mark_validated(path: str | Path, *, post_vcm_fix: bool,
                   training_data_hash: str, validated_by: str = "") -> dict:
    """Explicit human attestation. Loads the checkpoint, stamps the
    validation fields onto its existing `meta` dict (never fabricating
    `post_vcm_fix` -- the CALLER states it), re-saves in place.

    This does NOT retrain or otherwise change the checkpoint's weights --
    `encoder`/`heads` state dicts pass through byte-identical."""
    import time

    import torch

    p = Path(path)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    meta = dict(ck.get("meta") or {})
    meta.update(post_vcm_fix=bool(post_vcm_fix), validated=bool(post_vcm_fix),
               training_data_hash=training_data_hash,
               validated_by=validated_by or None,
               validated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               # hash of the WEIGHTS, computed BEFORE this save -- stable
               # regardless of what meta itself contains (see _weights_hash)
               checkpoint_hash=_weights_hash(ck))
    ck["meta"] = meta
    torch.save(ck, p)
    return meta
