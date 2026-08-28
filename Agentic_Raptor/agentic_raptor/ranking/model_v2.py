"""Stage 8 (2026-08-12): the promoted DPO V2 ranker's live-inference wrapper.

Stage 7.2B found the learned Level-2 ranker DPO_REJUSTIFIED using
POST_SAC_FEATURES_V2 (46 features: original 11 surrogate-derived + 35 rich
post-SAC candidate/trajectory/topology features) -- 77.80% run-grouped DEV
ranker-authority accuracy vs the deterministic selector's frozen 74.66%,
77 wins / 45 losses / 0 catastrophic errors. This module is the ONLY
authorized way to load that checkpoint for live inference.

Design constraint: agentic_raptor.ranking.post_sac.compare() calls
`model.score(spec, design, pred)` -- a fixed 3-argument interface it shares
with every other selector arm, and Level 1 (hard_safety_tier) is frozen and
untouched. PostSACRankerV2 satisfies that exact interface without needing
compare() to change: the extra per-candidate context V2 features require
(a completed branch's own trajectory + winning-candidate row + topology
family) is supplied ONCE per pipeline call, keyed by canonical_graph_hash,
so `.score()` can look up "which branch does this design belong to" from
`design.canonical_graph_hash` alone. Candidate A's score can therefore never
see candidate B's trajectory except through the final score difference --
each side's context is looked up independently.

Feature construction itself is NOT duplicated here: every call routes
through agentic_raptor.ranking.features_v2.features_v2(), the exact same
function Stage 7.2B trained against.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

#: the ONLY checkpoint Stage 8 may load for the live learned selector --
#: hardcoded, not a caller-suppliable default, so a stale/rejected
#: checkpoint can never be silently substituted.
PROMOTED_V2_DIR = (_ROOT / "artifacts/publication_v3/stage7_2b_dpo_repair"
                  / "post_sac_ranker_v2_combined__C_current")
PROMOTED_V2_CKPT = PROMOTED_V2_DIR / "ranker.pt"
PROMOTED_V2_NORM = PROMOTED_V2_DIR / "normalization.json"
PROMOTED_V2_MANIFEST = PROMOTED_V2_DIR / "training_manifest.json"
REQUIRED_V2_SHA256 = "0a82439bae42534358fda2450aefc384c7679f32d2f0d8bb1b334253ee593b74"
REQUIRED_FEATURE_SCHEMA = "POST_SAC_FEATURES_V2"


class RankerFeatureSchemaMismatch(RuntimeError):
    """The loaded checkpoint's declared feature schema does not match
    POST_SAC_FEATURES_V2 -- a V1/11-feature checkpoint must never be scored
    against the 46-feature V2 vector, or vice versa."""


class RankerCheckpointHashMismatch(RuntimeError):
    """The checkpoint on disk does not hash to the one promotion authorized.
    Never silently substitute a different checkpoint."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PostSACRankerV2:
    """Scores a sized design using POST_SAC_FEATURES_V2. Higher is better.
    Satisfies the same `.score(spec, design, pred)` interface as
    agentic_raptor.ranking.model.PostSACRanker (compare() is agnostic to
    which implementation it's holding)."""

    def __init__(self, model, checkpoint_hash: str, feature_mean: list,
                feature_std: list, branch_context: dict, arm: str = "combined"):
        self.model = model
        self.checkpoint_hash = checkpoint_hash
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.branch_context = branch_context   # {canonical_graph_hash: {...}}
        self.arm = arm
        self.feature_schema = REQUIRED_FEATURE_SCHEMA

    def _normalize(self, x):
        import torch
        mean = torch.tensor(self.feature_mean, dtype=torch.float32)
        std = torch.tensor(self.feature_std, dtype=torch.float32).clamp(min=1e-6)
        return (x - mean) / std

    def score(self, spec: dict, design, pred) -> float:
        import torch

        from agentic_raptor.ranking.features_v2 import features_v2
        ctx = self.branch_context.get(design.canonical_graph_hash, {})
        feats = features_v2(spec, pred, ctx.get("candidate_row"),
                            ctx.get("branch_rows"), ctx.get("family"), arm=self.arm)
        x = torch.tensor([feats], dtype=torch.float32)
        x = self._normalize(x)
        with torch.no_grad():
            return float(self.model(x)[0, 0])


def _build_net(hidden: int | None):
    import torch
    from agentic_raptor.ranking.features_v2 import FEATURE_DIM_V2
    if hidden is None:
        return torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM_V2, 1))
    return torch.nn.Sequential(torch.nn.Linear(FEATURE_DIM_V2, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, 1))


def load_promoted_v2(branch_context: dict) -> PostSACRankerV2:
    """The ONLY authorized loader for the live DPO V2 selector. Hard-fails
    (raises) rather than returning None on any mismatch -- Section 11: no
    silent fallback to the deterministic selector when ranker_mode='dpo'
    was explicitly requested."""
    if not PROMOTED_V2_CKPT.is_file():
        raise FileNotFoundError(
            f"promoted DPO V2 checkpoint missing: {PROMOTED_V2_CKPT}")
    actual_hash = _sha256(PROMOTED_V2_CKPT)
    if actual_hash != REQUIRED_V2_SHA256:
        raise RankerCheckpointHashMismatch(
            f"DPO V2 checkpoint hash mismatch: expected {REQUIRED_V2_SHA256}, "
            f"got {actual_hash} at {PROMOTED_V2_CKPT} -- refusing to load an "
            f"unauthorized checkpoint")
    if not PROMOTED_V2_MANIFEST.is_file():
        raise FileNotFoundError(
            f"promoted DPO V2 training manifest missing: {PROMOTED_V2_MANIFEST}")
    manifest = json.loads(PROMOTED_V2_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("feature_schema") != REQUIRED_FEATURE_SCHEMA:
        raise RankerFeatureSchemaMismatch(
            f"checkpoint declares feature_schema={manifest.get('feature_schema')!r}, "
            f"required {REQUIRED_FEATURE_SCHEMA!r}")
    if not PROMOTED_V2_NORM.is_file():
        raise FileNotFoundError(
            f"promoted DPO V2 normalization stats missing: {PROMOTED_V2_NORM} "
            f"-- inference must never refit normalization")
    norm = json.loads(PROMOTED_V2_NORM.read_text(encoding="utf-8"))

    import torch
    net = _build_net(manifest.get("hidden"))
    net.load_state_dict(torch.load(PROMOTED_V2_CKPT, weights_only=True))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    return PostSACRankerV2(net, actual_hash, norm["feature_mean"], norm["feature_std"],
                           branch_context, arm=manifest.get("arm", "combined"))
