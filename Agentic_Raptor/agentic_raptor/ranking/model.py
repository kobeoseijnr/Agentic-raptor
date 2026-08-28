"""The learned post-SAC ranker (Part 8).

`compare()` reaches Level 2 -- the learned selector -- whenever both designs
sit in the same hard-safety tier, which is the common case. Until now there
was no model to reach: no checkpoint and no trainer existed, so `compare()`
raised RankerCheckpointMissing and stage 8 of the canonical pipeline could
not run at all.

Trained with the DPO/Bradley-Terry pairwise objective on TRUSTED pairs only:
pairs where BOTH designs received a real ngspice measurement with complete
provenance. A preference model must be fitted to measured preferences; there
is no way to bootstrap it from predictions without learning the surrogate's
mistakes.

Features are read from the SurrogatePrediction ONLY. Reading anything from
the authoritative outcome at scoring time would reintroduce exactly the
leakage `types.py` exists to prevent -- the ranker would score using the
answer it is supposed to predict.
"""

from __future__ import annotations

from pathlib import Path

from agentic_raptor.ranking.types import (SurrogatePrediction,
                                          checkpoint_sha256,
                                          required_constraint_names)

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = _ROOT / "artifacts/publication_v2/post_sac_ranker/ranker.pt"

#: Ordered, documented feature vector. Every entry is derivable from a
#: SurrogatePrediction plus the spec. Missingness is an explicit feature
#: rather than an imputed value, so "unknown" cannot masquerade as "good".
FEATURE_NAMES = (
    "worst_violation",          # >=0, larger worse
    "n_satisfied",              # hard constraints predicted satisfied
    "n_missing",                # required constraints with NO prediction
    "uncertainty",              # MC-dropout spread, 1.0 when unknown
    "stability_p",              # 0.5 when unknown
    "margin_gain",
    "margin_pm",
    "margin_ugbw",
    "has_gain", "has_pm", "has_ugbw",
)
FEATURE_DIM = len(FEATURE_NAMES)

#: Stage 7.1 (Section 9) normalization audit: real trusted-pair data shows
#: `uncertainty` std ~0.017 vs `margin_gain` std ~1.57 -- roughly a 100x
#: scale gap, which L2 weight_decay actively fights (matching a small-scale
#: feature's influence needs proportionally larger weights, exactly what
#: weight_decay penalizes). Continuous, scale-heterogeneous features are
#: standardized (train-only mean/std); counts and binary presence flags are
#: left alone -- z-scoring a 0/1 indicator or a small integer count doesn't
#: correct a scale mismatch, it just relabels the same two states.
NORMALIZE_MASK = tuple(n in ("worst_violation", "uncertainty", "stability_p",
                             "margin_gain", "margin_pm", "margin_ugbw")
                       for n in FEATURE_NAMES)


def features(spec: dict, pred: SurrogatePrediction) -> list:
    """Feature vector for ONE design. Surrogate-only, never authoritative."""
    m = pred.normalized_margins or {}
    wv = pred.worst_predicted_violation
    req = required_constraint_names(spec)
    return [
        9.9 if wv is None else float(wv),
        float(pred.hard_constraints_satisfied()),
        float(sum(1 for n in req if m.get(n) is None)),
        1.0 if pred.predictive_uncertainty is None
        else float(pred.predictive_uncertainty),
        0.5 if pred.stability_probability is None
        else float(pred.stability_probability),
        float(m.get("gain", 0.0)),
        float(m.get("pm", 0.0)),
        float(m.get("ugbw", 0.0)),
        1.0 if m.get("gain") is not None else 0.0,
        1.0 if m.get("pm") is not None else 0.0,
        1.0 if m.get("ugbw") is not None else 0.0,
    ]


def fit_normalization(feature_rows: list) -> tuple:
    """Train-ONLY mean/std over NORMALIZE_MASK positions; identity (0/1)
    elsewhere. `feature_rows` must be TRAIN-split feature vectors only --
    fitting on dev/held-out rows would leak their distribution into the
    checkpoint."""
    import statistics as st
    n = FEATURE_DIM
    mean = [0.0] * n
    std = [1.0] * n
    for i in range(n):
        if not NORMALIZE_MASK[i]:
            continue
        col = [row[i] for row in feature_rows]
        mean[i] = st.mean(col) if col else 0.0
        std[i] = (st.pstdev(col) if len(col) > 1 else 1.0) or 1.0
    return mean, std


class PostSACRanker:
    """Scores a sized design. Higher is better.

    `compare()` only requires `.score(spec, design, pred)`; the design is
    accepted for interface compatibility and deliberately unused, because
    every feature must come from the prediction.
    """

    def __init__(self, model=None, checkpoint_hash: str | None = None,
                feature_mean: list | None = None,
                feature_std: list | None = None):
        self.model = model
        self.checkpoint_hash = checkpoint_hash
        # Stage 7.1 (Section 9): OPTIONAL train-only-fit standardization.
        # None on both means "no normalization" -- the exact behaviour every
        # checkpoint saved before this repair already has, so loading an old
        # .pt file (no sidecar) is bit-for-bit unchanged. A checkpoint that
        # DOES ship normalization stats gets them applied identically at
        # train and inference time; nothing here can invent a mismatch,
        # because both paths route through the same `_normalize`.
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    @staticmethod
    def build():
        import torch
        return torch.nn.Sequential(
            torch.nn.Linear(FEATURE_DIM, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 1))

    @staticmethod
    def normalization_path(checkpoint_path) -> Path:
        p = Path(checkpoint_path)
        return p.with_name(p.stem + "_normalization.json")

    @classmethod
    def load(cls, path=None) -> "PostSACRanker | None":
        """Return None when no checkpoint exists -- never a random model.

        A randomly initialised ranker would silently produce arbitrary
        selections that look like learned decisions.
        """
        p = Path(path or DEFAULT_CKPT)
        if not p.is_file():
            return None
        try:
            import torch
            net = cls.build()
            net.load_state_dict(torch.load(p, weights_only=True))
            net.eval()
            for q in net.parameters():
                q.requires_grad_(False)
            mean = std = None
            norm_path = cls.normalization_path(p)
            if norm_path.is_file():
                import json
                stats = json.loads(norm_path.read_text(encoding="utf-8"))
                mean, std = stats["feature_mean"], stats["feature_std"]
            return cls(net, checkpoint_sha256(p), feature_mean=mean, feature_std=std)
        except Exception:
            return None

    def _normalize(self, x):
        import torch
        if self.feature_mean is None or self.feature_std is None:
            return x
        # mean/std are pre-zeroed/one'd at the unmasked positions by
        # fit_normalization() below, so this is a no-op there and a
        # standard z-score on the masked (continuous) features.
        mean = torch.tensor(self.feature_mean, dtype=torch.float32)
        std = torch.tensor(self.feature_std, dtype=torch.float32).clamp(min=1e-6)
        return (x - mean) / std

    def score(self, spec: dict, design, pred: SurrogatePrediction) -> float:
        import torch
        x = torch.tensor([features(spec, pred)], dtype=torch.float32)
        x = self._normalize(x)
        with torch.no_grad():
            return float(self.model(x)[0, 0])
