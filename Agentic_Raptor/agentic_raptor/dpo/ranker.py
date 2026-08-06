"""Trainable pairwise preference ranker with a DPO-style objective.

Model: s_θ(x) — an MLP scoring one candidate's pre-SPICE feature vector
(topology-aware + specification-aware conditioning is in the features).

Objective (per chosen/rejected pair, confidence-weighted):

    L = − w · log σ( β · ( s_θ(x_chosen) − s_θ(x_rejected) ) )

This is the Bradley–Terry pairwise preference-optimization objective with a
DPO-style β temperature over score differences — a genuine trainable pairwise
preference model, NOT heuristic scalar scoring. (Classic DPO applies the same
logistic-difference loss to policy log-ratios; here the scored object is a
candidate representation rather than a generation policy, which is the
appropriate equivalent for a ranking module and is documented as such.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentic_raptor.dpo.preference_pairs import PreferencePair
from agentic_raptor.dpo.schemas import CandidateFeatures
from agentic_raptor.utils.seeding import apply_torch_omp_workaround


@dataclass
class DPOConfig:
    enabled: bool = False
    checkpoint_path: str | None = None
    update_mode: str = "periodic"          # "periodic" | "fixed"
    minimum_pairs_before_training: int = 8
    update_interval_episodes: int = 5
    beta: float = 2.0
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 20
    confidence_threshold: float = 0.4
    tie_margin: float = 0.02
    exploration_fraction: float = 0.2
    maximum_pairs_per_specification: int = 50
    hidden_dim: int = 64
    seed: int = 0


class DPORanker:
    """Scores and orders candidates before SPICE. Deterministic under a seed."""

    def __init__(self, feature_dim: int, config: DPOConfig) -> None:
        apply_torch_omp_workaround()
        import torch
        import torch.nn as nn

        self.torch = torch
        self.config = config
        self.feature_dim = feature_dim
        torch.manual_seed(config.seed)
        self.model = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.trained_pairs = 0

    # -- inference ----------------------------------------------------------
    def score(self, features: CandidateFeatures) -> float:
        torch = self.torch
        with torch.no_grad():
            x = torch.tensor([features.feature_vector()], dtype=torch.float32)
            return float(self.model(x).item())

    def rank(self, pool: list[CandidateFeatures]) -> list[tuple[CandidateFeatures, float, int]]:
        """(features, score, rank) best-first; deterministic tiebreak by id."""
        scored = sorted(
            ((f, self.score(f)) for f in pool),
            key=lambda pair: (-pair[1], pair[0].pool_candidate_id),
        )
        return [(f, s, rank) for rank, (f, s) in enumerate(scored)]

    # -- training -----------------------------------------------------------
    def train_on_pairs(self, pairs: list[PreferencePair]) -> dict[str, float]:
        torch = self.torch
        usable = [p for p in pairs if p.confidence >= self.config.confidence_threshold]
        if len(usable) < self.config.minimum_pairs_before_training:
            return {"skipped": 1.0, "usable_pairs": float(len(usable))}
        chosen = torch.tensor([p.chosen.features.feature_vector() for p in usable], dtype=torch.float32)
        rejected = torch.tensor([p.rejected.features.feature_vector() for p in usable], dtype=torch.float32)
        weights = torch.tensor([p.confidence for p in usable], dtype=torch.float32)
        last_loss = 0.0
        for _epoch in range(self.config.epochs):
            perm = torch.randperm(len(usable))
            for start in range(0, len(usable), self.config.batch_size):
                idx = perm[start:start + self.config.batch_size]
                diff = self.model(chosen[idx]).squeeze(-1) - self.model(rejected[idx]).squeeze(-1)
                loss = -(weights[idx] * torch.nn.functional.logsigmoid(self.config.beta * diff)).mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                last_loss = float(loss.detach().item())
        self.trained_pairs += len(usable)
        return {"loss": last_loss, "usable_pairs": float(len(usable))}

    def pair_accuracy(self, pairs: list[PreferencePair]) -> float:
        """Fraction of pairs where the chosen candidate outscores the rejected."""
        if not pairs:
            return 0.0
        correct = sum(1 for p in pairs if self.score(p.chosen.features) > self.score(p.rejected.features))
        return correct / len(pairs)

    # -- checkpointing ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "model": self.model.state_dict(),
                "feature_dim": self.feature_dim,
                "trained_pairs": self.trained_pairs,
                "config": vars(self.config),
            },
            p,
        )
        return p

    def load(self, path: str | Path) -> None:
        payload = self.torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload["feature_dim"] != self.feature_dim:
            raise ValueError(
                f"checkpoint feature_dim {payload['feature_dim']} != current {self.feature_dim}"
            )
        self.model.load_state_dict(payload["model"])
        self.trained_pairs = int(payload.get("trained_pairs", 0))
