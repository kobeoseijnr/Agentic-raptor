"""Gradient training of the topology policy–value network.

Losses (AlphaZero-style):
    L_policy = − Σ_a π_MCTS(a|s) · log π_θ(a|s)      (masked cross-entropy)
    L_value  = ( V_φ(s) − z )²

where z is the **final post-sizing reward** propagated onto each step by
:class:`agentic_raptor.learning.cross_level_credit.CrossLevelCreditAssigner`
(squashed through tanh to match the value head's (−1, 1) range).

These are real optimiser steps — ``train_on_steps`` runs backward() and
optimizer.step(); ``parameter_checksum`` lets tests verify parameters changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_raptor.topology_rl.policy_value_network import PolicyValueConfig
from agentic_raptor.topology_rl.trajectory import TrajectoryStep
from agentic_raptor.utils.seeding import apply_torch_omp_workaround


def parameter_checksum(module: Any) -> float:
    """Deterministic scalar over all parameters (for change-detection in tests)."""
    apply_torch_omp_workaround()
    import torch

    with torch.no_grad():
        return float(sum(p.abs().sum().item() for p in module.parameters()))


@dataclass
class TrainReport:
    policy_loss: float
    value_loss: float
    total_loss: float
    batch_size: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "total_loss": self.total_loss,
            "batch_size": self.batch_size,
        }


class PolicyValueTrainer:
    """Owns the optimiser for one PolicyValueNetwork."""

    def __init__(self, network: Any, config: PolicyValueConfig) -> None:
        apply_torch_omp_workaround()
        import torch

        self.torch = torch
        self.network = network
        self.config = config
        self.optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)

    def train_on_steps(self, steps: list[TrajectoryStep]) -> TrainReport:
        """One optimiser step over a batch of credit-assigned trajectory steps."""
        torch = self.torch
        if not steps:
            return TrainReport(0.0, 0.0, 0.0, 0)
        max_a = self.config.max_actions

        features = torch.tensor([s.state_features for s in steps], dtype=torch.float32)
        masks, targets, values = [], [], []
        for s in steps:
            mask = list(s.legal_action_mask)[:max_a] + [False] * max(0, max_a - len(s.legal_action_mask))
            dist = list(s.mcts_visit_distribution)[:max_a] + [0.0] * max(0, max_a - len(s.mcts_visit_distribution))
            total = sum(dist) or 1.0
            masks.append([1.0 if m else 0.0 for m in mask])
            targets.append([d / total for d in dist])
            z = s.discounted_return if s.discounted_return is not None else 0.0
            values.append(float(torch.tanh(torch.tensor(z)).item()))
        mask_t = torch.tensor(masks, dtype=torch.float32)
        target_pi = torch.tensor(targets, dtype=torch.float32)
        target_v = torch.tensor(values, dtype=torch.float32)

        logits, value_pred = self.network(features)
        # Masked log-softmax: illegal slots get -inf before normalization.
        masked_logits = logits.masked_fill(mask_t == 0.0, float("-inf"))
        log_pi = torch.log_softmax(masked_logits, dim=-1)
        log_pi = torch.where(mask_t > 0.0, log_pi, torch.zeros_like(log_pi))
        policy_loss = -(target_pi * log_pi).sum(dim=-1).mean()
        value_loss = torch.nn.functional.mse_loss(value_pred, target_v)
        loss = policy_loss + self.config.value_loss_weight * value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        return TrainReport(
            policy_loss=float(policy_loss.detach().item()),
            value_loss=float(value_loss.detach().item()),
            total_loss=float(loss.detach().item()),
            batch_size=len(steps),
        )
