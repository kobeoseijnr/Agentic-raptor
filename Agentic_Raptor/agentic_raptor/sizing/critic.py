"""Twin Q critics with target networks and Polyak soft updates."""

from __future__ import annotations

from agentic_raptor.utils.seeding import apply_torch_omp_workaround


def build_q_critic(state_dim: int, action_dim: int, hidden_dim: int = 128):
    apply_torch_omp_workaround()
    import torch
    import torch.nn as nn

    class QCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_dim = state_dim + action_dim
            self.norm = nn.LayerNorm(in_dim)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return self.net(self.norm(torch.cat([state, action], dim=-1)))

    return QCritic()


def build_twin_critics(state_dim: int, action_dim: int, hidden_dim: int = 128):
    """(q1, q2, q1_target, q2_target) with targets initialised from the live nets."""
    q1 = build_q_critic(state_dim, action_dim, hidden_dim)
    q2 = build_q_critic(state_dim, action_dim, hidden_dim)
    q1_target = build_q_critic(state_dim, action_dim, hidden_dim)
    q2_target = build_q_critic(state_dim, action_dim, hidden_dim)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    for target in (q1_target, q2_target):
        for p in target.parameters():
            p.requires_grad_(False)
    return q1, q2, q1_target, q2_target


def soft_update(live, target, tau: float) -> None:
    """Polyak update: θ_target ← (1 − τ)·θ_target + τ·θ_live."""
    apply_torch_omp_workaround()
    import torch

    with torch.no_grad():
        for p, pt in zip(live.parameters(), target.parameters(), strict=True):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)
