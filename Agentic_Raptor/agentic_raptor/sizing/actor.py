"""Stochastic Gaussian actor with tanh-bounded actions.

Implements the SAC actor requirements: mean and log-std heads,
reparameterised sampling (rsample), tanh squashing, and the log-probability
correction  log π(a|s) = log N(z; μ, σ) − Σ log(1 − tanh(z)² + ε).
"""

from __future__ import annotations

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def build_gaussian_actor(state_dim: int, action_dim: int, hidden_dim: int = 128):
    """Return a torch GaussianActor module (lazy torch import)."""
    apply_torch_omp_workaround()
    import torch
    import torch.nn as nn

    class GaussianActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(state_dim)
            self.base = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)

        def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.base(self.norm(state))
            mu = self.mu_head(h)
            log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
            return mu, log_std

        def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Reparameterised action + corrected log-prob. Differentiable."""
            mu, log_std = self(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mu, std)
            z = normal.rsample()
            action = torch.tanh(z)
            log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6)).sum(dim=-1, keepdim=True)
            return action, log_prob

        def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
            mu, _ = self(state)
            return torch.tanh(mu)

    return GaussianActor()
