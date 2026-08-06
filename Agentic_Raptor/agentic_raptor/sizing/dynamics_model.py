"""Learned dynamics model for model-based SAC.

Predicts, from (state, action):
* next state (as a delta on the state vector);
* immediate reward;
* termination probability (logit).

Trained on real sizing transitions; used for short imagined rollouts that are
labelled ``source="model"`` in the replay buffer. SPICE remains the final
authority on accepted results — model transitions only augment SAC training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_raptor.utils.seeding import apply_torch_omp_workaround


@dataclass
class DynamicsConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    grad_clip_norm: float = 5.0


def build_dynamics_model(state_dim: int, action_dim: int, config: DynamicsConfig | None = None):
    apply_torch_omp_workaround()
    import torch
    import torch.nn as nn

    cfg = config or DynamicsConfig()

    class DynamicsModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_dim = state_dim + action_dim
            self.norm = nn.LayerNorm(in_dim)
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
            )
            self.delta_head = nn.Linear(cfg.hidden_dim, state_dim)
            self.reward_head = nn.Linear(cfg.hidden_dim, 1)
            self.done_head = nn.Linear(cfg.hidden_dim, 1)

        def forward(
            self, state: torch.Tensor, action: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.trunk(self.norm(torch.cat([state, action], dim=-1)))
            next_state = state + self.delta_head(h)
            reward = self.reward_head(h).squeeze(-1)
            done_logit = self.done_head(h).squeeze(-1)
            return next_state, reward, done_logit

    return DynamicsModel()


class DynamicsTrainer:
    """Owns the optimiser for one dynamics model."""

    def __init__(self, model: Any, config: DynamicsConfig | None = None) -> None:
        apply_torch_omp_workaround()
        import torch

        self.torch = torch
        self.model = model
        self.config = config or DynamicsConfig()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def train_on_batch(
        self,
        states: list[list[float]],
        actions: list[list[float]],
        rewards: list[float],
        next_states: list[list[float]],
        dones: list[bool],
    ) -> dict[str, float]:
        torch = self.torch
        s = torch.tensor(states, dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.float32)
        r = torch.tensor(rewards, dtype=torch.float32)
        s2 = torch.tensor(next_states, dtype=torch.float32)
        d = torch.tensor([1.0 if x else 0.0 for x in dones], dtype=torch.float32)

        pred_s2, pred_r, pred_done_logit = self.model(s, a)
        state_loss = torch.nn.functional.mse_loss(pred_s2, s2)
        reward_loss = torch.nn.functional.mse_loss(pred_r, r)
        done_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_done_logit, d)
        loss = state_loss + reward_loss + done_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        return {
            "state_loss": float(state_loss.detach().item()),
            "reward_loss": float(reward_loss.detach().item()),
            "done_loss": float(done_loss.detach().item()),
            "total_loss": float(loss.detach().item()),
        }

    def imagine_rollout(
        self,
        actor: Any,
        start_state: list[float],
        horizon: int,
    ) -> list[dict[str, Any]]:
        """Short imagined rollout using the actor's stochastic policy.

        Returns transitions dicts with source="model".
        """
        torch = self.torch
        transitions: list[dict[str, Any]] = []
        state = torch.tensor([start_state], dtype=torch.float32)
        with torch.no_grad():
            for _t in range(horizon):
                action, _logp = actor.sample(state)
                next_state, reward, done_logit = self.model(state, action)
                done_prob = torch.sigmoid(done_logit)
                transitions.append(
                    {
                        "state": state.squeeze(0).tolist(),
                        "action": action.squeeze(0).tolist(),
                        "reward": float(reward.item()),
                        "next_state": next_state.squeeze(0).tolist(),
                        "done": bool(done_prob.item() > 0.5),
                        "source": "model",
                    }
                )
                if done_prob.item() > 0.5:
                    break
                state = next_state
        return transitions
