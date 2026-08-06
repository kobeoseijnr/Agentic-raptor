"""Graph-conditioned model-based Soft Actor-Critic.

Genuine SAC learning — all of the following are implemented and exercised by
the smoke test (see tests/test_mb_sac_training.py for parameter-change proofs):

* stochastic Gaussian actor, reparameterised, tanh-bounded, log-prob corrected;
* twin Q critics + twin frozen target critics;
* entropy-regularised Bellman target
      y = r + γ(1−d)[ min(Q'₁, Q'₂)(s', a') − α·log π(a'|s') ],  a' ~ π(·|s');
* actor objective  E[ α·log π(a|s) − min(Q₁, Q₂)(s, a) ];
* automatic entropy temperature α (learned log_alpha, target entropy −|A|);
* Polyak target updates;
* learned dynamics model + short imagined rollouts (source="model");
* configurable real/model batch mixture.

State vector (graph-conditioned):
    [ topology embedding | sizing vector | spec embedding | metrics | margins | budget ]

Design note: the legacy ``mb_sac.sac_agent.SACAgent`` was audited as genuine
SAC (docs/REPOSITORY_AUDIT.md §2.12) and can be swapped in through
``sizing.adapters.LegacySACBackend``; this class exists because the legacy
agent has fixed state/action dims and no jointly-learned dynamics model with
reward/termination heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.sizing.actor import build_gaussian_actor
from agentic_raptor.sizing.critic import build_twin_critics, soft_update
from agentic_raptor.sizing.dynamics_model import (
    DynamicsConfig,
    DynamicsTrainer,
    build_dynamics_model,
)
from agentic_raptor.sizing.parameter_space import SizingParameterSpace
from agentic_raptor.sizing.replay_buffer import (
    SizingReplayBuffer,
    SizingTransition,
)
from agentic_raptor.topology_rl.policy_value_network import (
    _spec_embedding,
    graph_feature_vector,
)
from agentic_raptor.utils.seeding import apply_torch_omp_workaround, make_rng

#: metrics exposed to the sizing state, fixed order
STATE_METRIC_KEYS: tuple[str, ...] = ("gain_db", "gbw_hz", "phase_margin_deg", "power_w")
STATE_MARGIN_KEYS: tuple[str, ...] = ("gain_db", "gbw_hz", "phase_margin_deg", "power_w")


@dataclass
class MBSACConfig:
    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    init_alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float | None = None  # default −action_dim
    grad_clip_norm: float = 10.0
    real_batch_fraction: float = 0.8
    rollout_horizon: int = 3
    replay_capacity: int = 100_000
    #: Stage 2: number of dynamics models; disagreement between them is the
    #: uncertainty signal for the real-SPICE query policy (>=1; 1 disables it).
    dynamics_ensemble_size: int = 2
    seed: int = 0
    device: str = "cpu"


@dataclass
class SizingStateContext:
    """Everything needed to build the graph-conditioned state vector."""

    graph: CircuitGraph
    spec: DesignSpecifications
    sizing_vector: list[float]
    metrics: dict[str, float] = field(default_factory=dict)
    constraint_margins: dict[str, float] = field(default_factory=dict)
    spice_budget_fraction: float = 1.0


def encode_sizing_state(ctx: SizingStateContext, action_dim: int) -> list[float]:
    """Deterministic state encoding; sizing vector is zero-padded to action_dim."""
    import math

    sizing = list(ctx.sizing_vector)[:action_dim]
    sizing += [0.0] * (action_dim - len(sizing))
    metrics = []
    for key in STATE_METRIC_KEYS:
        v = ctx.metrics.get(key, 0.0)
        metrics.append(math.copysign(math.log10(abs(v) + 1.0), v) / 12.0 if key == "gbw_hz" else v / 100.0)
    margins = [max(-2.0, min(2.0, ctx.constraint_margins.get(k, 0.0))) for k in STATE_MARGIN_KEYS]
    return (
        graph_feature_vector(ctx.graph)
        + sizing
        + _spec_embedding(ctx.spec)
        + metrics
        + margins
        + [float(ctx.spice_budget_fraction)]
    )


def sizing_state_dim(action_dim: int) -> int:
    from agentic_raptor.core.types import DeviceType

    graph_dim = len(tuple(DeviceType)) + 5
    spec_dim = len(DesignSpecifications.NUMERIC_FIELDS)
    return graph_dim + action_dim + spec_dim + len(STATE_METRIC_KEYS) + len(STATE_MARGIN_KEYS) + 1


class GraphConditionedMBSAC:
    """One agent instance per topology (the action space is topology-derived)."""

    def __init__(
        self,
        parameter_space: SizingParameterSpace,
        config: MBSACConfig | None = None,
    ) -> None:
        apply_torch_omp_workaround()
        import torch

        self.torch = torch
        self.config = config or MBSACConfig()
        self.space = parameter_space
        self.action_dim = max(parameter_space.dim, 1)
        self.state_dim = sizing_state_dim(self.action_dim)
        self.rng = make_rng(self.config.seed)
        torch.manual_seed(self.config.seed)

        cfg = self.config
        self.actor = build_gaussian_actor(self.state_dim, self.action_dim, cfg.hidden_dim)
        self.q1, self.q2, self.q1_target, self.q2_target = build_twin_critics(
            self.state_dim, self.action_dim, cfg.hidden_dim
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr
        )

        # Entropy temperature.
        self.target_entropy = (
            float(cfg.target_entropy) if cfg.target_entropy is not None else -float(self.action_dim)
        )
        if cfg.auto_alpha:
            self.log_alpha = torch.tensor(
                [float(torch.log(torch.tensor(cfg.init_alpha)))], requires_grad=True
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
        self._fixed_alpha = cfg.init_alpha

        dyn_cfg = DynamicsConfig(hidden_dim=cfg.hidden_dim, lr=cfg.lr)
        n_models = max(1, cfg.dynamics_ensemble_size)
        self.dynamics_models = [
            build_dynamics_model(self.state_dim, self.action_dim, dyn_cfg) for _ in range(n_models)
        ]
        self.dynamics_trainers = [DynamicsTrainer(m, dyn_cfg) for m in self.dynamics_models]
        # Stage 1 compatibility aliases (first ensemble member).
        self.dynamics = self.dynamics_models[0]
        self.dynamics_trainer = self.dynamics_trainers[0]
        self.replay = SizingReplayBuffer(capacity=cfg.replay_capacity)

    # -- policy -------------------------------------------------------------
    @property
    def alpha(self) -> float:
        if self.log_alpha is None:
            return self._fixed_alpha
        return float(self.log_alpha.exp().detach().item())

    def select_action(self, state: list[float], deterministic: bool = False) -> list[float]:
        torch = self.torch
        with torch.no_grad():
            s = torch.tensor([state], dtype=torch.float32)
            if deterministic:
                a = self.actor.deterministic_action(s)
            else:
                a, _ = self.actor.sample(s)
        return [float(v) for v in a.squeeze(0).tolist()]

    # -- data ---------------------------------------------------------------
    def add_transition(self, transition: SizingTransition) -> None:
        self.replay.add(transition)

    def generate_imagined_transitions(self, start_state: list[float], horizon: int | None = None) -> int:
        """Model rollout → replay entries labelled source='model'. Returns count."""
        rollout = self.dynamics_trainer.imagine_rollout(
            self.actor, start_state, horizon or self.config.rollout_horizon
        )
        for t in rollout:
            self.replay.add(SizingTransition(**t))
        return len(rollout)

    # -- learning -----------------------------------------------------------
    def update_dynamics(self, batch_size: int = 32) -> dict[str, float]:
        """One gradient step for EVERY ensemble member on REAL transitions only.

        Each member gets an independent bootstrap-style sample so their
        disagreement stays a meaningful uncertainty signal.
        """
        reports: list[dict[str, float]] = []
        for trainer in self.dynamics_trainers:
            real = [
                t for t in self.replay.sample_mixed(batch_size, self.rng, real_fraction=1.0)
                if t.source == "real"
            ]
            if not real:
                return {"skipped": 1.0}
            reports.append(
                trainer.train_on_batch(
                    states=[t.state for t in real],
                    actions=[t.action for t in real],
                    rewards=[t.reward for t in real],
                    next_states=[t.next_state for t in real],
                    dones=[t.done for t in real],
                )
            )
        merged = {k: sum(r[k] for r in reports) / len(reports) for k in reports[0]}
        merged["ensemble_size"] = float(len(reports))
        return merged

    def dynamics_uncertainty(self, state: list[float], action: list[float]) -> float:
        """Normalized next-state disagreement across the ensemble.

        Mean pairwise L2 distance between predicted next states, divided by
        sqrt(state_dim) so the scale is roughly per-feature. Single-member
        ensembles return 0.0 (uncertainty gating disabled).
        """
        if len(self.dynamics_models) < 2:
            return 0.0
        torch = self.torch
        s = torch.tensor([state], dtype=torch.float32)
        a = torch.tensor([action], dtype=torch.float32)
        with torch.no_grad():
            predictions = [model(s, a)[0] for model in self.dynamics_models]
        total, pairs = 0.0, 0
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                total += float(torch.dist(predictions[i], predictions[j]).item())
                pairs += 1
        return (total / pairs) / (self.state_dim ** 0.5) if pairs else 0.0

    def update(self, batch_size: int = 32) -> dict[str, float]:
        """One full SAC update: critics, actor, temperature, target networks."""
        torch = self.torch
        cfg = self.config
        batch = self.replay.sample_mixed(batch_size, self.rng, cfg.real_batch_fraction)
        if not batch:
            return {"skipped": 1.0}
        data = self.replay.as_batch(batch)
        s = torch.tensor(data["state"], dtype=torch.float32)
        a = torch.tensor(data["action"], dtype=torch.float32)
        r = torch.tensor(data["reward"], dtype=torch.float32).view(-1, 1)
        s2 = torch.tensor(data["next_state"], dtype=torch.float32)
        d = torch.tensor([1.0 if x else 0.0 for x in data["done"]], dtype=torch.float32).view(-1, 1)

        # --- critic update (entropy-regularised Bellman target) ---
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q_target = torch.min(self.q1_target(s2, a2), self.q2_target(s2, a2))
            y = r + cfg.gamma * (1.0 - d) * (q_target - self.alpha * logp2)
        q1_loss = torch.nn.functional.mse_loss(self.q1(s, a), y)
        q2_loss = torch.nn.functional.mse_loss(self.q2(s, a), y)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), cfg.grad_clip_norm
        )
        self.critic_optimizer.step()

        # --- actor update ---
        a_pi, logp_pi = self.actor.sample(s)
        q_pi = torch.min(self.q1(s, a_pi), self.q2(s, a_pi))
        actor_loss = (self.alpha * logp_pi - q_pi).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
        self.actor_optimizer.step()

        # --- entropy temperature update ---
        alpha_loss_value = 0.0
        if self.log_alpha is not None and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach().item())

        # --- Polyak target updates ---
        soft_update(self.q1, self.q1_target, cfg.tau)
        soft_update(self.q2, self.q2_target, cfg.tau)

        sources = data["source"]
        return {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss_value,
            "batch_real": float(sources.count("real")),
            "batch_model": float(sources.count("model")),
        }

    # -- introspection ------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "alpha": self.alpha,
            "replay": self.replay.counts(),
        }
