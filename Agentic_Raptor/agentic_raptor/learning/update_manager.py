"""Post-episode learning orchestration.

After each completed episode: credit-assign the topology trajectory, push it
into the topology replay buffer, run policy/value training steps, and report
what changed (parameter checksums included so callers can verify learning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from agentic_raptor.learning.cross_level_credit import CrossLevelCreditAssigner
from agentic_raptor.sizing.graph_conditioned_mb_sac import GraphConditionedMBSAC
from agentic_raptor.topology_rl.replay_buffer import TopologyReplayBuffer
from agentic_raptor.topology_rl.trainer import PolicyValueTrainer, parameter_checksum
from agentic_raptor.topology_rl.trajectory import TopologyTrajectory


@dataclass
class UpdateReport:
    credit: dict[str, Any] = field(default_factory=dict)
    policy_value: dict[str, Any] = field(default_factory=dict)
    sac: dict[str, Any] = field(default_factory=dict)
    dynamics: dict[str, Any] = field(default_factory=dict)
    parameters_changed: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)


class UpdateManager:
    def __init__(
        self,
        credit_assigner: CrossLevelCreditAssigner,
        topology_buffer: TopologyReplayBuffer,
        policy_value_trainer: PolicyValueTrainer | None,
        rng: Random,
        require_spice_for_training: bool = False,
    ) -> None:
        self.credit_assigner = credit_assigner
        self.topology_buffer = topology_buffer
        self.policy_value_trainer = policy_value_trainer
        self.rng = rng
        #: When True, trajectories whose final reward was not grounded in a
        #: SPICE evaluation (metadata reached_spice=False) are credited but
        #: NOT added to the topology training buffer. Default off (configurable).
        self.require_spice_for_training = require_spice_for_training

    def after_episode(
        self,
        trajectory: TopologyTrajectory,
        final_reward: float,
        sac: GraphConditionedMBSAC | None = None,
        policy_value_batch_size: int = 16,
        policy_value_steps: int = 1,
        sac_update_steps: int = 1,
        dynamics_update_steps: int = 1,
    ) -> UpdateReport:
        report = UpdateReport()

        # 1. Cross-level credit: final SPICE outcome → topology decisions.
        credit = self.credit_assigner.assign(trajectory, final_reward)
        report.credit = credit.to_dict()
        reached_spice = bool(trajectory.metadata.get("reached_spice", False))
        report.credit["reached_spice"] = reached_spice
        if self.require_spice_for_training and not reached_spice:
            report.credit["excluded_from_training"] = True
        else:
            self.topology_buffer.add_trajectory(trajectory)

        # 2. Topology policy/value gradient updates.
        if self.policy_value_trainer is not None and len(self.topology_buffer) > 0:
            before = parameter_checksum(self.policy_value_trainer.network)
            train_reports = []
            for _ in range(policy_value_steps):
                batch = self.topology_buffer.sample(policy_value_batch_size, self.rng)
                train_reports.append(self.policy_value_trainer.train_on_steps(batch).to_dict())
            after = parameter_checksum(self.policy_value_trainer.network)
            report.policy_value = {"steps": train_reports}
            report.parameters_changed["policy_value_network"] = before != after

        # 3. Sizing-level updates (dynamics first, then SAC on the mixture).
        if sac is not None:
            dyn_before = parameter_checksum(sac.dynamics)
            report.dynamics = {"steps": [sac.update_dynamics() for _ in range(dynamics_update_steps)]}
            report.parameters_changed["dynamics_model"] = dyn_before != parameter_checksum(sac.dynamics)

            actor_before = parameter_checksum(sac.actor)
            critic_before = parameter_checksum(sac.q1) + parameter_checksum(sac.q2)
            report.sac = {"steps": [sac.update() for _ in range(sac_update_steps)]}
            report.parameters_changed["sac_actor"] = actor_before != parameter_checksum(sac.actor)
            report.parameters_changed["sac_critics"] = (
                critic_before != parameter_checksum(sac.q1) + parameter_checksum(sac.q2)
            )
        return report
