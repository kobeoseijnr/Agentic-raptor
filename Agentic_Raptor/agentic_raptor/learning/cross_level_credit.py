"""Cross-level credit assignment: the key research mechanism.

The final post-sizing SPICE reward is propagated back onto every topology
decision in the trajectory as trajectory-level credit:

    z_t = γ^(T − 1 − t) · R_final   (+ optional shaping blend)

so the topology value network learns V(s) ≈ post-sizing achievable return and
the policy network learns from MCTS visit distributions whose subtrees were
evaluated under that value function.

Explicitly NOT done here: backpropagating SAC gradients through discrete
topology actions. Levels are coupled only through this scalar credit signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_raptor.topology_rl.trajectory import TopologyTrajectory


@dataclass
class CreditAssignmentReport:
    trajectory_id: str
    steps_credited: int
    final_reward: float
    returns: list[float]

    def to_dict(self) -> dict:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)


class CrossLevelCreditAssigner:
    """Attaches the final sizing outcome to all topology decisions."""

    def __init__(self, gamma: float = 0.97, shaping_blend: float = 0.0) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= shaping_blend < 1.0:
            raise ValueError("shaping_blend must be in [0, 1)")
        self.gamma = gamma
        #: fraction of the return taken from per-step shaping rewards stored in
        #: step.metadata["step_reward"]; 0.0 = pure final-outcome credit.
        self.shaping_blend = shaping_blend

    def assign(self, trajectory: TopologyTrajectory, final_reward: float) -> CreditAssignmentReport:
        trajectory.final_post_sizing_reward = final_reward
        horizon = len(trajectory.steps)
        returns: list[float] = []
        for index, step in enumerate(trajectory.steps):
            discounted_final = (self.gamma ** (horizon - 1 - index)) * final_reward
            if self.shaping_blend > 0.0:
                shaped = float(step.metadata.get("step_reward", 0.0))
                z = (1.0 - self.shaping_blend) * discounted_final + self.shaping_blend * shaped
            else:
                z = discounted_final
            step.final_post_sizing_reward = final_reward
            step.discounted_return = z
            returns.append(z)
        return CreditAssignmentReport(
            trajectory_id=trajectory.trajectory_id,
            steps_credited=horizon,
            final_reward=final_reward,
            returns=returns,
        )
