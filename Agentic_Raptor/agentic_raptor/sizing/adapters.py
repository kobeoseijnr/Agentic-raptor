"""Adapters to the legacy RAPTOR MB-SAC stack (audited genuine — see
docs/REPOSITORY_AUDIT.md §2.12). Read-only wrappers; never modify legacy code."""

from __future__ import annotations

from typing import Any

from agentic_raptor.adapters.legacy_raptor import ensure_repo_root_on_path, legacy_available


class LegacySACBackend:
    """Optional backend using the verified ``mb_sac.sac_agent.SACAgent``.

    Use when comparing against legacy checkpoints. Limitations vs
    GraphConditionedMBSAC: fixed dims per instance, no dynamics model with
    reward/termination heads (the legacy model-based part uses the offline
    surrogate ensemble at horizon 1).
    """

    @staticmethod
    def is_available() -> bool:
        return legacy_available("mb_sac.sac_agent")

    @staticmethod
    def create(state_dim: int, action_dim: int, **overrides: Any) -> Any:
        ensure_repo_root_on_path()
        from mb_sac.sac_agent import SACAgent, SACConfig  # noqa: PLC0415

        config = SACConfig(action_dim=action_dim, state_dim=state_dim, **overrides)
        return SACAgent(config)


class LegacySurrogateBackend:
    """Optional sizing-outcome estimator from the legacy surrogate ensemble."""

    @staticmethod
    def is_available() -> bool:
        return legacy_available("surrogate.surrogate_ensemble")

    @staticmethod
    def load(checkpoint_dir: str) -> Any:
        ensure_repo_root_on_path()
        from surrogate.surrogate_ensemble import SurrogateEnsemble  # noqa: PLC0415

        # TODO(stage-2): map legacy checkpoint layout; requires a results/ path
        # which we treat as read-only.
        return SurrogateEnsemble.load(checkpoint_dir)  # type: ignore[attr-defined]
