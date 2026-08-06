"""Guarded access to the legacy MB-SAC stack (audited genuine, §2.12 of the audit)."""

from __future__ import annotations

from typing import Any

from agentic_raptor.adapters.legacy_raptor import ensure_repo_root_on_path, legacy_available


def is_available() -> bool:
    return legacy_available("mb_sac.sac_agent")


def create_legacy_sac_agent(state_dim: int, action_dim: int, **overrides: Any) -> Any:
    """Instantiate the verified legacy SACAgent (read-only import)."""
    ensure_repo_root_on_path()
    from mb_sac.sac_agent import SACAgent, SACConfig  # noqa: PLC0415

    return SACAgent(SACConfig(action_dim=action_dim, state_dim=state_dim, **overrides))


def legacy_replay_classes() -> tuple[Any, Any]:
    """(ReplayBuffer, Transition) from the legacy stack."""
    ensure_repo_root_on_path()
    from mb_sac.replay_buffer import ReplayBuffer, Transition  # noqa: PLC0415

    return ReplayBuffer, Transition
