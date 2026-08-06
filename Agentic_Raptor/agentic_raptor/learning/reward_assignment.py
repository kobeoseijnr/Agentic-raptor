"""Configurable final-reward computation. Weights come from YAML, never code."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from agentic_raptor.core.budgets import BudgetState
from agentic_raptor.core.rewards import RewardComponents, feasibility_score, normalize_fom
from agentic_raptor.spice.interface import SimulationResult
from agentic_raptor.spice.pvt import PvtResult
from agentic_raptor.topology_validation.validator import ValidationResult
from agentic_raptor.utils.exceptions import ConfigurationError


@dataclass(frozen=True)
class RewardWeights:
    feasibility_weight: float = 1.0
    fom_weight: float = 0.5
    pvt_weight: float = 0.3
    spice_cost_weight: float = 0.1
    runtime_weight: float = 0.05
    invalidity_weight: float = 0.5
    validity_weight: float = 0.2
    sizing_progress_weight: float = 0.2
    fom_scale: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardWeights:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"unknown reward weight keys: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in data.items()})

    @classmethod
    def from_yaml(cls, path: str | Path) -> RewardWeights:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        section = data.get("reward_weights", data)
        if not isinstance(section, dict):
            raise ConfigurationError(f"no reward_weights mapping found in {path}")
        return cls.from_dict(section)


def compute_figure_of_merit(sim: SimulationResult, load_capacitance_f: float | None) -> float:
    """Classic OTA FoM = GBW · C_L / Power  (MHz·pF/mW ≈ dimensionless scale)."""
    gbw = sim.metrics.get("gbw_hz", 0.0)
    power = sim.metrics.get("power_w", 0.0)
    cl = load_capacitance_f or 1e-12
    if power <= 0:
        return 0.0
    return (gbw / 1e6) * (cl / 1e-12) / (power / 1e-3)


def assemble_components(
    validation: ValidationResult | None,
    sizing_progress: float,
    sim: SimulationResult | None,
    pvt: PvtResult | None,
    budgets: BudgetState,
    invalid_action_count: int,
    load_capacitance_f: float | None,
    fom_scale: float = 1.0,
) -> RewardComponents:
    """Collect unweighted reward signals from one completed episode."""
    components = RewardComponents()
    if validation is not None:
        components.topology_validity = 1.0 if validation.is_valid else 0.0
        components.topology_validity -= 0.05 * len(validation.warnings)
    components.sizing_progress = max(-1.0, min(1.0, sizing_progress))
    if sim is not None and sim.success:
        components.spice_feasibility = feasibility_score(sim.constraint_margins)
        components.normalized_fom = normalize_fom(
            compute_figure_of_merit(sim, load_capacitance_f), scale=fom_scale
        )
    if pvt is not None:
        components.pvt_robustness = pvt.pvt_score
    spice_cap = max(budgets.max_spice_calls, 1)
    components.normalized_spice_calls = budgets.used_spice_calls / spice_cap
    runtime_cap = max(budgets.max_runtime_s, 1e-9)
    components.normalized_runtime = min(1.0, budgets.elapsed_s() / runtime_cap)
    components.invalidity_penalty = float(invalid_action_count)
    return components


def compute_final_reward(components: RewardComponents, weights: RewardWeights) -> float:
    """The YAML-weighted final scalar (the configured formula, verbatim)."""
    return (
        weights.feasibility_weight * components.spice_feasibility
        + weights.fom_weight * components.normalized_fom
        + weights.pvt_weight * components.pvt_robustness
        + weights.validity_weight * components.topology_validity
        + weights.sizing_progress_weight * components.sizing_progress
        - weights.spice_cost_weight * components.normalized_spice_calls
        - weights.runtime_weight * components.normalized_runtime
        - weights.invalidity_weight * components.invalidity_penalty
    )
