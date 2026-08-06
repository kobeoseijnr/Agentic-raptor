"""Typed YAML configuration loading.

One dataclass per subsystem; ``AgenticConfig.from_yaml`` builds the whole tree
and fails loudly on unknown keys (catching typos early). All stochastic
behaviour flows from ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from agentic_raptor.utils.exceptions import ConfigurationError

T = TypeVar("T")


def _build(cls: type[T], data: dict[str, Any], context: str) -> T:
    from typing import get_type_hints

    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ConfigurationError(f"{context}: unknown keys {sorted(unknown)}")
    # PEP 563: field.type is a string under `from __future__ import annotations`;
    # resolve to real types so nested config dataclasses are constructed.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints.get(name)
        if isinstance(hint, type) and is_dataclass(hint):
            if not isinstance(value, dict):
                raise ConfigurationError(f"{context}.{name}: expected mapping")
            kwargs[name] = _build(hint, value, f"{context}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    decision_log: str = "outputs/decisions.jsonl"


@dataclass
class SpecificationConfig:
    text: str | None = None
    structured_specification_path: str | None = None
    table_path: str | None = None
    schematic_image_path: str | None = None
    netlist_path: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphLimitsConfig:
    max_nodes: int = 200
    max_edges: int = 800


@dataclass
class ActionLimitsConfig:
    max_edits: int = 6
    max_actions: int = 32
    invalid_action_penalty: float = -0.5


@dataclass
class MCTSSettings:
    num_simulations: int = 24
    c_puct: float = 1.5
    max_depth: int = 4
    pw_c: float = 2.0
    pw_alpha: float = 0.5
    pw_min: int = 3


@dataclass
class PolicyValueSettings:
    hidden_dim: int = 64
    encoder: str = "pooled"          # "pooled" | "message_passing"
    lr: float = 1.0e-3
    value_loss_weight: float = 1.0
    batch_size: int = 16
    updates_per_episode: int = 2
    #: Stage 2: exclude trajectories that never reached SPICE from training.
    require_spice_for_training: bool = False


@dataclass
class SACSettings:
    hidden_dim: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3.0e-4
    init_alpha: float = 0.2
    auto_alpha: bool = True
    real_batch_fraction: float = 0.8
    rollout_horizon: int = 3
    batch_size: int = 16
    updates_per_episode: int = 2
    dynamics_updates_per_episode: int = 2
    sizing_steps: int = 4
    # --- Stage 2: budgeted real-SPICE query policy ---
    dynamics_ensemble_size: int = 2
    real_spice_warmup_transitions: int = 2   # first K transitions per topology must be real
    dynamics_uncertainty_threshold: float = 0.5  # normalized disagreement above → query real SPICE
    real_query_interval: int = 3             # periodic real check every N model-guided steps


@dataclass
class SpiceConfig:
    simulator: str = "mock"          # "mock" | "ngspice"
    ngspice_exe: str | None = None   # no machine-specific default; discovery used when None
    analyses: list[str] = field(default_factory=lambda: ["op", "ac"])
    timeout_s: float = 10.0
    cache_path: str | None = None
    cache_enabled: bool = True
    pvt_corners: int = 3             # how many standard corners to run
    # --- Stage 2 ---
    model_library_path: str | None = None   # external model cards; None → built-in generic library
    technology_label: str = "generic_1u_level1"  # cache-key component; set when swapping libraries
    keep_workdirs: bool = False      # keep temporary simulation directories for debugging
    allow_mock_fallback: bool = False  # explicit opt-in: fall back to mock when real SPICE unavailable


@dataclass
class BudgetConfig:
    max_topology_edits: int = 6
    max_sizing_steps: int = 8
    max_spice_calls: int = 12
    max_runtime_s: float = 120.0
    max_generations: int = 3
    max_retrievals: int = 5


@dataclass
class RewardWeightsConfig:
    feasibility_weight: float = 1.0
    fom_weight: float = 0.5
    pvt_weight: float = 0.3
    spice_cost_weight: float = 0.1
    runtime_weight: float = 0.05
    invalidity_weight: float = 0.5
    validity_weight: float = 0.2
    sizing_progress_weight: float = 0.2
    fom_scale: float = 200.0
    credit_gamma: float = 0.97
    shaping_blend: float = 0.0


@dataclass
class RAGConfig:
    persist_path: str | None = None
    top_k: int = 3
    include_failures: bool = True
    seed_corpus: bool = True


@dataclass
class LLMConfig:
    generator: str = "mock"          # "mock" | "llm"
    number_of_candidates: int = 2
    max_retries: int = 2
    # --- Stage 2 provider settings (credentials come ONLY from env vars) ---
    provider: str = "openai_compatible"      # "mock" | "openai_compatible"
    model_name: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"      # name of the env var holding the key
    base_url_env: str = "AGENTIC_RAPTOR_LLM_BASE_URL"  # env var overriding the API base URL
    provider_timeout_s: float = 60.0
    provider_retries: int = 2
    image_parsing_enabled: bool = False      # provider-backed schematic understanding
    real_llm_enabled: bool = False           # gate: generator="llm" also requires this


@dataclass
class DPOSettings:
    enabled: bool = False
    checkpoint_path: str | None = None
    update_mode: str = "periodic"          # "periodic" | "fixed"
    minimum_pairs_before_training: int = 8
    update_interval_episodes: int = 2
    beta: float = 2.0
    learning_rate: float = 1.0e-3
    batch_size: int = 16
    epochs: int = 20
    confidence_threshold: float = 0.4
    tie_margin: float = 0.02
    exploration_fraction: float = 0.2
    maximum_pairs_per_specification: int = 50
    hidden_dim: int = 64
    pool_size: int = 4
    outcome_store_path: str | None = None  # default: <output_dir>/dpo_outcomes.jsonl


@dataclass
class CoordinatorConfig:
    accept_reward_threshold: float = 0.5
    stagnation_patience: int = 3
    max_repeated_failures: int = 3
    max_decisions: int = 40


@dataclass
class AgenticConfig:
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "outputs"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    specification: SpecificationConfig = field(default_factory=SpecificationConfig)
    graph_limits: GraphLimitsConfig = field(default_factory=GraphLimitsConfig)
    actions: ActionLimitsConfig = field(default_factory=ActionLimitsConfig)
    mcts: MCTSSettings = field(default_factory=MCTSSettings)
    policy_value: PolicyValueSettings = field(default_factory=PolicyValueSettings)
    sac: SACSettings = field(default_factory=SACSettings)
    spice: SpiceConfig = field(default_factory=SpiceConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    dpo: DPOSettings = field(default_factory=DPOSettings)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgenticConfig:
        return _build(cls, data, "config")

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgenticConfig:
        import yaml

        p = Path(path)
        if not p.is_file():
            raise ConfigurationError(f"config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"{p}: top level must be a mapping")
        config = cls.from_dict(data)
        config._base_dir = str(p.resolve().parent)  # type: ignore[attr-defined]
        return config

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config-relative path against the config file location."""
        base = getattr(self, "_base_dir", None)
        p = Path(relative)
        if p.is_absolute() or base is None:
            return p
        return Path(base) / p
