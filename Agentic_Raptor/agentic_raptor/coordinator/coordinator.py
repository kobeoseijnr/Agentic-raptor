"""Agentic coordinator: runs one full design episode.

Pipeline (the smoke test executes exactly this):
  parse multimodal input → retrieve → generate → validate → MCTS-guided edits
  → policy/value update → graph-conditioned MB-SAC sizing (real transitions,
  dynamics training, imagined rollout, SAC updates) → mock SPICE (+ PVT)
  → final reward → cross-level credit → memory update → stop.

Every decision is logged with its reason to a JSONL trace.

Sizing-step evaluations use the (cheap, deterministic) simulator as a stand-in
for a surrogate predictor and do not consume the SPICE budget; only RUN_SPICE
and RUN_PVT decisions consume SPICE calls. SPICE remains the final authority
for the reported reward.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from agentic_raptor.coordinator.budget_manager import BudgetManager
from agentic_raptor.coordinator.decision_policy import (
    DecisionContext,
    DecisionPolicy,
    RuleBasedDecisionPolicy,
)
from agentic_raptor.coordinator.state_machine import (
    CoordinatorState,
    Decision,
    StateMachine,
)
from agentic_raptor.core.budgets import BudgetState
from agentic_raptor.core.candidate import CircuitCandidate
from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.core.types import CandidateStatus, GenerationSource
from agentic_raptor.learning.cross_level_credit import CrossLevelCreditAssigner
from agentic_raptor.learning.reward_assignment import (
    RewardWeights,
    assemble_components,
    compute_figure_of_merit,
    compute_final_reward,
)
from agentic_raptor.learning.update_manager import UpdateManager
from agentic_raptor.rag.memory import CircuitMemory, spec_embedding
from agentic_raptor.rag.retriever import Retriever, seed_memory_for_smoke
from agentic_raptor.rag.schemas import MemoryEntry
from agentic_raptor.sizing.graph_conditioned_mb_sac import (
    GraphConditionedMBSAC,
    MBSACConfig,
    SizingStateContext,
    encode_sizing_state,
)
from agentic_raptor.sizing.parameter_space import SizingParameterSpace
from agentic_raptor.sizing.replay_buffer import SizingTransition
from agentic_raptor.specification import MultimodalDesignInput, parse_design_input
from agentic_raptor.spice.cache import SimulationCache
from agentic_raptor.spice.interface import SimulationResult
from agentic_raptor.spice.pvt import PvtResult, run_pvt, standard_corners
from agentic_raptor.spice.simulator_adapter import MockSpiceSimulator
from agentic_raptor.topology_generation.generator import (
    MockTopologyGenerator,
    TopologyGenerator,
)
from agentic_raptor.topology_rl.environment import TopologyEditEnv, TopologyEnvConfig
from agentic_raptor.topology_rl.mcts import MCTS, MCTSConfig
from agentic_raptor.topology_rl.policy_value_network import (
    NetworkEvaluator,
    PolicyValueConfig,
    build_policy_value_network,
    encode_state_features,
)
from agentic_raptor.topology_rl.replay_buffer import TopologyReplayBuffer
from agentic_raptor.topology_rl.trainer import PolicyValueTrainer
from agentic_raptor.topology_rl.trajectory import TopologyTrajectory, TrajectoryStep
from agentic_raptor.topology_validation.validator import TopologyValidator
from agentic_raptor.utils.config import AgenticConfig
from agentic_raptor.utils.exceptions import (
    BudgetExhaustedError,
    CoordinatorError,
    GenerationError,
    SpecificationError,
)
from agentic_raptor.utils.logging import JsonlEventLog, get_logger
from agentic_raptor.utils.seeding import make_rng, seed_everything
from agentic_raptor.utils.serialization import dump_json

logger = get_logger("agentic_raptor.coordinator")


@dataclass
class EpisodeResult:
    episode_id: str
    success: bool
    final_state: str
    final_reward: float | None = None
    best_candidate: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    update_report: dict[str, Any] = field(default_factory=dict)
    budget_snapshot: dict[str, Any] = field(default_factory=dict)
    memory_id: str | None = None
    summary_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        from agentic_raptor.utils.serialization import to_jsonable

        return to_jsonable(self)


class AgenticCoordinator:
    """Rule-based coordinator; the decision policy is injectable (learnable later)."""

    def __init__(
        self,
        config: AgenticConfig,
        generator: TopologyGenerator | None = None,
        memory: CircuitMemory | None = None,
        decision_policy: DecisionPolicy | None = None,
        simulator: Any | None = None,
    ) -> None:
        self.config = config
        seed_everything(config.seed)
        self.rng = make_rng(config.seed)

        out_dir = config.resolve_path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = out_dir
        self.decision_log = JsonlEventLog(config.resolve_path(config.logging.decision_log))

        self.memory = memory or CircuitMemory(
            config.resolve_path(config.rag.persist_path) if config.rag.persist_path else None
        )
        self.retriever = Retriever(self.memory, k=config.rag.top_k, include_failures=config.rag.include_failures)
        self.validator = TopologyValidator(
            max_nodes=config.graph_limits.max_nodes, max_edges=config.graph_limits.max_edges
        )
        self.generator = generator or self._build_generator(config)
        self.simulator, self.simulator_mode = self._build_simulator(config, simulator)
        self.cache = SimulationCache(
            config.resolve_path(config.spice.cache_path) if config.spice.cache_path else None,
            enabled=config.spice.cache_enabled,
        )
        self.policy = decision_policy or RuleBasedDecisionPolicy()

        # Topology RL stack.
        pv_cfg = PolicyValueConfig(
            max_actions=config.actions.max_actions,
            hidden_dim=config.policy_value.hidden_dim,
            encoder=config.policy_value.encoder,
            lr=config.policy_value.lr,
            value_loss_weight=config.policy_value.value_loss_weight,
            device=config.device,
        )
        self.pv_config = pv_cfg
        self.pv_network = build_policy_value_network(pv_cfg)
        self.pv_trainer = PolicyValueTrainer(self.pv_network, pv_cfg)
        self.pv_evaluator = NetworkEvaluator(self.pv_network, pv_cfg)
        self.topology_buffer = TopologyReplayBuffer()
        self.mcts = MCTS(
            self.pv_evaluator,
            self.validator,
            MCTSConfig(
                num_simulations=config.mcts.num_simulations,
                c_puct=config.mcts.c_puct,
                max_depth=config.mcts.max_depth,
                max_actions=config.actions.max_actions,
                pw_c=config.mcts.pw_c,
                pw_alpha=config.mcts.pw_alpha,
                pw_min=config.mcts.pw_min,
                seed=config.seed,
            ),
        )
        self.credit_assigner = CrossLevelCreditAssigner(
            gamma=config.reward_weights.credit_gamma,
            shaping_blend=config.reward_weights.shaping_blend,
        )
        self.update_manager = UpdateManager(
            self.credit_assigner,
            self.topology_buffer,
            self.pv_trainer,
            self.rng,
            require_spice_for_training=config.policy_value.require_spice_for_training,
        )
        # Pre-SPICE DPO preference ranking (disabled → original pipeline).
        self.dpo_ranker = None
        self.dpo_store = None
        self._dpo_episode_counter = 0
        if config.dpo.enabled:
            from agentic_raptor.dpo import FEATURE_DIM, DPOConfig, DPORanker, OutcomeStore

            dpo_cfg = DPOConfig(
                **{
                    k: v
                    for k, v in vars(config.dpo).items()
                    if k not in ("pool_size", "outcome_store_path")
                }
            )
            self.dpo_config = dpo_cfg
            self.dpo_ranker = DPORanker(FEATURE_DIM, dpo_cfg)
            if config.dpo.checkpoint_path:
                ckpt = config.resolve_path(config.dpo.checkpoint_path)
                if ckpt.is_file():
                    self.dpo_ranker.load(ckpt)
            store_path = config.dpo.outcome_store_path or f"{config.output_dir}/dpo_outcomes.jsonl"
            self.dpo_store = OutcomeStore(config.resolve_path(store_path))

        self.reward_weights = RewardWeights.from_dict(
            {
                k: v
                for k, v in vars(config.reward_weights).items()
                if k not in ("credit_gamma", "shaping_blend")
            }
        )

    # ------------------------------------------------------------------
    def _build_simulator(self, config: AgenticConfig, injected: Any | None) -> tuple[Any, str]:
        """Simulator per config. Real→mock fallback ONLY when explicitly allowed."""
        if injected is not None:
            return injected, getattr(injected, "mode_label", config.spice.simulator)
        if config.spice.simulator == "ngspice":
            from agentic_raptor.spice.ngspice_simulator import NgspiceSimulator

            sim = NgspiceSimulator(
                ngspice_exe=config.spice.ngspice_exe,
                model_library_path=(
                    str(config.resolve_path(config.spice.model_library_path))
                    if config.spice.model_library_path
                    else None
                ),
                technology_label=config.spice.technology_label,
                keep_workdirs=config.spice.keep_workdirs,
                seed=config.seed,
            )
            if not sim.is_available():
                if config.spice.allow_mock_fallback:
                    logger.warning(
                        "ngspice unavailable; falling back to mock (allow_mock_fallback=true)"
                    )
                    return MockSpiceSimulator(seed=config.seed), "mock"
                # Keep the real backend: every simulate() returns a structured
                # simulator_unavailable failure — never a silent mock swap.
                logger.warning("ngspice unavailable; runs will fail with simulator_unavailable")
            return sim, "ngspice"
        return MockSpiceSimulator(seed=config.seed), "mock"

    def _build_generator(self, config: AgenticConfig) -> Any:
        if config.llm.generator == "llm" and config.llm.real_llm_enabled:
            from agentic_raptor.topology_generation.generator import LLMTopologyGenerator
            from agentic_raptor.topology_generation.provider import (
                ProviderSettings,
                build_provider,
            )

            settings = ProviderSettings(
                provider=config.llm.provider,
                model_name=config.llm.model_name,
                api_key_env=config.llm.api_key_env,
                base_url_env=config.llm.base_url_env,
                timeout_s=config.llm.provider_timeout_s,
                retries=config.llm.provider_retries,
            )
            return LLMTopologyGenerator(
                build_provider(settings), self.validator, config.llm.max_retries
            )
        return MockTopologyGenerator(seed=config.seed)

    # ------------------------------------------------------------------
    def run_episode(
        self,
        design_input: MultimodalDesignInput | None = None,
        specifications: DesignSpecifications | None = None,
    ) -> EpisodeResult:
        cfg = self.config
        episode_id = f"ep-{uuid.uuid4().hex[:10]}"
        sm = StateMachine()
        budgets = BudgetState(
            max_topology_edits=cfg.budgets.max_topology_edits,
            max_sizing_steps=cfg.budgets.max_sizing_steps,
            max_spice_calls=cfg.budgets.max_spice_calls,
            max_runtime_s=cfg.budgets.max_runtime_s,
            max_generations=cfg.budgets.max_generations,
            max_retrievals=cfg.budgets.max_retrievals,
        )
        manager = BudgetManager(
            budgets,
            stagnation_patience=cfg.coordinator.stagnation_patience,
            max_repeated_failures=cfg.coordinator.max_repeated_failures,
        )
        result = EpisodeResult(episode_id=episode_id, success=False, final_state=sm.state.value)

        def log_decision(decision: str, reason: str, state: str, extra: dict[str, Any] | None = None) -> None:
            event = {
                "episode_id": episode_id,
                "state": state,
                "decision": decision,
                "reason": reason,
                "budgets": budgets.snapshot(),
                **(extra or {}),
            }
            self.decision_log.append(event)
            result.decisions.append(event)
            logger.info("[%s] %s → %s: %s", episode_id, state, decision, reason)

        # ---- PARSE_SPECIFICATIONS ----
        sm.transition(CoordinatorState.PARSE_SPECIFICATIONS)
        try:
            if specifications is not None:
                spec = specifications
                log_decision("USE_PROVIDED_SPEC", "specifications object supplied directly", sm.state.value)
            else:
                if design_input is None:
                    design_input = MultimodalDesignInput(
                        text=cfg.specification.text,
                        structured_specification_path=_maybe_path(cfg, cfg.specification.structured_specification_path),
                        table_path=_maybe_path(cfg, cfg.specification.table_path),
                        schematic_image_path=_maybe_path(cfg, cfg.specification.schematic_image_path),
                        netlist_path=_maybe_path(cfg, cfg.specification.netlist_path),
                    )
                spec, fusion_report, review = parse_design_input(
                    design_input, defaults=dict(cfg.specification.defaults)
                )
                log_decision(
                    "SPECIFICATIONS_PARSED",
                    f"fused {fusion_report.extracted_count} fields from "
                    f"{design_input.available_modalities()}; {len(fusion_report.conflicts)} conflicts; "
                    f"{len(review.checks)} plausibility warnings",
                    sm.state.value,
                    {"fusion": fusion_report.to_dict(), "review": review.to_dict()},
                )
        except SpecificationError as exc:
            sm.transition(CoordinatorState.FAILED)
            result.final_state = sm.state.value
            result.error = f"specification parsing failed: {exc}"
            log_decision("FAIL", result.error, sm.state.value)
            return self._finalize(result, budgets)

        # ---- RETRIEVE ----
        sm.transition(CoordinatorState.RETRIEVE)
        if cfg.rag.seed_corpus and len(self.memory) == 0:
            seeded = seed_memory_for_smoke(self.memory, spec)
            log_decision("SEED_MEMORY", f"seeded {len(seeded)} experience entries", sm.state.value)
        budgets.consume("retrievals")
        retrieval = self.retriever.retrieve(spec)
        log_decision(
            Decision.RETRIEVE_MORE.value,
            f"retrieved {len(retrieval.entries)} entries "
            f"({len(retrieval.successes())} successes, {len(retrieval.failures())} failures)",
            sm.state.value,
            {"retrieval": retrieval.to_dict()},
        )

        # ---- episode working state ----
        candidate: CircuitCandidate | None = None
        #: best SPICE-grounded evaluation so far: (reward, candidate, sim, pvt, progress)
        best_eval: tuple[float, CircuitCandidate, SimulationResult, PvtResult | None, float] | None = None
        env: TopologyEditEnv | None = None
        trajectory = TopologyTrajectory(trajectory_id=f"traj-{episode_id}")
        sac: GraphConditionedMBSAC | None = None
        space: SizingParameterSpace | None = None
        sizing_vector: list[float] = []
        sim_result: SimulationResult | None = None
        pvt_result: PvtResult | None = None
        invalid_actions = 0
        sizing_progress = 0.0
        first_margin_score: float | None = None
        dpo_pending: tuple[Any, Any] | None = None  # (SelectionResult, CandidateFeatures)

        sm.transition(CoordinatorState.GENERATE)

        for _decision_index in range(cfg.coordinator.max_decisions):
            ctx = self._build_context(candidate, sim_result, pvt_result, manager, budgets, retrieval, spec)
            decision, reason = self.policy.decide(ctx)
            log_decision(decision.value, reason, sm.state.value)

            if decision in (Decision.STOP_BUDGET_EXHAUSTED, Decision.STOP_SUCCESS, Decision.ACCEPT_CANDIDATE):
                if candidate is not None and decision != Decision.STOP_BUDGET_EXHAUSTED:
                    candidate.status = CandidateStatus.ACCEPTED
                break

            try:
                if decision == Decision.RETRIEVE_MORE:
                    budgets.consume("retrievals")
                    retrieval = self.retriever.retrieve(spec)

                elif decision == Decision.GENERATE_NEW_TOPOLOGY:
                    if sm.state != CoordinatorState.GENERATE:
                        sm.transition(CoordinatorState.GENERATE)
                    budgets.consume("generations")
                    try:
                        graphs = self.generator.generate(
                            spec, [s.entry for s in retrieval.entries], design_input, cfg.llm.number_of_candidates
                        )
                    except GenerationError as exc:
                        kind = getattr(exc, "kind", "generation_failed")
                        log_decision(
                            "GENERATION_FAILED",
                            f"{kind}: {exc} (retry limit inside generator reached)",
                            sm.state.value,
                        )
                        manager.record_failure()
                        continue
                    sm.transition(CoordinatorState.VALIDATE)
                    candidate = None
                    for graph in graphs:
                        validation = self.validator.validate(graph)
                        if validation.is_valid:
                            candidate = CircuitCandidate.create(graph, spec, GenerationSource.MOCK)
                            candidate.validation_result = validation.to_dict()
                            candidate.status = CandidateStatus.VALIDATED
                            break
                    if candidate is None and graphs:
                        graph = graphs[0]
                        candidate = CircuitCandidate.create(graph, spec, GenerationSource.MOCK)
                        candidate.validation_result = self.validator.validate(graph).to_dict()
                        candidate.status = CandidateStatus.INVALID
                    env = None
                    sac = None
                    sim_result = None
                    pvt_result = None
                    sizing_vector = []

                elif decision == Decision.REPAIR_OR_EDIT_TOPOLOGY:
                    assert candidate is not None
                    if sm.state != CoordinatorState.EDIT_TOPOLOGY:
                        sm.transition(CoordinatorState.EDIT_TOPOLOGY)
                    if env is None:
                        env = TopologyEditEnv(
                            candidate.topology,
                            spec,
                            self.validator,
                            TopologyEnvConfig(
                                max_edits=cfg.budgets.max_topology_edits,
                                max_actions=cfg.actions.max_actions,
                                invalid_action_penalty=cfg.actions.invalid_action_penalty,
                            ),
                        )
                        env.reset()
                    invalid_actions += self._mcts_edit_step(env, spec, trajectory, budgets)
                    assert env.state is not None
                    candidate.topology = env.state.graph
                    candidate.validation_result = env.state.validation.to_dict()
                    candidate.edit_history = list(env.state.edit_records)
                    candidate.status = (
                        CandidateStatus.VALIDATED if env.state.validation.is_valid else CandidateStatus.INVALID
                    )
                    sm.transition(CoordinatorState.VALIDATE)

                elif decision == Decision.CONTINUE_SIZING:
                    assert candidate is not None
                    if sm.state != CoordinatorState.SIZE:
                        sm.transition(CoordinatorState.SIZE)
                    if sac is None or space is None:
                        space = SizingParameterSpace.from_graph(candidate.topology)
                        sac = GraphConditionedMBSAC(
                            space,
                            MBSACConfig(
                                hidden_dim=cfg.sac.hidden_dim,
                                gamma=cfg.sac.gamma,
                                tau=cfg.sac.tau,
                                lr=cfg.sac.lr,
                                init_alpha=cfg.sac.init_alpha,
                                auto_alpha=cfg.sac.auto_alpha,
                                real_batch_fraction=cfg.sac.real_batch_fraction,
                                rollout_horizon=cfg.sac.rollout_horizon,
                                seed=cfg.seed,
                            ),
                        )
                        sizing_vector = space.default_vector()
                    progress = self._sizing_round(candidate, spec, sac, space, sizing_vector, budgets)
                    sizing_vector = progress["sizing_vector"]
                    sizing_progress = progress["progress"]
                    if first_margin_score is None:
                        first_margin_score = progress["first_margin_score"]
                    # --- pre-SPICE DPO ranking over the candidate pool ---
                    if self.dpo_ranker is not None and len(progress.get("pool", [])) >= 2:
                        selection, features = self._dpo_select(candidate, progress["pool"], budgets)
                        sizing_vector = selection.selected.sizing_vector
                        dpo_pending = (selection, features)
                        log_decision(
                            "DPO_SELECTION",
                            f"pool={len(progress['pool'])} selected rank {selection.dpo_rank} "
                            f"(score {selection.dpo_score:.3f}): {selection.reason}",
                            sm.state.value,
                        )
                    candidate.sizing_state = space.denormalize(sizing_vector)
                    candidate.topology.apply_sizing(candidate.sizing_state)
                    candidate.status = CandidateStatus.SIZING

                elif decision == Decision.RUN_SPICE:
                    assert candidate is not None
                    if sm.state != CoordinatorState.SIMULATE:
                        sm.transition(CoordinatorState.SIMULATE)
                    budgets.consume("spice_calls")
                    bound_candidate = candidate
                    sim_result, was_cached = self.cache.get_or_run(
                        candidate,
                        cfg.spice.analyses,
                        "typical",
                        self._simulator_fingerprint(spec),
                        lambda c=bound_candidate: self.simulator.simulate(c, cfg.spice.analyses, cfg.spice.timeout_s),
                    )
                    candidate.simulation_result = sim_result.to_dict()
                    candidate.status = CandidateStatus.SIMULATED
                    result.metrics = dict(sim_result.metrics)
                    if dpo_pending is not None and self.dpo_store is not None:
                        self._record_dpo_outcome(dpo_pending, sim_result, pvt_result, budgets, episode_id, spec)
                        dpo_pending = None
                    if sim_result.success:
                        log_decision(
                            "SPICE_RESULT",
                            f"backend={self.simulator_mode} success=True cached={was_cached} "
                            f"margins={ {k: round(v, 3) for k, v in sim_result.constraint_margins.items()} }",
                            sm.state.value,
                        )
                    else:
                        log_decision(
                            "SPICE_FAILURE",
                            f"backend={self.simulator_mode} error_type={sim_result.error_type}: "
                            f"{sim_result.error_message}",
                            sm.state.value,
                        )
                        manager.record_failure()
                        if sim_result.error_type == "simulator_unavailable":
                            sim_result = None  # cannot ground any reward in SPICE
                    sm.transition(CoordinatorState.DIAGNOSE)

                elif decision == Decision.RUN_PVT:
                    assert candidate is not None
                    corners = standard_corners()[: cfg.spice.pvt_corners]
                    corners = corners[: max(1, min(len(corners), budgets.remaining("spice_calls")))]
                    if sm.state != CoordinatorState.SIMULATE:
                        sm.transition(CoordinatorState.SIMULATE)
                    budgets.consume("spice_calls", len(corners))
                    pvt_result = run_pvt(self.simulator, candidate, cfg.spice.analyses, corners, cfg.spice.timeout_s)
                    log_decision(
                        "PVT_RESULT",
                        f"pvt_score={pvt_result.pvt_score:.2f} over {len(corners)} corners; "
                        f"worst margins={ {k: round(v, 3) for k, v in pvt_result.worst_case_margins.items()} }",
                        sm.state.value,
                    )
                    sm.transition(CoordinatorState.DIAGNOSE)

            except BudgetExhaustedError as exc:
                log_decision("BUDGET_HIT", str(exc), sm.state.value)
                manager.record_failure()
                continue

            # Track reward trend after each simulate/diagnose round; remember the
            # best SPICE-grounded candidate so regeneration cannot lose it.
            if sim_result is not None and candidate is not None:
                interim = self._final_reward(candidate, sizing_progress, sim_result, pvt_result, budgets, invalid_actions, spec)
                if sim_result.success and (best_eval is None or interim > best_eval[0]):
                    best_eval = (interim, candidate, sim_result, pvt_result, sizing_progress)
                improved = manager.record_reward(interim)
                if not improved and not sim_result.success:
                    manager.record_failure()

        # ---- final reward + learning updates + memory ----
        # If the current candidate lost its SPICE grounding (e.g. a later
        # regeneration never reached simulation), fall back to the best
        # SPICE-evaluated candidate: the reported design and the topology
        # return must be simulator-verified, never model- or heuristic-only.
        if sim_result is None and best_eval is not None:
            _r, candidate, sim_result, pvt_result, sizing_progress = best_eval
            log_decision(
                "RESTORED_BEST_CANDIDATE",
                "current candidate had no SPICE evaluation; reporting best SPICE-verified candidate",
                sm.state.value,
            )
        final_reward = self._final_reward(
            candidate, sizing_progress, sim_result, pvt_result, budgets, invalid_actions, spec
        )
        if sm.state not in (CoordinatorState.UPDATE_MEMORY, CoordinatorState.TERMINATE, CoordinatorState.FAILED):
            self._route_to_update_memory(sm)

        # Mark whether the final reward is grounded in a SPICE evaluation
        # (real or mock simulator) rather than validity-only shaping terms.
        trajectory.metadata["reached_spice"] = sim_result is not None
        trajectory.metadata["simulator_mode"] = self.config.spice.simulator
        for step in trajectory.steps:
            step.metadata["reached_spice"] = sim_result is not None

        update_report = self.update_manager.after_episode(
            trajectory,
            final_reward,
            sac=sac,
            policy_value_batch_size=cfg.policy_value.batch_size,
            policy_value_steps=cfg.policy_value.updates_per_episode,
            sac_update_steps=cfg.sac.updates_per_episode,
            dynamics_update_steps=cfg.sac.dynamics_updates_per_episode,
        )
        result.update_report = update_report.to_dict()

        # --- periodic DPO preference update from verified outcomes ---
        if self.dpo_ranker is not None and self.dpo_store is not None:
            from agentic_raptor.dpo import build_pairs

            self._dpo_episode_counter += 1
            dpo_report: dict[str, Any] = {"outcomes": len(self.dpo_store)}
            due = (
                self.dpo_config.update_mode == "periodic"
                and self._dpo_episode_counter % max(1, self.config.dpo.update_interval_episodes) == 0
            )
            if due:
                pairs = build_pairs(
                    self.dpo_store.records,
                    tie_margin=self.dpo_config.tie_margin,
                    max_pairs_per_specification=self.dpo_config.maximum_pairs_per_specification,
                    seed=self.config.seed,
                )
                dpo_report["pairs"] = len(pairs)
                dpo_report["train"] = self.dpo_ranker.train_on_pairs(pairs)
                if self.config.dpo.checkpoint_path:
                    self.dpo_ranker.save(self.config.resolve_path(self.config.dpo.checkpoint_path))
                    dpo_report["checkpoint_saved"] = True
            result.update_report["dpo"] = dpo_report

        memory_id = None
        if candidate is not None:
            entry = MemoryEntry.create(
                specifications=spec.to_dict(),
                topology=candidate.topology.to_dict(),
                sizing_values=candidate.sizing_state,
                edit_history=candidate.edit_history,
                spice_metrics=dict(sim_result.metrics) if sim_result else {},
                pvt_results=pvt_result.to_dict() if pvt_result else None,
                reward=final_reward,
                fom=compute_figure_of_merit(sim_result, spec.load_capacitance_f) if sim_result else None,
                success=bool(
                    sim_result is not None
                    and sim_result.success
                    and sim_result.constraint_margins
                    and all(m >= 0 for m in sim_result.constraint_margins.values())
                ),
                failure_reason=None
                if sim_result and all(m >= 0 for m in sim_result.constraint_margins.values())
                else _failure_reason(candidate, sim_result),
                generation_metadata={
                    "episode_id": episode_id,
                    "generator": candidate.topology.metadata.extra.get("generator", "mock"),
                    "memories_used": candidate.topology.metadata.extra.get("memories_used", []),
                },
                spec_embedding=spec_embedding(spec),
            )
            memory_id = self.memory.add(entry)
            log_decision("MEMORY_UPDATED", f"stored experience {memory_id}", sm.state.value)
        sm.transition(CoordinatorState.TERMINATE)

        candidate_ok = candidate is not None and candidate.status in (
            CandidateStatus.ACCEPTED,
            CandidateStatus.SIMULATED,
        )
        result.success = bool(candidate_ok and sim_result is not None and sim_result.success)
        result.final_state = sm.state.value
        result.final_reward = final_reward
        result.best_candidate = candidate.to_dict() if candidate is not None else None
        result.memory_id = memory_id
        return self._finalize(result, budgets)

    # ------------------------------------------------------------------
    def _mcts_edit_step(
        self,
        env: TopologyEditEnv,
        spec: DesignSpecifications,
        trajectory: TopologyTrajectory,
        budgets: BudgetState,
    ) -> int:
        """One MCTS-planned edit applied through the environment. Returns invalid count."""
        assert env.state is not None
        budgets.consume("topology_edits")
        mcts_result = self.mcts.run(env.state.graph, spec, env.edit_budget_fraction())
        legal = env.legal_actions()
        visit_dist = mcts_result.visit_distribution(legal)
        # Capture the decision state BEFORE env.step so graph_state, features,
        # mask, and visit distribution all describe the same pre-action state.
        pre_action_graph = env.state.graph.to_dict()
        features = encode_state_features(
            env.state.graph, spec, env.edit_budget_fraction(), env.state.validation.feature_vector()
        )
        try:
            action_index = next(
                i for i, a in enumerate(legal) if a.key() == mcts_result.best_action.key()
            )
        except StopIteration:
            action_index = 0  # TERMINATE fallback
        _obs, reward, _terminated, _truncated, info = env.step(action_index)
        trajectory.add_step(
            TrajectoryStep(
                graph_state=pre_action_graph,
                specification_state=spec.to_dict(),
                state_features=features,
                legal_action_mask=[i < len(legal) for i in range(self.config.actions.max_actions)],
                mcts_visit_distribution=visit_dist,
                selected_action=legal[action_index].to_dict(),
                metadata={"step_reward": reward, "mcts_root_value": mcts_result.root_value},
            )
        )
        return 1 if info.get("invalid_action") else 0

    def _sizing_round(
        self,
        candidate: CircuitCandidate,
        spec: DesignSpecifications,
        sac: GraphConditionedMBSAC,
        space: SizingParameterSpace,
        sizing_vector: list[float],
        budgets: BudgetState,
    ) -> dict[str, Any]:
        """Budgeted sizing round: real evaluations vs dynamics predictions.

        Mock mode: every evaluation uses the (free) mock simulator — Stage 1
        behaviour, no SPICE budget consumed.
        Real mode ("ngspice"): the RealSpiceQueryPolicy decides per transition
        between a real SPICE evaluation (consumes ``spice_calls`` budget,
        stored as source="real") and a dynamics-model prediction (stored as
        source="model", never used to accept a candidate). The vector the
        round returns as its best is the best REAL-evaluated one when real
        evaluations happened; final acceptance always goes through RUN_SPICE.
        """
        from agentic_raptor.sizing.spice_query_policy import RealSpiceQueryPolicy

        cfg = self.config
        real_backend = self.simulator_mode == "ngspice"
        policy = RealSpiceQueryPolicy(
            warmup_transitions=cfg.sac.real_spice_warmup_transitions,
            uncertainty_threshold=cfg.sac.dynamics_uncertainty_threshold,
            query_interval=cfg.sac.real_query_interval,
        )
        current = list(sizing_vector)
        real_evals = 0
        model_steps_since_real = 0
        query_log: list[str] = []
        pool_entries: list[dict[str, Any]] = []  # evaluated (pre-SPICE-selection) candidates

        def evaluate_real(vec: list[float]) -> tuple[dict[str, float], dict[str, float], float, bool]:
            probe = candidate.fork(GenerationSource.EDITED)
            probe.sizing_state = space.denormalize(vec)
            probe.topology.apply_sizing(probe.sizing_state)
            sim = self.simulator.simulate(probe, ["op", "ac"], cfg.spice.timeout_s)
            if not sim.success:
                return {}, {}, -1.0, False
            margins = sim.constraint_margins
            score = sum(min(m, 0.5) for m in margins.values()) / max(len(margins), 1)
            if len(pool_entries) < self.config.dpo.pool_size:
                pool_entries.append(
                    {"vector": list(vec), "margins": dict(margins), "metrics": dict(sim.metrics), "score": score}
                )
            return sim.metrics, margins, score, True

        # Anchor evaluation (always real-backend when in real mode; warm-up rule).
        if real_backend:
            budgets.consume("spice_calls")
        metrics, margins, score, ok = evaluate_real(current)
        if not ok and real_backend:
            return {
                "sizing_vector": current,
                "progress": 0.0,
                "first_margin_score": None,
                "error": "anchor real-SPICE evaluation failed",
            }
        first_margin_score = score
        best_real = (list(current), score) if real_backend else None

        steps = min(cfg.sac.sizing_steps, budgets.remaining("sizing_steps"))
        for _step in range(steps):
            budgets.consume("sizing_steps")
            ctx = SizingStateContext(
                graph=candidate.topology,
                spec=spec,
                sizing_vector=current,
                metrics=metrics,
                constraint_margins=margins,
                spice_budget_fraction=budgets.remaining("spice_calls") / max(budgets.max_spice_calls, 1),
            )
            state = encode_sizing_state(ctx, sac.action_dim)
            action = sac.select_action(state)
            new_vector = space.clip([c + 0.3 * a for c, a in zip(current, action, strict=True)])

            use_real = True
            if real_backend:
                decision = policy.decide(
                    transitions_done=real_evals,
                    model_steps_since_real=model_steps_since_real,
                    uncertainty=sac.dynamics_uncertainty(state, action),
                    spice_budget_remaining=budgets.remaining("spice_calls"),
                )
                use_real = decision.use_real
                query_log.append(("real: " if use_real else "model: ") + decision.reason)

            if use_real:
                if real_backend:
                    budgets.consume("spice_calls")
                new_metrics, new_margins, new_score, ok = evaluate_real(new_vector)
                if not ok:
                    # Failed real evaluation: penalty transition, state unchanged.
                    sac.add_transition(
                        SizingTransition(state=state, action=action, reward=-0.5,
                                         next_state=state, done=False, source="real")
                    )
                    continue
                real_evals += 1
                model_steps_since_real = 0
                reward = new_score - score
                source = "real"
            else:
                # Dynamics-model prediction guides the step (never accepted as final).
                torch = sac.torch
                with torch.no_grad():
                    pred_next, pred_reward, _done_logit = sac.dynamics(
                        torch.tensor([state], dtype=torch.float32),
                        torch.tensor([action], dtype=torch.float32),
                    )
                new_metrics, new_margins = metrics, margins  # unknown without SPICE
                new_score = score + float(pred_reward.item())
                reward = float(pred_reward.item())
                model_steps_since_real += 1
                source = "model"

            next_ctx = SizingStateContext(
                graph=candidate.topology,
                spec=spec,
                sizing_vector=new_vector,
                metrics=new_metrics,
                constraint_margins=new_margins,
                spice_budget_fraction=ctx.spice_budget_fraction,
            )
            next_state = encode_sizing_state(next_ctx, sac.action_dim)
            done = budgets.remaining("sizing_steps") == 0
            sac.add_transition(
                SizingTransition(state=state, action=action, reward=reward,
                                 next_state=next_state, done=done, source=source)  # type: ignore[arg-type]
            )
            if new_score >= score:
                current, metrics, margins, score = new_vector, new_metrics, new_margins, new_score
                if source == "real" and (best_real is None or new_score >= best_real[1]):
                    best_real = (list(new_vector), new_score)

        # Model-based components: dynamics learning + imagined rollout.
        sac.update_dynamics(batch_size=cfg.sac.batch_size)
        start_ctx = SizingStateContext(
            graph=candidate.topology,
            spec=spec,
            sizing_vector=current,
            metrics=metrics,
            constraint_margins=margins,
            spice_budget_fraction=budgets.remaining("spice_calls") / max(budgets.max_spice_calls, 1),
        )
        imagined = sac.generate_imagined_transitions(encode_sizing_state(start_ctx, sac.action_dim))
        sac.update(batch_size=cfg.sac.batch_size)

        # In real mode the returned vector is the best REAL-verified one.
        if real_backend and best_real is not None:
            current = best_real[0]
        logger.info(
            "sizing round [%s]: score %.4f → %.4f | real evals=%d | %d imagined | replay=%s",
            self.simulator_mode, first_margin_score, score, real_evals, imagined, sac.replay.counts(),
        )
        if query_log:
            logger.info("query policy: %s", "; ".join(query_log))
        return {
            "sizing_vector": current,
            "progress": score - (first_margin_score or 0.0),
            "first_margin_score": first_margin_score,
            "real_evaluations": real_evals,
            "query_log": query_log,
            "pool": pool_entries,
        }

    def _dpo_select(self, candidate: CircuitCandidate, pool: list[dict[str, Any]], budgets: BudgetState):
        """Build pre-SPICE features for each pool entry, rank, select."""
        from agentic_raptor.core.rewards import feasibility_score, normalize_fom
        from agentic_raptor.dpo import build_candidate_features, select_candidate

        features_pool = []
        for entry in pool:
            probe = candidate.fork(GenerationSource.EDITED)
            probe.simulation_result = None  # forked pre-selection view (leakage guard)
            metrics = entry["metrics"]
            power = metrics.get("power_w", 0.0)
            gbw = metrics.get("gbw_hz", 0.0)
            cl = candidate.specifications.load_capacitance_f or 1e-12
            fom = (gbw / 1e6) * (cl / 1e-12) / (power / 1e-3) if power > 0 else 0.0
            features_pool.append(
                build_candidate_features(
                    probe,
                    sizing_vector=entry["vector"],
                    predicted_margins=entry["margins"],
                    predicted_feasibility=feasibility_score(entry["margins"]),
                    predicted_fom=normalize_fom(fom, scale=self.reward_weights.fom_scale),
                    repair_burden=len(candidate.edit_history) / max(1, self.config.budgets.max_topology_edits),
                )
            )
        selection = select_candidate(features_pool, self.dpo_ranker, self.dpo_config, self.rng)
        return selection, selection.selected

    def _record_dpo_outcome(self, pending, sim_result, pvt_result, budgets, episode_id, spec) -> None:
        from agentic_raptor.dpo import OutcomeRecord

        selection, features = pending
        margins = sim_result.constraint_margins
        self.dpo_store.add(
            OutcomeRecord(
                features=features,
                dpo_score=selection.dpo_score,
                dpo_rank=selection.dpo_rank,
                selection_reason=selection.reason,
                spice_success=sim_result.success,
                passed_spec=bool(margins and all(m >= 0 for m in margins.values())),
                metrics=dict(sim_result.metrics),
                constraint_margins=dict(margins),
                pvt_pass_rate=pvt_result.pass_rate if pvt_result is not None else None,
                fom=features.predicted_fom,
                runtime_s=sim_result.runtime_s,
                spice_calls_total=budgets.used_spice_calls,
                episode_id=episode_id,
            )
        )

    @staticmethod
    def _route_to_update_memory(sm: StateMachine) -> None:
        """Walk a legal transition path from the current state to UPDATE_MEMORY."""
        from collections import deque

        from agentic_raptor.coordinator.state_machine import ALLOWED_TRANSITIONS

        target = CoordinatorState.UPDATE_MEMORY
        queue: deque[list[CoordinatorState]] = deque([[sm.state]])
        seen = {sm.state}
        while queue:
            path = queue.popleft()
            if path[-1] is target:
                for state in path[1:]:
                    sm.transition(state)
                return
            for nxt in sorted(ALLOWED_TRANSITIONS[path[-1]], key=lambda s: s.value):
                if nxt not in seen and nxt is not CoordinatorState.FAILED:
                    seen.add(nxt)
                    queue.append([*path, nxt])
        raise CoordinatorError(f"no legal path from {sm.state.value} to UPDATE_MEMORY")  # pragma: no cover

    def _simulator_fingerprint(self, spec: DesignSpecifications) -> dict[str, Any]:
        """Everything simulator-side that must participate in the cache key."""
        base: dict[str, Any] = {
            "simulator": self.simulator_mode,
            "seed": self.config.seed,
            "temperature_c": spec.temperature_c,
            "supply_voltage": spec.supply_voltage,
        }
        fingerprint = getattr(self.simulator, "config_fingerprint", None)
        if callable(fingerprint):
            base.update(fingerprint())
        return base

    def _final_reward(
        self,
        candidate: CircuitCandidate | None,
        sizing_progress: float,
        sim_result: SimulationResult | None,
        pvt_result: PvtResult | None,
        budgets: BudgetState,
        invalid_actions: int,
        spec: DesignSpecifications,
    ) -> float:
        validation = None
        if candidate is not None and candidate.validation_result is not None:
            from agentic_raptor.topology_validation.rules import ValidationIssue
            from agentic_raptor.topology_validation.validator import ValidationResult

            raw = candidate.validation_result
            validation = ValidationResult(
                is_valid=bool(raw.get("is_valid")),
                issues=[ValidationIssue(**i) for i in raw.get("issues", [])],
                graph_hash=raw.get("graph_hash"),
            )
        components = assemble_components(
            validation,
            sizing_progress,
            sim_result,
            pvt_result,
            budgets,
            invalid_actions,
            spec.load_capacitance_f,
            fom_scale=self.reward_weights.fom_scale,
        )
        return compute_final_reward(components, self.reward_weights)

    def _build_context(
        self,
        candidate: CircuitCandidate | None,
        sim_result: SimulationResult | None,
        pvt_result: PvtResult | None,
        manager: BudgetManager,
        budgets: BudgetState,
        retrieval: Any,
        spec: DesignSpecifications,
    ) -> DecisionContext:
        margins = sim_result.constraint_margins if sim_result else {}
        return DecisionContext(
            specifications_parsed=True,
            has_candidate=candidate is not None,
            candidate_valid=bool(candidate and candidate.status not in (CandidateStatus.INVALID,)),
            validation_error_count=len((candidate.validation_result or {}).get("issues", [])) if candidate else 0,
            sized=bool(candidate and candidate.sizing_state),
            simulated=sim_result is not None,
            last_sim_failed=bool(sim_result is not None and not sim_result.success),
            last_sim_error_type=sim_result.error_type if sim_result is not None else None,
            pvt_done=pvt_result is not None,
            all_constraints_met=bool(margins and all(m >= 0 for m in margins.values())),
            current_reward=manager.best_reward,
            best_reward=manager.best_reward,
            accept_reward_threshold=self.config.coordinator.accept_reward_threshold,
            worst_margin=min(margins.values()) if margins else None,
            budget_snapshot=budgets.snapshot(),
            budgets_exhausted=budgets.any_exhausted(),
            edit_budget_remaining=budgets.remaining("topology_edits"),
            sizing_budget_remaining=budgets.remaining("sizing_steps"),
            spice_budget_remaining=budgets.remaining("spice_calls"),
            generation_budget_remaining=budgets.remaining("generations"),
            retrieval_budget_remaining=budgets.remaining("retrievals"),
            retrieved_count=len(retrieval.entries) if retrieval else 0,
            topology_edits_done=budgets.used_topology_edits,
            repeated_failures=manager.repeated_failure_count,
            max_repeated_failures=manager.max_repeated_failures,
            stagnated=manager.stagnated,
        )

    def _finalize(self, result: EpisodeResult, budgets: BudgetState) -> EpisodeResult:
        result.budget_snapshot = budgets.snapshot()
        summary_path = self.output_dir / f"{result.episode_id}_summary.json"
        dump_json(result.to_dict(), summary_path)
        result.summary_path = str(summary_path)
        logger.info("episode %s finished: state=%s reward=%s", result.episode_id, result.final_state, result.final_reward)
        return result


def _maybe_path(cfg: AgenticConfig, value: str | None) -> str | None:
    return str(cfg.resolve_path(value)) if value else None


def _failure_reason(candidate: CircuitCandidate, sim: SimulationResult | None) -> str:
    if sim is None:
        return "never reached SPICE evaluation"
    failing = [k for k, v in sim.constraint_margins.items() if v < 0]
    return f"constraints not met: {failing}" if failing else (sim.error_message or "unknown")
