# Architecture

## Data flow

```text
Engineer
  │  text / YAML / JSON / CSV table / schematic image / SPICE netlist
  ▼
specification/            MultimodalDesignInput → per-modality ExtractedFields
  │                        → fusion (priority: structured > table > netlist > text > image,
  │                          conflict detection, provenance) → DesignSpecifications
  │                        → engineering plausibility review
  ▼
rag/                      CircuitMemory.retrieve(spec) — weighted similarity over
  │                        spec embedding, circuit class, technology, supply,
  │                        success bonus / failure relevance → top-k MemoryEntry
  ▼
topology_generation/      TopologyGenerator.generate(spec, retrieved, multimodal ctx, n)
  │                        → structured-JSON CircuitGraphs (mock 5T OTA | LLM shell
  │                          with prompt builder, output parser, validate+retry)
  ▼
topology_validation/      TopologyValidator: 14 deterministic rules → ValidationResult
  │                        (structured issues + WL graph hash). Invalid graphs never
  │                        enter RL.
  ▼
topology_rl/              TopologyEditEnv (gym-style, 13 typed reversible actions)
  │                        MCTS (PUCT + progressive widening + masking) guided by
  │                        PolicyValueNetwork; each applied edit records a
  │                        TrajectoryStep (features, mask, π_MCTS, action)
  ▼
sizing/                   SizingParameterSpace.from_graph → GraphConditionedMBSAC:
  │                        state = [topo embed | sizing vec | spec | metrics |
  │                        margins | budget]; real transitions → replay("real");
  │                        DynamicsModel training → imagined rollouts ("model");
  │                        SAC updates on the configured mixture
  ▼
spice/                    SpiceSimulator protocol → MockSpiceSimulator (deterministic)
  │                        | LegacyNgspiceSimulator (stage-2 seam); SimulationCache
  │                        keyed by (topo hash, sizing, analyses, corner, sim cfg);
  │                        run_pvt over corner set → PvtResult
  ▼
learning/                 assemble_components + YAML RewardWeights → final_reward;
  │                        CrossLevelCreditAssigner: z_t = γ^(T−1−t)·R_final onto every
  │                        TrajectoryStep; UpdateManager: PV training steps, SAC +
  │                        dynamics steps, parameter-change verification
  ▼
rag/                      MemoryEntry stored with topology, sizing, metrics, PVT,
  │                        reward, failure reason, embeddings, provenance
  ▼
coordinator/              StateMachine (12 states) + RuleBasedDecisionPolicy
                           (9 decisions, every one logged with a reason) +
                           BudgetManager (edits/sizing/SPICE/runtime/generations/
                           retrievals, stagnation, repeated failures)
```

## Module map

| Package | Key classes / functions |
|---|---|
| `core` | `DesignSpecifications`, `CircuitGraph/Node/Edge`, `TopologyMetadata`, `CircuitCandidate`, `BudgetState`, `RewardComponents` |
| `specification` | `MultimodalDesignInput`, `parse_text/structured/table/netlist`, `SchematicImageParser` (+mock), `fuse_fields`, `review_specification`, `parse_design_input` |
| `rag` | `MemoryEntry`, `CircuitMemory` (add/retrieve/update_outcome), `Retriever`, legacy converters |
| `topology_generation` | `TopologyGenerator` protocol, `MockTopologyGenerator`, `LLMTopologyGenerator`, `build_generation_prompt`, `parse_generator_output`, `TOPOLOGY_JSON_SCHEMA` |
| `topology_validation` | `TopologyValidator`, rule functions, connectivity BFS, `compute_graph_hash` |
| `topology_rl` | `ActionType`(13), `apply_action`, `enumerate_candidate_actions`, `TopologyEditEnv`, `MCTS`, `PolicyValueNetwork` (pooled / message-passing), `PolicyValueTrainer`, `TopologyReplayBuffer`, `TopologyTrajectory` |
| `sizing` | `SizingParameterSpace`, `GaussianActor`, twin critics + `soft_update`, `DynamicsModel/Trainer`, `SizingReplayBuffer`, `GraphConditionedMBSAC`, `LegacySACBackend` |
| `spice` | `SpiceSimulator` protocol, `SimulationResult`, `MockSpiceSimulator`, `LegacyNgspiceSimulator`, `SimulationCache`, `PvtCorner`/`run_pvt`, metric normalization + margins |
| `learning` | `RewardWeights` (YAML), `assemble_components`, `compute_final_reward`, `CrossLevelCreditAssigner`, `UpdateManager` |
| `coordinator` | `CoordinatorState`, `Decision`, `StateMachine`, `RuleBasedDecisionPolicy`, `BudgetManager`, `AgenticCoordinator.run_episode` |
| `adapters` | `legacy_raptor` (path + availability), `existing_mb_sac/spice/rag/surrogate` |
| `utils` | `AgenticConfig` (typed YAML), logging + JSONL event log, seeding (+OMP workaround), serialization, exceptions |

## Learning-signal wiring (the research core)

```text
π_MCTS(a|s)  ──────────────► L_policy = −Σ π_MCTS log π_θ         (trainer.py)
final SPICE reward R ──┐
                       ├──►  z_t = γ^(T−1−t)·R  ──► L_value = (V_φ − tanh-scaled z)²
sizing margins/score ──┘         (cross_level_credit.py)
real transitions ────────────► critic/actor/α updates (Bellman with entropy term)
                └────────────► dynamics model (state Δ, reward, done)
dynamics + actor ────────────► imagined transitions (source="model") → SAC mixture
```

Discrete topology actions and continuous sizing actions are **never** coupled by
gradients — only by the scalar cross-level return.
