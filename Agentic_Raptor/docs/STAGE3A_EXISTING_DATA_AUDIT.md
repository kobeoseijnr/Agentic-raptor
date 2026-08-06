# Stage 3A — Existing Data Audit (read-only; verified against the implementation)

## 0. Inspection evidence (file:line, gathered by direct repository scan)
- `agentic_raptor/utils/logging.py:31` `class JsonlEventLog` — decision-trace writer
- `agentic_raptor/coordinator/coordinator.py:131` decision log wiring; `:1032` `<episode_id>_summary.json` writer; `:207` `dpo_outcomes.jsonl` store path; `:581` `trajectory.metadata["reached_spice"]` stamp; `:141` `simulator_mode` resolution
- `agentic_raptor/spice/ngspice_simulator.py:99` `workdir_root` (defaults `%TEMP%/agentic_raptor_spice`) and `:230` `ngspice.log` raw-evidence path — confirms audit finding §4.8
- `agentic_raptor/spice/cache.py:15` `make_cache_key` (topology hash + sizing + analyses + corner + simulator fingerprint)
- `agentic_raptor/rag/memory.py:57` `CircuitMemory(persist_path)` JSONL persistence
- `agentic_raptor/dpo/schemas.py:23/116` `CandidateFeatures` / `OutcomeRecord`
- `agentic_raptor/topology_rl/trajectory.py:10` `TrajectoryStep` (pre-action `graph_state`, mask, `mcts_visit_distribution`, `discounted_return`)
- `agentic_raptor/sizing/replay_buffer.py:13` `SizingTransition` (real/model label; in-memory only — confirms §4.3 persistence gap)
- `agentic_raptor/finetuning/outcome_store.py:17` `EpisodeOutcome`
- `agentic_raptor/topology_generation/provider.py:45` `model_name` default (config-overridable via `LLMConfig`; Stage 2 verified model was gpt-4o-mini through `stage2_*.yaml`) and `:140` token-usage capture

## 1. Existing data sources
| Source | Location / producer | Format |
|---|---|---|
| Coordinator decision traces | `outputs/**/decisions.jsonl` (`JsonlEventLog`) | JSONL |
| Episode summaries | `outputs/**/<episode_id>_summary.json` (`EpisodeResult.to_dict`) | JSON |
| Topology trajectories | in-memory `TopologyReplayBuffer` (+`save/load` JSONL); steps embedded in episodes | JSONL-capable |
| MCTS traces | `MCTSResult.visit_counts` → stored per `TrajectoryStep.mcts_visit_distribution` | in-record |
| Sizing/MB-SAC replay | in-memory `SizingReplayBuffer` (real/model labelled) — **not persisted** | none (gap) |
| SPICE inputs/outputs | `%TEMP%/agentic_raptor_spice/<cand>_<corner>_<tag>_<n>/circuit.cir` + `ngspice.log` | text |
| Parsed SPICE metrics | `SimulationResult.to_dict` in episode summaries + cache | JSON |
| SPICE cache | `outputs/**/spice_cache.jsonl` (key = topo hash+sizing+analyses+corner+fingerprint) | JSONL |
| PVT results | `PvtResult.to_dict` in summaries (pass_rate, worst/mean margin, failed corners) | JSON |
| Reward traces | `RewardComponents` + weighted final in summaries/credit reports | JSON |
| RAG memory | `CircuitMemory` JSONL (`MemoryEntry`: spec, topology, sizing, metrics, PVT, reward, failure, embeddings) | JSONL |
| DPO outcomes | `outputs/**/dpo_outcomes.jsonl` (`OutcomeRecord` + `CandidateFeatures`) | JSONL |
| LLM prompts/outputs | prompt built in `prompt_builder`; provider logs token usage; raw output NOT persisted (gap) | — |
| Fine-tune episode store | `finetuning.EpisodeOutcomeStore` JSONL (schema exists; not yet wired into coordinator) | JSONL |
| Legacy evidence | `../RAPTOR_Legacy/{data,results}` (phase9 CSV, rag_memory, dpo_pairs) via adapters, read-only | CSV/JSONL |

## 2. Existing schemas (field sources)
specs `DesignSpecifications` (typed, units implicit in names, `source_metadata`); graphs `CircuitGraph/Node/Edge` + WL `structural_hash` (canonical: sorted nodes/edges, merged parallel terminals — rename-invariant); graph actions `ActionType`(13)+`EditRecord`; sizing actions normalized [-1,1] + `SizingParameterSpace` typed ids (`m1.width_m` etc. with bounds/log-scale); SPICE `SimulationResult` (+9-type failure taxonomy); rewards `RewardComponents`+YAML weights; RAG `MemoryEntry`; preferences `dpo.OutcomeRecord`/`PreferencePair` (a–i basis, confidence); trajectories `TrajectoryStep` (pre-action graph, mask, π_MCTS, `reached_spice`, discounted return); replay `SizingTransition` (real/model).

## 3. Generation entry points
CLI: `smoke-test, run-episode, run-stage2, generate-topology, run-spice, build-netlist, parse-schematic, validate, check-dependencies, inspect-repository`; scripts/ wrappers; configs `default/smoke/stage2_*` + `mb_sac/topology_rl/rag/spice.yaml`; tests as fixtures.

## 4. Existing data problems (Stage 3A must fix)
1. **No unified run record** — linkage via `episode_id` only; no `sample_id/spec hash/graph-id` chain.
2. **No canonical specification hash** (specs only embedded/serialized).
3. **Sizing transitions not persisted** — real-SPICE MB-SAC data is lost at episode end (biggest gap for Dataset 4).
4. **LLM raw generations not persisted** (only accepted graphs).
5. **Mock vs real**: `simulator_mode` on trajectories/summaries but NOT stamped per SPICE/memory/DPO record (`execution_mode` missing).
6. **No splits anywhere**; RAG has no eligibility manifest (leakage risk: seed corpus + episode write-back are unrestricted).
7. Metric units by naming convention (`*_hz`,`*_w`) — no explicit unit fields.
8. SPICE workdirs in `%TEMP%` — raw evidence not under `data/`; cleanup can remove non-log files.
9. Duplicate detection only via cache keys; no spec/prompt/trajectory dedup.
10. Topology families: **executable today: five-transistor OTA only** (mock generator + LLM path). Telescopic-/folded-cascode and two-stage Miller have schema/device support (all device types + validator + netlist builder are family-agnostic) but **no generator templates or verified testbench runs → document as unsupported until executed; do not fabricate records.**
11. PVT evidence sometimes absent (`pvt_evaluated` flag missing — `None` currently ambiguous).
12. Legacy CSV/JSONL schemas differ (dict-based); require explicit conversion + provenance labels.

## 5. Audit conclusion
Foundations required by §6–§22 largely exist (canonical graph hash, failure taxonomy, reward vectors, pre-action states, `reached_spice`, real/model labels). Stage 3A implementation order: (1) trace-ID + `execution_mode` stamping and unified run record, (2) persist sizing transitions + raw LLM generations + relocate SPICE raw evidence under `data/stage3a/raw/`, (3) canonical spec hashing, (4) derived-dataset builders + splits + validators. No data generated before these land.
