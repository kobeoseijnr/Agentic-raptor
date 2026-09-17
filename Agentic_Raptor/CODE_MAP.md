# RAPTOR — complete code map

Every module, grouped by pipeline stage. Written 2026-09-03 against the live
tree. Companion to `ARCHITECTURE_CURRENT.md` (which explains *what* each part
does; this one says *where it is*).

`agentic_raptor/` totals ~33,500 lines of Python across 24 packages.

---

## Entry point

| File | Lines | Role |
|---|---|---|
| `run_raptor_v2.py` | — | the pipeline. `run_pipeline()` at `:635`, `main()` at `:540` |
| `agentic_raptor/cli.py` | — | CLI surface |

Stage boundaries inside `run_pipeline`:
`:741` spec · `:797` RAG · `:803` propose+validate · `:906` select ·
`:1137` size · `:1241` rank · `:1356` verify · `:1582` training data

---

## Stage 1 — Specification

| File | Lines |
|---|---|
| `specification/parser.py` | 354 |
| `specification/fusion.py` | 127 |
| `specification/schematic.py` | 114 |
| `specification/validator.py` | 72 |
| `specification/multimodal_input.py` | 71 |
| `specification/__init__.py` | 65 |

## Stage 2 — Retrieval (RAG)

| File | Lines |
|---|---|
| `rag/memory.py` | 127 |
| `rag/retriever.py` | 96 |
| `rag/schemas.py` | 49 |
| `rag/adapters.py` | 48 |

Corpus the retriever draws on:

| File | Lines |
|---|---|
| `corpus/__init__.py` | 271 |
| `corpus/reextract_v2.py` | 257 |
| `corpus/operational_v3.py` | 216 |
| `corpus/hash_migration_v2.py` | 200 |

## Stages 3+4 — Topology proposal and validation

| File | Lines |
|---|---|
| `topology_generation/generator.py` | 247 |
| `topology_generation/provider.py` | 193 |
| `topology_generation/prompt_builder.py` | 82 |
| `topology_generation/output_parser.py` | 79 |
| `topology_generation/multimodal_inputs.py` | 37 |
| `topology_validation/rules.py` | 198 |
| `topology_validation/connectivity.py` | 98 |
| `topology_validation/validator.py` | 70 |
| `topology_validation/graph_hash.py` | 24 |

Model fine-tuning that produces the proposer:

| File | Lines |
|---|---|
| `finetuning/` | 550 (5 files) |
| `llm_dpo/` | 2904 (7 files) |

## Stage 5 — Topology selection

| File | Lines | Status |
|---|---|---|
| `topology_rl/bandit_selector.py` | 198 | **LIVE selector** |
| `topology_rl/alphazero.py` | 2356 | retired 2026-08-27 |
| `topology_rl/value_refresh.py` | 902 | value model refresh |
| `topology_rl/stage3e1.py` | 771 | structural edits |
| `topology_rl/stage3e2.py` | 749 | structural edits |
| `topology_rl/stage3e2_edits.py` | 554 | `qualify_device_graph` lives here |
| `topology_rl/actions.py` | 447 | edit action space |

> `topology_rl/` is the largest package (7452 lines) but most of it is the
> retired AlphaZero layer. The live selector is 198 lines.

## Stages 6+7 — Sizing

| File | Lines | Status |
|---|---|---|
| `mb_sac/spec_sizing.py` | 1308 | **LIVE sizer** — `sac_size()` at `:675` |
| `mb_sac/hybrid_sizing.py` | 389 | hybrid phase-2 SAC |
| `mb_sac/stage3d1.py` | 247 | staged development |
| `mb_sac/__init__.py` | 227 | 4-step smoke driver |
| `mb_sac/stage3d2.py` | 222 | staged development |

Model-based prototype — **not used by any pipeline**:

| File | Lines |
|---|---|
| `sizing/graph_conditioned_mb_sac.py` | 318 |
| `sizing/dynamics_model.py` | 137 |
| `sizing/parameter_space.py` | 115 |
| `sizing/replay_buffer.py` | 70 |
| `sizing/spice_query_policy.py` | 61 |
| `sizing/actor.py` | 56 |

## Stage 8 — Preference ranking

| File | Lines | Status |
|---|---|---|
| `ranking/post_sac.py` | 469 | post-sizing ranker |
| `ranking/pair_mining.py` | 395 | pair mining |
| `ranking/types.py` | 293 | shared types |
| `ranking/features_v2.py` | 235 | **promoted 46-feature set** |
| `ranking/surrogate.py` | 185 | performance predictor |
| `ranking/model.py` | 179 | superseded V1 |
| `ranking/model_v2.py` | 141 | **promoted V2 model** |
| `dpo/schemas.py` | 172 | record types |
| `dpo/preference_pairs.py` | 171 | pair construction |
| `dpo/ranker.py` | 137 | Bradley-Terry model |
| `dpo/selector.py` | 85 | branch A/B selection |

## Stages 9+10 — SPICE verification

| File | Lines |
|---|---|
| `electrical/__init__.py` | 440 |
| `electrical/pvt_eval.py` | 244 |
| `electrical/measurements.py` | 222 |
| `electrical/adjudication.py` | 183 |
| `electrical/pm_qualified.py` | 93 |
| `electrical/fom.py` | 75 |
| `spice/ngspice_simulator.py` | 263 |
| `spice/testbench_builder.py` | 205 |
| `spice/result_parser.py` | 179 |
| `spice/device_mapping.py` | 163 |
| `spice/simulator_adapter.py` | 151 |
| `spice/netlist_builder.py` | 111 |

## Orchestration

| File | Lines |
|---|---|
| `coordinator/coordinator.py` | 1047 |
| `coordinator/decision_policy.py` | 137 |
| `coordinator/state_machine.py` | 87 |
| `coordinator/budget_manager.py` | 51 |
| `agents/supervisor.py` | 363 |
| `agents/state.py` | 127 |
| `agents/critic.py` | 115 |
| `agents/planner.py` | 108 |
| `agents/recovery.py` | 75 |
| `agents/coordinator.py` | 62 |

## Shared types

| File | Lines |
|---|---|
| `core/circuit_graph.py` | 302 |
| `core/candidate.py` | 104 |
| `core/specifications.py` | 101 |
| `core/budgets.py` | 81 |
| `core/types.py` | 78 |
| `core/rewards.py` | 57 |
| `mapping/` | 714 (2 files) |
| `utils/` | 404 (6 files) |
| `adapters/` | 303 (7 files) |

## Experiments and self-improvement

| Package | Lines |
|---|---|
| `publication/` | 4748 (24 files) — baselines, ablations, reports |
| `selfimprove_v2/` | 1430 (5 files) |
| `learning/` | 268 (4 files) |
| `stage3a/` | 344 (2 files) |

---

## Reading order

To understand the system, follow the pipeline rather than the folder listing:

1. `run_raptor_v2.py:635` — `run_pipeline`, the spine
2. `topology_generation/generator.py` — where circuits are invented
3. `topology_rl/bandit_selector.py` — which two survive
4. `mb_sac/spec_sizing.py:675` — `sac_size`, how they get sized
5. `ranking/model_v2.py` + `ranking/features_v2.py` — which one is returned
6. `electrical/__init__.py` — how truth is established

## Two traps in this tree

**`topology_rl/alphazero.py` (2356 lines) is retired.** Deleted from the live
path 2026-08-27; `bandit_selector.py` (198 lines) replaced it. Size is not
significance here.

**`sizing/graph_conditioned_mb_sac.py` is not the sizer.** It is the genuinely
model-based prototype, imported by no pipeline. The live sizer is
`mb_sac/spec_sizing.py`, which despite the package name is **model-free**.
