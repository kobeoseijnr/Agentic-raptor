# Implementation Plan

## Stage 1 — DONE (this pass)

- [x] Repository audit incl. MB-SAC/MCTS genuineness verification (read-only)
- [x] Package scaffold, typed YAML config, CLI, packaging
- [x] Multimodal specification parsing (text/YAML/JSON/CSV/netlist; mock image parser),
      fusion with conflict detection + provenance, plausibility review
- [x] Core models: specifications, typed circuit graph (JSON/NetworkX/WL hash/sizing
      separation), candidate, budgets, reward components
- [x] Deterministic validator (14 rules, structured issues, stable hashing)
- [x] 13 typed reversible topology actions + gym-style edit environment
- [x] PUCT MCTS with progressive widening, masking, visit-count policy
- [x] Policy-value network (pooled + message-passing encoders) with genuine
      AlphaZero-style training updates
- [x] Genuine graph-conditioned MB-SAC: Gaussian actor, twin/target critics,
      auto-α, Polyak, dynamics model (reward+termination heads), imagined rollouts,
      real/model replay mixture
- [x] Mock SPICE + PVT corners + simulation cache; ngspice adapter seam
- [x] YAML reward weights, cross-level credit assigner, update manager with
      parameter-change verification
- [x] Structured RAG memory + weighted retrieval + outcome write-back + seed corpus
- [x] Rule-based coordinator (12 states / 9 decisions, all logged with reasons)
- [x] 100 tests incl. end-to-end smoke pipeline; ruff clean

## Stage 2 — Real SPICE in the loop

1. Typed `CircuitGraph` → legacy netlist export (bridge to
   `graph/export_graph_to_netlist.py`; verify with `parse_netlist_to_graph` round-trip).
2. Implement `LegacyNgspiceSimulator.simulate` via
   `controller.module_adapters.run_spice_validation` (ngspice exe from
   `configs/spice.yaml`, per-machine).
3. Map ngspice log parsing into `spice/result_parser.normalize_metrics`.
4. Replace sizing-round probe evaluations with the dynamics model / legacy surrogate
   (`adapters/existing_surrogate.py`) so SPICE calls stay budgeted.

## Stage 3 — Learning campaigns

5. Multi-episode driver: N episodes per spec family, persistent memory + replay,
   checkpointing of policy/value/SAC networks.
6. Baselines: (a) unsized-heuristic topology scoring, (b) legacy checkpoint-5d
   offline policy/value, (c) no-credit ablation (validity-only topology reward).
7. Metrics: final post-sizing reward vs. SPICE budget; feasibility rate; PVT score.

## Stage 4 — Multimodal LLM integration

8. Provider transports for `LLMTopologyGenerator` (openai/anthropic/ollama), reusing
   the legacy `llm/llm_client.py` env-var pattern; attach schematic image bytes for
   multimodal providers.
9. VLM-backed `SchematicImageParser`.
10. Retrieval-conditioned prompting ablations (successes only vs. +failures).

## Stage 5 — Scale and robustness

11. PyTorch Geometric encoder behind the existing `encoder:` switch.
12. Learned coordinator decision policy (the `DecisionPolicy` protocol is ready).
13. Monte Carlo analysis behind `AnalysisType.MONTE_CARLO`; mismatch-aware PVT score.
14. Legacy memory import at scale (`rag/adapters.import_legacy_items`).
