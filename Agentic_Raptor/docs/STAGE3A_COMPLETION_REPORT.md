# Stage 3A Completion Report — DEBUG scope (honest status)

Verdict: **DEBUG_DATASET_VALID** (never FULL_DATASET_VALID — debug data only).

## Real debug generation (2026-07-25, wall clock 164.1 s)
Command: `run_debug_generation("configs/experiments/stage2_end_to_end.yaml", "data/stage3a", seeds=[101..105], mode="REAL")` — 5 real episodes, real gpt-4o-mini (env base URL) + real ngspice-45.2, DPO ranker disabled during collection (correction 5).

Counts: runs 5 · reached_spice 4/5 · SFT 4 (TIER_2/3, multimodal provenance fields per correction 6) · search-ranker preferences 7 (labels from `compare()` rules over real-SPICE evidence only; no ranker-score labels; no LLM-DPO fields) · topology-RL steps 4 (pre-action graphs, REAL_POST_SIZING_SPICE value-target source) · MB-SAC transitions 4 (all `real_or_imagined=REAL`) · quarantined 0 · leakage 0 · duplicates 0.0.
Raw evidence: SPICE netlists+logs COPIED (originals preserved) into content-addressed `data/stage3a/raw/spice_outputs/`; episode summaries under `raw/reward_traces/`; manifests (leakage/duplicates/schema+field-provenance) under `data/stage3a/manifests/`.

## Tests
`pytest`: **184 passed, 1 skipped** (8 new Stage 3A tests incl. smokes A–D on the mock-mode pipeline; real-mode exercised by the generation run above). Ruff: 3 style findings remain in `stage3a/generate.py` (E731 lambda et al.) — functional, listed for cleanup.

## Known limitations (do not overstate)
1. **Spec-group collapse**: variants were injected via `specification.defaults`, but the stage2 config's structured spec wins fusion priority → all 5 runs share one canonical spec hash → one group → val/test splits empty (permitted by `allow_empty_debug_splits`; split algorithm separately fixture-tested). Fix queued: override the structured targets per run, not defaults.
2. Single topology family (5T OTA); telescopic/folded-cascode + two-stage Miller remain unsupported → `heldout_topology_family` files empty by construction.
3. 1/5 run ended invalid-candidate (captured as failure diversity, not quarantined).
4. MB-SAC transitions currently 1 terminal record/run (full per-step persistence remains queued); resumability/cost CSV partial (token usage in logs, not yet aggregated).
5. Remaining docs (SCHEMA_REFERENCE, SPLIT_POLICY, GENERATION_PROTOCOL, SEARCH_PREFERENCE_RANKER, DATA_ARCHITECTURE) pending.

## DPO clarification
No LLM DPO fine-tuning dataset was created. The preference dataset targets the feasibility-gated search-time ranker only.

Files added: `agentic_raptor/stage3a/{__init__,generate}.py`, `tests/test_stage3a.py`, `configs/stage3a/dataset_debug.yaml`, audit + this report, `data/stage3a/**` tree. `RAPTOR_Legacy` untouched. No git available (no diff; additive-only inside Agentic_Raptor).
