# Stage 3A Topology-Corpus Migration Report

## Previous implementation (verified, not assumed)
No five-topology registry existed: Stage 3A used 5 **specification variants** over one
5T-OTA template (+3 seed RAG entries). References found in 6 files (see
docs/stage3a_topology_migration.md). Restriction removed; generation/validation/RAG/
MCTS/LLM interfaces preserved; template retained for mock mode.

## Corpus load (dynamic discovery — count never hard-coded)
- Directories discovered / loaded: **57 / 57**, excluded: **0**
- Duplicate canonical hashes: **0**
- By source: analoggym 17 (literature) · generated clusters 40 (opamp_generator + cktgnn)
- With netlists: 17 · with schematics: 15 (AnalogGym only; absences recorded, no crashes)
- Electrical validation status: **57 unvalidated** (honest: structural ≠ electrical;
  fields stored separately)

## Hierarchical RAG index (`datasets/topology_rag/`)
- Level 1 topology records: **57** · Level 2 block records: **287** · Level 3 graph
  records: **57** · Level 4 simulation memory: **0 (interface reserved, empty)** —
  retrieval text is metadata/structure only; zero fabricated performance (tested).

## Splits (`datasets/topology_splits/`, seed 42, reproducible — tested)
- Family split: train **39** / validation **9** / test **9**, overlap **0**
- Benchmark holdout: **4 AnalogGym families** excluded from SFT, complete-topology
  RAG, selection examples, training trajectories (leakage_report.md)

## Selection
`select_topology_candidates`: top-K with score decomposition (literature/mapped/
stage/block/intent terms) + candidate tier `structurally_retrieved |
transistor_mapped | electrically_validated`; deterministic (tested).

## Files added
`agentic_raptor/corpus/__init__.py`, `tests/test_corpus.py`,
`configs/stage3a_topology_corpus.yaml`, `docs/stage3a_topology_migration.md`,
`datasets/topology_rag/*`, `datasets/topology_splits/*`, this report.
Modified: none of the legacy Stage 3A files were deleted; `dataset_debug.yaml`
remains valid (5T family is now one of 57).

## Tests
Full suite: **191 passed, 1 skipped** (7 new corpus tests: discovery, required/
optional files, filters+hash lookup, no-fabrication RAG, top-K tiers+determinism,
split reproducibility/leakage, MCTS-on-corpus-graph). 1 ruff style finding pending
in corpus module (non-functional).

## Scientific constraints upheld
57 families are NOT claimed electrically validated; 1,500 OPAMP-Generator rows and
400 CktGNN graphs are NOT claimed as independent families (they collapsed to 40
clustered block-level families, `approximate_dag_decode` marked); the 17
literature families are distinguished throughout.

## Limitations
Generated families lack transistor mappings/netlists; electrical validation of all
families pending (later sizing/SPICE stages); LLM-coordinator prompt wiring of the
retrieval documents is interface-ready (`retrieval_document()` → prompt_builder
MemoryEntry path) but full end-to-end episodes with the corpus retriever remain to
be exercised; Level-4 simulation memory awaits real labelled runs.
