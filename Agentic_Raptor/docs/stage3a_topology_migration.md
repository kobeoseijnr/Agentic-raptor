# Stage 3A topology migration

## What the "five topologies" actually were (Step 1 finding — verified by search)
There was never a five-topology registry, classifier, embedding index, or RAG store.
The previous Stage 3A used **five specification variants** (`DEBUG_SPEC_VARIANTS` in
`agentic_raptor/stage3a/generate.py`) over **one topology**: the built-in
five-transistor OTA template (`topology_generation/generator.py::build_five_transistor_ota`)
plus three seed RAG entries (`rag/retriever.py::seed_memory_for_smoke`).
Files referencing the old restriction: `stage3a/generate.py`,
`topology_generation/generator.py`, `configs/stage3a/dataset_debug.yaml`,
`tests/{conftest,test_provider,test_stage3a}.py`.

## Migration outcome
- Old assets are code templates, not data copies → nothing to archive under
  `deprecated/`; the 5T-OTA template REMAINS (it is also `topology_0041`-class
  entry in the corpus via AnalogGym-style parsing of the mock, and it still backs
  mock mode + Stage 1/2 tests). `legacy.enable_five_topology_mode: false` recorded.
- Corpus is now authoritative: `datasets/topology_library/` (dynamic discovery,
  57 families at build time — count never hard-coded) through
  `agentic_raptor/corpus/TopologyRegistry`.
- Hierarchical RAG: `datasets/topology_rag/` (family/block/graph levels; level-4
  simulation memory reserved and empty — no fabricated performance).
- Selection: `select_topology_candidates` returns top-K with score decomposition
  and candidate tier (`structurally_retrieved` / `transistor_mapped` /
  `electrically_validated`) — unvalidated graphs are never presented as verified.
- Splits: `datasets/topology_splits/` — seeded 70/15/15 family split + AnalogGym
  benchmark-holdout split with leakage report.
- Topology IDs changed from ad-hoc names to `topology_NNNN`; canonical
  `graph_hash` provides continuity (`find_by_graph_hash`).
- Experiment configs: new `configs/stage3a_topology_corpus.yaml`;
  `configs/stage3a/dataset_debug.yaml` unchanged (still valid for the 5T debug
  path, now one family among many).
- MCTS/validator/generator interfaces untouched; MCTS verified to run directly
  on corpus `CircuitGraph`s (one implementation for all sources).
- Sizing/SPICE not run in this migration, per constraints.
