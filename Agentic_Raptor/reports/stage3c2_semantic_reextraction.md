# Stage 3C.2 - Verified CktGNN Re-extraction & Migration Report

## Defect removed
tools/topology_extractor/cktgnn.py `_role` t%8 heuristic REPLACED with the
verified SUBG_NODE decoder; unknown codes raise `unknown_cktgnn_subgraph_type`
(never coerced). Regression test asserts "% 8" cannot return.

## Verified semantics (programmatic + source-cited)
agentic_raptor/corpus/cktgnn_semantics_v2.json: NODE_TYPE + 26-entry SUBG_NODE
basis from circuit_generation.py L33-70; gm notation (sign1=polarity,
sign2=ff/fb); zero-based subgraph ids = vertex 'type'; limitations noted
(codes 18-25 par/ser split from generation comments).

## Archive
artifacts/corpus_snapshots/stage3a_legacy_invalidated_v1/: 118 files with
SHA-256 manifest; reason, affected (39 cktgnn) and unaffected families listed;
read-only from migration code.

## Re-extraction (raw OCB ckt_bench_101, 400 graphs)
extracted_verified 400 - warnings 0 - unknown types 0 - malformed 0.
Attribute-aware canonical hashing (core fix: block_role now participates in WL
labels -> gm polarity/direction change the hash; verified by test).
Unique graphs 200 - duplicates removed 200 - clusters 142.

## Corrected corpus
**142 v2 families** (topology_v2_0001..0142) in artifacts/topology_registry_v2/
(graph/metadata/blocks/interpretation JSON each; semantic_version
verified_subgnode_v2; raw_source_ids preserved). Legacy count (39) NOT
targeted; legacy IDs never reused. Lineage: legacy_to_v2_lineage.jsonl -
honest many_to_many (confidence 0.5; legacy labels cannot support finer
mapping); AnalogGym/OPAMP-Generator families untouched.

## Interpretation (Stage 3C.1 rerun, unchanged gates)
142/142 interpretation_ready with valid FunctionalStageGraphs (100% gm
interpretation + polarity coverage; 2-4 main-path stages) - the verified
semantics unlock what wrong labels blocked. 0 ambiguous, 0 unsupported.

## Not yet run in 3C.2 (honest)
Transistor realization + ngspice qualification of the 142 (the Stage 3C
template bridge accepts FSG stage counts; execution deferred - no electrical
claims are made for v2 families). Registry-v2 splits/RAG rebuild pending;
default retrieval still serves v1 corpus (v2 lives in a separate registry
root; legacy cktgnn families remain marked withheld/invalidated in memory).

## Tests
26-entry decoder matrix (full, parameterized), unknown-type rejection, defect
removal, sign-sensitive hashing, v2 registry/lineage/archive checks.
Full suite: **249 passed, 1 skipped** - nothing weakened.
