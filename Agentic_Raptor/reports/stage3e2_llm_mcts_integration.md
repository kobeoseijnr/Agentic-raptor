# Stage 3E.2 LLM-DPO — mcts integration

Date: 2026-07-26

{
 "proposal_root_hash": "da58394d666835bc11341d51279f3120",
 "tree_nodes": 2,
 "root_visits": 2,
 "value_calls": 2,
 "validated_before_ingestion": true,
 "schema_version": "3e2L.1"
}
Path: proposal -> validator -> mapper -> CircuitGraph -> TopologySearchState -> policy/value + MCTS (value-only, no bypass). Full equal-budget root comparison (LLM vs registry vs RAG-only vs random) deferred: blocker = no valid model-generated proposals at CPU scale.
