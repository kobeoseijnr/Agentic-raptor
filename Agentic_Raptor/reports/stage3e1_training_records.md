# Stage 3E.1 — training records

Date: 2026-07-26

2 TopologyTrainingExample records (schema 3e1.1): state, legal ids, MCTS visit distribution (policy target, sums to 1), SPICE-backed structured outcome + scalar value target (1.057), component target, spec, budget, lineage, split, seed, provenance=stage3e1_mcts. Examples without a SPICE-backed outcome are skipped by train_step — never fabricated.
