# Stage 3E.2 — target sets

Date: 2026-07-26

datasets/target_sets_v1/: 80 records = 16 families x 5 tiers ['easy', 'boundary', 'hard', 'validation', 'heldout']. Splits: train (easy/boundary/hard), validation, test(heldout). Anchored to measured baselines with provenance; seed 42. Held-out TOPOLOGY split: all 16 stable families are needed for training; held-out TOPOLOGY split is technically unsupported at n=16 stable — held-out targets + unseen edited/proposed topologies serve instead. Leakage tests in tests/test_stage3e2.py::TestTargetsAndLeakage pass.
