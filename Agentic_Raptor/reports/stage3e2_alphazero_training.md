# Stage 3E.2 — alphazero training

Date: 2026-07-26

3 seeds (0,1,2), 1 episode/seed, 6 simulations, SPICE budget 4/episode. Per-seed training: {"0": {"policy_loss": 1.3928, "value_loss": 1.2945, "grad_norm": 4.491, "policy_entropy": 1.386, "heads_changed": true, "encoder_changed": true}, "1": {"policy_loss": 1.3924, "value_loss": 0.9046, "grad_norm": 3.768, "policy_entropy": 1.386, "heads_changed": true, "encoder_changed": true}, "2": {"policy_loss": 1.3944, "value_loss": 1.1541, "grad_norm": 3.6, "policy_entropy": 1.386, "heads_changed": true, "encoder_changed": true}}. PVs: {"0": ["a_sel_topology_0002"], "1": ["a_sel_topology_0002"], "2": ["a_sel_topology_0002"]}. Bounded engineering scale — one episode per seed is NOT claimed as convergence evidence; scaling requires only episodes_per_seed in configs/stage3e2/alphazero_training.yaml.
