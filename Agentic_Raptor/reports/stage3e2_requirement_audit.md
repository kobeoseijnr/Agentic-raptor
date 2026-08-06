# Stage 3E.2 — Requirement Completion Audit

Date: 2026-07-26. Statuses: C=complete, P=partially complete, D=deferred, M=was-missing→closed here.
Evidence paths are runtime artifacts, not class existence.

## Stage 3D.2 prompt requirements

| ID | Requirement | Pre-3E.2 | Action in 3E.2 | Evidence |
|---|---|---|---|---|
| D2-A | code freeze snapshot | P (manifest never completed) | M→C: verified in pre_stage3e1 + pre_stage3e2 snapshots | artifacts/code_snapshots/pre_stage3e2_full_generation/VERIFICATION.json |
| D2-B | MP encoder active w/ evidence | C | preserved | artifacts/stage3d2/SUMMARY.json |
| D2-C | versioned target sets (easy/boundary/hard/val/heldout) | M | M→C: 80 records, 16 families, splits, leakage tests | datasets/target_sets_v1/ |
| D2-D/E | multi-step Phase B/C | C (2 steps/family, 1 seed) | extended to 3 seeds in Phase-D | artifacts/stage3e2/phase_d.json |
| D2-F | shared Phase-D 16-family policy | M | M→C (bounded): shared actor/critics/encoder, 3 seeds, real SPICE | artifacts/stage3e2/phase_d.json |
| D2-G | dynamics calibration gates | M | M→C (source+global gates; topology gate honestly disabled: insufficient data) | artifacts/stage3e2/calibration.json |
| D2-H | equal-budget baselines/ablations | M | P: MCTS baselines 3-seed + ranker comparison done; model-free/topology-specific SAC ablations remain D (compute; command recorded) | artifacts/stage3e2/{mcts_baselines,ranker_comparison}.json |
| D2-I | honest DPO classification | C | terminology re-audited everywhere | stage3e2_edits docstring, reports |
| D2-J/K/L | feasibility-gated pairs + smoke | C | preserved; comparison added | artifacts/stage3e2/ranker_comparison.json |
| D2-M | DPO-assisted vs unassisted comparison | M | M→C (bounded): bt_ranker vs random vs scalar heuristic, equal budget | ranker_comparison.json |
| D2-N | held-out-target evaluation | M | M→C: 16 held-out targets, trained seed-0 actor, leakage-guarded | artifacts/stage3e2/heldout.json |
| D2-O/P/Q | score from bounded optimisation + repeatability + MCTS API | C (3D.2/3E.1) | preserved; repeatability extension D (compute) | stage3d1/stage3d2 artifacts |

## Stage 3E.1 prompt requirements

| ID | Requirement | Pre-3E.2 | Action | Evidence |
|---|---|---|---|---|
| E1-B/C | state+action schemas | C | preserved | stage3e1.py, tests |
| E1-C.exec | 8 structural categories *executable* | M (validator-gated) | M→C for 7 categories (all but general synthesis variants); real netlists + ngspice | artifacts/stage3e2/edit_demo.json |
| E1-D | rejection reasons | C | extended with EditRejected taxonomy | edit_demo.json rejected map |
| E1-G/H/I | policy/value/shared encoder | C | mixed-action embedding extension (edit features) | alphazero_campaign.json |
| E1-N..P | terminals/noise/result | C | preserved | stage3e1 tests |
| E1-Q/R | training records + losses | C (smoke) | 3-seed campaign | alphazero_campaign.json |
| E1-T | LLM boundary | P (interface only) | M→C: fixture provider + full validation path + real netlist + ngspice; external multimodal adapter D (no deterministic test possible; adapter slot documented) | artifacts/stage3e2/llm_demo.json |
| E1-V | baselines execute | C | 3-seed equal-budget comparison added | mcts_baselines.json |
| E1-W/X/Y/Z | CLI/configs/tests/reports | C | audited + extended for 3E.2 | reports/stage3e2_cli_and_configs.md |

## Known-limitation closures (items 1–21 of the 3E.2 prompt)

1 executable edits: **closed** (7 categories; see edit_demo). 2 generation vs selection: **closed at edit level** — new topologies created by edits + proposals reach real SPICE; free-form transistor synthesis remains out of scope by design. 3 target sets: **closed**. 4 Phase-D: **closed (bounded)**. 5 three-seed campaigns: **closed for AZ, Phase-D, MCTS baselines**. 6 held-out: **targets closed; held-out topology split technically unsupported at n=16 stable (all needed for training) — unseen edited/proposed topologies serve as the unseen-structure evaluation**. 7 calibration gates: **closed (global+source), topology gate blocked by data volume**. 8 model-free SAC comparison: **deferred — compute; exact command: `python -m agentic_raptor.topology_rl.stage3e2 --ablation model_free`** (config written). 9 topology-specific SAC: **deferred — compute (16 separate trainings)**. 10 ablations: **partially — pooled/no-rollout configs exist, campaigns deferred (compute)**. 11 ranker comparison: **closed (bounded)**. 12 extended repeatability: **deferred — compute**. 13 scalar calibration: **characterised ordinal; success-probability head deferred (needs campaign-scale outcome data)**. 14 AZ training: **3-seed campaign closed (bounded)**. 15 AZ vs baselines: **closed (bounded 3-seed)**. 16 fewer-SPICE-calls claim: **NOT claimed — bounded evidence insufficient; recorded as unsupported claim**. 17 LLM proposals: **closed via fixture provider (same validated path)**. 18 cross-level campaign scale: **partially — records written; scale deferred**. 19 repair curriculum: **sampled (5 families) — full 91 deferred (compute; same code path)**. 20 PVT: **closed (10-point matrix, separate accounting)**. 21 CLI/config/test audit: **closed** — see stage3e2_cli_and_configs.md.
