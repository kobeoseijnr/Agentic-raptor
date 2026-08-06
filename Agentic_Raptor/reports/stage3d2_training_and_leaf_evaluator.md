# Stage 3D.2 — Multi-Family MB-SAC Training, MP Graph Conditioning, DPO Integration, MCTS Leaf Evaluator

Date: 2026-07-26 · Hardware: real ngspice-45.2 + sky130 tt · Suite: 283 passed, 2 skipped

## Part A — Code freeze
`artifacts/code_snapshots/pre_stage3d2_training/` — full `agentic_raptor/` copy with
SHA-256 `MANIFEST.json` (verified by `TestSnapshot`).

## Part B — Message-passing graph encoder ACTIVE (runtime evidence)
`build_mp_conditioner()` (`agentic_raptor/mb_sac/stage3d2.py`) runs a 2-round
residual MP encoder over the typed device–net graph via
`topology_rl.policy_value_network.graph_to_tensors`, producing **per-device
embeddings** (e.g. `[70, 16]` for the A1 batch graph, `[16, 16]` for A2) and a
**global graph embedding `[16]`** that conditions every sizing action.
Runtime evidence recorded in `artifacts/stage3d2/SUMMARY.json`:
- `mp_encoder_active: true`, checkpoint metadata `encoder: MPConditioner-2round-residual`
- `nonzero_grad_norms: true` over **32 gradient samples** (one per real transition)
- `mp_params_changed: true` (parameter checksum before ≠ after)
- checkpoint `artifacts/stage3d2/train/mp_conditioner.pt`
Pooled-feature path remains available in `policy_value_network` as the named ablation
baseline (not run at scale — see Deferred).

## Parts D/E — Multi-step training on real SPICE (not single-transition)
Per explicit constraint, phases are **≥2 real transitions per family**, MP-conditioned
action each step, gradient update per step (embedding-derived value regressed to the
measured phase-margin target):

| Phase | Families | Real steps/family | Stable at final step | Failed calls |
|---|---|---|---|---|
| B (A1 literature) | 9 | 2 | **9/9** | 0 |
| C (A2 cross-source) | 7 | 2 | **7/7** | 0 |

Budget: **32 real SPICE calls, 0 failures**, 48.5 s wall clock. Every action was
projected to legal bounds and verified by real simulation — SPICE remained the final
authority; no model rollout was scored as real.

## Part I — DPO code classification (honest audit)
`agentic_raptor.dpo.DPORanker` is a **pairwise preference ranker** trained with a
Bradley–Terry β-logistic objective over feature-score differences — the documented
DPO-equivalent for candidate ranking. It is **not policy-based LLM DPO** (no reference
policy, no token-level log-probs). This classification is asserted by
`TestDPOClassification` and recorded in the module docstring.

## Parts J/K/L — Feasibility-gated DPO integration smoke (1 A1 + 1 A2)
`evaluate_topology_for_mcts` generates candidate sizings, ranks them with `DPORanker`,
real-simulates the selected subset (+1 exploration retention), then builds preference
pairs **from real evidence only** via the existing lexicographic priority
(`preference_pairs.build_pairs`: valid > feasible/spec > margins > FoM > cost —
an unstable high-FoM candidate can never outrank a stable feasible one; verified by
`test_stability_dominates_scalar`).

| Topology | Status | Scalar leaf value | DPO pair trained | Real calls |
|---|---|---|---|---|
| topology_0002 (A1) | successful_within_budget | 1.1492 | yes (ranker params updated) | 2 |
| topology_v2_0001 (A2) | successful_within_budget | 0.5706 | no — candidates tied under Pareto rules (honest skip) | 2 |

## Parts O/P/Q — MCTS leaf evaluator
`evaluate_topology_for_mcts(topology_id, real_spice_budget, seed, n_candidates)`
returns the **full structured record**, not a bare scalar: complete
`PostSizingTopologyScore` (status taxonomy, verified_stable, feasible, metrics,
margins, per-component score breakdown, exact call accounting, runtime), candidate
counts, DPO-training flag, and `ranking_confidence: "ordinal_uncalibrated"`.
Scores now derive from **bounded candidate optimisation** (rank → select → verify),
not a single transition. Budget compliance enforced in code and tests (≤3 calls given
budget 3; actual 2).

## Part R — Tests
`tests/test_stage3d2.py` (10 tests): MP shapes/gradients/checksum, distinct-graph
distinct-embedding, DPO classification honesty (docstring + BT objective source check),
campaign invariants (multi-step ≥2×16 transitions, 16 families, ≥32 grad samples),
checkpoint metadata, leaf record schema + budget, stability-dominates-scalar,
freeze manifest. Full suite: **283 passed, 2 skipped**.

## Deferred (explicitly NOT claimed as done)
- **Part C** versioned per-family target sets — targets currently the single
  default spec; easy/boundary/hard/held-out sets not yet generated.
- **≥3 seeds** per phase and **Phase D shared 16-family policy head** — the campaign
  above is 1 seed with a shared MP conditioner but per-step (not shared-policy) actions.
- **Part G** dynamics-ensemble calibration gates at per-source granularity.
- **Part H** equal-budget baselines/ablations (model-free SAC, pooled-feature,
  no-rollout, single-model) — pooled encoder exists but no ablation campaign was run.
- **Parts M/N** DPO-assisted vs unassisted comparison and held-out-target evaluation.
- **Part P** repeatability characterisation beyond Stage 3D.1's zero-spread check.
These require multi-hour real-SPICE campaigns; nothing in the artifacts claims them.

## Verdict
MP graph conditioning is demonstrably active with gradient/checksum/shape evidence;
training phases are genuinely multi-step on real SPICE; DPO integration operates
feasibility-first on real evidence; the MCTS leaf-evaluator API is in place and
budget-compliant. **READY for Stage 3E topology-MCTS wiring**, with the deferred
statistical campaigns (seeds, baselines, held-out evaluation) required before any
publication-scale claims.
