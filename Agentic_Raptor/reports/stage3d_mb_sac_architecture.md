# Stage 3D - Graph-Conditioned MB-SAC: Implementation Report

## Architecture (reuses the tested genuine core, wired to operational v3)
State: graph features + sizing vector + spec embedding + measured metrics +
constraint margins + SPICE-budget fraction + environment-mode flag
(spec-conditioned by construction). Actor: squashed-Gaussian (rsample, tanh,
log-prob correction), auto entropy temperature. Critics: two independently
initialised Q nets, clipped double-Q, Polyak targets (independence
test-verified). Dynamics: probabilistic-role ENSEMBLE (2 members, bootstrapped
real-only training, disagreement uncertainty). Rollout gate: disagreement
threshold; model transitions labelled source=model, never final authority.
Actions: SizingParameterSpace per topology (variable dimension, log-scale
bounds) with deterministic projection (W clamped to Sky130-safe range in the
smoke evaluator); no topology edits; no compensation insertion.

## Pools (from frozen snapshot pre_stage3d_mb_sac, checksummed)
NORMAL (16): 9 A1 literature + 7 A2 generated verified-stable.
REPAIR AUDIT (all 92 D2): repair_compensation_value_eligible 71 -
repair_sizing_eligible 15 - repair_combined_eligible 5 -
missing_controllable_stability_variable 1 (ineligible; retained in
failure-conditioned retrieval). => repair-training pool = 91.
EXCLUDED: 63 C1 withheld + 3 B2 (test-enforced out of all pools).

## Smoke experiment (Phase A, real ngspice-45.2 + sky130; topology_v2_0001)
5 real SPICE calls (0 failed) - 4 sizing transitions - actor update 1,
critic update 1, ensemble update 1 - uncertainty 0.163 -> gate PASSED ->
2 model transitions - checkpoint saved + deterministic resume verified.
Final (real, measured): 36.2 dB, UGBW 991 Hz @ 500 pF, PM +91.3 deg ->
verified_stable maintained. Mechanics validation only - NOT a performance
claim. Deviation noted: smoke ran on an A2 family (its emitted netlist
parameters are directly controllable); A1 param-file injection is the next
engineering step.

## Tests
Full suite: **267 passed, 1 skipped** (5 new Stage 3D tests: exact pools,
repair-audit policy, excluded-family enforcement, smoke mechanics incl.
critic independence + gated rollouts + resume, snapshot freeze).

## Limitations / next
A1 sizing-injection (design_variables writer); full curriculum phases B-F,
baselines and ablations not yet run; PVT reserved for final verification;
compound cache used for qualification, per-transition caching TBD; rewards
currently margin-sum scalar - full vector reward components pending.
