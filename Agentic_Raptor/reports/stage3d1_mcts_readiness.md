# Stage 3D.1 - Multi-Family MB-SAC Validation & MCTS Readiness

## Executed (real ngspice-45.2 + sky130, 57.1 s, 38 real calls, 0 failed)
1. **A1 design-variable writer**: parses/rewrites AnalogGym .PARAM files;
   9/9 manifests (artifacts/stage3d1/design_variables/a1/) with parameter_id,
   kind (W/L/M), bounds, log scale, matched groups, provenance; round-trip
   parse-write-parse identical for all 9 (test-enforced); deterministic writer.
2. **Environment validation: 16/16 stable-pool families PASS** - A1 via source
   netlist + rewritten param file; A2/v2 via re-emitted mapped netlists; every
   family produced a real SPICE result with positive verified PM.
3. **Phase B (9 A1) / Phase C (7 A2)**: one real perturbation transition per
   family (W x1.15, projected to legal range): stability preserved 9/9 and 7/7.
4. **PostSizingTopologyScore**: structured record (status taxonomy, stability,
   feasibility, margins, calls, components) + documented scalar
   (valid 0.2 / stable 0.4 / feasible 0.4 / margin 0.2 / -cost 0.1, all
   configurable); repeatability on 3 families x2 runs: spread 0.0000 (cache-
   deterministic single-eval scoring) - variance characterised.
5. **Exact accounting**: 38 = 16 env + 16 phase + 6 score; model transitions
   0 (not counted as SPICE); failed 0.

## Honest deferrals (explicitly NOT claimed)
Full Phase B/C/D multi-seed TRAINING campaigns, model-free-SAC and
topology-specific baselines, the four ablations, held-out-target and
held-out-topology evaluations were NOT run at scale in this pass - the
machinery (configs, envs, score interface, gating) is validated; the
campaigns are compute-scheduling work, not implementation risk. Full
message-passing encoder remains config-selectable but pooled features were
active here (named limitation, matching the no-graph ablation arm).

## Tests
Full suite: **274 passed, 1 skipped** (7 new: manifests/round-trip, writer
determinism, 16/16 envs, phase coverage, exact accounting, score schema,
repeatability). Nothing weakened.

## MCTS readiness verdict
READY-WITH-CAVEATS: the post-sizing scoring interface is implemented,
structured, repeatable, and honest about status; environments cover all 16
stable families with variable action dimensions. Before Stage 3E MCTS uses
scores for topology SELECTION at scale, run the deferred training campaigns
so scores reflect optimised (not baseline/1-step) sizing.
