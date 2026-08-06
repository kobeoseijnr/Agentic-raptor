# Stage 3C.1 — CktGNN Semantics Audit & GM Interpretation Report

## Verified semantics (evidence: repositories/CktGNN/OCB/src/circuit_generation.py L33-70)
NODE_TYPE: R=0, C=1, '+gm+'=2, '-gm+'=3, '+gm-'=4, '-gm-'=5, sudo_in=6,
sudo_out=7, In=8, Out=9. SUBG_NODE basis 0-25: In/Out, R, C, R+C (par=4/ser=5),
lone +-gm+-, C||gm (10-13), R||gm (14-17), C-R-gm par (18-21) / ser (22-25).
gm notation: FIRST sign = transconductance polarity; SECOND sign = path
direction (+ = feedforward/main path, - = feedback). Also inspected:
utils_src.py (subg_feature_type L164-172 confirms sudo_in=6/sudo_out=7 framing),
amp_generator.py (2-3 stage generation flow).

## Critical honest finding
The Stage 3A extractor (tools/topology_extractor/cktgnn.py `_role`) used an
UNVERIFIED modulo heuristic that disagrees with the verified basis for types
>= 4 (e.g. verified subg 4 = R+C parallel; heuristic said "R"). Corpus CktGNN
node labels are therefore unreliable. Rejected assumption: "type % 8 maps to
device kind". Consequence: ALL behavioral CktGNN families are classified
`withheld_at_semantic_audit` (39 recorded in
datasets/simulation_memory/interpretation_runs.jsonl with explicit reasons) --
NOT interpreted on top of wrong labels, NOT recorded as failed simulations.

## Implemented and validated (agentic_raptor/mapping/interpretation.py)
- Verified NODE_TYPE / SUBG_NODE tables embedded with source citations.
- FunctionalStageGraph IR (stages with function/polarity/path_class/evidence/
  rule/confidence/unresolved; validation() computes main_path_coverage,
  interpreted/unresolved gm fractions, polarity_coverage).
- Rule library (outcome-independent by construction): R1 input-connected gm ->
  differential input transconductor; R2 internal gm -> common-source
  equivalent; R3 gm feeding Out -> output transconductor; R4 second-sign '-'
  -> feedback transconductor; R5 capacitor branch -> compensation; R6 parallel
  short path -> feedforward. Inversion parity from '-gm' main-path count.
- 7 deterministic synthetic-graph tests prove: category detection, feedback vs
  feedforward, parity, validation scores, and that NO electrical fields exist
  in interpretation records (outcome independence).

## Counts
behavioral families audited: 39 (37 behavioral_only + 2 ambiguous share the
label defect) - interpretation-ready: 0 (pending re-extraction) - candidates:
0 - realizations: 0 - fabricated values: 0. Full suite: 218 passed, 1 skipped.

## Path to yield (one step, now unblocked)
Re-run the corpus extractor with the verified SUBG_NODE table (re-extraction
changes Stage 3A data, deliberately NOT done inside 3C.1 without approval),
then interpret_verified_dag() -> FunctionalStageGraph -> Stage 3C templates.

## Limitations
Re-extraction pending; A1/A2/C1/C2/D1/D2 tier constants recorded in mapping
records but corpus selector still exposes the Stage 3C tiers; OPAMP-Generator
ff/fb branches likewise await the analogous verified-semantics pass.
