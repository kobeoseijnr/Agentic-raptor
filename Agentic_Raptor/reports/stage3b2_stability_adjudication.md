# Stage 3B.2 — Polarity & Stability Adjudication Report

## Investigated (all functional families with verified PM < 0)
topology_0004 (-35.3), topology_0006 (-68.1), topology_0007 (-52.2),
topology_0010 (-18.2), topology_0017 (-64.2) [deg, convention A]

## Method (no circuit changes)
1. First attempt: input-slot swap -> DC loop became POSITIVE feedback -> railed
   bias, degenerate TF. Recorded as ORIENTATION EVIDENCE: convention A is the
   correct negative-feedback wiring per source testbench.
2. Adjudication runs: pure AC sign flip (ac 1 -> ac -1), DC loop untouched.
   Checks: |H| identical (<0.5 dB median), phase shift ~180 deg (<10 deg),
   PM reproduced within 2 deg (PM is DC-phase-referenced, convention-free).

## Verdicts
- polarity_mismatch: 0
- verified_unstable: 5 (all five: linear sign flip clean, PM reproduced,
  negative-feedback orientation confirmed) -- genuinely unstable OPEN-LOOP
  responses UNDER THE 500 pF ADM TESTBENCH; not pipeline artefacts.
- ambiguous / unsupported / insufficient_information: 0
- verified_stable annotated on the 9 functional families with PM >= 0.

## Level-4 updates
adjudication_runs.jsonl: 5 child records (parent_run_id links, polarity fields,
comparison, reasoning); originals preserved; summaries annotated with
stability_status / polarity_status / measurement_revision=stage3b2.
Retrieval can now distinguish "negative PM" from "polarity artefact" (none found).

## Scientific conclusion
The Stage 3B measurement pipeline is vindicated: every negative PM is a real
property of the standardized bench (heavy 500 pF load on multi-stage amplifiers
whose source compensation targets their own benches), not a sign error.

## Tests
tests/test_adjudication.py (5 deterministic invariants). Full suite: 206 passed, 1 skipped.
