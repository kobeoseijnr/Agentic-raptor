# Stage 3C — Transistor Mapping Report

## Architecture
CircuitGraph -> functional interpretation -> DeviceCircuitGraph (typed devices,
roles, matched groups pair1/mir1/tail1/cs_k, support-bias labelling, polarity
parity, provenance per assignment) -> prior-based sizing (AnalogGym-derived
priors, origin+confidence per device) -> Sky130 subckt (AnalogGym port
convention -> Stage 3B pipeline reused verbatim) -> static validation
(terminals, ports, shorts, duplicate IDs, model names, graph-preservation
score) -> real ngspice qualification. Bias scaffold (M6+IB1) labelled
generated_support_bias, preserved separately. Compensation only when C-type
blocks exist structurally (never inserted for stability). Polarity by
structural stage-inversion parity, explicitly NOT outcome-based.

## Audit of 40 generated families (reports basis: audit_generated)
- mapping_ready: 1  - partially_specified: 0  - behavioral_only: 37
  (CktGNN gm_pos/gm_neg abstractions need a gm->stage interpretation rule
  not yet implemented -> honestly NOT mapped)  - structurally_ambiguous: 2

## Mapping + qualification results (real ngspice, sky130)
- candidates generated: 1  - statically valid: 1
- electrically functional: 1 (topology_0057: measured 107.8 dB,
  UGBW 31.5 kHz under 500 pF, verified PM = -6.0 deg -> would classify
  verified_unstable per 3B.2 rules; reported as measured, not repaired)
- failed attempts: 0 unique failures (39 families correctly withheld at audit)
- Level-4: mapping_runs.jsonl (all attempts incl. withheld); artifacts under
  artifacts/stage3c/<tid>/<candidate>/ (netlist, device_graph, validation,
  provenance, raw ngspice run)

## Retrieval tiers
Mapped-functional generated families qualify as Tier A2 (recorded in mapping
records; corpus tier hook reads electrical_validation.json — wiring of A1/A2
sub-tiers into select_topology_candidates listed as follow-up).

## Tests
tests/test_mapping.py (5 deterministic tests: audit coverage, role/group
determinism, sky130 emission+static validation, unsupported-family refusal,
memory retention). Full suite: 211 passed, 1 skipped.

## Known limitations
1. gm-level (CktGNN) interpretation rule missing -> 37/40 families honestly
   unmapped this pass; that rule is the top Stage 3C.1 item.
2. Single mapping candidate per family (no alternatives yet); ff/fb branches
   of opamp-generator families unrealized (recorded unresolved).
3. Tier A1/A2 split not yet in corpus selector; adjudication not auto-invoked
   for mapped candidates (PM verdict recorded from measurement only).

## Recommendation — Stage 3D
Either (a) 3C.1: gm-block interpretation to unlock the 37 CktGNN families, or
(b) proceed to MB-SAC sizing on the 14+1 functional families (the mapped
candidate + literature set) since the sizing loop is the core research loop —
suggest (a) briefly then (b).

