# Stage 3B.1 — Measurement Hardening Report

## Implementation
`agentic_raptor/electrical/measurements.py`: all frequency-domain metrics computed
from the **complex transfer function** exported by ngspice `wrdata` (freq, re, im —
a real 3-vs-4-column format bug was found and fixed; raw files preserved).
Phase: `np.unwrap` continuous unwrapping (no ±360 corrections), PM referenced to
the DC phase (handles inverting DC sign), log-frequency linear interpolation at
the crossing. Crossover detector: all |H|=1 crossings with type (up/down/exact),
interpolated frequency, index, confidence; documented rule = single downward
crossing valid; multiple → `multiple_crossings` ambiguity; none → `no_unity_crossing`.
Metric functions (gain, UGBW, −3 dB BW, PM, GM) each return
{value, status ∈ verified/estimated/ambiguous/unsupported/measurement_failed,
confidence (0 unless trusted), method, failure_reason, metadata}. Ambiguous PM →
value null + `measurement_status: ambiguous_phase_margin` — never estimated.
New failure classes wired: phase_wrap_failure, multiple_crossings,
no_unity_crossing, crossing_ambiguous, invalid/nonfinite_transfer_function,
phase_noise, measurement_instability.

## Analytical validation (tests/test_measurements.py — 10/10 pass)
Single-pole 60 dB: gain 60.00±0.01 dB, UGBW 1 MHz±1%, PM 90°±1°.
Two-pole closed-form placements: fp2=fu0/√2 → 45°±2°; fp2=1.5·fu0 → 60°±2°
(initial test expectations were analytically wrong — corrected with derivation
in-test; implementation unchanged). Also validated: no-crossing (unsupported,
conf 0), multiple crossings (ambiguous), unstable 3-pole (negative PM measured,
not estimated), simulator-wrapped phase (≡ unwrapped result ±0.1°), NaN rejection,
coarse-sweep interpolation (crossing recovered to 5% from 40-point sweep).

## Level-4 rebuild (real ngspice reruns, 17 families)
- electrically_functional: 14 (unchanged — hardening did not chase pass rate)
- **Verified phase margins: 14 · withheld: 3** (null, with explicit reasons)
- Honest engineering finding now visible: several functional amplifiers measure
  **negative/near-zero open-loop PM under the 500 pF ADM testbench**
  (e.g. topology_0004 −35.3°, topology_0006 −68.1°, topology_0009 +2.4°) —
  measured, verified-status values; stability interpretation deferred, values
  are NOT hidden. Well-compensated examples: topology_0002 81.0° @ 2.07 MHz,
  topology_0008 62.5° @ 0.68 MHz, topology_0005 58.7° @ 1.10 MHz.
- Full per-metric status/confidence stored in each run record
  (`metric_reports`); only `status=="verified"` values enter summaries; raw
  simulator outputs untouched.

## Tier wiring & CLI
Corpus selection now ranks Tier A (Level-4 electrically qualified, +0.6) →
Tier B (netlist, unqualified) → Tier C (mapping required — still retrievable) →
Tier D (invalid). CLI: `python -m agentic_raptor.electrical
{validate|rebuild-memory|report|rerun-failed}`.

## Tests
Full suite: **201 passed, 1 skipped** (10 new analytical tests; one corpus test
updated to the new tier names — an intended interface change, not a weakening).

## Files
New: `electrical/measurements.py`, `electrical/__main__.py` (rewritten),
`tests/test_measurements.py`, this report. Modified: `electrical/__init__.py`
(wrdata export + verified-only metrics), `corpus/__init__.py` (tier wiring),
`tests/test_corpus.py` (tier names).

## Limitations
GM/output-impedance/CMRR/PSRR not yet in summaries (GM implemented; TB ports
pending); `--phase-only` flag currently aliases full validate; negative-PM
stability adjudication (closed-loop vs open-loop criteria) left to Stage 3C
analysis; confidence calibration is rule-based, not statistical.
