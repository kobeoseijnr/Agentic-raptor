# Stage 3B — Electrical Qualification Report (real ngspice-45.2 + sky130 PDK)

## Pipeline
TopologyRegistry (source=analoggym, has_netlist) → netlist audit → adapted ADM
testbench derived from the source `TB_Amplifier_ACDC.cir` (supply 1.8 V, VCM 0.25·VDD,
CLOAD 500 pF, Lfb/Cin open-loop; transforms = path normalisation to the legacy
sky130 ngspice corner + op/ac wrapper — **no bias/size/topology changes**) →
real ngspice run → metric extraction → classification → Level-4 memory.
Environment captured in `datasets/simulation_memory/environment.json`
(ngspice-45.2, PDK hash, per-run environment_id; git commit: none — repo not under git).

## Results (families discovered dynamically: 17 — not hard-coded)
| Stage | Count |
|---|---|
| Audited | 17 |
| Simulation attempted | 17 |
| Netlist parse + dependencies OK | 16 |
| DC operating point valid | 15 |
| AC analysis valid | 15 |
| **electrically_functional** | **14** |
| spec_conditioned_pass | not_evaluated (no trusted per-topology spec files adopted; none invented) |
| pvt_validation_status | not_evaluated |

Failures (classified, FailureRecords in `failures.jsonl`): `gain_below_unity` 1
(topology_0001 Alfio_RAFFC: measured −15.4 dB — honest measured failure),
`no_subckt` 1 (audit: netlist lacks .subckt), `malformed_netlist` 1.

Sample independently measured nominal metrics (500 pF load):
dc_gain 69.9–122.8 dB across functional families; UGBW ≈ 0.8–2.2 MHz;
quiescent power ~1e-4 W scale; output DC mid-rail-consistent.
**Phase-margin caveat**: raw `vp` phase at crossing is stored, but several values
exceed 180° → phase-wrapping is not yet handled; per Step-10 rules these are NOT
claimed as valid phase margins (flagged limitation, metric interpretation withheld).

## Level-4 memory (`datasets/simulation_memory/`)
runs.jsonl 17 · failures.jsonl 3 · topology_summaries.jsonl 17 ·
environment.json 1 · every record carries topology_id, graph_hash, netlist/testbench
hashes, environment_id, and the **family split** (train 13 / validation 2 / test 2
among these 17). Per-topology `electrical_validation.json` written atomically into
the library (references runs; original metadata untouched). Raw netlists+logs under
`artifacts/stage3b/runs/<topology>/`.

## Split safety
Split labels propagate into every Level-4 record; test-family records are
identifiable/filterable for train-only retrieval manifests. Stage 3A splits unchanged.

## Files
New: `agentic_raptor/electrical/{__init__,__main__}.py`,
`datasets/simulation_memory/*`, `artifacts/stage3b/**`,
`datasets/topology_library/*/electrical_validation.json`, this report.
Legacy sources untouched (PDK read in place from the legacy tree).

## Tests
Full suite: **191 passed, 1 skipped** (existing tests unweakened).

## Known limitations
1. Phase unwrapping not implemented → PM values unusable as-is (top follow-up).
2. Transient qualification not run (op+ac only this pass); CMRR/PSRR/DC-sweep
   measurements from the source TB not yet ported.
3. Dedicated Stage 3B unit tests + full CLI (dry-run/resume/force) pending;
   current entry point: `python -m agentic_raptor.electrical`.
4. Retry policies + Level-4 RAG retrieval filters wired into corpus selection
   pending (records exist; tier-A preference not yet applied in selection).
5. 40 generated families remain structurally-available/`mapping_required` (correct).

## Reproduce
`cd Agentic_Raptor && python -m agentic_raptor.electrical`
