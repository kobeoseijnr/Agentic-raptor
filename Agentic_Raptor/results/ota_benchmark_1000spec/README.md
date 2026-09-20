# RAPTOR main result on the current-mirror OTA benchmark (1,000 specs, budget 120)

`raptor_ota1000_b120_main.csv` — one row per target specification (1,000 rows), the design
RAPTOR returns for that spec, judged with the benchmark's strict pass rule and FoM.

| | |
|---|---|
| Benchmark | ORACLE current-mirror OTA, 45 nm PTM, ngspice, `specs_1000.pkl` (same file every baseline uses) |
| Run | RAPTOR (agents + SAC sizer), 120 SPICE calls per spec, seed 0 |
| Returned design | from each spec's 10-design portfolio, the passing design with the largest worst-case relative margin over (gain, UGBW, PM, Ibias) |
| Pass | 1000 / 1000 |
| Mean FoM | 13.186 |
| Runtime | 13.70 s per spec (recorded wall time) |

Columns: `spec_id`, `target_*` (gain dB, UGBW MHz, PM deg, Ibias A), `output_*` (delivered values),
`fom`, per-metric pass flags, `complete_pass`, `sims_used`, `first_pass_sim`, `budget`, `seed`,
`params_idx` (10 grid indices of the returned design: nf_bias|nf_tail|nf_in|nf_md|nf_mo|nf_sum|nf_cs|nf_csn|cc|rz),
`runtime_sec`, `sizing_parameters` (the same design decoded to physical values: finger counts, cc in pF, rz in kΩ), and the agent bookkeeping fields (`distinct_passing`, `committed_branch`,
`probe_verdict_a/b`, `banked_final`, `recovery_executed`, `plan_difficulty`).

Note: the benchmark netlist hard-codes cc = 1 pF and rz = 3 kΩ, so the last two indices are not
applied; the eight finger counts define the simulated circuit (true for every method on this benchmark).
