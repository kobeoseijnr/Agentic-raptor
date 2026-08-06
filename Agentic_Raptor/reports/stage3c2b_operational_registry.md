# Stage 3C.2b - Operational Registry, Realisation & Qualification Report

Registry v3: artifacts/topology_registry_operational_v3/ - EXACTLY 174 active
(17 AnalogGym + 15 og2 + 142 cktgnn_v2); topology_0057 lineage-only; CktGNN v1
excluded; hash_version 2 everywhere; manifest+summary checksummed.
Splits (seed 42): train 127 / validation 24 / test 23; AG assignments
preserved; og2 descendants share one inherited group; hash-leakage test clean.
RAG rebuilt from v3: 174 topology + 772 block + 174 graph records (v1
identities gone; level-4 references via simulation_memory files).

## Realisation & qualification (real ngspice-45.2 + sky130; 121.9 s)
Generated families processed: 157. Withheld (honest, pre-simulation): 63
(feedback-transconductor branches lack a verified template; zero-stage graphs)
- never counted as failures. Static validation: 94/94 passed (0 failures).
Simulated: 94 -> **94/94 electrically functional** (gain > unity, finite AC,
valid OP under the standardized 500 pF ADM bench).
Stability (3B.1/3B.2 rules, outcome-independent templates):
**7 verified_stable**, 87 verified_unstable, 0 ambiguous.

## Strict tiers (A requires verified_stable)
A1 = 9 (literature stable) - A2 = 7 (generated stable) - B1 = 0 - B2 = 3
(literature not fully qualified) - C1 = 63 (withheld/awaiting templates) -
C2 = 0 - D1 = 0 - D2 = 92 (functional-but-unstable + unstable literature).

## Evidence & consistency
AnalogGym: 17/17 evidence preserved, zero reruns. All raw artifacts under
artifacts/stage3c2b/<source>/<tid>/<realisation>/ + Level-4
stage3c2b_runs.jsonl. Consistency assertions in module + tests: 174 exact,
exclusions, hash-v2, og2 grouping, withheld/failed separation - all green.

## Tests
Full suite: **262 passed, 1 skipped** (7 new operational tests; none weakened).

## Stage 3D pools
- NORMAL SIZING POOL (16): 9 A1 literature + 7 A2 generated verified-stable.
- REPAIR-TRAINING POOL (92): D2 functional-but-unstable records (real,
  reproducible instability evidence - ideal for compensation-aware sizing).
- EXCLUDED (66): 63 C1 withheld + 3 B2 until templates/qualification exist.

## Limitations
Single realisation candidate per family (NMOS-input template only); feedback
branches untemplated (the 63); PM stability is single-bench (500 pF ADM);
og2/v2 sizing is initial_prior_sized by definition.
