# Stage 2 Implementation Report

Date: 2026-07-25. Environment: Windows 11, Python 3.14.2, torch CPU,
ngspice-45.2 (`ngspice_con.exe`), OpenAI-compatible provider (gpt-4o-mini via
`us.api.openai.com`). The parent repository is still **not under git**
(`git status` → fatal), so the `feature/agentic-raptor-stage2` branch could not
be created; safety is again additive-only, verified by filesystem scan.

Status legend used throughout: **implemented** / **tested with mocks** /
**tested with real ngspice** / **tested with real multimodal API** / **not yet tested**.

## 1–3. Files

**New files (Stage 2):**
- `agentic_raptor/spice/`: `device_mapping.py`, `model_library.py`, `netlist_builder.py`,
  `testbench_builder.py`, `ngspice_simulator.py`
- `agentic_raptor/sizing/spice_query_policy.py`
- `agentic_raptor/topology_generation/provider.py`
- `agentic_raptor/specification/schematic.py`
- `agentic_raptor/adapters/legacy_netlist.py`
- `configs/experiments/`: `stage2_real_spice.yaml`, `stage2_multimodal.yaml`,
  `stage2_end_to_end.yaml`, `inputs/stage2_spec.yaml`
- `tests/`: `test_trajectory_fix.py`, `test_netlist_builder.py`, `test_ngspice_simulator.py`,
  `test_stage2_learning.py`, `test_provider.py`, `test_coordinator_stage2.py`
- `docs/STAGE2_IMPLEMENTATION_REPORT.md`

**Modified inside Agentic_Raptor only:** `coordinator/coordinator.py` (pre-action
trajectory fix, simulator/generator selection, budgeted sizing loop, error handling,
best-candidate restoration, BFS state routing), `coordinator/decision_policy.py`
(hard-sim-failure regeneration, mandatory initial MCTS pass),
`learning/update_manager.py` (`require_spice_for_training`),
`sizing/graph_conditioned_mb_sac.py` (dynamics ensemble + uncertainty),
`spice/result_parser.py` (real ngspice parsing + failure taxonomy),
`spice/cache.py` (enable flag; success-only caching), `spice/pvt.py` (aggregation),
`topology_generation/generator.py` (provider-backed LLM generator),
`topology_generation/prompt_builder.py` (connectivity requirements + reference topology),
`topology_validation/rules.py`+`validator.py` (new `DC_FLOATING_NET` rule),
`utils/config.py` (Stage 2 fields), `cli.py` (6 new commands), `pyproject.toml`
(markers), plus test updates.

**Legacy RAPTOR files modified: NONE** — verified by a same-day recursive scan of
all 14 legacy source directories (`LEGACY UNTOUCHED (scan clean)`). Legacy access
remains read-only via `adapters/` (`graph.export_graph_to_netlist` inspected and
wrapped; its passives-only reconstruction is documented in the adapter and test).

## 4. Stage 1 bookkeeping fixes — implemented, tested with mocks + real ngspice
- `TrajectoryStep.graph_state` now captured **before** `env.step` (regression test
  `test_stored_graph_state_is_pre_action` re-validates action legality and feature
  re-encoding from the stored state).
- `reached_spice` recorded on trajectory + steps + credit report; semantics: the
  final reward was grounded in a simulator evaluation (success **or** simulator-judged
  failure). Filtering is configurable (`policy_value.require_spice_for_training`,
  default off) and tested both ways.

## 5. Netlist conversion — implemented, tested with mocks AND real ngspice
All 11 device types map (subcircuits via `attributes['subckt_name']`/`pin_nets`);
stable names (`M_m1`…), D-G-S-B ordering, node sanitisation with ground→`0`,
width/length/multiplicity/passive/bias emission, explicit `; defaults` markers for
parameter-space defaults, candidate/topology-hash header comments. Parameter space
remains graph-derived (no fixed topology). Un-emittable structures raise structured
`SimulationError`s.

## 6. Legacy exporter adapter — implemented, tested (integration marker)
`adapters/legacy_netlist.py`: typed→legacy dict translation, sizing transfer,
unsupported-structure reporting, structured errors. **Verified finding:** the legacy
exporter reconstructs only passive devices without preserved raw device lines
(consistent with its own conservative-by-design docstring), so it serves as a
cross-check/interop path; the native builder is the simulation path.

## 7. Real SPICE backend — implemented, tested with real ngspice-45.2
`NgspiceSimulator`: discovery (config → PATH `ngspice_con` first → known paths),
version probe, per-run temp workdirs with deterministic names, `-b` batch subprocess,
timeout kill, stdout/stderr capture, return-code validation, raw log preservation,
cleanup policy, structured `simulator_unavailable` error when absent. **No silent
mock fallback** — `spice.allow_mock_fallback` (default false) is the only fallback
path and it logs; tested in `test_coordinator_stage2`.

## 8. Analyses
- **op + ac: implemented, tested with real ngspice.** Open-loop testbench (huge-L/huge-C
  feedback, inverting input inferred or overridable), `.meas` extraction of DC gain,
  UGF/GBW, phase at UGF (degrees with radians fallback), supply current → power,
  output DC. Operating conditions from the unified spec; vcm default = vdd/2 explicitly
  marked. **tran (slew): implemented, tested with mocks; not yet run on real ngspice.**
- **noise / Monte Carlo: interfaces exist and deliberately raise** — not claimed.
- Output swing: not extracted (no swing testbench yet) — not claimed.

## 9. Result parser — implemented, tested with mocks (fixtures) + real runs
`parse_ngspice_stdout` + `metrics_from_ngspice` + failure taxonomy: convergence,
singular matrix, timestep, missing model, malformed netlist, timeout, missing
measurement, numerical overflow, unavailable executable. Failure signatures only
classify runs that actually failed to produce required measurements (a recovered
gmin-stepping note is not fatal). Failed simulations preserve `error_type` — never
converted to a silent low reward.

## 10. Cache — implemented, tested
Key = topology hash + sizing + analyses + corner + simulator fingerprint (backend,
exe, **version**, **model-library label**) + temperature + supply. Seven-way key
sensitivity test. `cache_enabled: false` is a pass-through; failures are never cached.

## 11. PVT — implemented, tested with mocks (real path shares the simulator code)
Configurable corner count; aggregation: pass_rate, worst/mean margin, worst corner,
failed corners. SPICE is the only authority; the dynamics model is never consulted
for PVT.

## 12–14. Budgeted real-SPICE MB-SAC — implemented, tested with mocks + real ngspice
`RealSpiceQueryPolicy` (documented ordered rules: warm-up → budget → ensemble
uncertainty → periodic re-anchor → terminal verification). Dynamics **ensemble**
(default 2) supplies genuine disagreement-based uncertainty; every member trains on
real transitions only. Real transitions consume the SPICE budget and are labelled
`real`; model-predicted steps are labelled `model` and can never produce the
accepted candidate (best-vector selection is real-only; final acceptance goes
through RUN_SPICE; regeneration restores the best SPICE-verified candidate).
Cross-level path verified end-to-end on real hardware: real SPICE metrics → final
reward → `CrossLevelCreditAssigner` → topology buffer → policy/value gradient steps
(parameter-change checks in the CLI verification).

## 15–17. Multimodal provider — implemented, tested with mocks AND real API
Provider-neutral `MultimodalTopologyModel` protocol; one real implementation
(`OpenAICompatibleModel`: chat-completions protocol, stdlib-only, JSON response
format, base64 image parts, timeouts, bounded retries, token-usage logging,
credentials only from env vars — a test asserts the key never appears in logs).
Missing credential → structured `missing_credential` naming the exact env var;
CLI `check-dependencies` reports it. `ProviderSchematicParser` extracts context
(class, devices, blocks, ports, printed values, hints, confidence, **unresolved**
rather than invented); output is multimodal context, and image fields keep lowest
fusion priority (test: image can never override an explicit value).
Real-API note: the account required the regional host — set
`AGENTIC_RAPTOR_LLM_BASE_URL=https://us.api.openai.com/v1` (the env override worked
as designed). Real image parsing through the API: **implemented, not yet tested**
(no schematic image was exercised against the real API; mock-tested).

## 19–20. LLM topology output — implemented, tested with mocks AND real API
Strict JSON only; rejects prose/invalid JSON/unsupported devices/missing ports/
malformed terminals; acceptance bar is *emittable* (no partially-connected devices,
isolated subgraphs, unpowered outputs, or DC-floating nets); rejection feedback is
fed into bounded retries. Deterministic validation always precedes RL. RAG: bounded
top-k entries with metrics, failures + reasons, and the best success's structure as
a reference (adaptation, not copying); memory IDs + scores recorded in candidate
lineage. Provider temperature pinned to 0.0 for reproducibility.

## 21. Coordinator — implemented, tested
New handling, every decision still logged with a reason: `GENERATION_FAILED`
(provider/malformed output, retry-limit), `SPICE_FAILURE` (typed), no-silent-fallback,
`RESTORED_BEST_CANDIDATE`, hard-sim-failure → regenerate, mandatory initial MCTS
pass (topology RL now always precedes sizing — this also closes the Stage 1 audit's
execution-order deviation), BFS legal-path routing to UPDATE_MEMORY.

## 13. Test results
```
python -m pytest tests -q  →  154 passed, 1 skipped   (Stage 1's 100 all preserved)
python -m ruff check agentic_raptor tests scripts  →  All checks passed
```
Markers: `unit`, `integration`, `requires_ngspice` (ran — ngspice installed),
`requires_multimodal_api` (**1 skipped**: needs `AGENTIC_RAPTOR_RUN_API_TESTS=1`
opt-in so the default suite never spends API credit — the real API was instead
exercised via the smoke runs below).

## 14. Smoke results (all runs real, none skipped)
- **Smoke A (real ngspice)** — PASSED. 5T OTA → netlist → ngspice-45.2 → parsed
  metrics: gain 51.59 dB, GBW 205 MHz, PM 83.5°, power 80 µW + margins. CLI `run-spice`.
- **Smoke B (real multimodal API)** — PASSED. gpt-4o-mini: attempt 1 rejected by
  deterministic validation → error-feedback retry → structurally valid 10-node OTA
  (confidence 0.85) saved as JSON. CLI `generate-topology`.
- **Smoke C (end-to-end real LLM + real ngspice)** — PASSED (`run-stage2`,
  episode ep-87c6614869): multimodal spec → RAG → real LLM generation (validated,
  incl. the new `DC_FLOATING_NET` rule that caught a genuinely singular bias net and
  whose feedback fixed the next generation) → mandatory MCTS pass → budgeted real-
  SPICE sizing (2 real + 3 model transitions, imagined rollout) → real nominal SPICE
  (gain margin +125%, PM +44%; GBW missed → honestly not accepted) → real-grounded
  final reward 1.237 → cross-level credit → **all four learnable components verified
  changed**. An earlier run (ep-75bf9d385e, pre-refinement-pass) additionally reached
  full `ACCEPT_CANDIDATE` with PVT pass-rate 1.00 and reward 1.94 — demonstrating the
  acceptance path on real hardware.

## 15. Known limitations
1. Generic LEVEL=1 model library — technology-neutral benchmark physics, not a PDK
   (swap via `spice.model_library_path` + `technology_label`).
2. tran/slew testbench implemented but not yet exercised on real ngspice; output
   swing not measured; noise/Monte Carlo are interfaces only.
3. LLM circuit quality varies; temperature 0 gives reproducibility, and validation
   feedback measurably improves outcomes, but acceptance is not guaranteed per episode.
4. Real image → real API schematic parsing untested (mock-tested only).
5. Single episode exercised end-to-end; multi-episode campaigns remain stage 3.
6. Not a git repository — no branch isolation (additive-only + scan instead).
7. `black`/`mypy` still not installed; ruff is the enforced gate.

## 16. Reproduction commands
```bash
cd Agentic_Raptor
python -m pytest tests -q
python -m ruff check agentic_raptor tests scripts
python -m agentic_raptor.cli check-dependencies
# Smoke A (needs ngspice):
python -m agentic_raptor.cli run-spice --graph outputs/smoke_a_graph.json --config configs/experiments/stage2_real_spice.yaml
# Smoke B (needs OPENAI_API_KEY; region-locked accounts: set AGENTIC_RAPTOR_LLM_BASE_URL=https://us.api.openai.com/v1):
python -m agentic_raptor.cli generate-topology --config configs/experiments/stage2_multimodal.yaml
# Smoke C (needs both):
python -m agentic_raptor.cli run-stage2 --config configs/experiments/stage2_end_to_end.yaml
# Mock mode (no dependencies), unchanged:
python -m agentic_raptor.cli smoke-test --config configs/experiments/smoke_test.yaml
# Opt-in real-API unit test:
set AGENTIC_RAPTOR_RUN_API_TESTS=1 && python -m pytest tests -m requires_multimodal_api
```
