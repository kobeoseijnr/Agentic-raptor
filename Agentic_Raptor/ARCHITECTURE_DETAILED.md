# Agentic RAPTOR — The Complete Architecture, Explained Simply

*Updated 2026-07-26. Companion to `ARCHITECTURE.md` (the compact diagram). This
document explains what every layer does, how it was actually implemented, and
why it was built that way.*

---

## 1. What is Agentic RAPTOR?

Agentic RAPTOR is an AI system that **designs analog circuits** (op-amps) the
way a team of engineers would:

1. Someone reads the requirements ("I need 60 dB of gain, stable, on 1.8 V").
2. Someone looks up similar past designs.
3. Someone sketches a circuit topology (which transistors, connected how).
4. Someone checks the sketch is legal and buildable.
5. Someone explores variations ("what if we add a second stage?").
6. Someone picks transistor sizes.
7. **The circuit simulator has the final word** — it either works or it doesn't.
8. Everything learned goes back into the team's shared knowledge.

Every one of those roles is a separate, verifiable component. The one
non-negotiable rule everywhere: **ngspice (a real circuit simulator) is the
only source of truth.** No neural network's prediction is ever recorded as a
measurement.

---

## 2. The Big Picture

```
                 ┌── Specifications (text) ─────────────┐
                 ├── RAG memory (past designs+outcomes) ─┤
                 ├── FunctionalStageGraph (serialized) ──┼─→ MULTIMODAL
                 ├── DeviceCircuitGraph (serialized) ────┤   TOPOLOGY LLM ──┐
                 └── Schematic image (real PIXELS →      │  (Qwen3-VL,     │
                     vision encoder, never captions) ────┘   SFT + DPO)    │
                                                                           │
                                    Registry roots ────────────────────────┤
                                                                           ▼
      Validators (parser → canonicaliser → topology → mapping)
                                                                           │
                                                                           ▼
      AlphaZero/MCTS search → Transistor realisation → MB-SAC sizing
      → BT candidate ranking → REAL NGSPICE → PostSizingTopologyScore
                                                     │
        ┌────────────────────────────────────────────┘
        ▼  (the score is the universal training signal)
      • MCTS value/policy targets        • MB-SAC replay + dynamics
      • surrogate training data          • BT-ranker preference pairs
      • L4 RAG simulation memory
      • LLM PREFERENCE PAIRS → offline DPO → the NEXT LLM adapter
        (SPICE-backed pairs are the gold tier — this is how the
         simulator's verdict fine-tunes the language model itself)
```

Think of it as a funnel with a return pipe: many ideas enter at the top, each
safety gate filters out the broken ones, only verified circuits come out the
bottom — and the simulator's verdict flows back to retrain *every* learner,
including the LLM that proposed the topology in the first place.

---

## 3. Layer by Layer

### 3.1 The Topology Corpus (the "library of known circuits")

**What it is:** 174 verified op-amp families in
`artifacts/topology_registry_operational_v3/` — 17 from AnalogGym (real
literature circuits with netlists), 15 unique families from OPAMP-Generator,
and 142 corrected families extracted from the CktGNN dataset.

**How it was built:** each circuit is stored as a `CircuitGraph`
(`agentic_raptor/core/circuit_graph.py`) — nodes are devices (NMOS, PMOS,
capacitor…), edges are terminal-to-net connections. Every graph gets a
**role-aware structural hash**: a fingerprint computed from device types,
connectivity, *and each device's functional role* (input pair, mirror, tail…).
Two circuits with the same fingerprint are the same topology.

**Hard lessons baked in:** two data defects were found by auditing and fixed
with full lineage: (1) the original CktGNN decoder used a wrong `t % 8`
formula — replaced with the verified 26-entry decoder, corpus re-extracted;
(2) the first hash ignored roles, silently merging different circuits — fixed,
and every legacy hash was preserved alongside the new one so nothing was
overwritten. Electrical results: 94 families simulated, 16 verified stable
(the "A1/A2" pools used for training), 91 flagged repair-eligible.

### 3.2 Hierarchical RAG (the "team's memory")

**What it is:** retrieval at four levels — L1 whole-corpus, L2 topology
family, L3 functional block, L4 simulation memory (every real SPICE result
ever recorded, in `datasets/simulation_memory/*.jsonl`).

**How it works:** when a new design request arrives, the system retrieves
similar families, their blocks, and — crucially — *past measured outcomes*,
including failures. Retrieval is disabled during held-out evaluation so the
model can't peek at the answers (leakage tests enforce this).

### 3.3 Target Datasets (the "exam papers")

`datasets/target_sets_v1/`: 80 versioned target records = 16 stable families ×
5 difficulty tiers (easy / boundary / hard / validation / held-out). Each
target is anchored to a *measured* baseline, so "hard" means hard relative to
what the real circuit actually does. Held-out targets are firewalled from
every trainer — replay buffers, ranker pairs, RAG, LLM datasets — and tests
prove it.

### 3.4 The Topology LLM (the "junior designer who proposes ideas")

**What it does:** turns a specification + RAG context into a **structured
TopologyProposal** — a strict JSON describing stages, blocks, ports, bias,
compensation, feedback. Never a netlist: free-text rationale is legal but is
*never* parsed as circuit connectivity, and generated netlist text is *never*
executed (both tested).

**How it was implemented** (`agentic_raptor/llm_dpo/`):
- **SFT (supervised fine-tuning):** a LoRA adapter (small trainable add-on,
  ~0.8 M params on an otherwise frozen model) learns "spec in → canonical
  proposal JSON out" from the verified corpus. Datasets are split-safe with
  graph-hash dedup and a held-out *structure* split (3-stage designs never
  seen in training).
- **True DPO (Direct Preference Optimization):** the genuine article — a
  trainable policy and a *frozen* reference copy of the same model; the loss
  `−w·logsigmoid(β·((logπ_pol(y⁺)−logπ_ref(y⁺))−(logπ_pol(y⁻)−logπ_ref(y⁻))))`
  operates on **response-token log-probabilities** (prompt and padding
  masked). Preference pairs always share one context; evidence is tiered
  (parseable > valid > mappable > stable > feasible > margins > FoM > cost),
  and real-SPICE-backed pairs are the gold tier. Tests prove the policy's
  parameters change and the reference stays bitwise frozen.
- **How SPICE results fine-tune the LLM (the closed loop):** measured
  qualification records are mechanically ingested by `scores_to_pairs()`
  (28 pairs from 8 real SPICE-backed sources currently queued in
  `llm_preference_queue/v2_mechanical.jsonl`); training the next adapter from
  this queue (`dpo_v2`) is one command away but has not yet been executed. Two proposals
  generated under the *same context* are compared with the hard-gated
  hierarchy — a proposal whose realised circuit measured *stable* on ngspice
  beats one that measured *unstable*, regardless of textual quality — and the
  winner/loser pair (with the score attached as provenance) is queued in
  `datasets/llm_preference_queue/`. Offline DPO then trains the next adapter
  version on these pairs, raising the log-probability of proposals that
  *physically worked* and lowering it for ones that failed. So the simulator
  literally teaches the language model, one verified comparison at a time.
- **Multimodal upgrade** (`llm_dpo/multimodal.py`, Stage 3E.4B): a
  vision-language model (**Qwen3-VL-4B primary**, running on the RTX 5070 Ti)
  receives *actual schematic image pixels* through its native vision
  encoder — never captions or OCR —
  jointly with the spec text and both graph serializations
  (`MultimodalTopologyContext`). Schematic images are rendered
  deterministically from device graphs into `datasets/schematic_images_v2/`
  with image↔graph↔proposal alignment. Proof instrumentation captures pixel
  tensor shapes, vision-encoder embedding shapes, and shows that ablating or
  swapping the image *changes the output logits*.
- **Results so far (honest):** SFT lifts structured validity from 0% (base
  model) to 100% at pilot scale; DPO perfectly learns the preference signal
  (held-out accuracy 1.0 across 3 seeds) but has not been shown to improve
  proposal quality; small models memorise rather than diversify — the
  GPU-scale campaign with a 3B model is the open next step.

### 3.5 The Validation Chain (the "gatekeepers")

Every proposal — from the LLM, the registry, or an edit — passes: **parser →
canonicaliser → topology validator → mapping validator.** The validator
(`TopologyValidationResult`) checks structural sanity, bias/supply
completeness, mapping supportability, and ancestor-cycle prevention, and
returns *explicit rejection reasons*. A rejected topology never costs a SPICE
call, and validator rejections are counted separately from simulator failures.

### 3.6 The Structural Edit Library (the "hands that modify circuits")

**What it is** (`topology_rl/stage3e2_edits.py`): eight versioned, typed edit
templates that *physically transform* a `DeviceCircuitGraph`: add a gain
stage, replace a stage (NMOS↔PMOS variants), replace a load, add/replace
Miller or RC-nulling compensation, add a source-follower output buffer,
connect local resistive feedback, and remove an added stage (the reversal).

**How each edit works:** check preconditions (rejects with a named reason) →
deep-copy the parent (immutability tested) → rewire nets and add/remove
devices deterministically → recompute the device-graph hash and the sizing
manifest (so the RL action space resizes correctly) → emit an audit record
with parent/child hashes and lineage. Add-then-remove provably recovers the
parent's hash. Every executed edit produced a real Sky130 netlist and ran on
real ngspice — including honest failures (a 3-stage edit measured PM −27°,
recorded as *verified unstable*, not hidden).

### 3.7 AlphaZero Topology Search (the "explorer")

**What it is** (`topology_rl/stage3e1.py` + `stage3e2.py`): a policy network
and value network sharing one message-passing graph encoder, driving PUCT
MCTS — the AlphaZero recipe applied to circuit topologies.

**How it works, simply:** the search tree's nodes are topologies; actions are
"keep this one", "switch to that registry family", "apply this executable
edit", or "stop". The **policy** suggests which actions look promising
(priors); the **value net** guesses how good a topology will be; **PUCT**
balances trying good-looking moves vs. unexplored ones
(`Q + 1.5·P·√N/(1+n)`); leaves are scored either cheaply (value net) or
expensively — a **real MB-SAC + SPICE evaluation** — under strict per-search
SPICE budgets with caching (a cache hit is never counted as a new call). After
search, the visit counts become the policy's training target and the
SPICE-backed outcome becomes the value target — that closed loop is what makes
it AlphaZero rather than plain tree search. Root Dirichlet noise adds
exploration during training only; resume is deterministic.

**Honest result:** MCTS correctly *rejected* a harmful edit (the destabilising
stage addition) by concentrating visits on KEEP — and no claim is made that
the still-barely-trained policy beats random search yet.

### 3.8 Transistor Realisation (the "draftsman")

`agentic_raptor/mapping/` converts an abstract topology into a real Sky130
netlist: an N-stage template (5-transistor first stage + common-source stages)
with AnalogGym-derived sizing priors, deterministic polarity bookkeeping
(structural, never outcome-based), static validation (no floating ports, no
supply shorts), then a `.subckt` netlist the simulator accepts. Edited and
LLM-proposed graphs use the same emitter, so everything downstream is
identical no matter where a topology came from.

### 3.9 MB-SAC Sizing (the "tuner")

**What it is:** a genuine Soft Actor-Critic agent, graph-conditioned and
model-based, that picks continuous transistor sizes (W/L/M within legal
bounds, log-scaled, matching-group aware).

**How it was implemented:** squashed-Gaussian actor (with the tanh log-prob
correction), twin critics with Polyak targets, automatic entropy temperature,
a 2-member probabilistic dynamics ensemble trained only on real transitions,
and uncertainty-gated model rollouts kept in a separate replay buffer from
real data. The **MP graph encoder sits inside the actor and critic losses** —
Phase-D training proved nonzero gradients flow from both objectives into the
encoder across 3 seeds (96 real SPICE transitions, stable-rate 0.979 ± 0.029).
Dynamics calibration gates rollouts per source: A2 passed, A1 failed →
rollouts disabled there, real-only learning continues (and that's logged, not
hidden).

### 3.10 Candidate Ranking — Two Separate Preference Systems

This distinction matters and is enforced by tests:

| | **True LLM DPO** | **Bradley–Terry Sizing Ranker** (`agentic_raptor/dpo/`) |
|---|---|---|
| Trains | the topology *language model* | a small ranker over *sizing candidates* |
| Operates on | token log-probabilities vs a frozen reference | numeric circuit features + real outcomes |
| Guarantee | reference stays bitwise frozen | **an unstable high-FoM candidate can never outrank a stable feasible one** (lexicographic hard gates, tested) |
| Shared data | none | none |

The sizing ranker (historically named `DPORanker` — kept for compatibility,
formally a *Feasibility-Gated Bradley–Terry Pairwise Preference Ranker*) sits
between MB-SAC's candidates and the final SPICE runs, always preserving one
exploration pick so the system never tunnel-visions.

### 3.11 Real SPICE — The Final Authority

**Setup:** ngspice-45.2 on Windows, sky130 PDK (tt/ss/ff/fs/sf corners), an
open-loop differential-mode testbench (1 T-Henry feedback inductor trick,
VCM = 0.25·VDD, 500 pF load), complex AC output captured via `wrdata`.

**Measurement hardening (hard-won):** the AC output is loaded as a *complex*
transfer function (an early bug dropped the imaginary part and corrupted every
phase margin); phase is unwrapped continuously; unity crossings are
log-interpolated; every metric carries a status (`verified` / `estimated` /
`ambiguous` / `measurement_failed`) and **only verified values are reported**.
Stability polarity is adjudicated with three preserved excitation artifacts
per circuit and a structural evidence chain — never chosen because it produces
a nicer number. PVT verification (corner × temperature × supply matrix) runs
separately and its SPICE calls are never mixed into nominal accounting.

### 3.12 PostSizingTopologyScore (the "report card")

One structured record per evaluation: status taxonomy (successful /
infeasible / unstable / simulator_failure / budget_exhausted / unsupported),
verified-stable and feasible flags, metric and margin vectors, exact SPICE
call counts, and a bounded ordinal scalar
(`0.2·valid + 0.4·stable + 0.4·feasible + 0.2·margin − 0.1·cost`) used for
MCTS backup and value targets. It is explicitly **ordinal and uncalibrated** —
never treated as a probability.

### 3.13 Cross-Level Feedback (the "learning loop")

Each real SPICE outcome fans out — with dedup keys so no artifact is counted
twice — to: AlphaZero policy/value targets, MB-SAC replay, dynamics-ensemble
and surrogate training data, Bradley–Terry ranker pairs, L4 simulation memory
for RAG, and a **versioned offline queue** of LLM preference evidence
(`datasets/llm_preference_queue/`). The deployed LLM adapter is frozen during
held-out evaluation; new evidence only trains the *next* adapter version.

### 3.14 Reproducibility & Honesty Infrastructure

- **Snapshots:** every stage begins by freezing the code and key artifacts
  with a SHA-256 manifest that is *re-read and re-verified* (e.g.
  `pre_stage3e4b_multimodal`: 208 files, 0 mismatches).
- **Tests:** 380+ passing, including "honesty tests" — no fabricated
  measurements, no unchecked netlist execution, leakage prevention, frozen
  references, exact call accounting.
- **Accounting:** validator calls, mapping attempts, value-net calls, MB-SAC
  transitions, model rollouts, cache hits, nominal SPICE, PVT SPICE — all
  counted separately; a model rollout or cache hit is never a SPICE call.
- **Reports:** 100+ per-stage markdown reports under `reports/`, each stating
  deferrals with their exact blocker (usually compute) and the exact command
  to lift it.

---

## 4. One Request, End to End

> "Design me an op-amp: 60 dB gain, 45° phase margin, 1.8 V supply, 500 pF load."

1. The spec becomes a target record; RAG retrieves similar families and their
   measured history.
2. The topology LLM (or the registry) proposes a structured 2-stage proposal
   with Miller compensation — pixels of its schematic included if the
   multimodal model is active.
3. Parser → canonicaliser → validators approve it; it becomes an MCTS root.
4. MCTS explores: keep it, swap to a sibling family, or apply a verified edit
   (add stage? RC nulling?). Expensive leaves get real bounded evaluations.
5. The winning topology is emitted as a Sky130 netlist.
6. MB-SAC proposes sizings; the surrogate screens; the BT ranker orders them
   (stability always beats raw FoM); one exploration pick survives.
7. ngspice measures the finalists. Verified numbers only.
8. The structured score goes back up the stack: MCTS backup, value targets,
   replay, ranker pairs, RAG memory, and the LLM's next preference dataset.
9. If the design passes nominal, the PVT matrix gets the last word.

---

## 5. Current Status at a Glance

| Layer | Status |
|---|---|
| Corpus, RAG, targets, validators, edits, realisation, SPICE, PVT, accounting | **Verified on real hardware** |
| AlphaZero mechanics, MB-SAC Phase-D, ranker, score, cross-level feedback | **Verified at engineering scale** (3 seeds, bounded budgets) |
| Text LLM SFT + true DPO | **Pilot-proven**; GPU 3B campaign now unblocked (RTX 5070 Ti live) |
| Multimodal (Qwen3-VL-4B) | **In progress** — vision-encoder evidence instrumented, first GPU run underway |
| Publication-scale statistics, 91-family repair, score calibration | **Deferred** — blockers and exact commands recorded |
