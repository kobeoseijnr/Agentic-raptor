# RAPTOR — Architecture (current)

Written 2026-09-03 against the live code.

**Read this instead of `ARCHITECTURE_DETAILED.md`.** That file is dated
2026-07-26 and is stale in one important way: it presents AlphaZero/PUCT MCTS as
the topology selector (§3.7). The AlphaZero/MCTS experiment layer was **deleted
on 2026-08-27** and the live selector is a linear contextual bandit
(`run_raptor_v2.py:971-982`). Anything in that document about MCTS driving
topology choice no longer describes the system.

---

## 1. What the system does

Given a target amplifier specification, RAPTOR **invents candidate
circuit topologies, chooses among them, sizes their components, and verifies the
result in SPICE.** This is the key difference from legacy RAPTOR, which sizes a
single fixed topology.

Five learned or learning components cooperate, each making one decision and
handing off to the next:

| # | Component | Decision it makes |
|---|---|---|
| 1 | Hierarchical RAG | which prior experience conditions the prompt |
| 2 | Topology LLM (SFT) | what circuit structures to propose |
| 3 | Contextual bandit | which 2 proposals are worth simulating |
| 4 | Soft actor-critic | what component values each topology gets |
| 5 | DPO preference ranker | which sized branch is returned |

SPICE is the final authority throughout: every pass/fail and every reported
metric comes from a real ngspice run, never from a learned model.

---

## 2. Pipeline

Entry point `run_pipeline()` — `run_raptor_v2.py:635`.

| Stage | Line | Purpose |
|---|---|---|
| 1 — Specification | `:741` | resolve the target; **one canonical C_load for the whole run** |
| 2 — RAG | `:797` | retrieve prior records (`rag_stage`, `:71`; `retrieve`, `:82`) |
| 3+4 — Propose & validate | `:803` | LLM emits 5 topology graphs; validator canonicalizes + dedupes (`propose_and_validate`, `:245`) |
| 5 — Topology selection | `:906` | contextual bandit keeps the top 2 |
| 6+7 — Sizing | `:1137` | each topology sized independently by SAC (`size_and_predict`, `:369`; `_size_one_branch`, `:421`) |
| 8 — Ranking | `:1241` | hard safety gate, then learned A/B choice (`compare`, `:1308`) |
| 9+10 — Verification | `:1356` | authoritative re-measurement; fall back to the backup branch on failure |
| 11 — Training data | `:1582` | ranker pairs recorded, trusted **only** if both branches were measured |

---

## 3. Components in detail

### 3.1 Hierarchical RAG — Stage 2

Retrieves prior design records and injects them into the LLM prompt.
`rag_stage()` at `run_raptor_v2.py:71`, `retrieve()` at `:82`.
Ablation `A1` replaces LLM proposal with pure library retrieval
(`retrieve_topology_candidates`, `:166`).

### 3.2 Topology LLM — Stages 3+4

A supervised-fine-tuned model proposes **5 distinct topology graphs** per
specification. A validation chain canonicalizes each graph, rejects malformed
ones, and deduplicates structurally identical proposals
(`propose_and_validate`, `run_raptor_v2.py:245`).

This is where circuit *structure* is invented — the capability legacy RAPTOR
does not have.

### 3.3 Contextual bandit selector — Stage 5

`agentic_raptor/topology_rl/bandit_selector.py` (198 lines).

A **linear contextual bandit** scores the validated proposals on 24 physical
features (`linear_value.physical_features_core`) and promotes the top 2 to
sizing. It shares its feature implementation with the frozen value probe so
offline training data and live scoring cannot drift.

Why it replaced the previous rule, from the module docstring: spec-disjoint on
held-out contexts it ranked measured topology outcomes at **12/12 pairwise
accuracy** against **1/12** for the frozen deterministic rule, which assigned
identical scores to same-family topologies and so could not rank within a family
at all — exactly where the measured regret concentrated.

Modes: `bandit_top2` (live default — no AlphaZero runs at all) and
`bandit_top2_az` (hybrid, AlphaZero as candidate generator only).

### 3.4 Soft actor-critic sizing — Stages 6+7

`agentic_raptor/mb_sac/spec_sizing.py` (1308 lines), `sac_size()` at `:675`.

Genuine SAC:

| Element | Where |
|---|---|
| actor `Linear(8,48) → ReLU → Linear(48,14)` | `:722` |
| twin critics `Linear(15,48) → ReLU → Linear(48,1)` | `:724-727` |
| Polyak target critics, frozen | `:738-740` |
| Adam lr `3e-3` (actor / critic / alpha) | `:741-747` |
| auto-tuned temperature, `target_entropy = -N_KNOBS` | `:748` |
| Bellman target, `gamma=0.99`, `tau=0.005` | `:687-688`, `:962-995` |

- **State** (8-dim, `_obs` at `:814`): gain target, PM target, load cap,
  remaining budget fraction, running best-closeness, and three per-constraint
  margins. Includes the target spec — the policy is spec-conditioned.
- **Action** (7 knobs, `:81-143`): multipliers on nominal values —
  `s1_w, s2_w, s1_l, s2_l, cap_x, ib_x, rz_x`. Absolute proposals, not deltas.
- **Reward** (`spec_reward`, `:359`): feasibility-conditioned with diminishing
  returns; `tanh` ramp while short of spec, small saturating credit past it,
  `+1.0` when all three constraints pass together.
- **Training**: one real ngspice call per step, then up to 6 gradient updates
  sampled from the episode's own transitions — a ~6:1 update-to-data ratio.
- **Persistence**: the canonical pipeline runs `persist=False`, so each sizing
  call cold-starts and discards its weights (`:1119-1120`).

> **Naming caution.** The package is called `mb_sac`, but this engine is
> **model-free**. There is no dynamics model and no imagined rollouts — grep for
> `imagine` / `rollout` / `source="model"` returns nothing, and every critic
> update uses a real ngspice measurement. `dynamics_surrogate_data.jsonl` is used
> for offline warm-start (advantage-weighted cloning toward historically good
> actions, `:634-638`), not for predicting next states. Do not describe this as
> model-based.

### 3.5 DPO preference ranker — Stage 8

A learned Bradley-Terry ranker chooses between the two sized branches using
measured evidence (`exact_spec_pass`, `best_distance`, FoM) plus its own score
(`compare`, `run_raptor_v2.py:1308`).

The code lives in **two** packages, which is easy to miss:

| Package | File | Lines | Role |
|---|---|---|---|
| `dpo/` | `ranker.py` | 137 | Bradley-Terry model |
| | `preference_pairs.py` | 171 | pair construction |
| | `selector.py` | 85 | branch A/B selection |
| | `schemas.py` | 172 | record types |
| `ranking/` | `post_sac.py` | 469 | post-sizing ranker |
| | `pair_mining.py` | 395 | preference-pair mining |
| | `features_v2.py` | 235 | **the promoted 46-feature set** |
| | `model_v2.py` | 141 | **promoted V2 model** |
| | `model.py` | 179 | superseded V1 |
| | `surrogate.py` | 185 | performance predictor |
| | `types.py` | 293 | shared types |

Version history matters (`run_raptor_v2.py:686-699`): the original 11-feature
ranker was rejected; the richer **46-feature V2** (`ranking/features_v2.py`,
`ranking/model_v2.py`) was re-justified in Stage 7.2B and is the promoted one.
Resolution never falls back to the old `ranker.pt` — `ranking/model.py` is V1 and
is not live.

A hard safety gate runs before the ranker (`:1241`), so a learned preference can
never promote an unsafe design.

### 3.6 SPICE — final authority

| File | Lines | Role |
|---|---|---|
| `electrical/__init__.py` | 440 | netlist emission, ngspice invocation, load handling |
| `electrical/measurements.py` | 222 | gain / UGBW / phase-margin extraction |
| `electrical/pvt_eval.py` | 244 | PVT corner evaluation |
| `electrical/adjudication.py` | 183 | pass/fail adjudication |
| `electrical/pm_qualified.py` | 93 | phase-margin qualification |
| `electrical/fom.py` | 75 | FoM computation |

Stages 9+10 re-measure the selected design authoritatively and fall back to the
backup branch if it fails.

---

## 4. What is NOT the architecture

**`agentic_raptor/sizing/graph_conditioned_mb_sac.py`** is a genuinely
model-based SAC — dynamics ensemble (`dynamics_ensemble_size=2`), imagined
rollouts (`generate_imagined_transitions`), replay with real/model separation
(`real_batch_fraction=0.8`), graph-conditioned state. It is **not used**: dated
2026-07-25, driven only by a 4-step smoke (`mb_sac/__init__.py:105,162-222`), and
imported by none of `run_raptor_v2.py`, `topology_rl/alphazero.py`, or
`publication/sizing_baselines.py`. It produced none of the reported results.

**AlphaZero / PUCT MCTS topology search** — deleted 2026-08-27. Retired modes
now raise rather than silently degrade (`run_raptor_v2.py:971-982`).

---

## 5. Honest notes for a write-up

**SAC's isolated contribution is small.** `artifacts/publication_v3/stage6_1_repair_recheck/STAGE6_1_RECHECK_REPORT.json`
compares `mbsac_fixed` against `nosac` (`tpe_lite`, a non-RL "sample near the top
quantile of history" baseline) at equal budget, 4 problems x 2 seeds. SAC wins
clearly in **1 of 8**; in several passing cases the non-RL baseline attains a
*higher* FoM; in 2 of 8 both fail. The system's performance is not attributable
to SAC alone — topology proposal and selection plausibly carry at least as much.

**Not every step inside a "SAC episode" is actor-driven.** Step 0 is a scripted
nominal anchor; the final third switches to scripted log-normal perturbation
around the best anchor (`spec_sizing.py:851-916`). In one recorded 16-step run,
1 nominal-anchor + 5 exploitation-tail steps were scripted, leaving ~10 genuine
actor/ranker-selected steps.

**Electrical-environment versioning.** Artifacts are split PRE/POST
`C_LOAD_FIX_V1`; pre-fix data ran at a fixed 500 pF regardless of spec and is
frozen as historical evidence, not used (`spec_sizing.py:49-57`).
