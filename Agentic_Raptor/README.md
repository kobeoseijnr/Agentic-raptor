# Agentic RAPTOR

An agentic analog circuit design framework extending the existing RAPTOR research
codebase — without modifying it. Multimodal specification parsing feeds an
engineering experience memory, a multimodal topology generator, AlphaZero-inspired
topology reinforcement learning, graph-conditioned model-based SAC sizing, SPICE/PVT
evaluation, and cross-level credit assignment, coordinated by a rule-based agentic
controller.

## Central hypothesis

> A topology should be evaluated according to the circuit performance it can achieve
> **after continuous sizing and SPICE verification**, rather than only according to
> its unsized structural features.

The topology policy and value networks therefore learn from the **final post-sizing
SPICE return**, propagated onto every topology decision by trajectory-level credit
assignment (`learning/cross_level_credit.py`).

## Architecture

```text
Multimodal Design Input (text / YAML / JSON / CSV table / schematic image / netlist)
        ↓  specification/          — parse per modality, fuse, detect conflicts
Unified DesignSpecifications (canonical, provenance-tagged)
        ↓  rag/                    — structured engineering experience memory
Retrieved successes, failures, reusable blocks
        ↓  topology_generation/    — mock or LLM generator → structured JSON graphs
Candidate CircuitGraph
        ↓  topology_validation/    — deterministic rules; invalid graphs never reach RL
        ↓  topology_rl/            — policy net + value net + MCTS planning over edits
Refined topology
        ↓  sizing/                 — GraphConditionedMBSAC (genuine SAC + dynamics model)
Sized candidate
        ↓  spice/                  — mock or (stage-2) ngspice; PVT corners; caching
        ↓  learning/               — YAML-weighted final reward; cross-level credit
        ↓  rag/                    — memory updated with the full outcome
        ↓  coordinator/            — regenerate / retrieve / edit / resize / stop
```

## Three levels, three different jobs

1. **Topology generation** proposes complete structural candidates (graphs), guided
   by specifications and retrieved experience.
2. **Topology editing** refines a candidate with discrete, typed, validated graph
   edits (add/remove devices, bias branches, compensation, gain stages...). It never
   touches continuous sizes.
3. **Sizing** optimizes continuous device parameters (W/L/multiplicity/I/R/C) for a
   *fixed* topology. The parameter space is generated dynamically from the graph.

## Why MCTS is planning, not RL by itself

MCTS (`topology_rl/mcts.py`) is a *search* procedure: given the current networks, it
expands an edit tree using PUCT `Q(s,a) + c_puct·P(s,a)·√N(s)/(1+N(s,a))`, with
legal-action masking, progressive widening, and value-network leaf evaluation. It
learns nothing by itself. The *reinforcement-learning* components are the policy and
value networks (`policy_value_network.py`), trained by real gradient steps
(`trainer.py`): the policy on MCTS root visit-count distributions
(`L_policy = −Σ π_MCTS log π_θ`), the value on final post-sizing returns
(`L_value = (V_φ − z)²`). This mirrors the AlphaZero pattern: search improves the
networks' targets; the networks improve the search's priors and evaluations.

## What "graph-conditioned MB-SAC" means

`sizing/graph_conditioned_mb_sac.py` is a genuine Soft Actor-Critic — reparameterised
tanh-squashed Gaussian actor with log-prob correction, twin + target critics,
entropy-regularised Bellman targets, automatic temperature, Polyak updates — whose
**state is conditioned on the topology**: `[topology embedding | sizing vector |
spec embedding | metrics | constraint margins | SPICE-budget fraction]`, and whose
**action space is derived from the topology** (one bounded, normalized dimension per
sizable device parameter). *Model-based*: a jointly trained dynamics model predicts
next state, reward, and termination; short imagined rollouts add `source="model"`
transitions mixed with real ones at a configurable ratio. SPICE remains the final
authority for accepted results.

## Cross-level SPICE reward mechanism

After sizing + SPICE (+ PVT), a YAML-configured reward is computed
(`learning/reward_assignment.py`):

```python
final_reward = (feasibility_weight * feasibility_score
                + fom_weight * normalized_fom
                + pvt_weight * pvt_score
                - spice_cost_weight * normalized_spice_calls
                - runtime_weight * normalized_runtime
                - invalidity_weight * invalidity_penalty)   # + validity/progress terms
```

`CrossLevelCreditAssigner` writes `z_t = γ^(T−1−t) · final_reward` onto every
topology decision, and the value network trains against those `z` targets. SAC
gradients are **not** backpropagated through discrete topology actions — coupling is
purely through this trajectory-level scalar credit.

## Setup

```bash
cd Agentic_Raptor
python -m pip install -r requirements.txt      # numpy, networkx, pyyaml, torch, pytest, ruff
# optional editable install:
python -m pip install -e .
```

Python 3.11+ required (developed and tested on 3.14 / torch CPU / Windows).

## Running

From the `Agentic_Raptor` directory:

```bash
python -m agentic_raptor.cli inspect-repository
python -m agentic_raptor.cli parse-specification --config configs/experiments/smoke_test.yaml
python -m agentic_raptor.cli validate --graph path/to/graph.json
python -m agentic_raptor.cli smoke-test --config configs/experiments/smoke_test.yaml
python -m agentic_raptor.cli run-episode --config configs/default.yaml

python -m pytest tests -q          # 100 tests
python -m ruff check agentic_raptor tests scripts
```

The smoke test needs **no external APIs and no SPICE installation**; it verifies that
the policy/value network, SAC actor, SAC critics, and dynamics model all actually
changed parameters during the episode, and fails otherwise.

## Current implementation status

Functional: multimodal parsing/fusion (text, YAML/JSON, CSV, netlist; mock image
parser), structured RAG memory with weighted retrieval + outcome write-back, mock
topology generator (validator-clean 5T OTA) + provider-neutral LLM shell, 14-rule
deterministic validator with WL structural hashing, 13 typed reversible edit actions,
Gymnasium-style edit environment, PUCT MCTS with progressive widening, genuine
policy/value training, genuine graph-conditioned MB-SAC with dynamics model and
imagined rollouts, deterministic mock SPICE + PVT + caching, YAML reward +
cross-level credit, rule-based coordinator with fully logged decisions, CLI, 100
passing tests.

## Known limitations

- The **mock simulator** is physics-flavoured, not physics: absolute metric values
  are only meaningful for exercising the learning machinery.
- **LLM generation** and **schematic-image parsing** are interfaces + mocks; provider
  transports are stage 2 (env-var config points are in place).
- **Legacy ngspice integration** is an explicit seam (`LegacyNgspiceSimulator`)
  pending typed-graph → legacy-netlist export.
- Single-episode learning is exercised, not convergence; no training campaigns yet.
- The parent repository is **not a git repository** — the requested feature branch
  could not be created (documented in `docs/REPOSITORY_AUDIT.md`).
- `black`/`mypy` were not available in the environment; `ruff` (lint + import
  sorting) passes clean.

## Next development stages

1. Wire `LegacyNgspiceSimulator` through `graph/export_graph_to_netlist.py` +
   `controller.module_adapters.run_spice_validation` (real SPICE in the loop).
2. Multi-episode training campaigns: retrieval-seeded generation → search → credit →
   improvement curves vs. the legacy checkpoint-5d offline pipeline.
3. Real LLM transports (reusing the legacy `llm/llm_client.py` provider pattern) and
   a VLM schematic parser behind `SchematicImageParser`.
4. Import legacy RAG memory (`rag/adapters.py`) to warm-start the experience store.
5. PyTorch Geometric encoder behind the existing `encoder:` config switch.

See `docs/` for the architecture, audit, decisions, novelty statement, and plan.
