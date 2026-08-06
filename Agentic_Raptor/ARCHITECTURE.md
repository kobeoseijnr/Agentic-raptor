# Agentic RAPTOR — System Architecture

Updated: 2026-07-26 (through Stage 3E.4A). Status labels: **[V]** verified on real
hardware, **[P]** pilot-scale mechanics proven, **[D]** deferred (blocker recorded).

## End-to-end pipeline

```
                         ┌────────────────────────────────────┐
                         │       Design Specifications        │
                         │  gain / UGBW / PM / power / load   │
                         │  datasets/target_sets_v1  [V]      │
                         └──────────────────┬─────────────────┘
                                            │
                         ┌──────────────────▼─────────────────┐
                         │        Hierarchical RAG  [V]       │
                         │  L1 corpus → L2 family → L3 block  │
                         │  → L4 simulation memory            │
                         │  agentic_raptor/corpus             │
                         └──────┬───────────────────┬─────────┘
                                │                   │
              ┌─────────────────▼──────────────┐   ┌────────▼────────────────────┐
              │  MULTIMODAL Topology LLM       │   │  Verified Registry Roots    │
              │  MultimodalTopologyContext:    │   │  174 families, role-aware   │
              │   spec text + RAG evidence     │   │  hashes, tiers A1/A2/…  [V] │
              │   + FunctionalStageGraph       │   │  operational_v3             │
              │   + DeviceCircuitGraph         │   └────────┬────────────────────┘
              │   + schematic image PIXELS ────┼─┐          │
              │  image → native processor →    │ │ datasets/schematic_images_v2 │
              │  vision encoder → projector →  │ │ (image↔graph↔proposal        │
              │  language backbone  [P]        │ │  alignment, SHA-256)  [P]    │
              │  frozen base + frozen vision   │ └─                             │
              │  encoder; language-side LoRA;  │            │
              │  SFT + true token-level DPO    │            │
              │  (frozen multimodal reference) │            │
              │  pilot: SmolVLM-256M [P];      │            │
              │  primary: Qwen2.5-VL-3B [D:GPU]│            │
              │  llm_dpo/multimodal.py         │            │
              └──────┬─────────────────────────┘            │
                     │  structured TopologyProposal │
                     │  (JSON; rationale ≠ graph;   │
                     │   netlist text NEVER runs)   │
              ┌──────▼───────────────────────────────▼──────┐
              │   Parser → Canonicaliser → Topology         │
              │   Validator → Mapping Validator  [V]        │
              │   stage3e2_edits.validate_proposal          │
              │   topology_rl/stage3e1.validate_candidate   │
              └──────────────────┬──────────────────────────┘
                                 │ TopologySearchState (lineage, budgets)
              ┌──────────────────▼──────────────────────────┐
              │   AlphaZero Topology RL  [V]                │
              │   policy+value heads on shared MP encoder;  │
              │   PUCT MCTS over mixed actions:             │
              │   SELECT / KEEP / TERMINATE +               │
              │   7 executable structural edits             │
              │   (add/replace stage, load, compensation,   │
              │    output buffer, local feedback)  [V]      │
              │   topology_rl/{stage3e1,stage3e2,_edits}    │
              └──────────────────┬──────────────────────────┘
                                 │ candidate DeviceCircuitGraph
              ┌──────────────────▼──────────────────────────┐
              │   Transistor Realisation  [V]               │
              │   template mapper + edit library →          │
              │   Sky130 netlist + sizing manifest          │
              │   mapping/ + stage3e2_edits                 │
              └──────────────────┬──────────────────────────┘
                                 │
              ┌──────────────────▼──────────────────────────┐
              │   Graph-conditioned MB-SAC Sizing  [V]      │
              │   squashed-Gaussian actor, twin critics,    │
              │   dynamics ensemble w/ calibration gates,   │
              │   MP-encoder gradients from actor+critic    │
              │   mb_sac/ + sizing/                         │
              └──────────────────┬──────────────────────────┘
                                 │ sizing candidates
              ┌──────────────────▼──────────────────────────┐
              │   Surrogate screen [P] +                    │
              │   Feasibility-Gated Bradley–Terry           │
              │   Sizing Ranker  [V]  (NOT LLM DPO)         │
              │   valid > stable > feasible > margins >     │
              │   FoM > cost; exploration preserved         │
              │   dpo/ (DPORanker)                          │
              └──────────────────┬──────────────────────────┘
                                 │ selected candidates
              ┌──────────────────▼──────────────────────────┐
              │   Real ngspice-45.2 + sky130  [V]           │
              │   open-loop ADM testbench; verified-only    │
              │   measurements; PVT matrix separate  [V]    │
              │   electrical/                               │
              └──────────────────┬──────────────────────────┘
                                 │ PostSizingTopologyScore
              ┌──────────────────▼──────────────────────────┐
              │   Cross-Level Feedback  [V/P]               │
              │   → AlphaZero value/policy targets          │
              │   → MB-SAC replay, dynamics, surrogate      │
              │   → BT ranker pairs (real evidence only)    │
              │   → L4 simulation memory / RAG              │
              │   → LLM preference queue (offline dpo_v2)   │
              └─────────────────────────────────────────────┘
```

## Two preference systems (strictly separated)

| | True LLM DPO (`llm_dpo/`) | BT Sizing Ranker (`dpo/`) |
|---|---|---|
| Operates on | token log-probs, policy vs frozen reference | circuit-candidate features |
| Improves | topology proposals | sizing-candidate selection |
| Objective | −w·logsigmoid(β·Δ(policy−ref) margins) | BT logistic over feature scores |
| Records | same-context response pairs | real-SPICE outcome pairs |
| Shared data | **none** (tested) | **none** (tested) |

## Safety and honesty boundaries
- No LLM output bypasses parser → validator → mapping validation; generated
  netlist text is never executed (tested).
- SPICE is the final authority; surrogate/dynamics predictions are never stored
  as measurements; withheld ≠ failed; validator rejections ≠ simulator failures.
- Cache hits and model rollouts are never counted as SPICE calls.
- Polarity/stability adjudication is structural, never outcome-based.
- Every stage freezes a SHA-256-verified code snapshot before changes.

## Key artifacts
`artifacts/topology_registry_operational_v3` (corpus) · `datasets/target_sets_v1`
· `datasets/simulation_memory/*.jsonl` · `datasets/schematic_images_v1` ·
`artifacts/stage3d*|3e*` (campaign evidence) · `reports/` (80+ stage reports) ·
`tests/` (369+ passing).

## Open deferrals
GPU campaign for 1–7B (text) and Qwen2.5-VL-3B (multimodal) topology LLMs;
publication-scale AZ self-play and MB-SAC ablations; full 91-family repair
curriculum; success-probability calibration head.
