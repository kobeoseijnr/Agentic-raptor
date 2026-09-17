# RAPTOR — System Architecture

Updated: 2026-09-16

This document describes the **RAPTOR architecture evaluated in the paper**:

> **RAPTOR: Retrieval-Augmented Preference-Guided Multi-Agent Topology and Sizing Optimization via Reinforcement Learning for Analog Circuits**

RAPTOR jointly coordinates analog circuit **topology generation** and **circuit sizing** under a shared SPICE budget. The system contains four main agents:

1. **Topology Generation Agent**
2. **Topology Critic Agent**
3. **Topology Selection Agent**
4. **Circuit Sizing Agent**

These agents are followed by preference ranking, PVT analysis, final SPICE verification, and a bounded recovery mechanism.

> **Scope note:** This document describes the architecture used in the current RAPTOR paper and evaluation. Experimental extensions such as AlphaZero-style topology RL, MCTS topology editing, multimodal/VLM generation, and cross-level RL training are not part of the evaluated paper architecture and should be documented separately.

---

## End-to-End Pipeline

```text
┌──────────────────────────────────────────────┐
│              Design Specifications           │
│                                              │
│  gain / UGBW / phase margin / load           │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          Topology Generation Agent           │
│                                              │
│  • Analyze target specifications             │
│  • Retrieve similar measured designs         │
│  • Up to 6 RAG examples                      │
│  • Maximum 2 examples per topology family    │
│  • SFT-adapted Qwen3-4B-Instruct-2507        │
│  • Generate 4–6 candidate topologies         │
│  • Structured topology representation        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│            Topology Critic Agent             │
│                                              │
│  • Structural validity checks                │
│  • Stage-count diversity                     │
│  • Compensation diversity                    │
│  • Specification-aware refinement            │
│  • Limited regeneration rounds               │
│  • Validation and deduplication              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          Topology Selection Agent            │
│                                              │
│  • 24-D topology/specification context       │
│  • Contextual-bandit ranking                 │
│  • Offline ridge-regression reward model     │
│  • Select highest-ranked topology            │
│  • Select structurally diverse alternative   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Primary Topology  │
             └─────────┬─────────┘
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      Primary Candidate   Alternative Candidate
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             Circuit Sizing Agent             │
│                                              │
│  • Probe both selected topologies            │
│  • Compare SPICE-measured performance        │
│  • Allocate remaining simulation budget      │
│  • MB-SAC continuous sizing                  │
│  • Online surrogate guidance                 │
│  • Reserve budget for refinement/recovery    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              Preference Ranking              │
│                                              │
│  • Remove invalid/unstable designs           │
│  • Compare feasible candidates using:        │
│      - specification margins                 │
│      - FoM                                   │
│      - efficiency                            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             Final Refinement                 │
│                                              │
│  Remaining SPICE budget may be used to       │
│  further improve the selected design.        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                PVT Analysis                  │
│                                              │
│  Process / voltage / temperature evaluation  │
│  More robust feasible design is preferred    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          Final SPICE Verification            │
└──────────────────────┬───────────────────────┘
                       │
                 verification fails
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              Recovery Attempt                │
│                                              │
│  • Revisit alternative topology              │
│  • Re-optimize with remaining budget         │
│  • Reserve one SPICE call for final check    │
│  • At most one recovery attempt              │
└──────────────────────────────────────────────┘