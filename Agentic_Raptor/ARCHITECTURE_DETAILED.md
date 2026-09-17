# RAPTOR — Current Paper Architecture

Updated: 2026-09-16

This document describes the architecture evaluated in the current RAPTOR paper:

> **RAPTOR: Retrieval-Augmented Preference-Guided Multi-Agent Topology and Sizing Optimization via Reinforcement Learning for Analog Circuits**

RAPTOR jointly coordinates **analog circuit topology generation** and **continuous circuit sizing** under a shared SPICE simulation budget.

The evaluated system consists of four agents:

1. **Topology Generation Agent**
2. **Topology Critic Agent**
3. **Topology Selection Agent**
4. **Circuit Sizing Agent**

The four-agent pipeline is followed by preference ranking, optional final refinement, PVT analysis, final SPICE verification, and one bounded recovery attempt.

SPICE remains the final authority for circuit performance.

---

## 1. What the System Does

Given a target analog amplifier specification, RAPTOR:

1. analyzes the requested performance,
2. retrieves relevant previously measured designs,
3. generates several candidate circuit topologies,
4. critiques and refines the candidate pool,
5. ranks the candidates,
6. selects two structurally diverse topologies,
7. probes both candidates with SPICE,
8. allocates more sizing budget to the stronger candidate,
9. performs continuous sizing with reinforcement learning,
10. ranks feasible sized designs,
11. evaluates PVT robustness,
12. performs final SPICE verification,
13. performs one recovery attempt if final verification fails.

The important distinction from a fixed-topology sizing system is that RAPTOR makes both:

- **structural decisions** about topology, and
- **continuous decisions** about circuit parameters.

---

## 2. End-to-End Pipeline

```text
┌──────────────────────────────────────────────┐
│              Design Specifications           │
│                                              │
│ gain / UGBW / phase margin / load            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          Topology Generation Agent           │
│                                              │
│ • Analyze specification requirements         │
│ • Retrieve similar measured designs          │
│ • Condition SFT-adapted LLM                   │
│ • Generate 4–6 topology candidates           │
│ • Validate and deduplicate outputs            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│            Topology Critic Agent             │
│                                              │
│ • Structural validity                        │
│ • Stage-count diversity                      │
│ • Compensation diversity                     │
│ • Specification-aware feedback               │
│ • Limited regeneration rounds                │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          Topology Selection Agent            │
│                                              │
│ • 24-D context representation                │
│ • Contextual-bandit ranking                  │
│ • Offline ridge reward model                 │
│ • Select highest-ranked candidate            │
│ • Select diverse alternative candidate       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Two Candidates   │
              └────────┬─────────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
         Candidate 1       Candidate 2
              │                 │
              └────────┬────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             Circuit Sizing Agent             │
│                                              │
│ • Probe both candidates with SPICE           │
│ • Compare measured performance               │
│ • Allocate remaining SPICE budget            │
│ • MB-SAC continuous sizing                   │
│ • Online surrogate guidance                  │
│ • Preserve calls for refinement/recovery     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              Preference Ranking              │
│                                              │
│ • Remove invalid/unstable candidates         │
│ • Compare feasible designs using             │
│   margins, FoM, and efficiency               │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              Final Refinement                │
│                                              │
│ Remaining simulation budget may be used      │
│ to further improve the selected design.      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                PVT Analysis                  │
│                                              │
│ process / voltage / temperature              │
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
│ • Revisit alternative topology               │
│ • Re-optimize with remaining budget          │
│ • Reserve one final SPICE call               │
│ • Maximum one recovery attempt               │
└──────────────────────────────────────────────┘