# RAPTOR — Novelty and Claims

Updated: 2026-09-16

This document defines the novelty and claims for the architecture evaluated in the current RAPTOR paper:

> **RAPTOR: Retrieval-Augmented Preference-Guided Multi-Agent Topology and Sizing Optimization via Reinforcement Learning for Analog Circuits**

This document intentionally covers only the system evaluated in the paper. Experimental extensions such as AlphaZero topology RL, MCTS topology editing, multimodal/VLM generation, token-level LLM DPO, and cross-level topology-policy training are outside the scope of the current paper.

---

## Central Hypothesis

> A topology should not be selected only because its unsized structure appears promising. Its usefulness depends on whether it can be efficiently sized to satisfy the target specifications under SPICE verification.

Topology generation and circuit sizing are therefore treated as coupled decisions.

A generated topology may be structurally valid but difficult or impossible to size to the requested specifications. Likewise, an effective sizing algorithm cannot overcome fundamental limitations of an unsuitable topology.

RAPTOR addresses this coupling by generating multiple candidate topologies, refining them, selecting promising and structurally diverse candidates, probing their measured performance, and allocating the available SPICE budget according to downstream sizing potential.

---

## Core Contribution

RAPTOR integrates:

```text
Design Specifications
        +
Retrieval-Augmented Topology Generation
        +
SFT-Adapted LLM
        +
Topology Critic Agent
        +
Contextual-Bandit Topology Selection
        +
Two-Candidate SPICE Probing
        +
Shared SPICE-Budget Allocation
        +
MB-SAC Continuous Circuit Sizing
        +
Online Surrogate Guidance
        +
Preference Ranking
        +
PVT Analysis
        +
Final SPICE Verification
        +
Bounded Recovery