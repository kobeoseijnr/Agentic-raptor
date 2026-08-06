# Novelty and Claims

## Central hypothesis (defensible)

> A topology should be judged by its final performance after continuous sizing and
> SPICE verification, not only by its unsized graph structure.

Operationally: the topology-level value function is trained against returns computed
from **post-sizing SPICE outcomes** (`z_t = γ^(T−1−t) · R_final`), never against
purely structural heuristics. Structural validity contributes only a small shaping
term whose weight is configuration, not code.

## Candidate contribution

An agentic analog-design framework coupling:

```text
Multimodal specification parsing
+
RAG-conditioned multimodal topology generation
+
AlphaZero-inspired topology reinforcement learning
+
Graph-conditioned model-based Soft Actor-Critic sizing
+
Post-sizing SPICE-based cross-level credit assignment
+
Adaptive regenerate/edit/resize/stop coordination
```

The claimed novelty candidate is the **integration**: a single closed loop in which
(a) the discrete topology policy learns from the continuous sizing level's final
simulator-verified outcome via trajectory-level credit, and (b) an agentic
coordinator allocates edit/sizing/SPICE budgets across both levels with logged,
auditable decisions.

## Claims explicitly NOT made

We do **not** claim any of the following:

- first multimodal circuit generator;
- first LLM circuit generator;
- first RL topology generator;
- first use of MCTS in circuit generation;
- first joint topology-and-sizing method;
- first agentic circuit-design framework.

Prior work inside this very repository already contains an LLM netlist generator,
graph-edit MCTS, learned topology policy/value training (offline, checkpoint 5d),
MB-SAC sizing, RAG memory, and a rule-based controller — the audit
(`REPOSITORY_AUDIT.md`) documents exactly what existed before this extension.

## Status of performance claims

**All performance claims are hypotheses until experiments are complete.** The
current implementation runs end-to-end with a deterministic mock simulator; no
comparative results exist yet. Planned evidence: improvement curves of final
post-sizing reward across episodes vs. (i) unsized-heuristic topology scoring and
(ii) the legacy offline policy/value pipeline, under matched SPICE budgets.

## Differences from the existing RAPTOR pipeline (factual, audit-based)

| Aspect | Legacy RAPTOR | Agentic RAPTOR |
|---|---|---|
| Specification input | Netlist/text-centric scripts | Multimodal parse + fusion + provenance + conflict detection |
| Topology value targets | Inner-sizing evaluator estimates (offline datasets) | Final post-sizing SPICE return via credit assignment |
| Policy/value training | Offline dataset pipeline (checkpoint 5d) | In-loop updates after every episode |
| MCTS | PUCT, top-k priors | PUCT + progressive widening + visit-count policy targets |
| Sizing | Genuine SAC, surrogate horizon-1 synthetic data | Genuine SAC + jointly trained dynamics model (reward + termination heads), actor-driven imagined rollouts |
| Orchestration | Rule-based controller script | State-machine coordinator with budget manager, logged decisions, learnable-policy interface |
| Memory | Simulator-grounded JSONL items | Typed entries with embeddings, PVT, edit trajectories, outcome write-back |
