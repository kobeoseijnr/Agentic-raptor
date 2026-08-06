# Pre-SPICE DPO Ranking & One-Time LoRA Fine-Tuning

## Updated architecture (DPO stage inserted before SPICE)

```
Specs → RAG → LLM generation → validation → MCTS refinement → MB-SAC sizing
  → Candidate pool (evaluated sizing vectors, agentic_raptor/dpo/schemas.py)
  → Pre-SPICE DPO preference ranking (dpo/ranker.py — Bradley–Terry/β pairwise
    objective; NOT heuristic scoring; leakage-guarded features)
  → Feasibility/uncertainty/diversity screening + exploration quota (dpo/selector.py)
  → Candidate selection → SPICE + PVT → OutcomeRecord storage (dpo_outcomes.jsonl)
  → Cross-level updates: MB-SAC, dynamics (surrogate role), DPO pairs, RAG,
    topology policy/value credit
```

Preference pairs: lexicographic a–i hierarchy (`preference_pairs.py`), tie-tolerant
Pareto, ambiguous pairs excluded, confidence weights, group-safe splits,
checkpoint/rollback, periodic or fixed updates. Config: `dpo:` section
(`utils/config.py::DPOSettings`); `enabled: false` reproduces the original
pipeline exactly (tested).

## Ablation matrix (set in any experiment YAML)

| Ablation | Config |
|---|---|
| No DPO (baseline) | `dpo.enabled: false` |
| With DPO | `dpo.enabled: true` |
| DPO w/o surrogate signals | `dpo.enabled: true` + zero feasibility/uncertainty terms: set `sac.dynamics_ensemble_size: 1` |
| Surrogate w/o DPO | `dpo.enabled: false` (selector falls back to predicted feasibility) |
| DPO + surrogate | `dpo.enabled: true`, `sac.dynamics_ensemble_size: 2` |
| DPO w/o exploration | `dpo.exploration_fraction: 0.0` |
| Periodic vs fixed | `dpo.update_mode: periodic` vs `fixed` |

Metrics available from episode summaries + `dpo_outcomes.jsonl`: pass rate, Pass@K,
calls-to-first-pass, total SPICE calls, runtime, best FoM, PVT pass rate,
`DPORanker.pair_accuracy` (preference accuracy; ranking↔outcome correlation),
false-positive selections (selected-but-failed records), pool diversity, and DPO
overhead (ranker runtime is negligible vs SPICE).

## LoRA fine-tuning (agentic_raptor/finetuning/) — one-time, post-collection

Pipeline: `EpisodeOutcomeStore` → attribution A (original success, ≤1 edit) /
B (corrected) / C (failed; never a target) → tiered SFT dataset (elite /
verified_success / verified_corrected; prompts contain deployment-time inputs
only) → group-safe splits by spec family → `inject_lora` (frozen base, adapters
on q/v projections, adapter-only gradients — test-verified) → `train_lora`
(teacher-forced) → versioned `AdapterManager` (save/load/switch/rollback, never
merged) → `acceptance_gate` (validity/pass-rate/diversity/seed criteria; rejects
otherwise). **Honest caveat**: the deployed generator is API-backed (gpt-4o-mini);
LoRA attaches to a *local* model interface (`build_toy_topology_decoder` stand-in
in tests) and becomes production-relevant when a local multimodal model is
adopted. No continual per-episode fine-tuning; pre-SPICE DPO remains the only
candidate ranker.
