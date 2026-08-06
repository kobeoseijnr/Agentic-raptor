# Stage 3E.3 — LLM Requirement Audit

Date: 2026-07-26. Traces the LLM-DPO amendment against runtime artifacts (not class existence).
Fixture-backed generation is NEVER counted as trained-LLM generation.

| Req | Requested | Pre-3E.3 status | 3E.3 action | Evidence |
|---|---|---|---|---|
| L1 model | open-weight versioned model | C (distilgpt2, apache-2.0) | preserved; hardware limits re-recorded | stage3e2_llm/SUMMARY.json |
| L2 schema | structured proposal | C | preserved; FORBIDDEN section added to prompt | stage3e3/sft_dataset_v2.json |
| L3 SFT dataset | split-safe, per-source stats, held-out structure | P (small, no structure holdout) | **closed**: A1+A2 sourced, dedup, per-source/stage stats, 3-stage structure holdout | sft_dataset_v2.json |
| L4 SFT | gate beyond LM loss | P (40 steps, 0% validity) | **closed at CPU scale**: 220 steps, validation loss, greedy structured eval | sft_v2.json |
| G generation campaign | multi-candidate same-context decoding | M | **closed**: 3 ctx x 3 seeds, records w/ parse/validator/novelty | generation_campaign.json |
| H realisation | trained proposal -> real ngspice | M | **closed** (if valid generation exists; else honest skip recorded) | realised_proposal.json |
| I preference pairs | evidence-tiered pairs | C | rebuilt (18; 6 real-SPICE-backed) | stage3e2_llm/preference_pairs.json |
| J/K true DPO | token-level, frozen ref, 3 seeds | C | re-run from SFT-v2 checkpoint | stage3e2_llm/dpo.json |
| L refinement | proposal->MCTS->edit->SPICE before/after | M (deferred hop) | **closed**: KEEP-vs-EDIT one-ply PUCT, real SPICE both sides, hard-gated verdict | refinement.json |
| M comparison | base/SFT/SFT+DPO equal budget | C (0% all) | re-run greedy on v2 ckpts | comparison_v2.json |
| N novelty | classify generations | P (edits only) | **closed**: per-generation novelty class incl. exact-memorisation detection | generation_campaign.json |
| O feedback queue | versioned, frozen-adapter | C | re-queued v2 | datasets/llm_preference_queue/v1.jsonl |

Blocked items (unchanged, exact blockers): GPU-scale SFT/DPO campaigns (no CUDA);
multimodal image input (model class); statistical DPO-improvement claim (needs
GPU-scale valid-generation rates).
