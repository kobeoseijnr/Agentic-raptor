# Stage 3E.2 LLM-DPO — sft dataset

Date: 2026-07-26

Split-safe SFT dataset: train=14, validation=2, held-out targets excluded, graph-hash dedup enforced (leakage tests). Held-out topology split: unsupported at n=16 stable (all in training corpus). Contexts: spec + RAG refs + allowed blocks; outputs: canonical proposals.
