# Stage 3E.2 LLM-DPO — true dpo

Date: 2026-07-26

Objective (exact): -w*logsigmoid(beta*((lp_pol(y+)-lp_ref(y+))-(lp_pol(y-)-lp_ref(y-)))), beta=0.1, reference-free disabled. Sequence log-prob = sum of token log-softmax over response tokens only (prompt+padding masked; hand-checked loss(0)=log2 in tests). Trainable LoRA policy from SFT checkpoint; frozen SFT reference (checksum-proved unchanged).
