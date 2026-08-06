# Stage 3E.3 — true dpo

Date: 2026-07-26

Objective: -w*logsigmoid(beta*((lp_pol(y+)-lp_ref(y+))-(lp_pol(y-)-lp_ref(y-)))), beta=0.1, response-only masks, reference-free disabled, frozen SFT-v2 reference (checksums bitwise stable), trainable LoRA policy.
