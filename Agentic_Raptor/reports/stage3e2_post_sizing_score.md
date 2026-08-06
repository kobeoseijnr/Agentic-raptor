# Stage 3E.2 — post sizing score

Date: 2026-07-26

Scalar = 0.2*valid + 0.4*stable + 0.4*feasible + 0.2*clip(worst margin) - 0.1*min(1,calls/20): bounded [-1.3,1.3], ordinal, monotonic under hard-gate ordering (stability dominates FoM — tested), budget-dependent via cost term, suitable for value regression and MCTS backup. NOT a probability; success-probability head deferred until campaign-scale outcome counts exist (exact blocker: data volume).
