# Stage 3E.1 — policy network

Date: 2026-07-26

Pointer-style per-action scorer: shared 2-round residual MP encoder (16-d) -> context [graph 16 + spec 5 + budget 3] concat [action one-hot 11 + action-graph embedding 16] -> MLP(51->64->1) -> masked softmax over the deterministically ordered legal set. Variable action-set sizes supported; probabilities sum to 1 (tested); illegal actions never enter the distribution (removed pre-softmax). Provides MCTS priors only.
