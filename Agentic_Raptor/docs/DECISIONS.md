# Design Decisions

**D1 — Additive-only integration.** The parent repo is not a git repository (no
branch isolation possible), holds ~580k artifact files, and mixes many past
experiment stages. All new code lives in `Agentic_Raptor/`; legacy code is reached
only through read-only guarded adapters (`agentic_raptor/adapters/`).

**D2 — Dataclasses over Pydantic.** Both are allowed by the brief; dataclasses with
explicit `validate()`/`from_dict()` keep the dependency surface minimal (Pydantic
v2 behaviours differ across environments) and are fully typed. Config loading gets
strict unknown-key checking via `utils/config._build` using `get_type_hints` (PEP
563 makes `field.type` a string — resolved explicitly).

**D3 — Graph encoder without PyTorch Geometric.** PyG is not installed on this
machine (and is heavy to build on Windows/3.14). Two plain-torch encoders are
provided behind a config switch: `pooled` (default; permutation-invariant feature
pooling) and `message_passing` (2-round dense mean-aggregation over the bipartite
device–net adjacency). `CircuitGraph.to_networkx()` is the future PyG bridge
(`from_networkx`), so upgrading is additive.

**D4 — Terminal-to-net edges.** `CircuitEdge = (node, terminal) → net` matches
SPICE semantics, converts directly to the legacy `bipartite_device_net` view, makes
nets first-class (supply-short and floating checks become trivial), and keeps
hashing well-defined. Parallel attachments (diode-connected MOS) are merged with
combined sorted labels in the NetworkX view so WL hashing is insertion-order
independent.

**D5 — WL hashing, not exact canonical labelling.** Weisfeiler–Lehman hashes are
stable, rename-invariant, and fast; rare collisions are acceptable for caching and
dedup at these graph sizes. Exact isomorphism can be layered later without touching
callers (`topology_validation/graph_hash.py` is the single seam).

**D6 — Mock-first multimodal boundary.** Text/YAML/JSON/CSV/netlist parsing is
fully implemented; schematic-image parsing is a Protocol with a mock (a VLM call is
an integration, not a research risk). The LLM generator shell enforces structured
JSON output with validation + retry hooks; provider transports read env vars
(`AGENTIC_RAPTOR_LLM_*`) at call time — no keys in code or config.

**D7 — Fresh genuine SAC instead of wrapping the legacy agent.** The audit verified
the legacy `SACAgent` is genuine, but it fixes state/action dims per instance and
its model-based part is a horizon-1 surrogate with random actions. Graph-derived
action spaces change per topology, and the v2 brief requires a jointly-learned
dynamics model with reward/termination heads and actor-driven rollouts. The legacy
agent remains available as a backend via `sizing/adapters.LegacySACBackend` for A/B
comparisons.

**D8 — Sizing-step evaluations vs. SPICE budget.** During sizing rounds, candidate
evaluations use the cheap deterministic simulator as a stand-in for a surrogate
predictor and consume the *sizing* budget; only explicit RUN_SPICE / RUN_PVT
decisions consume the *SPICE* budget. This keeps the cost signal in the reward
meaningful while the smoke pipeline stays fast. With a real simulator, the sizing
loop would call the surrogate/dynamics model instead — same seam.

**D9 — Environment API.** `TopologyEditEnv` follows the Gymnasium 5-tuple
convention without subclassing `gymnasium.Env`: the legal-action set is dynamic
(index into `legal_actions()`), which fits poorly into static `spaces` declarations;
a wrapper can add formal spaces later without changing the core.

**D10 — Coordinator is rule-based but injectable.** `DecisionPolicy` is a Protocol;
`RuleBasedDecisionPolicy` implements transparent ordered rules, every decision
logged with its reason to JSONL. A learned policy can be swapped in without
touching the episode loop.

**D11 — KMP_DUPLICATE_LIB_OK workaround.** The legacy repo sets this before torch
imports on this machine (OpenMP runtime collision); replicated in
`utils/seeding.apply_torch_omp_workaround()` and used before every lazy torch import.

**D12 — Tooling.** `pytest` and `ruff` were installed into the environment (they
were absent); `black` and `mypy` were not installed — ruff covers lint + import
sorting; mypy adoption is listed as follow-up work.
