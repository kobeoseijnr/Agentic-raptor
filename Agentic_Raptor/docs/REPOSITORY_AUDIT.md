# Repository Audit — existing RAPTOR codebase

Date: 2026-07-25. Based on direct inspection of `c:\Users\kobeo\OneDrive\Desktop\raptor1`.

## 0. Version control status

**The repository is NOT a Git repository** (`git status` → `fatal: not a git repository`).
The requested branch `feature/agentic-raptor` therefore cannot be created. No `git init`
was performed (initializing version control is a repository-owner decision). All safety
guarantees in this pass are achieved by *additive-only* changes: nothing outside
`Agentic_Raptor/` was created, modified, renamed, or deleted.

Recommendation: run `git init`, commit the existing tree as a baseline, then branch.

## 1. Top-level layout (Python file counts)

| Directory | .py files | Notes |
|---|---|---|
| `AnalogGym/` | 157 | Vendored external benchmark suite (own README/LICENSE/PDK) |
| `experiments/` | 154 | Checkpoint-numbered experiment runners (checkpoint_3 … 8f) |
| `RGNN_RL/` | 42 | MOSFET-model / RGNN code (3k+ data files) |
| `mb_sac/` | 30 | Model-based SAC sizing stack |
| `graph_search/` | 19 | MCTS + policy-value topology search |
| `rag/` | 18 | RAG memory, retrieval, snapshot manager |
| `AutoCkt/` | 13 | Vendored external baseline (own env/README) |
| `dpo/` | 12 | Sizing-level DPO reranking |
| `scripts/` | 11 | Candidate generation / diagnostics |
| `graph/` | 9 | Circuit-graph schema, edit grammar, validation |
| `surrogate/` | 8 | Surrogate ensemble + uncertainty |
| `llm/` | 6 | Provider-neutral LLM client, netlist generation/repair |
| `controller/` | 6 | Rule-based RAG+DPO controller (closest ancestor of our coordinator) |
| `topology_dpo/` | 6 | Topology-level DPO |
| `results/` | 0 (545,554 files) | Experiment outputs/checkpoints — **do not touch** |
| `data/` | 0 (22,838 files) | Datasets — **do not touch** |
| `outputs/` | 0 (11,286 files) | Run outputs — **do not touch** |
| `configs/` | 8 YAML | Checkpoint experiment configs |
| `archive/` | — | `legacy_before_rag_dpo_cleanup/` |

## 2. Existing modules by concern

### 2.1 RAG
- `rag/memory_schema.py` — `RagMemoryItem` dataclass; memory types `valid_design`,
  `failed_design`, `repair_case`, `topology_template`, `sizing_rule`, `testbench_rule`,
  `trajectory_step`; simulator-grounded fields (`netlist_text`, `metrics_json`,
  `spice_success`, `pass_flag`, spec gaps).
- `rag/rag_knowledge_store.py`, `build_rag_memory.py`, `build_rag_knowledge_base.py` — store construction.
- `rag/retrieve_root_spec.py`, `retrieve_search_state.py`, `retrieve_failure_cases.py`,
  `retrieve_rag_evidence.py` — three retrieval modes (root spec / search state / failure).
- `rag/update_memory.py`, `update_intervention_memory.py` — outcome write-back.
- `rag/state_similarity.py` — similarity used for retrieval.

### 2.2 MB-SAC (sizing)
- `mb_sac/sac_agent.py` — `SACAgent` + `SACConfig` (twin critics, auto-alpha, LayerNorm MLPs;
  uses `KMP_DUPLICATE_LIB_OK=TRUE` workaround before importing torch — we replicate this).
- `mb_sac/sizing_environment.py` — sizing env; **invokes ngspice via subprocess** (this is the
  real SPICE execution site, together with validation scripts in `graph_search/`).
- `mb_sac/train_mb_sac.py`, `train_true_mb_sac.py` — training loops (model-based rollouts).
- `mb_sac/model_based_rollout.py`, `model_replay_buffer.py`, `replay_buffer.py`,
  `uncertainty_estimator.py`, `surrogate_model.py` — MB components.
- `mb_sac/state_encoder.py`, `rich_state_encoder.py`, `augmented_decision_state.py` — state encoding.
- `mb_sac/reward_function.py` — sizing reward.
- `mb_sac/action_space.py`, `transition_schema.py` — action/transition schemas.

### 2.3 MCTS / topology search
- `graph_search/graph_mcts.py` — `run_graph_mcts(...)`: PUCT selection, top-k prior expansion,
  value backup over dict-based graph states. No progressive widening; no trained network required.
- `graph_search/mcts_node.py` — `MCTSNode` (select_child/expand/backup).
- `graph_search/graph_policy_value.py` — heuristic `score_graph_action` / `estimate_graph_value`.
- `graph_search/topology_policy_value_model.py`, `train_topology_policy_value.py`,
  `learned_policy_value_adapter.py`, `topology_policy_value_dataset.py` — learned policy/value
  (AlphaZero-style direction already started at checkpoint 5d).
- `graph_search/inner_sizing_evaluator.py`, `topology_leaf_evaluator.py` — leaf evaluation via
  inner sizing (the ancestor of our cross-level credit mechanism).
- `graph_search/topology_dpo_mcts_prior.py` — DPO-based MCTS prior.

### 2.4 Circuit graph and edits
- `graph/graph_schema.py` — `CircuitGraph`/`GraphNode`/`GraphEdge` dataclasses,
  **bipartite device–net representation** (`graph_type="bipartite_device_net"`); carries
  netlist text, specs, metrics, rewards, flags in one object (topology and outcomes mixed).
- `graph/graph_edit_grammar.py` — `_ALLOWED_ACTIONS` registry (compensation caps, series RC,
  bleeder/shunt resistors, …) with preconditions and risk levels; `enumerate_valid_graph_actions`.
- `graph/apply_graph_edit.py` — applies edits to dict graphs.
- `graph/validate_graph.py` — `validate_circuit_graph(graph_dict) -> {errors, warnings}`
  (missing ground/supply/output, terminal counts, degree checks).
- `graph/parse_netlist_to_graph.py`, `export_graph_to_netlist.py` — netlist ⇄ graph.
- `graph/graph_features.py` — feature extraction.

### 2.5 SPICE
- No standalone `spice/` package. Simulation is embedded in:
  - `mb_sac/sizing_environment.py` (ngspice subprocess; `bsim4v5.out` at repo root is an ngspice artifact),
  - `graph_search/validate_refined_candidates_spice.py`, `validate_refined_candidates_exported_netlists_spice.py`,
  - `controller/module_adapters.py::run_spice_validation` (takes `exported_netlist_path`,
    `ngspice_exe`, timeout; returns dict with `error_message` etc.).
- Netlists are exported to files and simulated externally; results parsed from ngspice output.

### 2.6 Surrogate models
- `surrogate/surrogate_ensemble.py`, `train_surrogate.py`, `evaluate_surrogate.py`,
  `uncertainty.py`, `feature_schema.py`, `transition_dataset.py`, `schema_validator.py`.
- Also `mb_sac/surrogate_model.py` (dynamics-side surrogate).

### 2.7 DPO / ranking
- Sizing level: `dpo/` (preference pairs, `preference_model.py`, `rag_conditioned_dpo_reranker.py`,
  train/eval scripts).
- Topology level: `topology_dpo/` (dataset, model, interface, train/eval).

### 2.8 LLM generation
- `llm/llm_client.py` — provider-neutral `call_llm`; provider from `LLM_PROVIDER` env var
  (`openai` | `ollama` | `mock`, default **mock**); models/base-URLs from env vars; no hard-coded
  keys; logs calls to `results/logs/llm_calls.jsonl`.
- `llm/generate_netlist.py`, `parse_llm_output.py`, `repair_netlist.py`, `prompts.py`.
- Generation is **netlist-text-first** (then parsed to graph), not graph-JSON-first, and not multimodal.

### 2.9 Controller / orchestration
- `controller/rag_dpo_controller.py` — rule-based orchestration of retrieve → MCTS → sizing →
  surrogate screen → DPO rank → SPICE → memory update.
- `controller/module_adapters.py` — the integration hub; wraps every subsystem as functions
  (`run_graph_mcts_topology_search`, `run_mb_sac_sizing`, `run_spice_validation`,
  `run_surrogate_screen`, `rank_sizing_candidates_with_dpo`, `update_rag_memory`, …).
- `controller/controller_state.py`, `controller_decision_schema.py`, `controller_trace_logger.py`.

### 2.10 Configuration & experiments
- `configs/*.yaml` — per-checkpoint experiment configs (no unified typed config loader found).
- `experiments/` — 154 checkpoint-numbered scripts; hard-wired to `results/` tables/paths.
- Root-level scratch: `tmp_*.py`, `_tmp_dpo_debug.py`, `inspect_results.py`.

### 2.11 Datasets & checkpoints
- `results/` (545k files), `data/` (22k), `outputs/` (11k) — left strictly untouched.
- `RGNN_RL/mosfet_model/`, `AnalogGym/`, `AutoCkt/`, `baselines/` — vendored/external; untouched.

## 2.12 Genuineness audit — is the existing MB-SAC real RL? Is the MCTS AlphaZero-style?

### MB-SAC (`mb_sac/sac_agent.py`, verified line-by-line)
**Verdict: genuine SAC.** Verified present:
- stochastic Gaussian actor with `mu`/`log_std` heads (clamped) — yes (`_Actor`, lines 65–85);
- reparameterised sampling — yes (`dist.rsample()`, line 125);
- tanh-bounded actions with log-prob squash correction — yes (lines 126–131);
- twin Q critics + twin target critics initialised from the live critics — yes (lines 88–94);
- Bellman target **with entropy term** `r + γ(1-d)[min(Q'₁,Q'₂) − α·logπ]` — yes (lines 158–165);
- critic MSE update with grad clipping — yes (lines 167–179);
- actor loss `α·logπ − min(Q₁,Q₂)` — yes (lines 182–206);
- automatic entropy temperature (learned `log_alpha`, target entropy `−|A|`) — yes (lines 100–111, 208–214);
- Polyak soft target updates — yes (lines 216–222);
- save/load of all networks — yes.

### Model-based part (`mb_sac/train_true_mb_sac.py`)
**Verdict: real but partial.** `TrueMBSACTrainer` keeps separate real/model replay buffers,
generates **horizon-1 synthetic transitions from the surrogate ensemble** with an
uncertainty-rejection gate (`synthetic_max_uncertainty`), and mixes real/synthetic batches
(80/20 default). Limitations vs. a full MBPO-style loop: rollout horizon fixed to 1; synthetic
actions are random Gaussian, not actor-sampled; the "dynamics model" is the offline-trained
surrogate ensemble — there is **no jointly-trained dynamics network predicting reward and
termination**. The new `agentic_raptor.sizing.dynamics_model` fills exactly this gap.

### MCTS (`graph_search/graph_mcts.py`, `mcts_node.py`)
**Verdict: genuine PUCT planning, not yet a full AlphaZero loop.**
Present: PUCT selection, prior-scored top-k expansion, value backup, depth limit.
Missing: progressive widening; root visit-count distribution extraction as a policy target;
legal-action masking as an explicit mask (it enumerates only valid actions instead);
Dirichlet root noise.

### Learned policy/value (`graph_search/train_topology_policy_value.py`)
**Verdict: real gradient training exists** (`loss.backward()` + optimizer step on a combined
policy + weighted value loss, with validation-based model selection) — but it is an **offline
dataset pipeline** (checkpoint 5d datasets collected from MCTS trajectories). The value targets
come from inner-sizing evaluator estimates, **not** from final post-sizing SPICE returns
propagated by trajectory-level credit assignment, and there is no integrated
search → train → search self-play loop. This is precisely the gap Agentic RAPTOR's
`topology_rl.trainer` + `learning.cross_level_credit` close.

## 3. Reuse decisions

### Reusable directly (via import, guarded adapters)
| Legacy entry point | Used for |
|---|---|
| `controller.module_adapters` | Single choke-point to reach MCTS/sizing/SPICE/DPO/RAG pipelines |
| `mb_sac.sac_agent.SACAgent` | Backing agent for `GraphConditionedMBSAC` |
| `rag.memory_schema.RagMemoryItem` + `rag.rag_knowledge_store` | Persistent memory backend |
| `llm.llm_client.call_llm` | Provider-neutral LLM transport (env-var config, mock default) |
| `graph.parse_netlist_to_graph` / `export_graph_to_netlist` | Netlist ⇄ graph bridging |
| `surrogate.surrogate_ensemble` | Sizing-outcome estimate input to policy-value net |

### Requires adapters (new code wraps, does not duplicate)
- Legacy dict-based `CircuitGraph` (bipartite device–net, untyped, outcome-mixed) ⇄ new typed
  `agentic_raptor.core.circuit_graph.CircuitGraph`. Adapter: `adapters/legacy_raptor.py`.
- ngspice execution (`mb_sac/sizing_environment.py`, `module_adapters.run_spice_validation`) behind
  the new `SpiceSimulator` protocol. Adapter: `spice/simulator_adapter.py::LegacyNgspiceSimulator`
  (stub in this pass; the smoke test uses the deterministic mock).
- Legacy MCTS (`graph_search.graph_mcts`) — superseded by the new MCTS (adds progressive widening,
  visit-count policy, pluggable network evaluator) but kept importable for A/B comparison.

### Compatibility risks
1. **Not a git repo** — no branch isolation; mitigated by additive-only policy.
2. **Python 3.14.2** environment (new code targets 3.11+; legacy code appears 3.10+-style — fine).
3. Missing dev deps at audit time: `pytest`, `ruff` (installed during this pass), `black`, `mypy`,
   `torch_geometric` (NOT installed — fallback pooled/message-passing encoder used, see DECISIONS.md).
4. Torch/OpenMP collision on this machine — legacy sets `KMP_DUPLICATE_LIB_OK=TRUE`; replicated in
   `agentic_raptor.utils.seeding` and all lazy torch imports.
5. Legacy modules do `sys.path.insert(0, repo_root)` and import each other by top-level package
   name (`rag.…`, `graph.…`) — adapters must add the repo root to `sys.path`, which is done lazily
   and only inside `adapters/`.
6. `results/` paths are hard-coded inside legacy modules (e.g. controller reads
   `results/tables/checkpoint_5d_*.csv`) — legacy calls may fail if those artifacts move; we never
   move them.
7. OneDrive filesystem is slow for large recursive scans; new outputs stay inside
   `Agentic_Raptor/outputs/`.

## 3.1 Recommended read-only import paths

All legacy access goes through `agentic_raptor.adapters.*`, which appends the repo root to
`sys.path` lazily and never writes to legacy modules or their data. Recommended imports:

```python
from mb_sac.sac_agent import SACAgent, SACConfig            # verified-genuine SAC backend
from mb_sac.replay_buffer import ReplayBuffer, Transition   # sizing transitions
from rag.memory_schema import RagMemoryItem                 # legacy memory entries
from rag.rag_knowledge_store import ...                     # legacy store (read paths only)
from llm.llm_client import call_llm                         # provider-neutral LLM transport
from graph.parse_netlist_to_graph import ...                # netlist → legacy graph
from graph.export_graph_to_netlist import ...               # legacy graph → netlist
from graph_search.graph_mcts import run_graph_mcts          # baseline MCTS (A/B reference)
from surrogate.surrogate_ensemble import SurrogateEnsemble  # sizing-outcome estimates
from controller.module_adapters import run_spice_validation # ngspice execution site
```

Cautions: several legacy modules read hard-coded `results/...` paths at import/call time and
some import torch at module import; adapters therefore import them inside functions, guarded
by availability checks, and treat every legacy artifact as read-only.

## 4. Assumptions recorded
- ngspice executable path is machine-specific and passed as an argument in legacy code; the new
  scaffold keeps it in YAML (`configs/spice.yaml`) with no default absolute path.
- The five-transistor OTA used by the mock generator is representative of the amplifier class the
  legacy checkpoints target (checkpoint 5x operates on op-amp topologies).
- Legacy `graph_type="bipartite_device_net"` is the canonical structural view; the new typed graph
  converts to the same bipartite form for NetworkX/WL-hashing, keeping the two worlds alignable.
