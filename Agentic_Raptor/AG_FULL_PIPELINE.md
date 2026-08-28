# AG-FULL — the Agentic RAPTOR full pipeline (verified 2026-08-27)

The single authoritative description of what executes on an AG_FULL run.
Source-verified against run_raptor_v2.py, agentic_raptor/agents/*, 
agentic_raptor/topology_rl/bandit_selector.py, agentic_raptor/mb_sac/spec_sizing.py.

**Definition**: AG_FULL = A0's exact frozen pipeline configuration + all four
agents (Design Planner, Topology Critic, Optimization Supervisor, Recovery),
with a BudgetLedger capping total spend at A0's own envelope — agents
reallocate, they never spend more (run_ablation_v3.py AGENTIC_ARMS).

```
SPEC (gain / UGBW / PM / CL targets)
 |
 |- Stage 1   Spec intake -- targets + C_LOAD resolution
 |
 |- AGENT: DESIGN PLANNER (agents/planner.py)
 |            spec -> StrategyPlan: difficulty tier, SOFT stage-count priors
 |            (TRAIN-measured ceilings, discourage-never-ban), sizing budget
 |            class, written rationale
 |
 |- Stage 2   RAG retrieval -- clean frozen provenance-verified memory
 |
 |- Stage 3   TOPOLOGY PROPOSAL -- SFT-finetuned LLM generates circuit graphs
 |            (exclusion-conditioned diversity ladder, 20 attempts)
 |     |- AGENT: TOPOLOGY CRITIC (agents/critic.py): critique -> re-prompt
 |            loop; validator canonicalises + dedups; a short set is REPORTED
 |            (below SELECT_K the run aborts) -- never enumeration-filled
 |
 |- Screen    Coordinator soft strategy screen (agents/coordinator.py):
 |            plan-discouraged candidates dropped ONLY if >= 2 preferred
 |            candidates exist
 |
 |- Stage 5   SELECTION -- LINEAR CONTEXTUAL BANDIT top-2
 |            (topology_rl/bandit_selector.py, promoted weights =
 |            BANDIT_TOP2_V2, SHA-256 hash-pinned, loader hard-fails on
 |            mismatch or feature drift):
 |            each proposal -> 24 physical+spec features
 |            (linear_value.physical_features_core: 13 structural, 5 spec
 |            context, 6 interactions) -> z-scored linear scoresheet ->
 |            deterministic rank -> top-2 DISTINCT topologies. 0 SPICE calls.
 |            V2 promotion gate (2026-08-16, spec-disjoint vs V1):
 |            top-2-contains-passing-family 10/10 vs 6/10, pairwise 80/88 vs
 |            41/88. (AlphaZero/MCTS modes = opt-in ablation arms only; they
 |            do NOT run in AG_FULL. Trace key "stage5_alphazero" is a
 |            historical schema name; its content records
 |            search_topology=bandit_top2_linear_scoresheet + weights sha.)
 |
 |- AGENT: OPTIMIZATION SUPERVISOR (agents/supervisor.py)
 |            probe both branches (PROBE_BUDGET=3 real SPICE calls each) ->
 |            probe_verdict from REAL measurements (improving/stalled/
 |            hopeless/pathological, trajectory-relative) -> allocate():
 |            commit the whole remainder to the better branch, ONE
 |            full-length run, no early stop, select_by="fom" (v4.2,
 |            campaign-validated); bank unspent calls. All spends flow
 |            through the BudgetLedger.
 |
 |- Stage 6   SIZING -- TRUE SOFT ACTOR-CRITIC RL, spec-conditioned
 |            (mb_sac/spec_sizing.py sac_size, driven per branch via
 |            run_raptor_v2._size_one_branch):
 |              * tanh-squashed Gaussian actor (mean + log-std,
 |                reparameterized, tanh log-prob correction)
 |              * twin Q-critics + Polyak-averaged frozen target critics
 |                (tau=0.005)
 |              * entropy regularization with auto-tuned temperature
 |                (log_alpha -> target_entropy = -|A|)
 |              * off-policy replay over the episode buffer, bootstrapped
 |                Bellman targets y = r + gamma(1-done)(min Q_t - alpha logpi)
 |              * every transition = one real ngspice measurement; stored
 |                actions must decode back to the exact knobs simulated
 |                (hard invariant); parameter checksums prove training
 |            + Stage 7 spec-conditioned surrogate screening
 |            + Stage 8 DPO/Bradley-Terry ranker trained on measured outcomes
 |
 |- Stage 9   FRESH AUTHORITATIVE VERIFICATION -- a NEW ngspice call on the
 |            returned design (pair-provenance checked)
 |     |- AGENT: RECOVERY (agents/recovery.py): ONLY after a failed
 |            verification -- one bounded re-size of the backup branch,
 |            funded from BANKED calls only (>= 4), at most once per run;
 |            honest final answer either way
 |
 |- Stage 11  FEEDBACK -- the verified A/B result routed into the
 |            self-improvement streams (accept-or-rollback gates)
 v
ANSWER: one verified design + full audit trace (plan, probe verdicts,
        allocation, ledger entries, bandit ranking + weights hash,
        SAC transitions, recovery log)
```

**Tier-3 measured result of this exact system** (HELDOUT29, 3 seeds):
pass 69/87 (79%) vs A0's 87/87; FoM median 13,665 vs 4,921 (2.8x, paired
76/11 wins, p<0.001); fastest arm wall-clock (182 s/run).

**Fairness invariant**: B_spice(AG_FULL) <= B_spice(A0), enforced by the
ledger; banking never creates budget (calls are charged at allocation and
drawn without a second charge).
