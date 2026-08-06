# Stage 3E.1 — mcts

Date: 2026-07-26

PUCT: Q + c_puct*P*sqrt(N_parent)/(1+N_child), c_puct=1.5, deterministic tie-break by node_id. Selection/expansion(validator-gated)/evaluation(value|SPICE hybrid)/backup. Root Dirichlet noise alpha=0.4 eps=0.25 training-only (tested off in deterministic mode). Terminal reasons: terminate_action, no_legal_actions, max_depth, search/spice_budget_exhausted, acceptance_met, unsupported. Smoke tree: 33 nodes, depth<=2, 8 simulations, root dist {'a_keep': 0.125, 'a_sel_topology_0003': 0.125, 'a_sel_topology_0005': 0.25, 'a_sel_topology_0008': 0.125, 'a_sel_topology_0009': 0.125, 'a_term': 0.25}, PV ['a_sel_topology_0005', 'a_keep'].
