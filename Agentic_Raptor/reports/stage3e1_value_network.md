# Stage 3E.1 — value network

Date: 2026-07-26

Value head: context(24) -> MLP(64) -> tanh scalar (ordinal) + aux4: feasibility_logit_UNCALIBRATED, stability_logit_UNCALIBRATED, expected_spice_cost, budget_exhaustion_logit. Scalar target derives from SPICE-backed PostSizingTopologyScore; full component vector preserved in nodes and training records. No auxiliary output is called a calibrated probability.
