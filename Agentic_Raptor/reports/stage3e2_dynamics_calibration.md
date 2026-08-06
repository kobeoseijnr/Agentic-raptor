# Stage 3E.2 — dynamics calibration

Date: 2026-07-26

{
 "pm_abs_error_deg_by_source": {
  "A1": 30.75,
  "A2": 14.31
 },
 "global_pm_error": 22.53,
 "ensemble_disagreement_deg": 17.17,
 "rollout_gates": {
  "global": false,
  "source_A1": false,
  "source_A2": true,
  "topology_level": "insufficient_data_gate_disabled"
 },
 "gate_rule": "enable model rollouts only if held-out one-step |pm error| < 15 deg; disabled contexts keep real-only learning (not simulator failures)",
 "surrogate_note": "same held-out split (seed 2); predictions never stored as verified measurements"
}
