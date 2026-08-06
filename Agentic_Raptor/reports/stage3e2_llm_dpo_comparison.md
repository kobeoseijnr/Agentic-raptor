# Stage 3E.2 LLM-DPO — dpo comparison

Date: 2026-07-26

Equal decode budget (temp 0.7, top-p 0.9, seed 0): {
 "base": {
  "attempts": 2,
  "parseable": 0,
  "schema_valid": 0,
  "validator_pass": 0
 },
 "sft": {
  "attempts": 2,
  "parseable": 0,
  "schema_valid": 0,
  "validator_pass": 0
 },
 "sft_dpo": {
  "attempts": 2,
  "parseable": 0,
  "schema_valid": 0,
  "validator_pass": 0
 }
}
Structured-validity is 0 at this CPU-bounded scale for ALL variants — no proposal-quality improvement is claimed from DPO loss alone; the measurable DPO effect is held-out implicit-reward accuracy: {"0": 1.0, "1": 1.0, "2": 1.0}.
