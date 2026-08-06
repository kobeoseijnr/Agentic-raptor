# Stage 3E.3 — base sft dpo comparison

Date: 2026-07-26

{
 "base": {
  "attempts": 2,
  "parseable": 0,
  "schema_valid": 0,
  "validator_pass": 0,
  "unsupported_block_halluc": 0
 },
 "sft": {
  "attempts": 2,
  "parseable": 2,
  "schema_valid": 2,
  "validator_pass": 2,
  "unsupported_block_halluc": 0
 },
 "sft_dpo": {
  "attempts": 2,
  "parseable": 2,
  "schema_valid": 2,
  "validator_pass": 2,
  "unsupported_block_halluc": 0
 }
}
Equal contexts/decoding/budget. SFT lifted structured validity 0/2 -> 2/2; DPO preserved 2/2 and adds preference alignment (held-out implicit-reward acc 1.0 x3 seeds). DPO>SFT on proposal quality NOT claimed.
