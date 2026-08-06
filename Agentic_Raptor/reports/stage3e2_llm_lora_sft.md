# Stage 3E.2 LLM-DPO — lora sft

Date: 2026-07-26

{
 "lora": {
  "r": 8,
  "alpha": 16,
  "dropout": 0.05,
  "modules": [
   "c_attn"
  ]
 },
 "trainable_params": 147456,
 "lr": 0.0002,
 "optimiser": "AdamW",
 "batch": 1,
 "grad_accum": 1,
 "seq_len": 512,
 "precision": "fp32",
 "steps": 40,
 "seed": 0,
 "hardware": "cpu",
 "loss_first_last": [
  3.624,
  2.269
 ],
 "generation_eval": {
  "attempts": 2,
  "parseable": 0,
  "schema_valid": 0,
  "validator_pass": 0
 },
 "checkpoint": "C:\\Users\\kobeo\\OneDrive\\Desktop\\raptor1\\Agentic_Raptor\\artifacts\\stage3e2_llm\\sft_adapter",
 "wall_clock_s": 29.8,
 "dataset": {
  "train": 14,
  "validation": 2
 },
 "dpo_gate_note": "structured-output reliability at this CPU-bounded scale is below the primary-DPO gate; DPO below is MECHANICS validation on curated pairs \u2014 GPU-scale SFT deferred"
}
