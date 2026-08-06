# Stage 3E.2 LLM-DPO — dpo training

Date: 2026-07-26

{
 "objective": "-w*logsigmoid(beta*((lp_pol(y+)-lp_ref(y+))-(lp_pol(y-)-lp_ref(y-))))",
 "beta": 0.1,
 "optimiser": "AdamW",
 "lr": 5e-05,
 "batch": 1,
 "grad_accum": 1,
 "steps": 16,
 "reference_free": false,
 "train_pairs": 13,
 "heldout_pairs": 5,
 "per_seed": {
  "0": {
   "loss_first_last": [
    0.6706,
    0.3936
   ],
   "reward_margin_last": 7.293,
   "implicit_reward_accuracy_heldout": 1.0,
   "policy_params_changed": true,
   "reference_unchanged": true,
   "checkpoint": "C:\\Users\\kobeo\\OneDrive\\Desktop\\raptor1\\Agentic_Raptor\\artifacts\\stage3e2_llm\\dpo_adapter_seed0"
  },
  "1": {
   "loss_first_last": [
    0.7158,
    0.3953
   ],
   "reward_margin_last": 7.241,
   "implicit_reward_accuracy_heldout": 1.0,
   "policy_params_changed": true,
   "reference_unchanged": true,
   "checkpoint": "C:\\Users\\kobeo\\OneDrive\\Desktop\\raptor1\\Agentic_Raptor\\artifacts\\stage3e2_llm\\dpo_adapter_seed1"
  },
  "2": {
   "loss_first_last": [
    0.7009,
    0.3907
   ],
   "reward_margin_last": 7.382,
   "implicit_reward_accuracy_heldout": 1.0,
   "policy_params_changed": true,
   "reference_unchanged": true,
   "checkpoint": "C:\\Users\\kobeo\\OneDrive\\Desktop\\raptor1\\Agentic_Raptor\\artifacts\\stage3e2_llm\\dpo_adapter_seed2"
  }
 },
 "hardware": "cpu fp32",
 "wall_clock_s": 49.4,
 "schema_version": "3e2L.1"
}
