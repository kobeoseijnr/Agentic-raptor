# Stage 3E.3 — dpo training

Date: 2026-07-26

{
 "objective": "-w*logsigmoid(beta*((lp_pol(y+)-lp_ref(y+))-(lp_pol(y-)-lp_ref(y-))))",
 "beta": 0.1,
 "optimiser": "AdamW",
 "lr": 5e-05,
 "batch": 1,
 "grad_accum": 1,
 "steps": 12,
 "reference_free": false,
 "train_pairs": 13,
 "heldout_pairs": 5,
 "per_seed": {
  "0": {
   "acc": 1.0,
   "pol": true,
   "ref": true,
   "loss": [
    0.6948,
    0.3842
   ]
  },
  "1": {
   "acc": 1.0,
   "pol": true,
   "ref": true,
   "loss": [
    0.6986,
    0.3967
   ]
  },
  "2": {
   "acc": 1.0,
   "pol": true,
   "ref": true,
   "loss": [
    0.6595,
    0.3663
   ]
  }
 },
 "hardware": "cpu fp32",
 "wall_clock_s": 38.0,
 "schema_version": "3e2L.1"
}
