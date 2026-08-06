"""Qwen DPO v2: shuffled mini-batches, per-epoch loss curve, frozen reference,
then the metric that matters — spec-structure-match rate on held-out contexts,
SFT vs SFT+DPO, with a bootstrap CI over contexts.

Run from Agentic_Raptor:  python run_dpo_qwen.py
(with AGENTIC_RAPTOR_TOPOLOGY_LLM='Qwen/Qwen2.5-3B-Instruct')
"""
import json
import random
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from agentic_raptor.llm_dpo import MODEL_ID, OUT, seq_logprob

O4 = Path("artifacts/stage3e4")
EPOCHS, BATCH, LR, BETA, SEED = 8, 4, 5e-5, 0.1, 0
random.seed(SEED)
torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token


def load(adapter, trainable):
    # frozen reference goes FULLY to CPU: two 4B models don't fit 12GB VRAM,
    # and partial offload triggers a peft adapter-loading bug
    dm = "auto" if trainable else {"": "cpu"}
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                             device_map=dm)
    return PeftModel.from_pretrained(m, adapter, is_trainable=trainable)


policy = load(str(O4 / "sft_adapter"), True)
ref = load(str(O4 / "sft_adapter"), False)
for p in ref.parameters():
    p.requires_grad_(False)
ref_ck = float(sum(p.abs().sum().float() for p in ref.parameters()))

pairs = [json.loads(x) for x in
         Path("datasets/llm_preference_queue/v2_mechanical.jsonl").read_text().splitlines()
         if x.strip()]
pairs += [p for p in json.loads((OUT / "preference_pairs.json").read_text())["pairs"]
          if p["split"] == "train"]
# spec-conditioned only: a promptless pair would train an unconditional
# structure preference (the fallback-prompt bug that collapsed the exam)
pairs = [p for p in pairs if p.get("prompt")]
print(f"pairs: {len(pairs)}  epochs: {EPOCHS}  batch: {BATCH}")

opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)
epoch_losses = []
for epoch in range(EPOCHS):
    random.shuffle(pairs)
    losses = []
    for i in range(0, len(pairs), BATCH):
        batch = pairs[i:i + BATCH]
        loss = 0.0
        for pr in batch:
            prompt = pr["prompt"]
            lp_p, _ = seq_logprob(policy, tok, prompt, pr["preferred"])
            lp_r, _ = seq_logprob(policy, tok, prompt, pr["rejected"])
            with torch.no_grad():
                lr_p, _ = seq_logprob(ref, tok, prompt, pr["preferred"])
                lr_r, _ = seq_logprob(ref, tok, prompt, pr["rejected"])
            lr_p, lr_r = lr_p.to(lp_p.device), lr_r.to(lp_r.device)
            loss = loss - pr.get("confidence", 0.9) * torch.nn.functional.logsigmoid(
                BETA * ((lp_p - lr_p) - (lp_r - lr_r)))
        loss = loss / len(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    epoch_losses.append(round(sum(losses) / len(losses), 4))
    print(f"epoch {epoch + 1}/{EPOCHS} mean loss {epoch_losses[-1]}")

policy.save_pretrained(str(O4 / "dpo_adapter_qwen"))
ref_ok = float(sum(p.abs().sum().float() for p in ref.parameters())) == ref_ck
del policy, ref
torch.cuda.empty_cache()

# ---- the metric that matters: spec-structure match on held-out contexts ----
from agentic_raptor.llm_dpo.stage3e4 import run_diversity_campaign, bootstrap_ci


def match_rates(div):
    """Per-context rate of candidates whose structure matches the
    spec-appropriate target structure for that context."""
    rates = []
    for row in div["rows"]:
        cands = [c for c in row["candidates"] if c["valid"]]
        if cands:
            rates.append(sum(c["graph_hash"] == row["target_variant"]
                             for c in cands) / len(cands))
    return rates


d_dpo = run_diversity_campaign(str(O4 / "dpo_adapter_qwen"), k=3, label="cmp_dpo_v2")
d_sft = json.loads((O4 / "diversity_sft.json").read_text())
r_dpo, r_sft = match_rates(d_dpo), match_rates(d_sft)
out = {"epoch_mean_losses": epoch_losses, "pairs_used": len(pairs),
       "reference_unchanged": ref_ok,
       "dpo": {"valid_rate": d_dpo["valid"] / d_dpo["generated"],
               "unique_valid": d_dpo["unique_valid_canonical"],
               "spec_match_rate": round(sum(r_dpo) / max(1, len(r_dpo)), 3),
               "spec_match_ci95": bootstrap_ci(r_dpo)},
       "sft": {"spec_match_rate": round(sum(r_sft) / max(1, len(r_sft)), 3),
               "spec_match_ci95": bootstrap_ci(r_sft)},
       "note": "DPO helps only if dpo.spec_match_ci95 clears sft.spec_match_ci95"}
(O4 / "dpo_qwen_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
