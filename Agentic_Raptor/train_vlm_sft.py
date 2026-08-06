"""Train the multimodal (vision) topology model properly: 300 SFT steps on
(spec text + real schematic IMAGE -> circuit JSON), then check the validity
gate. Saves the adapter the full pipeline uses in multimodal mode.

Run:  python train_vlm_sft.py        (~30-40 min on GPU)
"""
import json
import random
import torch
from pathlib import Path

from agentic_raptor.llm_dpo import parse_proposal_text, proposal_dict_valid
from agentic_raptor.llm_dpo.multimodal import (IMG_DIR, build_image_dataset,
                                               load_vlm, mm_seq_logprob,
                                               multimodal_inputs)

OUT = Path("artifacts/stage3e4b")
STEPS = 300
build_image_dataset(limit=20)          # render schematics for A1+A2 families
corpus = json.loads(Path("artifacts/stage3e4/corpus.json").read_text())
train = [r for r in corpus["records"] if r["split"] == "train"
         and (IMG_DIR / f"{r['topology_id']}.png").is_file()]
print(f"train records with images: {len(train)}")

proc, model, rec = load_vlm(lora=True, seed=0)
print("model:", rec["model_id"], "| trainable:", rec["trainable_params"],
      "| vision frozen:", rec["vision_frozen"])
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=2e-4)
rng = random.Random(0)
losses = []
for step in range(STEPS):
    r = train[rng.randrange(len(train))]
    img = IMG_DIR / f"{r['topology_id']}.png"
    lp, n = mm_seq_logprob(model, proc, r["prompt"], img, r["response"])
    loss = -lp / n
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
    if (step + 1) % 25 == 0:
        print(f"step {step+1}/{STEPS} loss {sum(losses[-25:])/25:.3f}")

model.save_pretrained(str(OUT / "vlm_sft_adapter"))

# validity gate: greedy generation on 4 held-out contexts WITH their images
held = [r for r in corpus["records"] if r["split"] == "heldout"
        and (IMG_DIR / f"{r['topology_id']}.png").is_file()][:4]
ok = 0
for r in held:
    inputs = multimodal_inputs(proc, r["prompt"],
                               IMG_DIR / f"{r['topology_id']}.png", model)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=260, do_sample=False)
    text = proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
    obj = parse_proposal_text(text)
    valid = bool(obj) and proposal_dict_valid(obj)[0]
    ok += int(valid)
    print("heldout:", r["context_id"][:30], "valid:", valid)
result = {"steps": STEPS, "loss_first_last": [round(losses[0], 3),
                                              round(losses[-1], 3)],
          "heldout_valid": f"{ok}/{len(held)}",
          "gate_passed": ok == len(held) and len(held) > 0,
          "adapter": str(OUT / "vlm_sft_adapter")}
(OUT / "vlm_sft_result.json").write_text(json.dumps(result, indent=1),
                                         encoding="utf-8")
print(json.dumps(result, indent=1))
