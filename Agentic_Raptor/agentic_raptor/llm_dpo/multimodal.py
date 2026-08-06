"""Stage 3E.4A: TRUE multimodal topology generator.

Replaces ONLY the topology-generation model: real schematic IMAGES pass through
the model's vision encoder (never converted to text), jointly with spec text,
RAG context, and serialized Functional/Device graphs. All downstream interfaces
(TopologyProposal schema, parser, validators, MCTS, MB-SAC, ranker, SPICE) are
unchanged.

Preferred model: Qwen/Qwen2.5-VL-3B-Instruct. Pilot default is the smallest
true VLM (SmolVLM-256M: SigLIP vision encoder + projector + causal LM) so the
mechanics run on this host; set AGENTIC_RAPTOR_VLM to use Qwen2.5-VL on GPU.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[2]
OM = _ROOT / "artifacts" / "stage3e4b"
IMG_DIR = _ROOT / "datasets" / "schematic_images_v2"
SCHEMA = "3e4b.1"

import hashlib
from dataclasses import dataclass, field


@dataclass
class MultimodalTopologyContext:
    """Typed joint context: pixels are ALWAYS loaded and processed by the
    model's native image processor - never OCR'd, captioned or described."""
    target_id: str
    gain_db: float
    ugbw_hz: float
    pm_deg: float
    supply_v: float
    load_f: float
    process: str
    allowed_blocks: tuple[str, ...]
    rag_success_ids: tuple[str, ...]
    rag_failure_ids: tuple[str, ...]
    functional_graph: str          # FunctionalStageGraph serialization
    device_graph: str              # DeviceCircuitGraph serialization
    image_path: str
    image_sha256: str
    graph_hashes: tuple[str, ...]
    source_family: str
    slew_rate: float | None = None
    power_w: float | None = None
    schema_version: str = SCHEMA

    def prompt(self) -> str:
        return (f"SPEC gain>={self.gain_db}dB ugbw>={self.ugbw_hz:.0e} "
                f"pm>={self.pm_deg}deg vdd={self.supply_v} cl={self.load_f} "
                f"proc={self.process}\nRAG +{','.join(self.rag_success_ids)} "
                f"-{','.join(self.rag_failure_ids)}\nFSG {self.functional_graph}\n"
                f"DCG {self.device_graph}\nBLOCKS {','.join(self.allowed_blocks)}\n"
                f"Output canonical TopologyProposal JSON only.")
VLM_ID = os.environ.get("AGENTIC_RAPTOR_VLM", "Qwen/Qwen3-VL-4B-Instruct")
PREFERRED_GPU_VLM = "Qwen/Qwen3-VL-4B-Instruct"


# ------------------------- schematic image rendering -------------------------
def render_schematic(g_dev, path: Path, size: int = 384) -> dict[str, Any]:
    """Deterministic schematic image from a DeviceCircuitGraph: device symbols
    as labelled boxes on rails, nets as routed lines. A real raster image for
    the vision encoder - NOT text."""
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    d.line([(10, 20), (size - 10, 20)], fill="red", width=3)          # vdda rail
    d.line([(10, size - 20), (size - 10, size - 20)], fill="black", width=3)  # gnda
    nets = sorted({n for dev in g_dev.devices for n in dev.nets.values()
                   if n not in ("vdda", "gnda")})
    ny = {n: 50 + i * (size - 100) // max(1, len(nets)) for i, n in enumerate(nets)}
    for n, y in ny.items():
        d.line([(30, y), (size - 30, y)], fill="lightgray")
        d.text((4, y - 6), n[:6], fill="gray")
    color = {"nmos": "blue", "pmos": "purple", "cap": "green", "res": "orange",
             "isrc": "brown"}
    for i, dev in enumerate(sorted(g_dev.devices, key=lambda x: x.device_id)):
        x = 40 + (i * (size - 80)) // max(1, len(g_dev.devices))
        ys = [20 if n == "vdda" else size - 20 if n == "gnda" else ny.get(n, size // 2)
              for n in dev.nets.values()]
        y0 = sum(ys) // len(ys)
        d.rectangle([x - 12, y0 - 12, x + 12, y0 + 12],
                    outline=color.get(dev.kind, "black"), width=2)
        d.text((x - 11, y0 - 6), dev.device_id[:4], fill="black")
        for y in ys:
            d.line([(x, y0), (x, y)], fill=color.get(dev.kind, "black"))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return {"path": str(path), "size": size, "devices": len(g_dev.devices)}


def build_image_dataset(limit: int = 20) -> dict[str, Any]:
    """image <-> device-graph <-> topology mapping for verified topologies
    (registry A1/A2 + edit-derived children)."""
    from agentic_raptor.corpus import TopologyRegistry
    from agentic_raptor.mb_sac import load_pools
    from agentic_raptor.mb_sac.stage3d2 import V3
    from agentic_raptor.topology_rl.stage3e2 import map_root, new_costs
    from agentic_raptor.topology_rl.stage3e2_edits import apply_edit, device_graph_hash

    reg = TopologyRegistry(V3)
    pools = load_pools()
    rows = []
    for r in (pools["A1"] + pools["A2"])[:limit]:
        tid = r["topology_id"]
        try:
            g = map_root(reg, tid, new_costs())
        except Exception as exc:
            rows.append({"topology_id": tid, "status": f"map_failed:{exc}"})
            continue
        info = render_schematic(g, IMG_DIR / f"{tid}.png")
        rows.append({"topology_id": tid, "source": "registry",
                     "device_hash": device_graph_hash(g),
                     "graph_ref": f"registry:{tid}", **info})
    # edit-derived children (verified structural edits)
    base = map_root(reg, "topology_v2_0001", new_costs())
    for et in ("ADD_VERIFIED_STAGE", "ADD_SUPPORTED_OUTPUT_STAGE"):
        ng, audit = apply_edit(base, et)
        info = render_schematic(ng, IMG_DIR / f"edit_{et.lower()[:12]}.png")
        rows.append({"topology_id": f"edit:{et}", "source": "edit_derived",
                     "device_hash": audit["child_hash"],
                     "graph_ref": f"edit:{audit['child_hash'][:12]}", **info})
    OM.mkdir(parents=True, exist_ok=True)
    (IMG_DIR / "MAPPING.json").write_text(json.dumps(
        {"rows": rows, "schema_version": SCHEMA}, indent=0), encoding="utf-8")
    ok = [r for r in rows if "path" in r]
    return {"images": len(ok), "failed": len(rows) - len(ok), "dir": str(IMG_DIR)}


# ------------------------- multimodal model provider -------------------------
def load_vlm(lora: bool = True, seed: int = 0):
    """Load VLM + processor; freeze base and vision encoder; LoRA on the
    language backbone (projector optionally trainable)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(seed)
    proc = AutoProcessor.from_pretrained(VLM_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        VLM_ID, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    record = {"model_id": VLM_ID, "revision": "main",
              "preferred_gpu_model": PREFERRED_GPU_VLM,
              "parameters": sum(p.numel() for p in model.parameters()),
              "vision_encoder": type(model.model.vision_model).__name__
              if hasattr(model.model, "vision_model") else "visual",
              "cuda": torch.cuda.is_available(), "schema_version": SCHEMA}
    for p in model.parameters():
        p.requires_grad_(False)                     # freeze base
    vis = getattr(model.model, "vision_model", None) or getattr(model, "visual", None)
    vis_params = set()
    if vis is not None:
        for p in vis.parameters():
            vis_params.add(id(p))                   # vision encoder stays frozen
    if lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                         target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        for n, p in model.named_parameters():       # LoRA only on language side
            if p.requires_grad and ("vision" in n or "visual" in n):
                p.requires_grad_(False)
    record["trainable_params"] = sum(p.numel() for p in model.parameters()
                                     if p.requires_grad)
    record["vision_frozen"] = all(not p.requires_grad for p in
                                  (vis.parameters() if vis is not None else []))
    return proc, model, record


def multimodal_inputs(proc, prompt: str, image_path: Path, model=None):
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=text, images=[img], return_tensors="pt")
    if model is not None:
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
    return inputs


def mm_seq_logprob(model, proc, prompt: str, image_path: Path, response: str):
    """Response-only token log prob with the IMAGE through the vision encoder."""
    import torch
    inputs = multimodal_inputs(proc, prompt, image_path, model)
    resp_ids = proc.tokenizer(response, return_tensors="pt").input_ids.to(
        inputs["input_ids"].device)
    ids = torch.cat([inputs["input_ids"], resp_ids], dim=1)
    extra = {}
    n_prompt = inputs["input_ids"].shape[1]
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        # per-token tensors (e.g. mm_token_type_ids) must extend to cover the
        # appended response tokens — response tokens are text type (0)
        if hasattr(v, "dim") and v.dim() == 2 and v.shape[1] == n_prompt:
            pad = torch.zeros((v.shape[0], resp_ids.shape[1]),
                              dtype=v.dtype, device=v.device)
            v = torch.cat([v, pad], dim=1)
        extra[k] = v
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra).logits
    n_p = inputs["input_ids"].shape[1]
    n_r = resp_ids.shape[1]
    lsm = torch.log_softmax(out[0, n_p - 1: n_p - 1 + n_r], dim=-1)
    return lsm[torch.arange(n_r), ids[0, n_p: n_p + n_r]].sum(), n_r


def run_pilot(sft_steps: int = 8, dpo_steps: int = 4) -> dict[str, Any]:
    """Mechanics pilot: real image -> vision encoder -> joint generation;
    LoRA SFT steps; multimodal DPO steps with frozen reference."""
    import torch
    t0 = time.time()
    ds = build_image_dataset(limit=4)
    mapping = json.loads((IMG_DIR / "MAPPING.json").read_text())["rows"]
    rows = [r for r in mapping if "path" in r][:3]
    proc, model, record = load_vlm(lora=True)
    from agentic_raptor.llm_dpo import proposal_to_text
    prompt = ("SPEC gain>=60dB pm>=45deg sky130. RAG rag_l2. "
              "GRAPH stages=2 comp=miller. Output canonical TopologyProposal JSON.")
    target = proposal_to_text(2, True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    sft_losses = []
    for s in range(sft_steps):
        r = rows[s % len(rows)]
        lp, n = mm_seq_logprob(model, proc, prompt, Path(r["path"]), target)
        loss = -lp / n
        opt.zero_grad(); loss.backward(); opt.step()
        sft_losses.append(round(float(loss.detach()), 3))
    # multimodal DPO: same image+text context, frozen reference
    import copy
    reference = copy.deepcopy(model)
    for p in reference.parameters():
        p.requires_grad_(False)
    ref_ck = float(sum(p.abs().sum() for p in reference.parameters()))
    rej = proposal_to_text(2, True).replace("cs_gain_stage", "quantum_stage")
    dpo_losses = []
    for s in range(dpo_steps):
        r = rows[s % len(rows)]
        img = Path(r["path"])
        lp_p, _ = mm_seq_logprob(model, proc, prompt, img, target)
        lp_r, _ = mm_seq_logprob(model, proc, prompt, img, rej)
        with torch.no_grad():
            lr_p, _ = mm_seq_logprob(reference, proc, prompt, img, target)
            lr_r, _ = mm_seq_logprob(reference, proc, prompt, img, rej)
        margin = (lp_p - lr_p) - (lp_r - lr_r)
        loss = -torch.nn.functional.logsigmoid(0.1 * margin)
        opt.zero_grad(); loss.backward(); opt.step()
        dpo_losses.append(round(float(loss.detach()), 4))
    ref_ok = float(sum(p.abs().sum() for p in reference.parameters())) == ref_ck
    # ---- multimodality proof: vision hooks + image ablation + mismatch ----
    shapes = {}
    vis = getattr(model.base_model.model.model, "vision_model", None) if hasattr(
        model, "base_model") else None
    handles = []
    if vis is not None:
        handles.append(vis.register_forward_hook(
            lambda m, i, o: shapes.__setitem__(
                "visual_embedding", list(getattr(o, "last_hidden_state", o[0]).shape))))
    img0 = Path(rows[0]["path"])
    inp_ok = multimodal_inputs(proc, prompt, img0, model)
    blank = OM / "blank.png"
    Image.new("RGB", (384, 384), "white").save(blank)
    with torch.no_grad():
        lg_ok = model(**{k: v for k, v in inp_ok.items()}).logits[0, -1]
        inp_bl = multimodal_inputs(proc, prompt, blank, model)
        lg_bl = model(**{k: v for k, v in inp_bl.items()}).logits[0, -1]
        inp_mm = multimodal_inputs(proc, prompt, Path(rows[-1]["path"]), model)
        lg_mm = model(**{k: v for k, v in inp_mm.items()}).logits[0, -1]
    for h in handles:
        h.remove()
    evidence = {
        "image_tensor_shape": list(inp_ok["pixel_values"].shape),
        "image_dtype": str(inp_ok["pixel_values"].dtype),
        "visual_embedding_shape": shapes.get("visual_embedding", "hook_point_varies"),
        "image_sha256": hashlib.sha256(img0.read_bytes()).hexdigest()[:16],
        "logit_delta_vs_ablated": round(float((lg_ok - lg_bl).abs().mean()), 5),
        "logit_delta_vs_mismatched": round(float((lg_ok - lg_mm).abs().mean()), 5),
        "image_conditioning_measurable": bool(
            float((lg_ok - lg_bl).abs().mean()) > 0
            and float((lg_ok - lg_mm).abs().mean()) > 0),
        "lora_grads_language_only": all(
            "vision" not in n and "visual" not in n
            for n, p in model.named_parameters() if p.requires_grad),
    }
    # image-conditioned generation through the SAME validation path
    inputs = multimodal_inputs(proc, prompt, Path(rows[0]["path"]), model)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=120,
                             do_sample=False)
    text = proc.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
    from agentic_raptor.llm_dpo import parse_proposal_text, proposal_dict_valid
    obj = parse_proposal_text(text)
    valid = bool(obj) and proposal_dict_valid(obj)[0]
    out = {"model_record": record, "image_dataset": ds,
           "multimodality_evidence": evidence,
           "primary_label": ("PRIMARY-GPU" if torch.cuda.is_available()
                             else "NON-PRIMARY CPU smoke (interface validation only; "
                                  "primary campaign requires the GPU VLM)"),
           "pixel_values_shape": list(inputs["pixel_values"].shape),
           "sft_losses": sft_losses, "dpo_losses": dpo_losses,
           "reference_unchanged": ref_ok,
           "generation_parseable": obj is not None, "generation_valid": valid,
           "generation_sample": text[:200],
           "wall_clock_s": round(time.time() - t0, 1), "schema_version": SCHEMA}
    (OM / "PILOT.json").write_text(json.dumps(out, indent=1, default=str),
                                   encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run_pilot(), indent=1, default=str))
