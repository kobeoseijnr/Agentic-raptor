"""Genuine LoRA mechanics on plain torch + adapter manager + acceptance gate.

IMPORTANT (inspection finding, docs/STAGE2_IMPLEMENTATION_REPORT.md): the
deployed topology generator is API-backed (no local weights), so LoRA cannot
attach to it. This module implements the real pipeline against a *pluggable
local decoder interface*: `inject_lora` wraps any torch model's named Linear
layers (default: attention q/v projections by name pattern) with frozen base
weights + trainable low-rank A/B adapters. `ToyTopologyDecoder` is the
executable stand-in used by tests; a HF/PEFT model drops in when adopted.
One-time training only — no per-episode continual fine-tuning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_raptor.utils.seeding import apply_torch_omp_workaround


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")   # optional: k_proj, o_proj, ff, projector
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    batch_size: int = 4
    gradient_accumulation: int = 1
    epochs: int = 3
    max_sequence_length: int = 512
    mixed_precision: bool = False
    gradient_clip: float = 1.0
    seed: int = 0
    early_stopping_patience: int = 3


def build_lora_linear(base: Any, rank: int, alpha: float, dropout: float) -> Any:
    apply_torch_omp_workaround()
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)          # frozen base
            self.lora_a = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
            self.scaling = alpha / rank
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) + self.drop(x) @ self.lora_a.T @ self.lora_b.T * self.scaling

    return LoRALinear()


def inject_lora(model: Any, config: LoRAConfig) -> list[str]:
    """Freeze the whole model, then wrap matching Linear submodules. Returns wrapped names."""
    apply_torch_omp_workaround()
    import torch.nn as nn

    for p in model.parameters():
        p.requires_grad_(False)
    wrapped: list[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t in full for t in config.target_modules):
                setattr(module, child_name, build_lora_linear(child, config.rank, config.alpha, config.dropout))
                wrapped.append(full)
    return wrapped


def lora_parameters(model: Any):
    return [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def lora_state_dict(model: Any) -> dict[str, Any]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


def load_lora_state_dict(model: Any, state: dict[str, Any]) -> None:
    apply_torch_omp_workaround()
    import torch

    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


def clear_lora(model: Any) -> None:
    """Zero the B matrices → adapter contributes nothing (base behaviour restored)."""
    apply_torch_omp_workaround()
    import torch

    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_b" in name:
                p.zero_()


# ---------------------------------------------------------------------------
# Executable stand-in decoder + one-time training run
# ---------------------------------------------------------------------------
def build_toy_topology_decoder(vocab_size: int = 128, dim: int = 32):
    """Tiny char-level decoder with q_proj/v_proj names (LoRA target stand-in)."""
    apply_torch_omp_workaround()
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.o_proj = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf")), diagonal=1)
            attn = torch.softmax(q @ k.transpose(-2, -1) / dim**0.5 + mask, dim=-1)
            return x + self.o_proj(attn @ v)

    class ToyTopologyDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.block = Block()
            self.head = nn.Linear(dim, vocab_size)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.head(self.block(self.embed(tokens)))

    return ToyTopologyDecoder()


def train_lora(model: Any, texts: list[tuple[str, str]], config: LoRAConfig) -> dict[str, float]:
    """One teacher-forced supervised LoRA run (adapter-only gradients)."""
    apply_torch_omp_workaround()
    import torch

    torch.manual_seed(config.seed)
    params = lora_parameters(model)
    if not params:
        raise ValueError("no trainable LoRA parameters; call inject_lora first")
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    sequences = []
    for prompt, target in texts:
        ids = [min(ord(c), 127) for c in (prompt + "\x1f" + target)][: config.max_sequence_length]
        if len(ids) > 1:
            sequences.append(torch.tensor(ids))
    last = 0.0
    for _epoch in range(config.epochs):
        for seq in sequences:
            tokens = seq.unsqueeze(0)
            logits = model(tokens[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            last = float(loss.detach().item())
    return {"final_loss": last, "examples": float(len(sequences))}


# ---------------------------------------------------------------------------
# Adapter manager + acceptance gate
# ---------------------------------------------------------------------------
class AdapterManager:
    """Versioned save/load/switch/rollback of LoRA adapters (never merges into base)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, name: str, model: Any, config: LoRAConfig, manifest: dict[str, Any]) -> Path:
        apply_torch_omp_workaround()
        import torch

        adapter_dir = self.root / name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state_dict(model), adapter_dir / "adapter.pt")
        (adapter_dir / "manifest.json").write_text(
            json.dumps({"adapter": name, "lora_config": vars(config) | {"target_modules": list(config.target_modules)}, **manifest}, indent=2, default=str),
            encoding="utf-8",
        )
        return adapter_dir

    def load(self, name: str, model: Any) -> dict[str, Any]:
        apply_torch_omp_workaround()
        import torch

        adapter_dir = self.root / name
        load_lora_state_dict(model, torch.load(adapter_dir / "adapter.pt", weights_only=False))
        return json.loads((adapter_dir / "manifest.json").read_text(encoding="utf-8"))

    def versions(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "adapter.pt").is_file()) if self.root.is_dir() else []


@dataclass
class AcceptanceCriteria:
    min_validity_rate_delta: float = 0.0      # no regression in structural validity
    min_pass_rate_delta: float = 0.0          # improved final pass rate OR
    min_spice_efficiency_delta: float = 0.0   # improved SPICE efficiency
    max_diversity_drop: float = 0.1           # no significant diversity reduction
    required_seeds: int = 2                   # reproducibility across seeds


@dataclass
class EvalSummary:
    validity_rate: float
    final_pass_rate: float
    mean_spice_calls: float
    diversity: float
    seeds: int = 1
    extra: dict[str, float] = field(default_factory=dict)


def acceptance_gate(base: EvalSummary, adapter: EvalSummary, criteria: AcceptanceCriteria) -> tuple[bool, list[str]]:
    """Deploy only when held-out criteria hold; reasons explain any rejection."""
    reasons: list[str] = []
    if adapter.validity_rate < base.validity_rate + criteria.min_validity_rate_delta:
        reasons.append(f"validity regression: {adapter.validity_rate:.3f} < {base.validity_rate:.3f}")
    improved_pass = adapter.final_pass_rate > base.final_pass_rate + criteria.min_pass_rate_delta
    improved_eff = adapter.mean_spice_calls < base.mean_spice_calls - criteria.min_spice_efficiency_delta
    if not (improved_pass or improved_eff):
        reasons.append("no improvement in final pass rate or SPICE efficiency")
    if adapter.diversity < base.diversity - criteria.max_diversity_drop:
        reasons.append(f"diversity drop {base.diversity - adapter.diversity:.3f} exceeds allowance")
    if adapter.seeds < criteria.required_seeds:
        reasons.append(f"only {adapter.seeds} seed(s); {criteria.required_seeds} required")
    return (not reasons), reasons
