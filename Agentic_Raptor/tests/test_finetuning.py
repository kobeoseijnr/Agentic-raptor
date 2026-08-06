"""LoRA fine-tuning pipeline: freezing, attribution, splits, adapters, gate."""

from __future__ import annotations

import pytest

from agentic_raptor.finetuning import (
    AcceptanceCriteria,
    AdapterManager,
    EpisodeOutcome,
    EpisodeOutcomeStore,
    EvalSummary,
    LoRAConfig,
    acceptance_gate,
    attribute,
    build_dataset,
    build_toy_topology_decoder,
    clear_lora,
    group_safe_split,
    inject_lora,
    serialize_example,
    train_lora,
)


def _episode(eid="e1", passed=True, valid=True, edits=0, spec=None, margins=None) -> EpisodeOutcome:
    return EpisodeOutcome(
        episode_id=eid,
        specifications=spec or {"circuit_class": "ota", "technology": "g", "supply_voltage": 1.8},
        original_topology={"graph_id": "orig"},
        original_valid=valid,
        final_topology={"graph_id": "final"} if passed else None,
        constraint_margins=margins or {"gain_db": 0.2},
        pvt_pass_rate=1.0,
        passed=passed,
        n_mcts_edits=edits,
        spice_calls=3,
        validation_errors=[] if valid else ["MISSING_GROUND_PORT"],
    )


def test_attribution_categories():
    assert attribute(_episode(edits=0)).category == "A"
    assert attribute(_episode(edits=1)).category == "A"
    assert attribute(_episode(edits=5, valid=False)).category == "B"
    assert attribute(_episode(passed=False)).category == "C"


def test_heavily_repaired_not_labeled_original_success():
    heavy = attribute(_episode(edits=6, valid=True))
    assert heavy.category == "B", "heavy repair must never count as a direct success"
    examples = build_dataset([_episode(edits=6, valid=True)])
    assert all(e.kind == "corrected" for e in examples)


def test_failed_finals_never_targets_and_no_future_leakage():
    examples = build_dataset([_episode(passed=False), _episode(eid="ok")])
    assert len(examples) == 1 and examples[0].target == {"graph_id": "final"}
    prompt_text, _ = serialize_example(examples[0])
    for banned in ("spice_metrics", "constraint_margins", "pvt"):
        assert banned not in prompt_text, "post-simulation data must not enter prompts"


def test_group_safe_split():
    specs = [
        {"circuit_class": "ota", "technology": "g", "supply_voltage": v} for v in (1.2, 1.8, 3.3, 5.0)
    ]
    episodes = [_episode(eid=f"e{i}", spec=s) for i, s in enumerate(specs * 2)]
    splits = group_safe_split(build_dataset(episodes), seed=0)
    m = splits.manifest()
    assert not (set(m["train_groups"]) & set(m["test_groups"]))
    assert not (set(m["val_groups"]) & set(m["test_groups"]))


def test_base_frozen_and_only_lora_trains():
    model = build_toy_topology_decoder()
    wrapped = inject_lora(model, LoRAConfig(target_modules=("q_proj", "v_proj")))
    assert wrapped == ["block.q_proj", "block.v_proj"]
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable), "only LoRA params may train"
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert any("k_proj" in n for n in frozen) and any("embed" in n for n in frozen)


def test_training_changes_only_lora():

    model = build_toy_topology_decoder()
    inject_lora(model, LoRAConfig(epochs=2))
    base_before = sum(
        float(p.abs().sum()) for n, p in model.named_parameters() if "lora_" not in n
    )
    lora_before = parameter_checksum_named(model, "lora_")
    report = train_lora(model, [("spec: ota", '{"graph_id": "final"}')], LoRAConfig(epochs=2))
    assert report["examples"] == 1.0
    base_after = sum(
        float(p.abs().sum()) for n, p in model.named_parameters() if "lora_" not in n
    )
    assert base_after == pytest.approx(base_before), "base must stay frozen"
    assert parameter_checksum_named(model, "lora_") != lora_before


def parameter_checksum_named(model, key: str) -> float:
    return sum(float(p.abs().sum()) for n, p in model.named_parameters() if key in n)


def test_adapter_save_reload_and_disable(tmp_path):
    import torch

    model = build_toy_topology_decoder()
    inject_lora(model, LoRAConfig())
    train_lora(model, [("a", "b")], LoRAConfig(epochs=1))
    manager = AdapterManager(tmp_path)
    manager.save("topology_lora_v1", model, LoRAConfig(), {"dataset": "test"})
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        adapted = model(tokens).clone()
    clear_lora(model)
    with torch.no_grad():
        base_out = model(tokens).clone()
    assert not torch.allclose(adapted, base_out), "clearing adapter must change output"
    manifest = manager.load("topology_lora_v1", model)
    assert manifest["adapter"] == "topology_lora_v1"
    with torch.no_grad():
        assert torch.allclose(model(tokens), adapted), "reloaded adapter must reproduce output"
    assert manager.versions() == ["topology_lora_v1"]


def test_acceptance_gate_rejects_and_accepts():
    base = EvalSummary(validity_rate=0.8, final_pass_rate=0.4, mean_spice_calls=10, diversity=0.5, seeds=2)
    worse = EvalSummary(validity_rate=0.6, final_pass_rate=0.4, mean_spice_calls=10, diversity=0.2, seeds=1)
    ok, reasons = acceptance_gate(base, worse, AcceptanceCriteria())
    assert not ok and len(reasons) >= 3
    better = EvalSummary(validity_rate=0.85, final_pass_rate=0.55, mean_spice_calls=8, diversity=0.5, seeds=2)
    ok, reasons = acceptance_gate(base, better, AcceptanceCriteria())
    assert ok and not reasons


def test_outcome_store_roundtrip(tmp_path):
    store = EpisodeOutcomeStore(tmp_path / "episodes.jsonl")
    store.add(_episode())
    assert len(EpisodeOutcomeStore(tmp_path / "episodes.jsonl")) == 1
