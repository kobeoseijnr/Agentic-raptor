"""One-time post-sizing LoRA fine-tuning pipeline for the topology generator.

Distinct from the pre-SPICE DPO ranker (agentic_raptor.dpo): DPO ranks
candidates before simulation; this module performs one supervised,
outcome-grounded LoRA run after enough verified episodes exist.
"""

from agentic_raptor.finetuning.attribution import Attribution, attribute
from agentic_raptor.finetuning.dataset_builder import (
    DatasetSplits,
    SFTExample,
    build_dataset,
    group_safe_split,
    serialize_example,
)
from agentic_raptor.finetuning.lora import (
    AcceptanceCriteria,
    AdapterManager,
    EvalSummary,
    LoRAConfig,
    acceptance_gate,
    build_toy_topology_decoder,
    clear_lora,
    inject_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    train_lora,
)
from agentic_raptor.finetuning.outcome_store import EpisodeOutcome, EpisodeOutcomeStore

__all__ = [
    "AcceptanceCriteria", "AdapterManager", "Attribution", "DatasetSplits",
    "EpisodeOutcome", "EpisodeOutcomeStore", "EvalSummary", "LoRAConfig",
    "SFTExample", "acceptance_gate", "attribute", "build_dataset",
    "build_toy_topology_decoder", "clear_lora", "group_safe_split",
    "inject_lora", "load_lora_state_dict", "lora_parameters",
    "lora_state_dict", "serialize_example", "train_lora",
]
