"""Topology generators: protocol, deterministic mock, and LLM-backed shell.

The mock builds a real, validator-clean five-transistor OTA so every
downstream stage (validation, MCTS editing, sizing, mock SPICE) operates on a
meaningful circuit. Candidate metadata (reasoning summary, memories used,
confidence) travels in ``CircuitGraph.metadata.extra``.
"""

from __future__ import annotations

from typing import Any, Protocol

from agentic_raptor.core.circuit_graph import (
    CircuitEdge,
    CircuitGraph,
    CircuitNode,
    TopologyMetadata,
)
from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.core.types import DeviceType, TerminalType
from agentic_raptor.rag.schemas import MemoryEntry
from agentic_raptor.specification.multimodal_input import MultimodalDesignInput
from agentic_raptor.topology_generation.output_parser import parse_generator_output
from agentic_raptor.topology_generation.prompt_builder import build_generation_prompt
from agentic_raptor.topology_validation.validator import TopologyValidator
from agentic_raptor.utils.exceptions import GenerationError

#: Environment variables for future multimodal LLM integration (no keys in code).
ENV_PROVIDER = "AGENTIC_RAPTOR_LLM_PROVIDER"        # "openai" | "anthropic" | "ollama" | ...
ENV_MODEL = "AGENTIC_RAPTOR_LLM_MODEL"
ENV_API_KEY = "AGENTIC_RAPTOR_LLM_API_KEY"          # read at call time, never stored
ENV_BASE_URL = "AGENTIC_RAPTOR_LLM_BASE_URL"


class TopologyGenerator(Protocol):
    def generate(
        self,
        specifications: DesignSpecifications,
        retrieved_examples: list[MemoryEntry],
        multimodal_context: MultimodalDesignInput | None,
        number_of_candidates: int,
    ) -> list[CircuitGraph]: ...


# ---------------------------------------------------------------------------
# Deterministic template: five-transistor OTA
# ---------------------------------------------------------------------------
def build_five_transistor_ota(graph_id: str, with_load_cap: bool = True) -> CircuitGraph:
    """Classic 5T OTA: NMOS diff pair, PMOS mirror load, NMOS tail, bias branch."""
    n = [
        CircuitNode("vdd", DeviceType.SUPPLY_PORT),
        CircuitNode("gnd", DeviceType.GROUND_PORT),
        CircuitNode("inp", DeviceType.INPUT_PORT),
        CircuitNode("inn", DeviceType.INPUT_PORT),
        CircuitNode("out", DeviceType.OUTPUT_PORT),
        CircuitNode("m1", DeviceType.NMOS, block_role="diff_pair"),
        CircuitNode("m2", DeviceType.NMOS, block_role="diff_pair"),
        CircuitNode("m3", DeviceType.PMOS, block_role="mirror_load"),
        CircuitNode("m4", DeviceType.PMOS, block_role="mirror_load"),
        CircuitNode("m5", DeviceType.NMOS, block_role="tail_source"),
        CircuitNode("ib1", DeviceType.CURRENT_SOURCE, block_role="bias_branch"),
    ]
    e: list[tuple[str, str, TerminalType, str]] = [
        ("e_vdd", "vdd", TerminalType.PORT, "n_vdd"),
        ("e_gnd", "gnd", TerminalType.PORT, "n_gnd"),
        ("e_inp", "inp", TerminalType.PORT, "n_inp"),
        ("e_inn", "inn", TerminalType.PORT, "n_inn"),
        ("e_out", "out", TerminalType.PORT, "n_out"),
        # Diff pair
        ("e_m1g", "m1", TerminalType.GATE, "n_inp"),
        ("e_m1d", "m1", TerminalType.DRAIN, "n_mirror"),
        ("e_m1s", "m1", TerminalType.SOURCE, "n_tail"),
        ("e_m1b", "m1", TerminalType.BULK, "n_gnd"),
        ("e_m2g", "m2", TerminalType.GATE, "n_inn"),
        ("e_m2d", "m2", TerminalType.DRAIN, "n_out"),
        ("e_m2s", "m2", TerminalType.SOURCE, "n_tail"),
        ("e_m2b", "m2", TerminalType.BULK, "n_gnd"),
        # PMOS mirror (m3 diode-connected)
        ("e_m3g", "m3", TerminalType.GATE, "n_mirror"),
        ("e_m3d", "m3", TerminalType.DRAIN, "n_mirror"),
        ("e_m3s", "m3", TerminalType.SOURCE, "n_vdd"),
        ("e_m3b", "m3", TerminalType.BULK, "n_vdd"),
        ("e_m4g", "m4", TerminalType.GATE, "n_mirror"),
        ("e_m4d", "m4", TerminalType.DRAIN, "n_out"),
        ("e_m4s", "m4", TerminalType.SOURCE, "n_vdd"),
        ("e_m4b", "m4", TerminalType.BULK, "n_vdd"),
        # Tail current source transistor + bias branch
        ("e_m5g", "m5", TerminalType.GATE, "n_bias"),
        ("e_m5d", "m5", TerminalType.DRAIN, "n_tail"),
        ("e_m5s", "m5", TerminalType.SOURCE, "n_gnd"),
        ("e_m5b", "m5", TerminalType.BULK, "n_gnd"),
        ("e_ib1p", "ib1", TerminalType.PLUS, "n_vdd"),
        ("e_ib1n", "ib1", TerminalType.MINUS, "n_bias"),
    ]
    nodes = list(n)
    edges = [CircuitEdge(eid, nid, term, net) for eid, nid, term, net in e]
    if with_load_cap:
        nodes.append(CircuitNode("cl", DeviceType.CAPACITOR, block_role="load"))
        edges.append(CircuitEdge("e_clp", "cl", TerminalType.PLUS, "n_out"))
        edges.append(CircuitEdge("e_cln", "cl", TerminalType.MINUS, "n_gnd"))
    # Tie n_bias to gnd through a diode-connected NMOS mirror to keep every
    # terminal referenced (m6 mirrors ib1 into m5).
    nodes.append(CircuitNode("m6", DeviceType.NMOS, block_role="bias_branch"))
    edges += [
        CircuitEdge("e_m6g", "m6", TerminalType.GATE, "n_bias"),
        CircuitEdge("e_m6d", "m6", TerminalType.DRAIN, "n_bias"),
        CircuitEdge("e_m6s", "m6", TerminalType.SOURCE, "n_gnd"),
        CircuitEdge("e_m6b", "m6", TerminalType.BULK, "n_gnd"),
    ]
    return CircuitGraph(
        graph_id=graph_id,
        nodes=nodes,
        edges=edges,
        metadata=TopologyMetadata(
            name="five_transistor_ota",
            description="NMOS-input 5T OTA with current mirror bias and optional load cap",
            circuit_family="ota",
            source="mock_generator",
        ),
    )


class MockTopologyGenerator:
    """Deterministic generator for tests and the smoke pipeline.

    Candidate 0 is the plain 5T OTA; later candidates add a small deterministic
    variation (output compensation cap) so multiple candidates differ.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def generate(
        self,
        specifications: DesignSpecifications,
        retrieved_examples: list[MemoryEntry],
        multimodal_context: MultimodalDesignInput | None,
        number_of_candidates: int,
    ) -> list[CircuitGraph]:
        graphs: list[CircuitGraph] = []
        for index in range(max(1, number_of_candidates)):
            graph = build_five_transistor_ota(f"mock-ota-s{self.seed}-c{index}")
            if index % 2 == 1:
                comp = CircuitNode(f"cc{index}", DeviceType.CAPACITOR, block_role="compensation")
                graph.add_node(comp)
                graph.add_edge(CircuitEdge(f"e_cc{index}p", comp.node_id, TerminalType.PLUS, "n_out"))
                graph.add_edge(CircuitEdge(f"e_cc{index}n", comp.node_id, TerminalType.MINUS, "n_gnd"))
            graph.metadata.extra.update(
                {
                    "reasoning_summary": (
                        "Deterministic 5T OTA template selected for "
                        f"{specifications.circuit_class}; variation index {index}."
                    ),
                    "confidence": 0.7 if index == 0 else 0.55,
                    "memories_used": [e.memory_id for e in retrieved_examples],
                    "generator": "mock",
                }
            )
            graphs.append(graph)
        return graphs


class LLMTopologyGenerator:
    """Real multimodal LLM generator over the provider-neutral interface.

    Stage 2: the injected ``model`` implements
    :class:`agentic_raptor.topology_generation.provider.MultimodalTopologyModel`
    (real OpenAI-compatible provider or a mock for tests). Schematic images and
    netlist context ride along; output must parse into typed CircuitGraphs and
    pass deterministic validation before it can enter topology RL. Rejected:
    prose-only responses, invalid JSON, unsupported device types, missing
    ports, malformed terminal references (all surfaced as GenerationError and
    fed back into the retry prompt).
    """

    def __init__(
        self,
        model: Any,
        validator: TopologyValidator | None = None,
        max_retries: int = 2,
        extra_context: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.validator = validator or TopologyValidator()
        self.max_retries = max_retries
        self.extra_context = dict(extra_context or {})

    def generate(
        self,
        specifications: DesignSpecifications,
        retrieved_examples: list[MemoryEntry],
        multimodal_context: MultimodalDesignInput | None,
        number_of_candidates: int,
    ) -> list[CircuitGraph]:
        import json as _json
        from pathlib import Path

        from agentic_raptor.topology_generation.multimodal_inputs import build_context_summary

        prompt = build_generation_prompt(
            specifications, retrieved_examples, multimodal_context, number_of_candidates
        )
        images: list[bytes] = []
        if multimodal_context and multimodal_context.schematic_image_path:
            image_path = Path(multimodal_context.schematic_image_path)
            if image_path.is_file():
                images.append(image_path.read_bytes())
        structured_context = {**build_context_summary(multimodal_context), **self.extra_context}

        last_error: Exception | None = None
        rejected_reasons: list[str] = []
        for attempt in range(1 + self.max_retries):
            attempt_prompt = prompt if attempt == 0 else (
                f"{prompt}\n\n## Previous attempt failed — fix and return corrected JSON only\n{last_error}"
            )
            try:
                raw = self.model.generate_structured(attempt_prompt, images, structured_context)
                graphs = parse_generator_output(_json.dumps(raw))
            except GenerationError as exc:
                last_error = exc
                rejected_reasons.append(str(exc))
                continue
            valid: list[CircuitGraph] = []
            #: warnings that make a graph un-emittable as a netlist — the LLM
            #: acceptance bar is "simulatable", stricter than plain validity.
            blocking_warnings = {"PARTIALLY_CONNECTED_DEVICE", "ISOLATED_SUBGRAPH", "OUTPUT_PATH_UNPOWERED"}
            for graph in graphs:
                validation = self.validator.validate(graph)
                blockers = [i for i in validation.issues if i.severity == "error" or i.code in blocking_warnings]
                if not blockers:
                    valid.append(graph)
                else:
                    rejected_reasons.append(
                        f"{graph.graph_id}: " + "; ".join(f"{i.code} {i.message}" for i in blockers[:6])
                    )
            if valid:
                for g in valid:
                    g.metadata.extra.setdefault("memories_used", [e.memory_id for e in retrieved_examples])
                    g.metadata.extra.setdefault("generator", "llm")
                    g.metadata.extra.setdefault("generation_attempt", attempt + 1)
                return valid
            last_error = GenerationError(
                f"all candidates failed deterministic validation: {rejected_reasons[-3:]}"
            )
        raise GenerationError(
            f"LLM generation failed after {self.max_retries + 1} attempts: {last_error}"
        )
