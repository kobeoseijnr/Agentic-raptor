"""Prompt construction for multimodal LLM topology generation."""

from __future__ import annotations

import json

from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.rag.schemas import MemoryEntry
from agentic_raptor.specification.multimodal_input import MultimodalDesignInput
from agentic_raptor.topology_generation.multimodal_inputs import build_context_summary
from agentic_raptor.topology_generation.output_parser import TOPOLOGY_JSON_SCHEMA

_SYSTEM_ROLE = (
    "You are an analog circuit topology designer. You output ONLY structured JSON "
    "circuit graphs conforming to the provided schema — no free-form prose."
)


def build_generation_prompt(
    spec: DesignSpecifications,
    retrieved: list[MemoryEntry],
    context: MultimodalDesignInput | None,
    number_of_candidates: int,
) -> str:
    """Deterministic prompt with specs, retrieved experience, and context."""
    sections: list[str] = [_SYSTEM_ROLE, "", "## Target specifications", json.dumps(spec.to_dict(), indent=2)]

    successes = [e for e in retrieved if e.success is True]
    failures = [e for e in retrieved if e.success is False]
    if successes:
        sections.append("\n## Retrieved successful designs (reuse proven structure)")
        for entry in successes[:3]:
            sections.append(
                json.dumps(
                    {
                        "memory_id": entry.memory_id,
                        "metrics": entry.spice_metrics,
                        "reward": entry.reward,
                        "topology_nodes": len((entry.topology or {}).get("nodes", [])),
                        "reusable_blocks": entry.reusable_blocks,
                    },
                    indent=2,
                )
            )
        # Include the best success's full structure as a reference (bounded).
        # The model may adapt it; it is not copied automatically.
        best = successes[0]
        if best.topology:
            reference = json.dumps(
                {"nodes": best.topology.get("nodes", []), "edges": best.topology.get("edges", [])}
            )
            if len(reference) <= 6000:
                sections.append(
                    "\n### Reference topology from the best retrieved success "
                    f"(memory {best.memory_id}; adapt as needed, every terminal connected):\n{reference}"
                )
    if failures:
        sections.append("\n## Retrieved failures (avoid these mistakes)")
        for entry in failures[:3]:
            sections.append(f"- {entry.failure_reason or 'unspecified failure'} (metrics: {entry.spice_metrics})")

    summary = build_context_summary(context)
    if summary:
        sections.append("\n## Additional multimodal context")
        for key, value in summary.items():
            sections.append(f"### {key}\n{value}")

    sections += [
        "\n## Output requirements",
        f"Return exactly {number_of_candidates} candidate topologies as JSON matching this schema:",
        json.dumps(TOPOLOGY_JSON_SCHEMA, indent=2),
        "Device types: NMOS, PMOS, RESISTOR, CAPACITOR, CURRENT_SOURCE, VOLTAGE_SOURCE, "
        "INPUT_PORT, OUTPUT_PORT, SUPPLY_PORT, GROUND_PORT, SUBCIRCUIT_BLOCK.",
        "MOS terminals: D, G, S, B. Two-terminal devices: P, N. Ports: PORT.",
        "Every candidate must include supply, ground, input and output ports, and a "
        "reasoning_summary plus confidence in [0, 1]. Do NOT include sizing values.",
        "EVERY terminal of EVERY device must be connected to a net: each MOS needs all of "
        "D, G, S and B connected (tie B to the source's rail: NMOS bulk to ground net, PMOS "
        "bulk to supply net); each two-terminal device needs both P and N connected. "
        "Candidates with unconnected terminals or floating subcircuits are REJECTED.",
    ]
    return "\n".join(sections)
