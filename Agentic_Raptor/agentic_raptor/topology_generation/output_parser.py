"""Structured JSON output format for generated topologies + strict parser."""

from __future__ import annotations

import json
from typing import Any

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.utils.exceptions import GenerationError, GraphError

#: The exact schema generators must emit (also embedded in prompts).
TOPOLOGY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["graph_id", "nodes", "edges"],
                "properties": {
                    "graph_id": {"type": "string"},
                    "reasoning_summary": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["node_id", "device_type"],
                            "properties": {
                                "node_id": {"type": "string"},
                                "device_type": {"type": "string"},
                                "block_role": {"type": ["string", "null"]},
                                "attributes": {"type": "object"},
                            },
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["edge_id", "node_id", "terminal", "net"],
                        },
                    },
                },
            },
        }
    },
}


def parse_generator_output(text: str) -> list[CircuitGraph]:
    """Parse strict JSON generator output into typed graphs.

    Raises :class:`GenerationError` with a precise message on any malformation
    (fed back into the retry hook so an LLM can self-correct).
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"generator output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "candidates" not in data:
        raise GenerationError("generator output must be an object with a 'candidates' array")
    candidates = data["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise GenerationError("'candidates' must be a non-empty array")

    graphs: list[CircuitGraph] = []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, dict):
            raise GenerationError(f"candidate {index} is not an object")
        try:
            graph = CircuitGraph.from_dict(raw)
        except (GraphError, KeyError, ValueError) as exc:
            raise GenerationError(f"candidate {index} ({raw.get('graph_id')}): {exc}") from exc
        graph.metadata.extra.setdefault("reasoning_summary", str(raw.get("reasoning_summary", "")))
        graph.metadata.extra.setdefault("confidence", float(raw.get("confidence", 0.5)))
        graphs.append(graph)
    return graphs
