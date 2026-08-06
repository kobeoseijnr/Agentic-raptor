"""Simulation cache keyed by everything that determines a result."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentic_raptor.core.candidate import CircuitCandidate
from agentic_raptor.spice.interface import SimulationResult


def make_cache_key(
    topology_hash: str,
    sizing_state: dict[str, dict[str, float]],
    analyses: list[str],
    corner: str,
    simulator_config: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "topology": topology_hash,
            "sizing": {nid: {p: round(v, 12) for p, v in sorted(params.items())} for nid, params in sorted(sizing_state.items())},
            "analyses": sorted(analyses),
            "corner": corner,
            "simulator": simulator_config,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class SimulationCache:
    """In-memory cache with optional JSONL persistence.

    Key coverage (Stage 2): topology hash, sizing vector, analysis config,
    corner, and the ``simulator_config`` dict — callers must include netlist
    configuration, technology/model-library label, simulator version, and
    temperature there (``NgspiceSimulator.config_fingerprint()`` provides the
    backend parts). Any differing value produces a different key.
    ``enabled=False`` turns the cache into a pass-through (every run executes).
    """

    def __init__(self, persist_path: str | Path | None = None, enabled: bool = True) -> None:
        self._store: dict[str, SimulationResult] = {}
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.is_file():
            with self.persist_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        self._store[record["key"]] = SimulationResult.from_dict(record["result"])

    def get(self, key: str) -> SimulationResult | None:
        result = self._store.get(key)
        if result is not None:
            self.hits += 1
        return result

    def put(self, key: str, result: SimulationResult) -> None:
        self.misses += 1
        self._store[key] = result
        if self.persist_path:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self.persist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "result": result.to_dict()}, ensure_ascii=False) + "\n")

    def get_or_run(
        self,
        candidate: CircuitCandidate,
        analyses: list[str],
        corner: str,
        simulator_config: dict[str, Any],
        run: Callable[[], SimulationResult],
    ) -> tuple[SimulationResult, bool]:
        """(result, was_cached). ``run`` is only called on a miss."""
        if not self.enabled:
            return run(), False
        key = make_cache_key(
            candidate.topology.structural_hash(),
            candidate.sizing_state or candidate.topology.sizing_state(),
            analyses,
            corner,
            simulator_config,
        )
        cached = self.get(key)
        if cached is not None:
            return cached, True
        result = run()
        # Only successful results are cached: transient failures (simulator
        # unavailable, timeout, convergence) must be retryable.
        if result.success:
            self.put(key, result)
        else:
            self.misses += 1
        return result, False

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses}
