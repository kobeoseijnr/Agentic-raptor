"""Stable topology hashing (thin wrapper over CircuitGraph.structural_hash).

Kept as its own module so hashing policy can evolve (e.g. exact canonical
labelling) without touching the validator.
"""

from __future__ import annotations

from agentic_raptor.core.circuit_graph import CircuitGraph


def compute_graph_hash(graph: CircuitGraph) -> str:
    """Weisfeiler–Lehman hash of the bipartite device–net structure.

    Properties:
    - invariant to node/net renaming;
    - sensitive to device types, terminal roles, and connectivity;
    - excludes continuous sizing values and metadata.

    Note: WL hashing can (rarely) collide for non-isomorphic graphs; for the
    graph sizes used here (tens of devices) this is acceptable for caching and
    dedup. Exact isomorphism can be layered on later if needed.
    """
    return graph.structural_hash()
