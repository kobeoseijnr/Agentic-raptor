"""Trainable policy–value network for topology editing.

Graph encoder decision (docs/DECISIONS.md D3): PyTorch Geometric is NOT
installed in this environment, so two plain-PyTorch encoders are provided:

* ``pooled``  — permutation-invariant pooled feature encoder (default; fast, deterministic);
* ``message_passing`` — 2-round mean-aggregation message passing over the
  device–net bipartite adjacency, implemented with dense torch ops.

Inputs (concatenated feature vector, see :func:`encode_state_features`):
topology features, target-spec embedding, remaining edit-budget fraction,
validation features, optional predicted sizing outcome.

Outputs: policy logits over a fixed number of action slots (masked to the
current legal actions) and a scalar value in (-1, 1) — trained against the
final post-sizing reward mapped through tanh scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_raptor.core.circuit_graph import CircuitGraph
from agentic_raptor.core.specifications import DesignSpecifications
from agentic_raptor.core.types import DeviceType
from agentic_raptor.topology_rl.actions import TopologyAction
from agentic_raptor.utils.seeding import apply_torch_omp_workaround

_DEVICE_ORDER: tuple[DeviceType, ...] = tuple(DeviceType)


def _spec_embedding(spec: DesignSpecifications) -> list[float]:
    """Normalized fixed-length spec embedding (log-compressed where wide-range)."""
    raw = spec.feature_vector()
    out: list[float] = []
    for name, value in zip(DesignSpecifications.NUMERIC_FIELDS, raw, strict=True):
        if value == 0.0:
            out.append(0.0)
        elif name in ("target_gbw_hz", "minimum_slew_rate_v_per_s"):
            out.append(math.log10(abs(value)) / 12.0)
        elif name in ("maximum_power_w", "load_capacitance_f", "maximum_area_um2"):
            out.append(math.log10(abs(value) + 1e-18) / -18.0)
        elif name == "target_gain_db":
            out.append(value / 120.0)
        elif name == "minimum_phase_margin_deg":
            out.append(value / 180.0)
        elif name == "temperature_c":
            out.append(value / 200.0)
        else:
            out.append(value / 10.0)
    return out


def graph_feature_vector(graph: CircuitGraph) -> list[float]:
    """Pooled structural features: per-type device counts, net/edge counts, degree stats."""
    counts = {t: 0 for t in _DEVICE_ORDER}
    for node in graph.nodes.values():
        counts[node.device_type] += 1
    nets = graph.nets()
    degrees = [len(v) for v in nets.values()] or [0]
    return (
        [counts[t] / 10.0 for t in _DEVICE_ORDER]
        + [
            len(graph.nodes) / 20.0,
            len(nets) / 20.0,
            len(graph.edges) / 50.0,
            max(degrees) / 10.0,
            (sum(degrees) / len(degrees)) / 10.0,
        ]
    )


def encode_state_features(
    graph: CircuitGraph,
    spec: DesignSpecifications,
    edit_budget_fraction: float,
    validation_features: list[float] | None = None,
    sizing_outcome_estimate: float | None = None,
) -> list[float]:
    """Full network input vector. Deterministic, pure python."""
    val = list(validation_features) if validation_features is not None else [0.0, 0.0, 0.0]
    return (
        graph_feature_vector(graph)
        + _spec_embedding(spec)
        + [float(edit_budget_fraction)]
        + val
        + [0.0 if sizing_outcome_estimate is None else float(sizing_outcome_estimate)]
    )


#: len(graph features) + len(spec embedding) + budget + validation(3) + sizing estimate
STATE_FEATURE_DIM = (len(_DEVICE_ORDER) + 5) + len(DesignSpecifications.NUMERIC_FIELDS) + 1 + 3 + 1


@dataclass
class PolicyValueConfig:
    max_actions: int = 32
    hidden_dim: int = 128
    encoder: str = "pooled"  # "pooled" | "message_passing"
    mp_rounds: int = 2
    mp_node_dim: int = 32
    device: str = "cpu"
    lr: float = 1e-3
    value_loss_weight: float = 1.0
    grad_clip_norm: float = 5.0


class PolicyValueEvaluator(Protocol):
    """What MCTS needs: priors over legal actions and a state value."""

    def evaluate(
        self,
        graph: CircuitGraph,
        spec: DesignSpecifications,
        edit_budget_fraction: float,
        validation_features: list[float],
        legal_actions: list[TopologyAction],
    ) -> tuple[list[float], float]: ...


class HeuristicEvaluator:
    """Torch-free fallback: uniform priors; value from validation features."""

    def evaluate(
        self,
        graph: CircuitGraph,
        spec: DesignSpecifications,
        edit_budget_fraction: float,
        validation_features: list[float],
        legal_actions: list[TopologyAction],
    ) -> tuple[list[float], float]:
        n = max(len(legal_actions), 1)
        validity = validation_features[0] if validation_features else 0.0
        errors = validation_features[1] if len(validation_features) > 1 else 0.0
        value = max(-1.0, min(1.0, validity - 0.1 * errors))
        return [1.0 / n] * len(legal_actions), value


def build_policy_value_network(config: PolicyValueConfig) -> Any:
    """Construct the torch module (lazy torch import with OMP workaround)."""
    apply_torch_omp_workaround()
    import torch
    import torch.nn as nn

    class _MessagePassingEncoder(nn.Module):
        """Dense mean-aggregation message passing over device–net adjacency."""

        def __init__(self, node_feat_dim: int, node_dim: int, rounds: int):
            super().__init__()
            self.rounds = rounds
            self.embed = nn.Linear(node_feat_dim, node_dim)
            self.update = nn.ModuleList(
                [nn.Sequential(nn.Linear(2 * node_dim, node_dim), nn.ReLU()) for _ in range(rounds)]
            )

        def forward(self, node_feats: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.embed(node_feats))
            degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
            for layer in self.update:
                messages = adjacency @ h / degree
                h = layer(torch.cat([h, messages], dim=-1))
            return h.mean(dim=0)  # permutation-invariant pooling

    class PolicyValueNetwork(nn.Module):
        def __init__(self, cfg: PolicyValueConfig):
            super().__init__()
            self.cfg = cfg
            in_dim = STATE_FEATURE_DIM
            self.mp_encoder: nn.Module | None = None
            if cfg.encoder == "message_passing":
                # one-hot device type (+1 for net nodes) + degree scalar
                self.mp_encoder = _MessagePassingEncoder(len(_DEVICE_ORDER) + 2, cfg.mp_node_dim, cfg.mp_rounds)
                in_dim += cfg.mp_node_dim
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(cfg.hidden_dim, cfg.max_actions)
            self.value_head = nn.Linear(cfg.hidden_dim, 1)

        def forward(
            self, features: torch.Tensor, graph_tensors: tuple[torch.Tensor, torch.Tensor] | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = features
            if self.mp_encoder is not None:
                if graph_tensors is None:
                    zeros = torch.zeros(features.shape[0], self.cfg.mp_node_dim, device=features.device)
                    x = torch.cat([features, zeros], dim=-1)
                else:
                    node_feats, adjacency = graph_tensors
                    graph_embed = self.mp_encoder(node_feats, adjacency).unsqueeze(0)
                    x = torch.cat([features, graph_embed.expand(features.shape[0], -1)], dim=-1)
            h = self.trunk(x)
            return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)

    return PolicyValueNetwork(config)


def graph_to_tensors(graph: CircuitGraph, device: str = "cpu") -> tuple[Any, Any]:
    """Node-feature matrix + adjacency for the message-passing encoder.

    Rows: device nodes then net nodes. Features: one-hot device type
    (index len(_DEVICE_ORDER) marks a net node) + normalized degree.
    """
    apply_torch_omp_workaround()
    import torch

    node_ids = sorted(graph.nodes)
    nets = graph.nets()
    net_names = sorted(nets)
    n = len(node_ids) + len(net_names)
    feats = torch.zeros(n, len(_DEVICE_ORDER) + 2)
    adjacency = torch.zeros(n, n)
    index = {nid: i for i, nid in enumerate(node_ids)}
    index.update({f"net::{name}": len(node_ids) + j for j, name in enumerate(net_names)})
    for nid in node_ids:
        feats[index[nid], _DEVICE_ORDER.index(graph.nodes[nid].device_type)] = 1.0
    for name in net_names:
        row = index[f"net::{name}"]
        feats[row, len(_DEVICE_ORDER)] = 1.0
        for node_id, _terminal in nets[name]:
            adjacency[row, index[node_id]] = 1.0
            adjacency[index[node_id], row] = 1.0
    degree = adjacency.sum(dim=-1, keepdim=True)
    feats[:, -1:] = degree / 10.0
    return feats.to(device), adjacency.to(device)


class NetworkEvaluator:
    """Adapts the torch network to the MCTS evaluator protocol."""

    def __init__(self, network: Any, config: PolicyValueConfig):
        self.network = network
        self.config = config

    def evaluate(
        self,
        graph: CircuitGraph,
        spec: DesignSpecifications,
        edit_budget_fraction: float,
        validation_features: list[float],
        legal_actions: list[TopologyAction],
    ) -> tuple[list[float], float]:
        apply_torch_omp_workaround()
        import torch

        features = torch.tensor(
            [encode_state_features(graph, spec, edit_budget_fraction, validation_features)],
            dtype=torch.float32,
        )
        graph_tensors = (
            graph_to_tensors(graph, self.config.device) if self.config.encoder == "message_passing" else None
        )
        with torch.no_grad():
            logits, value = self.network(features, graph_tensors)
        n = min(len(legal_actions), self.config.max_actions)
        masked = logits[0, :n]
        priors = torch.softmax(masked, dim=-1).tolist() if n else []
        priors += [0.0] * (len(legal_actions) - n)  # overflow actions get zero prior
        return priors, float(value.item())
