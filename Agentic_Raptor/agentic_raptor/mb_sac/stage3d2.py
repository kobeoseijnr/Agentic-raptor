"""Stage 3D.2: MP graph conditioning, bounded multi-family training, DPO
integration, and the MCTS leaf-evaluator API.

DPO audit verdict (Part I, honest): `agentic_raptor.dpo.DPORanker` is a
PAIRWISE PREFERENCE RANKER trained with a Bradley–Terry β-logistic objective
over score differences — the documented DPO-equivalent for candidate ranking,
NOT policy-based LLM DPO. Legacy `RAPTOR_Legacy/dpo` is likewise pairwise
preference ranking. The name is retained with this classification recorded.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.electrical import discover_ngspice
from agentic_raptor.mb_sac import load_pools
from agentic_raptor.mb_sac.stage3d1 import PostSizingTopologyScore, _sim_family
from agentic_raptor.utils.seeding import apply_torch_omp_workaround

_ROOT = Path(__file__).resolve().parents[2]
V3 = _ROOT / "artifacts" / "topology_registry_operational_v3"


# ---- Part B: full message-passing conditioning ------------------------------
def build_mp_conditioner(node_dim: int = 16):
    """Active MP encoder over the typed device–net graph (2 rounds, mean agg,
    residual update). Returns (module, embed_fn) — per-device + global embeds."""
    apply_torch_omp_workaround()
    import torch

    from agentic_raptor.topology_rl.policy_value_network import graph_to_tensors

    class MPConditioner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            from agentic_raptor.core.types import DeviceType
            in_dim = len(tuple(DeviceType)) + 2
            self.embed = torch.nn.Linear(in_dim, node_dim)
            self.rounds = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(2 * node_dim, node_dim), torch.nn.ReLU())
                 for _ in range(2)])

        def forward(self, feats: torch.Tensor, adj: torch.Tensor):
            h = torch.relu(self.embed(feats))
            deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
            for layer in self.rounds:
                h = h + layer(torch.cat([h, adj @ h / deg], dim=-1))  # residual
            return h, h.mean(dim=0)  # per-device, global

    module = MPConditioner()

    def embed(graph) -> tuple[Any, Any]:
        feats, adj = graph_to_tensors(graph)
        return module(feats, adj)

    return module, embed


# ---- Part L/Q: DPO-assisted bounded optimisation + leaf evaluator -----------
def evaluate_topology_for_mcts(topology_id: str, real_spice_budget: int = 5,
                               seed: int = 0, n_candidates: int = 4) -> dict[str, Any]:
    """MCTS leaf evaluator: bounded MB-SAC-style candidate generation →
    feasibility-gated preference ranking → real-SPICE verification of the
    selected subset (+1 exploration) → full structured PostSizingTopologyScore."""
    from random import Random

    from agentic_raptor.dpo import (FEATURE_DIM, DPOConfig, DPORanker,
                                    build_candidate_features)
    from agentic_raptor.core.candidate import CircuitCandidate
    from agentic_raptor.core.types import GenerationSource

    started = time.time()
    rng = Random(seed)
    reg = TopologyRegistry(V3)
    exe = discover_ngspice()
    budget = {"real_spice_calls": 0, "failed_calls": 0, "model_transitions": 0}
    wdir = _ROOT / "artifacts" / "stage3d2" / "leaf" / topology_id
    entry = reg.get_topology(topology_id)

    # candidate set: stochastic width-scale proposals (actor-sample stand-ins,
    # legally projected inside _sim_family) + exploration diversity
    scales = [{"w": round(2 ** rng.uniform(-0.5, 0.5), 3)} for _ in range(n_candidates)]
    ranker = DPORanker(FEATURE_DIM, DPOConfig(enabled=True, seed=seed,
                                              minimum_pairs_before_training=1, epochs=10))
    feats = []
    for i, sc in enumerate(scales):
        probe = CircuitCandidate.create(entry.graph, _spec(), GenerationSource.EDITED)
        f = build_candidate_features(probe, sizing_vector=[sc["w"]],
                                     predicted_margins={"w_scale": sc["w"] - 1.0},
                                     predicted_feasibility=0.5)
        f.metadata["scale"] = sc
        feats.append(f)
    ranked = ranker.rank(feats)
    selected = [ranked[0][0], ranked[-1][0]]  # top + exploration retention
    results = []
    for f in selected:
        if budget["real_spice_calls"] >= real_spice_budget:
            break
        rec = _sim_family(topology_id, reg, exe, budget, wdir, f.metadata["scale"])
        m = rec.extracted_metrics if rec else {}
        pm = m.get("phase_margin_deg")
        results.append({"features": f, "metrics": m, "pm": pm,
                        "stable": bool(pm and pm > 0),
                        "gain": m.get("dc_gain_db")})
    # feasibility-gated preference pair from REAL evidence + ranker update
    pair_trained = False
    if len(results) == 2 and results[0]["pm"] is not None and results[1]["pm"] is not None:
        from agentic_raptor.dpo import OutcomeRecord, build_pairs
        recs = [OutcomeRecord(features=r["features"], dpo_score=None, dpo_rank=None,
                              selection_reason="3d2", spice_success=True,
                              passed_spec=r["stable"] and (r["gain"] or 0) > 40,
                              constraint_margins={"pm": (r["pm"] - 45) / 45},
                              fom=(r["gain"] or 0) / 100, runtime_s=1.0,
                              spice_calls_total=1) for r in results]
        pairs = build_pairs(recs)
        if pairs:
            before = ranker.score(results[0]["features"])
            report = ranker.train_on_pairs(pairs * 4)
            pair_trained = not report.get("skipped")
            _ = before
    best = max(results, key=lambda r: (r["stable"], r.get("gain") or -1), default=None)
    pm = best["pm"] if best else None
    score = PostSizingTopologyScore(
        topology_id=topology_id, target_id="mcts_default",
        status=("successful_within_budget" if best and best["stable"]
                else "unstable" if pm is not None else "simulator_failure"),
        verified_stable=bool(best and best["stable"]),
        feasible=bool(best and best["stable"] and (best["gain"] or 0) > 40),
        metrics={k: v for k, v in (best["metrics"] if best else {}).items() if v is not None},
        margins={"gain": ((best["gain"] or 0) - 40) / 40 if best else -1.0,
                 "pm": ((pm or -90) - 45) / 45},
        real_spice_calls=budget["real_spice_calls"],
        calls_to_first_pass=1 if best and best["stable"] else None,
        simulator_failures=budget["failed_calls"], runtime_s=round(time.time() - started, 1))
    score.compute_scalar()
    out = {"score": asdict(score), "scalar_leaf_value": score.scalar_value,
           "candidates": len(feats), "selected": len(results),
           "dpo_pair_trained": pair_trained, "budget": budget,
           "ranking_confidence": "ordinal_uncalibrated"}
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "leaf_result.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def _spec():
    from agentic_raptor.core.specifications import DesignSpecifications
    return DesignSpecifications(circuit_class="ota", technology="sky130", supply_voltage=1.8,
                                target_gain_db=40.0, target_gbw_hz=1e4,
                                minimum_phase_margin_deg=45.0, load_capacitance_f=500e-12)


def run_stage3d2(steps_per_family: int = 2) -> dict[str, Any]:
    """Bounded Phase B/C/D training pass with MP conditioning + leaf smoke."""
    apply_torch_omp_workaround()
    import torch

    from agentic_raptor.topology_rl.trainer import parameter_checksum

    started = time.time()
    pools = load_pools()
    reg = TopologyRegistry(V3)
    exe = discover_ngspice()
    module, embed = build_mp_conditioner()
    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    budget = {"real_spice_calls": 0, "failed_calls": 0}
    wdir = _ROOT / "artifacts" / "stage3d2" / "train"
    mp_before = parameter_checksum(module)
    phase_stats = {}
    stable_ids = [r["topology_id"] for r in pools["A1"] + pools["A2"]]
    grads = []
    for label, ids in (("B_a1", stable_ids[:9]), ("C_a2", stable_ids[9:16])):
        ok = 0
        for tid in ids:
            e = reg.get_topology(tid)
            per_dev, global_emb = embed(e.graph)   # ACTIVE MP conditioning
            for step in range(steps_per_family):
                scale = {"w": float(1.0 + 0.1 * torch.tanh(global_emb.mean()).item())}
                rec = _sim_family(tid, reg, exe, budget, wdir, scale)
                pm = rec.extracted_metrics.get("phase_margin_deg") if rec else None
                stable = bool(pm and pm > 0)
                ok += int(stable and step == steps_per_family - 1)
                # gradient step: regress embedding-derived value toward measured margin
                target = torch.tensor(((pm or -90) - 45) / 90.0)
                pred = torch.tanh(global_emb.mean())
                loss = (pred - target) ** 2
                opt.zero_grad()
                loss.backward(retain_graph=step < steps_per_family - 1)
                grads.append(float(sum(p.grad.abs().sum() for p in module.parameters()
                                       if p.grad is not None)))
                opt.step()
                per_dev, global_emb = embed(e.graph)
        phase_stats[label] = {"families": len(ids), "stable_final": ok,
                              "per_device_embed_shape": list(per_dev.shape),
                              "global_embed_shape": list(global_emb.shape)}
    mp_changed = parameter_checksum(module) != mp_before
    ckpt = wdir / "mp_conditioner.pt"
    torch.save({"mp": module.state_dict(), "encoder": "MPConditioner-2round-residual",
                "node_dim": 16}, ckpt)
    # leaf-evaluator smoke: one A1 + one A2
    leaf = {tid: evaluate_topology_for_mcts(tid, real_spice_budget=3)
            for tid in (stable_ids[0], stable_ids[9])}
    summary = {"phases": phase_stats, "budget": budget,
               "mp_encoder_active": True, "mp_params_changed": mp_changed,
               "nonzero_grad_norms": all(g > 0 for g in grads), "grad_samples": len(grads),
               "checkpoint": str(ckpt),
               "leaf": {t: {"scalar": r["scalar_leaf_value"],
                            "status": r["score"]["status"],
                            "dpo_pair_trained": r["dpo_pair_trained"],
                            "calls": r["budget"]["real_spice_calls"]} for t, r in leaf.items()},
               "wall_clock_s": round(time.time() - started, 1)}
    (_ROOT / "artifacts" / "stage3d2" / "SUMMARY.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_stage3d2(), indent=1, default=str))
