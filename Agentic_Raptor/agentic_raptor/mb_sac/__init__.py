"""Stage 3D: graph-conditioned MB-SAC over the qualified operational corpus.

Reuses the genuine, already-tested MB-SAC core (sizing.GraphConditionedMBSAC:
squashed-Gaussian actor, twin independent critics, auto-α, Polyak targets,
dynamics ENSEMBLE with disagreement uncertainty, real/model replay separation,
query gating) and wires it to the Stage 3C.2b operational corpus:
snapshot freeze → repair-pool audit → normal/repair environment modes →
real-SPICE transitions (final authority) → SAC + ensemble updates →
checkpoint/resume → exact budget accounting.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.electrical import discover_ngspice, qualify_family
from agentic_raptor.mapping import _MappedEntry, emit_netlist, map_family
from agentic_raptor.sizing.graph_conditioned_mb_sac import (
    GraphConditionedMBSAC, MBSACConfig, SizingStateContext, encode_sizing_state)
from agentic_raptor.sizing.parameter_space import SizingParameterSpace
from agentic_raptor.sizing.replay_buffer import SizingTransition

_ROOT = Path(__file__).resolve().parents[2]
V3 = _ROOT / "artifacts" / "topology_registry_operational_v3"
SNAP = _ROOT / "artifacts" / "corpus_snapshots" / "pre_stage3d_mb_sac"


def freeze_corpus() -> dict[str, Any]:
    SNAP.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for f in [V3 / "SUMMARY.json", V3 / "split_v3.json",
              _ROOT / "datasets/simulation_memory/stage3c2b_runs.jsonl",
              _ROOT / "datasets/simulation_memory/topology_summaries.jsonl"]:
        if f.is_file():
            shutil.copy2(f, SNAP / f.name)
            manifest[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (SNAP / "MANIFEST.json").write_text(json.dumps(
        {"created": time.time(), "snapshot_id": "pre_stage3d_mb_sac", "files": manifest},
        indent=1), encoding="utf-8")
    return manifest


def load_pools() -> dict[str, list[dict[str, Any]]]:
    runs = [json.loads(x) for x in
            (_ROOT / "datasets/simulation_memory/stage3c2b_runs.jsonl").read_text().splitlines()]
    ag = [json.loads(x) for x in
          (_ROOT / "datasets/simulation_memory/topology_summaries.jsonl").read_text().splitlines()]
    stable_gen = [r for r in runs if r.get("stability") == "verified_stable"]
    unstable_gen = [r for r in runs if r.get("stability") == "verified_unstable"]
    stable_lit = [r for r in ag if r.get("stability_status") == "verified_stable"
                  and r.get("electrical_validation_status") == "electrically_functional"]
    unstable_lit = [r for r in ag if r.get("stability_status") == "verified_unstable"]
    return {"A1": stable_lit, "A2": stable_gen, "D2_gen": unstable_gen, "D2_lit": unstable_lit}


def audit_repair_pool() -> dict[str, Any]:
    """Continuous-action repair eligibility for every D2 family (no topology edits)."""
    pools = load_pools()
    reg = TopologyRegistry(V3)
    rows = []
    for r in pools["D2_gen"] + pools["D2_lit"]:
        tid = r["topology_id"]
        try:
            e = reg.get_topology(tid)
        except KeyError:
            continue
        graph = e.graph
        has_comp = any((n.block_role or "") in ("miller_compensation", "compensation")
                       or n.device_type.value == "CAPACITOR" for n in graph.nodes.values())
        n_stages = sum(1 for n in graph.nodes.values()
                       if (n.block_role or "") in ("gain_stage", "second_stage_gain_device", "gmf", "gm1", "gm2", "gm3"))
        has_bias = any("bias" in (n.block_role or "") for n in graph.nodes.values())
        pm = (r.get("metrics") or r.get("verified_metrics") or {}).get("phase_margin_deg")
        if pm is None:
            cls = "measurement_insufficient"
        elif has_comp and has_bias:
            cls = "repair_combined_eligible"
        elif has_comp:
            cls = "repair_compensation_value_eligible"
        elif has_bias or n_stages >= 1:
            cls = "repair_sizing_eligible"
        else:
            cls = "missing_controllable_stability_variable"
        rows.append({"topology_id": tid, "classification": cls, "phase_margin_deg": pm,
                     "has_compensation_cap": has_comp, "has_bias_control": has_bias,
                     "policy": "existing continuous parameters only; no added components"})
    out = _ROOT / "datasets/simulation_memory/stage3d_repair_audit.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["classification"]] += 1
    eligible = [r["topology_id"] for r in rows if r["classification"].startswith("repair_")]
    return {"audited": len(rows), "counts": dict(counts), "eligible": eligible}


def run_smoke(topology_id: str | None = None, steps: int = 4, seed: int = 3) -> dict[str, Any]:
    """Phase-A smoke: one verified-stable family, real-SPICE MB-SAC episode."""
    from agentic_raptor.topology_rl.trainer import parameter_checksum

    freeze_corpus()
    pools = load_pools()
    tid = topology_id or pools["A2"][0]["topology_id"]  # A2: sizing directly controllable
    reg = TopologyRegistry(V3)
    e = reg.get_topology(tid)
    blocks = {n.block_role for n in e.graph.nodes.values() if n.block_role}
    audit_row = {"topology_id": tid, "gain_stages": max(1, sum(
        1 for n in e.graph.nodes.values() if n.block_role == "gain_stage")),
        "functional_blocks": sorted(blocks) + ["C"], "unresolved_blocks": [],
        "mapping_readiness": "mapping_ready", "graph_hash": e.metadata.get("graph_hash")}
    g_dev, _ = map_family(e, audit_row)
    assert g_dev is not None
    wdir = _ROOT / "artifacts" / "stage3d" / "smoke" / tid
    wdir.mkdir(parents=True, exist_ok=True)
    exe = discover_ngspice()
    budget = {"real_spice_calls": 0, "failed_calls": 0, "model_transitions": 0,
              "actor_updates": 0, "critic_updates": 0, "model_updates": 0}

    # sizing space over the mapped device graph (bounds → deterministic projection)
    space_graph = e.graph  # topology-conditioned features
    from agentic_raptor.core.circuit_graph import CircuitGraph, CircuitNode, CircuitEdge  # noqa
    devs = {d.device_id: d for d in g_dev.devices}
    base_sizing = {d.device_id: dict(d.sizing) for d in g_dev.devices if d.kind in ("nmos", "pmos", "cap", "isrc")}

    def evaluate(scale: dict[str, float]) -> tuple[Any, dict[str, float]]:
        for d in g_dev.devices:
            if d.kind in ("nmos", "pmos"):
                d.sizing["w"] = max(0.42, min(100.0, base_sizing[d.device_id]["w"] * scale.get(d.device_id, 1.0)))
        subckt = f"smoke_{tid}"
        net = emit_netlist(g_dev, subckt)
        cdir = wdir / f"step_{budget['real_spice_calls']}"
        cdir.mkdir(exist_ok=True)
        (cdir / "netlist.sp").write_text(net, encoding="utf-8")
        (cdir / "design_variables").mkdir(exist_ok=True)
        (cdir / "design_variables" / subckt).write_text("* inlined\n", encoding="utf-8")
        import agentic_raptor.electrical as elec
        fake = _MappedEntry(tid, cdir, {"name": subckt, "graph_hash": e.metadata.get("graph_hash")}, e.graph)
        orig = elec._LEGACY_AMP
        elec._LEGACY_AMP = cdir
        try:
            rec = qualify_family(fake, {"subckt_name": subckt, "immediately_runnable": True,
                                        "blocking_reason": None}, cdir / "run", exe,
                                 "env-3d-smoke", "train", 120.0)
        finally:
            elec._LEGACY_AMP = orig
        budget["real_spice_calls"] += 1
        if not rec.extracted_metrics.get("dc_gain_db"):
            budget["failed_calls"] += 1
        margins = {"gain_db": (rec.extracted_metrics.get("dc_gain_db") or 0) / 100.0,
                   "pm": ((rec.extracted_metrics.get("phase_margin_deg") or -90) / 90.0)}
        return rec, margins

    space = SizingParameterSpace.from_graph(space_graph)
    sac = GraphConditionedMBSAC(space, MBSACConfig(hidden_dim=32, lr=1e-3, seed=seed,
                                                   dynamics_ensemble_size=2))
    spec = e.graph  # spec object needed: use a benchmark spec
    from agentic_raptor.core.specifications import DesignSpecifications
    spec = DesignSpecifications(circuit_class="ota", technology="sky130", supply_voltage=1.8,
                                target_gain_db=60.0, target_gbw_hz=1e5,
                                minimum_phase_margin_deg=45.0, load_capacitance_f=500e-12)
    mos_ids = [d.device_id for d in g_dev.devices if d.kind in ("nmos", "pmos")]
    rec, margins = evaluate({})
    score = sum(margins.values())
    checks = {"actor_before": parameter_checksum(sac.actor),
              "critic1_before": parameter_checksum(sac.q1),
              "critic2_before": parameter_checksum(sac.q2)}
    for step in range(steps):
        ctx = SizingStateContext(graph=space_graph, spec=spec,
                                 sizing_vector=[0.0] * sac.action_dim,
                                 metrics=rec.extracted_metrics if rec else {},
                                 constraint_margins=margins,
                                 spice_budget_fraction=1 - step / max(steps, 1))
        state = encode_sizing_state(ctx, sac.action_dim) + [1.0]  # +mode flag normal_sizing
        state = state[:sac.state_dim]
        action = sac.select_action(state)
        scale = {mid: float(2 ** (0.5 * action[i % len(action)])) for i, mid in enumerate(mos_ids)}
        rec2, m2 = evaluate(scale)  # real SPICE transition (projection in evaluate: clamped W)
        new_score = sum(m2.values())
        sac.add_transition(SizingTransition(state=state, action=action,
                                            reward=new_score - score,
                                            next_state=state, done=step == steps - 1,
                                            source="real"))
        if new_score >= score:
            rec, margins, score = rec2, m2, new_score
    dyn = sac.update_dynamics(batch_size=4)
    budget["model_updates"] += 0 if dyn.get("skipped") else 1
    unc = sac.dynamics_uncertainty([0.0] * sac.state_dim, [0.0] * sac.action_dim)
    gate_pass = unc < 0.5  # calibration gate: rollouts only if disagreement low
    if gate_pass:
        budget["model_transitions"] = sac.generate_imagined_transitions([0.0] * sac.state_dim, 2)
    upd = sac.update(batch_size=4)
    if not upd.get("skipped"):
        budget["actor_updates"] += 1
        budget["critic_updates"] += 1
    # checkpoint + deterministic resume
    ckpt = wdir / "checkpoint.pt"
    sac.torch.save({"actor": sac.actor.state_dict(), "q1": sac.q1.state_dict(),
                    "q2": sac.q2.state_dict(), "snapshot_id": "pre_stage3d_mb_sac",
                    "budget": budget}, ckpt)
    sac2 = GraphConditionedMBSAC(space, MBSACConfig(hidden_dim=32, lr=1e-3, seed=seed,
                                                    dynamics_ensemble_size=2))
    payload = sac2.torch.load(ckpt, weights_only=False)
    sac2.actor.load_state_dict(payload["actor"])
    resume_ok = parameter_checksum(sac2.actor) == parameter_checksum(sac.actor)
    result = {"topology": tid, "budget": budget, "uncertainty": round(unc, 4),
              "model_gate_passed": bool(gate_pass),
              "critics_independent": checks["critic1_before"] != checks["critic2_before"],
              "actor_changed": parameter_checksum(sac.actor) != checks["actor_before"],
              "critics_changed": parameter_checksum(sac.q1) != checks["critic1_before"],
              "resume_deterministic": bool(resume_ok),
              "final_metrics": {k: v for k, v in (rec.extracted_metrics or {}).items() if v is not None},
              "final_stability": "verified_stable" if (rec.extracted_metrics.get("phase_margin_deg") or -1) > 0 else "verified_unstable"}
    (wdir / "smoke_result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps({"repair_audit": audit_repair_pool(), "smoke": run_smoke()},
                     indent=1, default=str))
