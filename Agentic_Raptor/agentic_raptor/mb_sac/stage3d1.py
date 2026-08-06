"""Stage 3D.1: multi-family MB-SAC validation before topology MCTS.

A1 design-variable writer (AnalogGym .PARAM files) + round-trip verification,
16-family environment validation, bounded Phase B/C/D training passes, exact
SPICE accounting, and the PostSizingTopologyScore interface for Stage 3E.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.electrical import _LEGACY_AMP, discover_ngspice, qualify_family, audit_family
from agentic_raptor.mb_sac import load_pools
from agentic_raptor.mapping import _MappedEntry, emit_netlist, map_family

_ROOT = Path(__file__).resolve().parents[2]
V3 = _ROOT / "artifacts" / "topology_registry_operational_v3"
_PARAM_RE = re.compile(r"([A-Za-z0-9_]+)=([-\w.']+)")


# ---------------- A1 design variables ---------------------------------------
def parse_params(text: str) -> dict[str, str]:
    return dict(_PARAM_RE.findall(text))


def write_params(params: dict[str, str]) -> str:
    items = [f"{k}={v}" for k, v in params.items()]
    lines = [".PARAM"]
    for i in range(0, len(items), 2):
        lines.append("+ " + " ".join(items[i:i + 2]) + " ")
    return "\n".join(lines) + "\n"


def _kind(name: str) -> str:
    if "_W_" in name:
        return "width"
    if "_L_" in name:
        return "length"
    if "_M_" in name:
        return "multiplicity"
    return "other"


def build_a1_manifests() -> dict[str, Any]:
    out = _ROOT / "artifacts" / "stage3d1" / "design_variables" / "a1"
    out.mkdir(parents=True, exist_ok=True)
    pools = load_pools()
    reg = TopologyRegistry(V3)
    results = {}
    for r in pools["A1"]:
        tid = r["topology_id"]
        name = reg.get_metadata(tid).get("name")
        pfile = _LEGACY_AMP / "design_variables" / name
        params = parse_params(pfile.read_text(encoding="utf-8", errors="replace"))
        manifest = [{
            "parameter_id": k, "netlist_element": "_".join(k.split("_")[:2]) or k,
            "parameter_kind": _kind(k), "original_value": v,
            "lower_bound": 1 if _kind(k) == "multiplicity" else 0.5,
            "upper_bound": 64 if _kind(k) == "multiplicity" else 100.0,
            "scale": "log", "projection_rule": "clamp+grid",
            "matched_group": re.sub(r"_(W|L|M)_", "_", k),
            "fixed": _kind(k) == "other",
            "provenance": "analoggym design_variables file",
        } for k, v in params.items()]
        (out / f"{tid}.json").write_text(json.dumps(manifest, indent=0), encoding="utf-8")
        # round-trip: parse → rewrite → reparse must be identical
        rt = parse_params(write_params(params))
        results[tid] = {"params": len(params),
                        "widths": sum(1 for k in params if _kind(k) == "width"),
                        "lengths": sum(1 for k in params if _kind(k) == "length"),
                        "mults": sum(1 for k in params if _kind(k) == "multiplicity"),
                        "roundtrip_identical": rt == params}
    return results


# ---------------- environment validation + bounded training -----------------
@dataclass
class PostSizingTopologyScore:
    topology_id: str
    target_id: str
    status: str            # successful_within_budget | infeasible | unstable |
    #                        simulator_failure | budget_exhausted | unsupported_action_space
    verified_stable: bool
    feasible: bool
    metrics: dict[str, float]
    margins: dict[str, float]
    real_spice_calls: int
    calls_to_first_pass: int | None
    simulator_failures: int
    scalar_value: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    runtime_s: float = 0.0

    def compute_scalar(self, weights: dict[str, float] | None = None) -> float:
        w = weights or {"valid": 0.2, "stable": 0.4, "feasible": 0.4,
                        "margin": 0.2, "cost": 0.1}
        worst = min(self.margins.values(), default=-1.0)
        self.components = {
            "valid": w["valid"] * (1.0 if self.status != "simulator_failure" else 0.0),
            "stable": w["stable"] * (1.0 if self.verified_stable else 0.0),
            "feasible": w["feasible"] * (1.0 if self.feasible else 0.0),
            "margin": w["margin"] * max(-1.0, min(1.0, worst)),
            "cost": -w["cost"] * min(1.0, self.real_spice_calls / 20.0),
        }
        self.scalar_value = round(sum(self.components.values()), 4)
        return self.scalar_value


def _sim_family(tid: str, reg: TopologyRegistry, exe: str, budget: dict[str, int],
                wdir: Path, scale: dict[str, float] | None = None) -> Any:
    """One real evaluation of a normal-pool family (A1 via source netlist +
    param rewrite; A2/v2 via re-emitted mapped netlist)."""
    e = reg.get_topology(tid)
    if tid.startswith("topology_00"):  # A1 literature
        name = e.metadata.get("name")
        cdir = wdir / tid
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "netlist.sp").write_text(
            (_LEGACY_AMP / "spice_netlist" / name).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8")
        (cdir / "design_variables").mkdir(exist_ok=True)
        params = parse_params((_LEGACY_AMP / "design_variables" / name)
                              .read_text(encoding="utf-8", errors="replace"))
        if scale:  # multiply matched W params (grid-projected int-safe strings kept numeric)
            for k in params:
                if _kind(k) == "width":
                    try:
                        params[k] = f"{max(0.5, min(100.0, float(params[k]) * scale.get('w', 1.0))):.4g}"
                    except ValueError:
                        pass
        (cdir / "design_variables" / name).write_text(write_params(params), encoding="utf-8")
        audit = {"subckt_name": name.lower(), "immediately_runnable": True, "blocking_reason": None}
        entry = _MappedEntry(tid, cdir, {"name": name, "graph_hash": e.metadata.get("graph_hash")}, e.graph)
    else:  # generated: re-map + emit with scaled sizing
        blocks = {n.block_role for n in e.graph.nodes.values() if n.block_role}
        row = {"topology_id": tid, "gain_stages": max(1, sum(
            1 for n in e.graph.nodes.values() if n.block_role == "gain_stage")),
            "functional_blocks": sorted(blocks) + ["C"], "unresolved_blocks": [],
            "mapping_readiness": "mapping_ready", "graph_hash": e.metadata.get("graph_hash")}
        g_dev, _ = map_family(e, row)
        if g_dev is None:
            return None
        if scale:
            for d in g_dev.devices:
                if d.kind in ("nmos", "pmos"):
                    d.sizing["w"] = max(0.42, min(100.0, d.sizing["w"] * scale.get("w", 1.0)))
        cdir = wdir / tid
        cdir.mkdir(parents=True, exist_ok=True)
        sub = f"d31_{tid}"
        (cdir / "netlist.sp").write_text(emit_netlist(g_dev, sub), encoding="utf-8")
        (cdir / "design_variables").mkdir(exist_ok=True)
        (cdir / "design_variables" / sub).write_text("* inlined\n", encoding="utf-8")
        audit = {"subckt_name": sub, "immediately_runnable": True, "blocking_reason": None}
        entry = _MappedEntry(tid, cdir, {"name": sub, "graph_hash": e.metadata.get("graph_hash")}, e.graph)
    import agentic_raptor.electrical as elec
    orig = elec._LEGACY_AMP
    elec._LEGACY_AMP = cdir
    try:
        rec = qualify_family(entry, audit, cdir / "run", exe, "env-3d1", "train", 120.0)
    finally:
        elec._LEGACY_AMP = orig
    budget["real_spice_calls"] += 1
    if not rec.extracted_metrics.get("dc_gain_db"):
        budget["failed_calls"] += 1
    return rec


def run_stage3d1(phase_steps: int = 1) -> dict[str, Any]:
    started = time.time()
    exe = discover_ngspice()
    reg = TopologyRegistry(V3)
    pools = load_pools()
    a1 = build_a1_manifests()
    budget = {"real_spice_calls": 0, "failed_calls": 0, "model_transitions": 0}
    wdir = _ROOT / "artifacts" / "stage3d1" / "envs"
    env_results = {}
    stable_ids = [r["topology_id"] for r in pools["A1"] + pools["A2"]]
    for tid in stable_ids:  # env validation: baseline sim per family
        rec = _sim_family(tid, reg, exe, budget, wdir)
        pm = rec.extracted_metrics.get("phase_margin_deg") if rec else None
        env_results[tid] = {
            "spice_success": bool(rec and rec.extracted_metrics.get("dc_gain_db") is not None),
            "phase_margin_deg": pm,
            "stable": pm is not None and pm > 0}
    # Phase B/C/D-lite: one perturbation transition per family (real), grouped
    phase = {}
    for label, ids in (("B_a1", [r["topology_id"] for r in pools["A1"]]),
                       ("C_a2", [r["topology_id"] for r in pools["A2"]])):
        ok = 0
        for tid in ids:
            for _s in range(phase_steps):
                rec = _sim_family(tid, reg, exe, budget, wdir / "phase", {"w": 1.15})
                if rec and (rec.extracted_metrics.get("phase_margin_deg") or -1) > 0:
                    ok += 1
        phase[label] = {"families": len(ids), "stable_after_action": ok}
    # post-sizing score demo on 3 families (repeatability x2)
    scores = []
    for tid in stable_ids[:3]:
        for rep in range(2):
            calls0 = budget["real_spice_calls"]
            rec = _sim_family(tid, reg, exe, budget, wdir / f"score{rep}")
            m = rec.extracted_metrics if rec else {}
            pm = m.get("phase_margin_deg")
            margins = {"gain": ((m.get("dc_gain_db") or 0) - 40) / 40,
                       "pm": ((pm or -90) - 45) / 45}
            s = PostSizingTopologyScore(
                topology_id=tid, target_id="t_easy_0",
                status="successful_within_budget" if pm and pm > 0 and margins["gain"] > 0
                else "unstable" if pm is not None and pm <= 0 else "infeasible",
                verified_stable=bool(pm and pm > 0),
                feasible=bool(margins["gain"] > 0 and pm and pm > 45),
                metrics={k: v for k, v in m.items() if v is not None}, margins=margins,
                real_spice_calls=budget["real_spice_calls"] - calls0,
                calls_to_first_pass=1 if pm and pm > 0 else None, simulator_failures=0)
            s.compute_scalar()
            scores.append(asdict(s))
    (wdir.parent / "post_sizing_scores.jsonl").write_text(
        "\n".join(json.dumps(s, default=str) for s in scores), encoding="utf-8")
    rep_var = defaultdict(list)
    for s in scores:
        rep_var[s["topology_id"]].append(s["scalar_value"])
    summary = {
        "a1_manifests": {t: {k: v for k, v in r.items()} for t, r in a1.items()},
        "a1_roundtrip_all_identical": all(r["roundtrip_identical"] for r in a1.values()),
        "env_validation": env_results,
        "env_pass": sum(1 for r in env_results.values() if r["spice_success"]),
        "phase_results": phase, "budget": budget,
        "score_repeatability": {t: {"values": v, "spread": round(max(v) - min(v), 4)}
                                for t, v in rep_var.items()},
        "wall_clock_s": round(time.time() - started, 1),
    }
    (_ROOT / "artifacts" / "stage3d1" / "SUMMARY.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = run_stage3d1()
    print(json.dumps({k: v for k, v in out.items() if k != "a1_manifests"}, indent=1, default=str))
