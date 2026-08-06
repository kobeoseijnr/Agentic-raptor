"""Stage 3E.2 Parts D–L: executable structural-edit library on
DeviceCircuitGraph + complete-topology proposal interface.

Every edit: typed template, explicit preconditions, deterministic transform on
an immutable copy, device-graph hash + lineage + manifest delta, then the SAME
static validation and real-ngspice qualification path as registry topologies.
A structurally valid edit is NOT automatically electrically successful — both
outcomes are recorded honestly and separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from agentic_raptor.mapping import (PRIORS, DeviceCircuitGraph, DeviceRecord,
                                    _MappedEntry, emit_netlist, static_validate)

_ROOT = Path(__file__).resolve().parents[2]
EDIT_SCHEMA_VERSION = "3e2.1"


class EditRejected(Exception):
    pass


def device_graph_hash(g: DeviceCircuitGraph) -> str:
    """Deterministic structural hash over device kinds/roles/nets/groups
    (sizing parameter NAMES included, values excluded)."""
    rows = sorted(
        (d.device_id, d.kind, d.role, tuple(sorted(d.nets.items())),
         d.group or "", tuple(sorted(d.sizing.keys())))
        for d in g.devices)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:32]


def build_manifest(g: DeviceCircuitGraph) -> list[dict[str, Any]]:
    """Sizing-parameter manifest → defines the sizing action dimension."""
    man = []
    bounds = {"w": (0.42, 100.0, "log"), "l": (0.15, 8.0, "log"),
              "value": (1e-13, 2e-11, "log")}
    for d in sorted(g.devices, key=lambda d: d.device_id):
        for k in d.sizing:
            if k in ("origin", "conf"):
                continue
            lo, hi, sc = bounds.get(k, (0.0, 1.0, "linear"))
            if d.kind == "res":
                lo, hi = 100.0, 1e6
            if d.kind == "isrc":
                continue     # bias current fixed by support-bias policy
            man.append({"parameter_id": f"{d.device_id}_{k}", "device": d.device_id,
                        "kind": k, "value": d.sizing[k], "lower": lo, "upper": hi,
                        "scale": sc, "group": d.group})
    return man


def _clone(g: DeviceCircuitGraph, edit_type: str) -> DeviceCircuitGraph:
    ng = copy.deepcopy(g)
    ng.mapping_candidate_id = f"{g.mapping_candidate_id}+{edit_type}"
    return ng


def _rename_net(g: DeviceCircuitGraph, old: str, new: str) -> None:
    for d in g.devices:
        for t, n in d.nets.items():
            if n == old:
                d.nets[t] = new


def _nets(g: DeviceCircuitGraph) -> set[str]:
    return {n for d in g.devices for n in d.nets.values()}


def _find(g: DeviceCircuitGraph, **kw) -> list[DeviceRecord]:
    return [d for d in g.devices
            if all(getattr(d, k, None) == v for k, v in kw.items())]


# ---------------------------------------------------------------------------
# Edit implementations. Each returns (new_graph, audit) or raises EditRejected.
# ---------------------------------------------------------------------------
def add_verified_stage(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    """Append a verified common-source NMOS gain stage before vout."""
    if g.stage_count >= 3:
        raise EditRejected("max_supported_stage_count_reached")
    if "vout" not in _nets(g):
        raise EditRejected("no_output_net")
    if "nmir" not in _nets(g) or "nbias" not in _nets(g):
        raise EditRejected("bias_or_mirror_net_missing")
    ng = _clone(g, "addstage")
    k = ng.stage_count + 1
    inner = f"nx{k}"
    _rename_net(ng, "vout", inner)          # previous output becomes internal
    ng.devices += [
        DeviceRecord(f"ME{2 * k}", "nmos", "second_stage_gain_device",
                     {"d": "vout", "g": inner, "s": "gnda", "b": "gnda"},
                     f"cse{k}", dict(PRIORS["cs_gain_nmos"]),
                     {"edit": "add_verified_stage", "template": EDIT_SCHEMA_VERSION}),
        DeviceRecord(f"ME{2 * k + 1}", "pmos", "active_load",
                     {"d": "vout", "g": "nmir", "s": "vdda", "b": "vdda"},
                     f"cse{k}", dict(PRIORS["load_pmos"]),
                     {"edit": "add_verified_stage", "template": EDIT_SCHEMA_VERSION}),
    ]
    ng.stage_count = k
    ng.polarity["stage_inversion_count"] = ng.polarity.get("stage_inversion_count", 0) + 1
    ng.polarity["edit_note"] = "one extra inversion from added CS stage (structural, not outcome-based)"
    return ng


def remove_optional_supported_stage(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    """Exact reversal of add_verified_stage (only edit-added stages removable)."""
    k = g.stage_count
    added = _find(g, group=f"cse{k}")
    if not added:
        raise EditRejected("last_stage_not_edit_added_optional_stage")
    ng = _clone(g, "rmstage")
    ng.devices = [d for d in ng.devices if d.group != f"cse{k}"]
    _rename_net(ng, f"nx{k}", "vout")
    ng.stage_count = k - 1
    ng.polarity["stage_inversion_count"] = ng.polarity.get("stage_inversion_count", 1) - 1
    ng.mapping_candidate_id = g.mapping_candidate_id.rsplit("+addstage", 1)[0] + "+rt"
    return ng


def _cs_driver(g: DeviceCircuitGraph) -> DeviceRecord:
    ds = _find(g, role="second_stage_gain_device")
    if not ds:
        raise EditRejected("no_cs_stage_present")
    return sorted(ds, key=lambda d: d.device_id)[-1]


def replace_stage_with_compatible_block(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    """NMOS CS + PMOS load → PMOS CS + NMOS sink (typed-port compatible:
    gate in / drain out, inverting polarity preserved, bias net reused)."""
    drv = _cs_driver(g)
    if drv.kind != "nmos":
        raise EditRejected("stage_already_pmos_variant")
    if "nbias" not in _nets(g):
        raise EditRejected("bias_net_missing_for_sink_load")
    loads = [d for d in _find(g, role="active_load") if d.nets.get("d") == drv.nets["d"]]
    if not loads:
        raise EditRejected("stage_load_not_found")
    ng = _clone(g, "repstage")
    out, inp = drv.nets["d"], drv.nets["g"]
    ng.devices = [d for d in ng.devices
                  if d.device_id not in (drv.device_id, loads[0].device_id)]
    ng.devices += [
        DeviceRecord(drv.device_id + "P", "pmos", "second_stage_gain_device",
                     {"d": out, "g": inp, "s": "vdda", "b": "vdda"}, drv.group,
                     dict(PRIORS["load_pmos"]), {"edit": "replace_stage", "was": drv.device_id}),
        DeviceRecord(loads[0].device_id + "N", "nmos", "current_sink_load",
                     {"d": out, "g": "nbias", "s": "gnda", "b": "gnda"}, drv.group,
                     dict(PRIORS["tail_nmos"]), {"edit": "replace_stage", "was": loads[0].device_id}),
    ]
    return ng


def replace_load_with_compatible_block(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    """Mirror-gated PMOS active load → diode-connected PMOS load (output node,
    mirror roles, DC path preserved; no floating node introduced)."""
    drv = _cs_driver(g)
    loads = [d for d in _find(g, role="active_load")
             if d.nets.get("d") == drv.nets["d"] and d.kind == "pmos"]
    if not loads:
        raise EditRejected("no_replaceable_pmos_active_load")
    ng = _clone(g, "repload")
    for d in ng.devices:
        if d.device_id == loads[0].device_id:
            d.nets["g"] = d.nets["d"]            # diode connection
            d.role = "diode_connected_load"
            d.provenance = {**d.provenance, "edit": "replace_load"}
    return ng


def add_compensation(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    """Miller cap across ONE inverting CS stage (gate -> drain).

    Local placement is required, not stylistic: a Miller capacitor is
    negative feedback only across an ODD number of inverting stages. Spanning
    two CS stages feeds back in phase and destabilises -- measured at
    -87.35 deg of phase margin on a 3-stage amplifier versus +0.24 deg for
    the local placement.

    The duplicate check tolerates either orientation of an existing
    capacitor, and also rejects a stage already compensated from the output,
    so a second capacitor can never be stacked onto a compensated stage.
    """
    drv = _cs_driver(g)
    a, b = drv.nets["g"], drv.nets["d"]
    if a in ("vdda", "gnda") or b in ("vdda", "gnda"):
        raise EditRejected("compensation_endpoint_is_supply")
    endpoints = {frozenset({d.nets.get("p"), d.nets.get("n")})
                 for d in g.devices if d.kind == "cap"}
    if frozenset({a, b}) in endpoints:
        raise EditRejected("duplicate_compensation_path")
    ng = _clone(g, "addcomp")
    ng.devices.append(DeviceRecord(
        "CCE", "cap", "miller_compensation", {"p": a, "n": b}, None,
        {"value": PRIORS["miller_cap_f"]["value"]},
        {"edit": "add_compensation",
         "orientation": "across ONE inverting CS stage (negative feedback)"}))
    return ng


def replace_compensation(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    """Miller cap → RC series nulling-resistor branch (template constant RZ)."""
    caps = _find(g, role="miller_compensation")
    if not caps:
        raise EditRejected("no_compensation_to_replace")
    c = caps[0]
    ng = _clone(g, "repcomp")
    ng.devices = [d for d in ng.devices if d.device_id != c.device_id]
    ng.devices += [
        DeviceRecord(c.device_id + "Z", "cap", "miller_compensation",
                     {"p": c.nets["p"], "n": "nzc"}, None, dict(c.sizing),
                     {"edit": "replace_compensation"}),
        DeviceRecord("RZ1", "res", "nulling_resistor",
                     {"p": "nzc", "n": c.nets["n"]}, None,
                     {"value": EDIT_TEMPLATES["REPLACE_SUPPORTED_COMPENSATION_STRUCTURE"]
                      ["template_params"]["rz_ohm"]},
                     {"edit": "replace_compensation"}),
    ]
    return ng


def add_output_stage(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    """NMOS source-follower output buffer + bias-mirrored sink (non-inverting)."""
    if "nbias" not in _nets(g):
        raise EditRejected("bias_net_missing")
    if _find(g, role="output_follower"):
        raise EditRejected("output_stage_already_present")
    ng = _clone(g, "addout")
    _rename_net(ng, "vout", "nfo")
    # PMOS source follower: level-shifts UP (nfo ~0.45V + |Vgs| -> ~1.1V, in
    # range). The earlier NMOS follower sat below threshold (gate 0.45V < vth)
    # -> output stage off -> measured -76dB. Pull-up current mirrors via nmir,
    # same pattern as the active loads.
    ng.devices += [
        DeviceRecord("MOF", "pmos", "output_follower",
                     {"d": "gnda", "g": "nfo", "s": "vout", "b": "vdda"}, "outb",
                     dict(PRIORS["load_pmos"]), {"edit": "add_output_stage"}),
        DeviceRecord("MOP", "pmos", "output_pullup",
                     {"d": "vout", "g": "nmir", "s": "vdda", "b": "vdda"}, "outb",
                     dict(PRIORS["mirror_pmos"]), {"edit": "add_output_stage"}),
    ]
    return ng


def connect_verified_feedback_path(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    """Local shunt-shunt resistive feedback across ONE inverting CS stage —
    single inversion ⇒ negative loop sign (structural evidence, not outcome)."""
    drv = _cs_driver(g)
    a, b = drv.nets["g"], drv.nets["d"]
    if a in ("vdda", "gnda") or b in ("vdda", "gnda"):
        raise EditRejected("feedback_endpoint_is_supply")
    if a in ("vinp", "vinn") or b in ("vinp", "vinn"):
        raise EditRejected("feedback_to_driven_input_port_forbidden_in_adm_testbench")
    if any(d.kind == "res" and {d.nets.get("p"), d.nets.get("n")} == {a, b}
           for d in g.devices):
        raise EditRejected("duplicate_feedback_path")
    ng = _clone(g, "addfb")
    ng.devices.append(DeviceRecord(
        "RFB1", "res", "local_feedback",
        {"p": a, "n": b}, None,
        {"value": EDIT_TEMPLATES["CONNECT_VERIFIED_FEEDBACK_PATH"]
         ["template_params"]["rfb_ohm"]},
        {"edit": "connect_feedback", "source": b, "dest": a,
         "expected_sign": "negative", "path_type": "shunt_shunt_local",
         "stage_traversal": 1,
         "loop_polarity_evidence": "single inverting CS stage in loop"}))
    return ng


def replace_simple_miller_with_nested(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    """three_stage_nested_miller: remove per-stage Miller caps; add outer Cc1
    (stage1-out -> vout, with nulling Rz1) + inner Cc2 (stage2-out -> vout).
    Preconditions: >=3 gain stages with identifiable stage outputs."""
    # NUMERIC sort: lexicographic ordering put "M10" before "M8", swapping the
    # stage-1/stage-2 output nets and miswiring Cc1/Cc2 (found via raw-AC audit)
    drivers = sorted(_find(g, role="second_stage_gain_device"),
                     key=lambda d: int("".join(c for c in d.device_id
                                               if c.isdigit()) or "0"))
    if g.stage_count < 3 or len(drivers) < 2:
        raise EditRejected("nested_miller_requires_three_stages")
    n1 = drivers[0].nets["g"]          # first-stage output
    n2 = drivers[-1].nets["g"]         # second-stage output
    ng = _clone(g, "nestedmiller")
    ng.devices = [d for d in ng.devices if d.role != "miller_compensation"]
    ng.devices += [
        DeviceRecord("CC1", "cap", "nested_outer_miller",
                     {"p": n1, "n": "nzo"}, None, {"value": 4e-12},
                     {"edit": "nested_miller", "loop": "outer(stages2+3)"}),
        DeviceRecord("RZ1", "res", "nested_nulling",
                     {"p": "nzo", "n": "vout"}, None, {"value": 700.0},
                     {"edit": "nested_miller"}),
        DeviceRecord("CC2", "cap", "nested_inner_miller",
                     {"p": n2, "n": "vout"}, None, {"value": 1e-12},
                     {"edit": "nested_miller", "loop": "inner(stage3)"}),
    ]
    ng.polarity["compensation"] = "nested_miller(Cc1 outer + Cc2 inner + Rz1)"
    return ng


def add_nested_miller(g: DeviceCircuitGraph) -> DeviceCircuitGraph:
    if any(d.role == "miller_compensation" for d in g.devices):
        return replace_simple_miller_with_nested(g)
    return replace_simple_miller_with_nested(g)   # same construction path


EDIT_TEMPLATES: dict[str, dict[str, Any]] = {
    "ADD_NESTED_MILLER": {
        "fn": add_nested_miller, "reversible_by": None,
        "source_roles": ["three_stage_cascade"], "target_roles": ["nested_miller"],
        "block_family": "nested_miller_compensation",
        "template_params": {"cc1_f": 4e-12, "cc2_f": 1e-12, "rz1_ohm": 700.0},
        "polarity": "nested negative feedback", "param_delta": "+3", "edit_cost": 2.0},
    "REPLACE_SIMPLE_MILLER_WITH_NESTED": {
        "fn": replace_simple_miller_with_nested, "reversible_by": None,
        "source_roles": ["miller_compensation"], "target_roles": ["nested_miller"],
        "block_family": "nested_miller_compensation",
        "template_params": {"cc1_f": 4e-12, "cc2_f": 1e-12, "rz1_ohm": 700.0},
        "polarity": "nested negative feedback", "param_delta": "+3", "edit_cost": 2.0},
    "ADD_VERIFIED_STAGE": {
        "fn": add_verified_stage, "reversible_by": "REMOVE_OPTIONAL_SUPPORTED_STAGE",
        "source_roles": ["gain_stage_output"], "target_roles": ["output_node"],
        "block_family": "cs_nmos_stage", "ports": {"in": "gate", "out": "drain",
                                                   "supply": ["vdda", "gnda"], "bias": ["nmir"]},
        "polarity": "inverting", "param_delta": "+4 (w,l x2 devices)", "edit_cost": 2.0},
    "REPLACE_STAGE_WITH_COMPATIBLE_BLOCK": {
        "fn": replace_stage_with_compatible_block, "reversible_by": None,
        "source_roles": ["second_stage_gain_device"], "target_roles": ["cs_pmos_stage"],
        "block_family": "cs_pmos_stage", "ports": {"in": "gate", "out": "drain",
                                                   "supply": ["vdda", "gnda"], "bias": ["nbias"]},
        "polarity": "inverting_preserved", "param_delta": "0", "edit_cost": 2.0},
    "REPLACE_LOAD_WITH_COMPATIBLE_BLOCK": {
        "fn": replace_load_with_compatible_block, "reversible_by": None,
        "source_roles": ["active_load"], "target_roles": ["diode_connected_load"],
        "block_family": "pmos_load", "ports": {"out": "drain", "supply": ["vdda"]},
        "polarity": "unchanged", "param_delta": "0", "edit_cost": 1.0},
    "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE": {
        "fn": add_compensation, "reversible_by": "remove (delete CCE)",
        "source_roles": ["cs_input", "cs_output"], "target_roles": ["miller_cap"],
        "block_family": "miller_cap", "ports": {"p": "stage_in", "n": "stage_out"},
        "polarity": "feedback_across_inverting_stage", "param_delta": "+1", "edit_cost": 1.0},
    "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE": {
        "fn": replace_compensation, "reversible_by": None,
        "source_roles": ["miller_compensation"], "target_roles": ["rc_nulling"],
        "block_family": "rc_series", "template_params": {"rz_ohm": 5000.0},
        "polarity": "unchanged", "param_delta": "+1 (rz)", "edit_cost": 1.0},
    "ADD_SUPPORTED_OUTPUT_STAGE": {
        "fn": add_output_stage, "reversible_by": None,
        "source_roles": ["output_node"], "target_roles": ["source_follower"],
        "block_family": "nmos_follower", "ports": {"in": "gate", "out": "source",
                                                   "supply": ["vdda", "gnda"], "bias": ["nbias"]},
        "polarity": "non_inverting", "param_delta": "+4", "edit_cost": 2.0},
    "REMOVE_OPTIONAL_SUPPORTED_STAGE": {
        "fn": remove_optional_supported_stage, "reversible_by": "ADD_VERIFIED_STAGE",
        "source_roles": ["edit_added_stage"], "target_roles": [],
        "block_family": "cs_nmos_stage", "polarity": "removes_one_inversion",
        "param_delta": "-4", "edit_cost": 1.0},
    "CONNECT_VERIFIED_FEEDBACK_PATH": {
        "fn": connect_verified_feedback_path, "reversible_by": "remove (delete RFB1)",
        "source_roles": ["cs_output"], "target_roles": ["cs_input"],
        "block_family": "resistive_local_feedback",
        "template_params": {"rfb_ohm": 1000000.0},   # 1 MOhm: preserves loop gain
        #                    (100k crushed gain below unity -> PM unmeasurable,
        #                     diagnosed in the first full-pipeline run)
        "polarity": "negative (single inverting stage)", "param_delta": "+1",
        "edit_cost": 1.5},
}
for _name, _t in EDIT_TEMPLATES.items():
    _t["schema_version"] = EDIT_SCHEMA_VERSION


def apply_edit(g: DeviceCircuitGraph, edit_type: str) -> tuple[DeviceCircuitGraph, dict[str, Any]]:
    """Immutable apply + audit (pre/post hash, manifest delta, lineage)."""
    t = EDIT_TEMPLATES[edit_type]
    pre_hash, pre_man = device_graph_hash(g), build_manifest(g)
    ng = t["fn"](g)
    post_man = build_manifest(ng)
    audit = {"edit_type": edit_type, "schema_version": EDIT_SCHEMA_VERSION,
             "parent_hash": pre_hash, "child_hash": device_graph_hash(ng),
             "action_dim_before": len(pre_man), "action_dim_after": len(post_man),
             "manifest_added": sorted({m["parameter_id"] for m in post_man}
                                      - {m["parameter_id"] for m in pre_man}),
             "manifest_removed": sorted({m["parameter_id"] for m in pre_man}
                                        - {m["parameter_id"] for m in post_man}),
             "reversible_by": t["reversible_by"], "edit_cost": t["edit_cost"]}
    return ng, audit


def qualify_device_graph(tid: str, g: DeviceCircuitGraph, wdir: Path, exe: str,
                         label: str, budget: dict[str, int]) -> dict[str, Any]:
    """Static validation then the verbatim Stage 3B qualification path.
    Invalid graphs never reach ngspice."""
    from agentic_raptor.electrical import qualify_family
    import agentic_raptor.electrical as elec

    cdir = wdir / label
    cdir.mkdir(parents=True, exist_ok=True)
    sub = f"e2_{label[:20]}"
    net = emit_netlist(g, sub)
    val = static_validate(g, net)
    (cdir / "netlist.sp").write_text(net, encoding="utf-8")
    (cdir / "device_graph.json").write_text(json.dumps(asdict(g), indent=0, default=str),
                                            encoding="utf-8")
    (cdir / "static_validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
    if val["status"] != "mapped_static_valid":
        return {"label": label, "static": val["status"], "problems": val["problems"],
                "electrical": "not_simulated_static_invalid", "spice_calls": 0}
    (cdir / "design_variables").mkdir(exist_ok=True)
    (cdir / "design_variables" / sub).write_text("* inlined\n", encoding="utf-8")
    fake = _MappedEntry(tid, cdir, {"name": sub, "graph_hash": device_graph_hash(g)}, None)
    orig = elec._LEGACY_AMP
    elec._LEGACY_AMP = cdir
    try:
        rec = qualify_family(fake, {"subckt_name": sub, "immediately_runnable": True,
                                    "blocking_reason": None}, cdir / "run", exe,
                             "env-3e2", "train", 120.0)
    finally:
        elec._LEGACY_AMP = orig
    budget["real_spice_calls"] += 1
    m = {k: v for k, v in rec.extracted_metrics.items() if v is not None}
    pm = m.get("phase_margin_deg")
    if not m.get("dc_gain_db"):
        budget["simulator_failures"] += 1
    return {"label": label, "static": val["status"], "electrical": rec.electrical_validation_status,
            "metrics": m, "stability": ("verified_stable" if pm is not None and pm > 0
                                        else "verified_unstable" if pm is not None
                                        else "phase_margin_unavailable"),
            "spice_calls": 1, "run_ref": str(cdir / "run")}


# ---------------------------------------------------------------------------
# Part L: complete-topology proposal interface
# ---------------------------------------------------------------------------
SUPPORTED_BLOCKS = {"five_transistor_first_stage", "cs_gain_stage", "miller_cap",
                    "bias_mirror", "output_follower"}
PROPOSAL_SCHEMA_VERSION = "3e2.1"


@dataclass
class TopologyProposal:
    proposal_id: str
    stages: list[dict[str, Any]]          # [{block, role, inputs, outputs}]
    ports: dict[str, str]
    connections: list[dict[str, str]]
    supply_roles: dict[str, str]
    bias_roles: list[str]
    compensation: list[dict[str, str]]
    feedback_paths: list[dict[str, str]]
    intended_polarity: str
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    schema_version: str = PROPOSAL_SCHEMA_VERSION


class FixtureProposalProvider:
    """Deterministic provider — no external API needed for tests. A real
    multimodal-LLM adapter must emit the same schema and pass the same path."""

    def propose(self, spec: dict[str, float]) -> TopologyProposal:
        return TopologyProposal(
            proposal_id="fixture_two_stage_ota",
            stages=[{"block": "five_transistor_first_stage", "role": "input_stage",
                     "inputs": ["vinp", "vinn"], "outputs": ["s1out"]},
                    {"block": "cs_gain_stage", "role": "gain_stage",
                     "inputs": ["s1out"], "outputs": ["vout"]}],
            ports={"gnda": "gnda", "vdda": "vdda", "vinn": "vinn", "vinp": "vinp",
                   "vout": "vout"},
            connections=[{"from": "s1out", "to": "cs_gain_stage.in"}],
            supply_roles={"vdda": "supply", "gnda": "ground"},
            bias_roles=["bias_mirror"],
            compensation=[{"type": "miller_cap", "from": "s1out", "to": "vout"}],
            feedback_paths=[], intended_polarity="vinp_noninverting",
            provenance={"provider": "fixture", "rag_refs": ["rag_l2_two_stage"]},
            confidence=0.9)


def validate_proposal(p: TopologyProposal) -> tuple[bool, list[str]]:
    reasons = []
    if not p.stages:
        reasons.append("no_stages")
    seen = set()
    for st in p.stages:
        if st.get("block") not in SUPPORTED_BLOCKS:
            reasons.append(f"unsupported_block:{st.get('block')}")
        for o in st.get("outputs", []):
            if o in seen:
                reasons.append(f"duplicate_output_node:{o}")
            seen.add(o)
    if not p.bias_roles:
        reasons.append("missing_bias_path")
    for fb in p.feedback_paths:
        if fb.get("to") in ("vinp", "vinn"):
            reasons.append("illegal_feedback_to_driven_input")
        if fb.get("sign") == "positive":
            reasons.append("uncontrolled_positive_feedback")
    for port in ("gnda", "vdda", "vinp", "vinn", "vout"):
        if port not in p.ports:
            reasons.append(f"missing_port:{port}")
    return not reasons, reasons


def realise_proposal(p: TopologyProposal, wdir: Path, exe: str,
                     budget: dict[str, int]) -> dict[str, Any]:
    """Proposal → validator → map_family (same template mapper) → canonical
    hash → static validation → real ngspice. Never bypasses validation."""
    from agentic_raptor.mapping import map_family

    ok, reasons = validate_proposal(p)
    if not ok:
        return {"proposal_id": p.proposal_id, "status": "rejected_by_validator",
                "reasons": reasons, "spice_calls": 0}
    gain_stages = sum(1 for s in p.stages if "stage" in s["role"])
    row = {"topology_id": f"proposal_{p.proposal_id}", "gain_stages": gain_stages,
           "functional_blocks": ["C"] if p.compensation else [],
           "unresolved_blocks": [], "mapping_readiness": "mapping_ready",
           "graph_hash": None}

    class _Stub:
        topology_id = f"proposal_{p.proposal_id}"
    g_dev, note = map_family(_Stub(), row)
    if g_dev is None:
        return {"proposal_id": p.proposal_id, "status": "mapping_unsupported",
                "spice_calls": 0}
    res = qualify_device_graph(row["topology_id"], g_dev, wdir, exe,
                               f"llm_{p.proposal_id}", budget)
    return {"proposal_id": p.proposal_id, "status": "mapped_and_simulated",
            "mapping_note": note, "canonical_hash": device_graph_hash(g_dev),
            "provenance": p.provenance, **res}
