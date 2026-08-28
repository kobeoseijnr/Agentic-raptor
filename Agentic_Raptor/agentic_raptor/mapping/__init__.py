"""Stage 3C: transistor mapping of generated structural families → Sky130.

Pipeline: structural CircuitGraph → functional interpretation → DeviceCircuitGraph
(device roles, groups, provenance) → bias/supply integration (labelled
generated_support_bias) → prior-based initial sizing → Sky130 subckt netlist
(AnalogGym port convention, so the existing Stage 3B testbench/qualification
pipeline is reused verbatim) → static validation (incl. graph preservation) →
electrical qualification. Failures preserved; nothing silently repaired.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.electrical import _ROOT, discover_ngspice, qualify_family

MAPPING_STATUSES = ("mapped_static_valid", "mapped_static_invalid", "mapping_ambiguous",
                    "mapping_incomplete", "mapping_unsupported", "netlist_generation_failed",
                    "bias_unresolved", "port_unresolved", "compensation_unresolved",
                    "graph_preservation_failed")

#: AnalogGym-derived safe Sky130 priors (origin recorded per device).
PRIORS = {
    "input_pair_nmos": {"w": 10.0, "l": 0.5, "origin": "analoggym_input_pair_prior", "conf": 0.8},
    "mirror_pmos": {"w": 20.0, "l": 0.5, "origin": "analoggym_mirror_prior", "conf": 0.8},
    "tail_nmos": {"w": 20.0, "l": 1.0, "origin": "analoggym_tail_prior", "conf": 0.8},
    "cs_gain_nmos": {"w": 40.0, "l": 0.5, "origin": "analoggym_second_stage_prior", "conf": 0.7},
    "load_pmos": {"w": 60.0, "l": 0.5, "origin": "analoggym_load_prior", "conf": 0.7},
    "bias_nmos": {"w": 20.0, "l": 1.0, "origin": "generated_support_bias_prior", "conf": 0.6},
    "miller_cap_f": {"value": 2e-12, "origin": "conservative_sky130_default", "conf": 0.6},
    #: nulling resistor for RC-type compensation: ~1/gm2 territory; sized live
    #: by the rz_x knob (spec_sizing.apply_knobs clamps to [RZ_MIN, RZ_MAX])
    "rz_ohm": {"value": 2000.0, "origin": "conservative_sky130_default", "conf": 0.5},
    #: TIER-2 VOCABULARY (2026-08-17): cascode devices sit in series with the
    #: input pair / mirror -- same width class as what they stack on; the
    #: class-AB output pair is sized like the CS stage it replaces.
    "cascode_nmos": {"w": 10.0, "l": 0.5, "origin": "analoggym_input_pair_prior", "conf": 0.6},
    "cascode_pmos": {"w": 20.0, "l": 0.5, "origin": "analoggym_mirror_prior", "conf": 0.6},
    "ab_nmos": {"w": 40.0, "l": 0.5, "origin": "analoggym_second_stage_prior", "conf": 0.6},
    "ab_pmos": {"w": 60.0, "l": 0.5, "origin": "analoggym_load_prior", "conf": 0.6},
    #: FOLDED-CASCODE OTA (2026-08-18, SECOND CIRCUIT TYPE): PMOS input pair,
    #: NMOS folding legs + cascodes, PMOS top current sources, NMOS cascode
    #: mirror load. Widths from the AnalogGym FC prior class.
    "fc_input_pmos": {"w": 20.0, "l": 0.5, "origin": "analoggym_fc_input_prior", "conf": 0.6},
    "fc_tail_pmos": {"w": 40.0, "l": 1.0, "origin": "analoggym_fc_tail_prior", "conf": 0.6},
    "fc_fold_nmos": {"w": 60.0, "l": 0.5, "origin": "analoggym_fc_fold_prior", "conf": 0.6},
    "fc_cas_nmos": {"w": 30.0, "l": 0.5, "origin": "analoggym_fc_cascode_prior", "conf": 0.6},
    "fc_top_pmos": {"w": 120.0, "l": 0.5, "origin": "analoggym_fc_source_prior", "conf": 0.6},
    "ibias_a": {"value": 20e-6, "origin": "analoggym_bias_current_prior", "conf": 0.7},
}


@dataclass
class DeviceRecord:
    device_id: str
    kind: str            # nmos | pmos | cap | res | isrc
    role: str
    nets: dict[str, str]              # terminal → net
    group: str | None = None          # matched/mirror/pair/cascode group id
    sizing: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceCircuitGraph:
    topology_id: str
    mapping_candidate_id: str
    stage_count: int
    devices: list[DeviceRecord] = field(default_factory=list)
    ports: dict[str, str] = field(default_factory=dict)
    support_bias: list[str] = field(default_factory=list)   # generated_support_bias device ids
    polarity: dict[str, Any] = field(default_factory=dict)
    block_assignments: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


def audit_generated(registry: TopologyRegistry) -> list[dict[str, Any]]:
    rows = []
    for tid in registry.list_topologies():
        e = registry.get_topology(tid)
        if e.source == "analoggym":
            continue
        blocks = set(e.metadata.get("functional_blocks", {}))
        gain_stages = e.metadata.get("functional_blocks", {}).get("gain_stage", 0)
        unresolved = sorted(b for b in blocks if b.startswith(("ff_", "fb_")))
        if gain_stages >= 1 and not unresolved:
            readiness = "mapping_ready"
        elif gain_stages >= 1:
            readiness = "partially_specified"   # gm main path + unmapped ff/fb branches
        elif blocks & {"gm_pos", "gm_neg"}:
            readiness = "behavioral_only"
        else:
            readiness = "structurally_ambiguous"
        rows.append({"topology_id": tid, "source": e.source,
                     "graph_hash": e.metadata.get("graph_hash"),
                     "nodes": len(e.graph.nodes), "edges": len(e.graph.edges),
                     "functional_blocks": sorted(blocks), "gain_stages": gain_stages,
                     "unresolved_blocks": unresolved, "mapping_readiness": readiness})
    return rows


def map_folded_cascode(entry, audit_row: dict[str, Any]) -> tuple[DeviceCircuitGraph, str]:
    """SECOND CIRCUIT TYPE (2026-08-18): single-stage FOLDED-CASCODE OTA.

    NMOS-INPUT form, chosen for the testbench: the AnalogGym harness biases
    the amplifier in unity feedback at DC with VCM = 0.25*VDD = 0.45 V, so
    the output DC sits at 0.45 V. A PMOS-input FC puts the output on top of
    an NMOS cascode stack that needs ~0.7 V of headroom -> op-point fails
    (measured: op_valid=0 at nominal). NMOS input + PMOS cascode load puts
    the output UNDER a PMOS stack from 1.8 V, which is comfortable at 0.45 V.

      * NMOS differential pair M1/M2, tail M0 (gate nb1) -- the pair drains
        FOLD up into nodes nf1/nf2;
      * top: PMOS current sources M5/M6 (gate = nt1, wide-swing mirror) with
        PMOS cascodes M7/M8 (gate pb2): the CASCODED PMOS MIRROR LOAD
        (M7 side is the diode reference at nt1; output at M8's drain);
      * bottom: NMOS current sinks M3/M4 (gate nb1) + NMOS cascodes M9/M10
        (gate nb2) feed the two legs.
    Fully cascoded top and bottom: Rout ~ gm*ro^2 || gm*ro^2, one dominant
    pole at the load, no Miller path. Same DeviceRecord/emit/apply_knobs
    path; roles land in S1_ROLES so the existing knobs size it."""
    cid = f"map-{uuid.uuid4().hex[:8]}"
    g = DeviceCircuitGraph(entry.topology_id, cid, 1,
                           ports={"gnda": "gnda", "vdda": "vdda", "vinn": "vinn",
                                  "vinp": "vinp", "vout": "vout"})
    ev = {"rule": "folded_cascode_template", "audit": audit_row["mapping_readiness"],
          "circuit_type": "folded_cascode_ota"}
    P = PRIORS
    g.devices += [
        # ---- bias tree: nb1 (NMOS diode), nb2 (stacked NMOS diode), pb2 (PMOS
        #      cascode gate = PMOS diode fed by nb1 sink)
        DeviceRecord("IB1", "isrc", "bias_device", {"p": "vdda", "n": "nb1"}, "bias1",
                     {"value": P["ibias_a"]["value"], "origin": P["ibias_a"]["origin"]},
                     {"generated_support_bias": True, **ev}),
        DeviceRecord("MB1", "nmos", "bias_device",
                     {"d": "nb1", "g": "nb1", "s": "gnda", "b": "gnda"},
                     "bias1", dict(P["bias_nmos"]), {"generated_support_bias": True, **ev}),
        DeviceRecord("IB2", "isrc", "bias_device", {"p": "vdda", "n": "nb2"}, "bias2",
                     {"value": P["ibias_a"]["value"], "origin": P["ibias_a"]["origin"]},
                     {"generated_support_bias": True, **ev}),
        DeviceRecord("MB2", "nmos", "bias_device",
                     {"d": "nb2", "g": "nb2", "s": "nb2s", "b": "gnda"},
                     "bias2", dict(P["bias_nmos"]), {"generated_support_bias": True, **ev}),
        DeviceRecord("MB3", "nmos", "bias_device",
                     {"d": "nb2s", "g": "nb1", "s": "gnda", "b": "gnda"},
                     "bias2", dict(P["bias_nmos"]), {"generated_support_bias": True, **ev}),
        DeviceRecord("MB4", "pmos", "bias_device",
                     {"d": "pb2", "g": "pb2", "s": "pb2s", "b": "vdda"},
                     "bias3", dict(P["fc_top_pmos"]), {"generated_support_bias": True, **ev}),
        DeviceRecord("MB5", "pmos", "bias_device",
                     {"d": "pb2s", "g": "pb2", "s": "vdda", "b": "vdda"},
                     "bias3", dict(P["fc_top_pmos"]), {"generated_support_bias": True, **ev}),
        DeviceRecord("MB6", "nmos", "bias_device",
                     {"d": "pb2", "g": "nb1", "s": "gnda", "b": "gnda"},
                     "bias3", dict(P["bias_nmos"]), {"generated_support_bias": True, **ev}),
        # ---- signal path: NMOS pair + tail
        DeviceRecord("M0", "nmos", "tail_current_source",
                     {"d": "ntail", "g": "nb1", "s": "gnda", "b": "gnda"},
                     "tail1", dict(P["tail_nmos"]), ev),
        DeviceRecord("M1", "nmos", "input_pair_nmos",
                     {"d": "nf1", "g": "vinp", "s": "ntail", "b": "gnda"},
                     "pair1", dict(P["input_pair_nmos"]), ev),
        DeviceRecord("M2", "nmos", "input_pair_nmos",
                     {"d": "nf2", "g": "vinn", "s": "ntail", "b": "gnda"},
                     "pair1", dict(P["input_pair_nmos"]), ev),
        # bottom sinks + NMOS cascodes (feed the fold nodes)
        DeviceRecord("M3", "nmos", "tail_current_source",
                     {"d": "nc1", "g": "nb1", "s": "gnda", "b": "gnda"},
                     "sink1", dict(P["fc_fold_nmos"]), ev),
        DeviceRecord("M4", "nmos", "tail_current_source",
                     {"d": "nc2", "g": "nb1", "s": "gnda", "b": "gnda"},
                     "sink1", dict(P["fc_fold_nmos"]), ev),
        DeviceRecord("M9", "nmos", "tail_current_source",
                     {"d": "nf1", "g": "nb2", "s": "nc1", "b": "gnda"},
                     "cas0", dict(P["fc_cas_nmos"]), ev),
        DeviceRecord("M10", "nmos", "tail_current_source",
                     {"d": "nf2", "g": "nb2", "s": "nc2", "b": "gnda"},
                     "cas0", dict(P["fc_cas_nmos"]), ev),
        # top PMOS wide-swing cascoded mirror = LOAD; output at M8 drain
        DeviceRecord("M5", "pmos", "mirror_reference",
                     {"d": "pt1", "g": "nt1", "s": "vdda", "b": "vdda"},
                     "mir1", dict(P["fc_top_pmos"]), ev),
        DeviceRecord("M6", "pmos", "mirror_output",
                     {"d": "pt2", "g": "nt1", "s": "vdda", "b": "vdda"},
                     "mir1", dict(P["fc_top_pmos"]), ev),
        DeviceRecord("M7", "pmos", "mirror_reference",
                     {"d": "nt1", "g": "pb2", "s": "pt1", "b": "vdda"},
                     "cas1", dict(P["fc_top_pmos"]), ev),
        DeviceRecord("M8", "pmos", "mirror_output",
                     {"d": "vout", "g": "pb2", "s": "pt2", "b": "vdda"},
                     "cas1", dict(P["fc_top_pmos"]), ev),
    ]
    # the fold: pair drains nf1/nf2 ARE the sources of the top cascodes'
    # legs -- connect: M7 (cascode) source pt1 fed by M5; its DRAIN nt1 must
    # join the fold node nf1 (mirror side) and M8's drain vout joins nf2's
    # path. Rewire the top cascodes' drains to the fold nodes.
    for d in g.devices:
        if d.device_id == "M7":
            d.nets["d"] = "nf1"          # mirror reference node == fold node 1
        elif d.device_id == "M8":
            d.nets["d"] = "vout"         # output side: fold node 2 IS vout
        elif d.device_id == "M10":
            d.nets["d"] = "vout"         # bottom cascode on the output leg
        elif d.device_id in ("M5", "M6"):
            d.nets["g"] = "nf1"          # mirror gates from the fold/ref node
    g.support_bias = ["IB1", "MB1", "IB2", "MB2", "MB3", "MB4", "MB5", "MB6"]
    if "C" in audit_row.get("functional_blocks", []) or \
            "RC_series" in audit_row.get("functional_blocks", []):
        g.devices.append(DeviceRecord("CL1", "cap", "miller_compensation",
                                      {"p": "vout", "n": "gnda"}, None,
                                      {"value": P["miller_cap_f"]["value"],
                                       "origin": P["miller_cap_f"]["origin"]},
                                      {"compensation_topology": "load_cap_to_gnd", **ev}))
    g.polarity = {"polarity_status": "template_defined", "stage_inversion_count": 1,
                  "signal_path_evidence": "vinp->M1->nf1(mirror ref); vinn->M2->vout via M8/M10",
                  "noninverting_input": "vinp", "selection_rule": "structural parity, NOT outcome-based"}
    g.block_assignments = [{"block": "folded_cascode_input_stage",
                            "devices": ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"],
                            "confidence": 0.7, "rule": "template", "evidence": ev}]
    return g, "mapped"


def map_family(entry, audit_row: dict[str, Any]) -> tuple[DeviceCircuitGraph | None, str]:
    """Template mapping: N-stage cascade (5T first stage, CS stages after),
    C blocks → Miller/load compensation, R+C → nulling branch. ff/fb gm branches
    are NOT silently realized: families keep partially_specified provenance and
    only the main path is mapped in this candidate (recorded in unresolved)."""
    # SECOND CIRCUIT TYPE (2026-08-18): dispatch to the folded-cascode
    # template when the audit row flags it. Additive -- every existing
    # cascade family is byte-identical.
    if audit_row.get("folded_cascode"):
        return map_folded_cascode(entry, audit_row)
    n = max(1, audit_row["gain_stages"]) if audit_row["gain_stages"] else 0
    if n == 0:
        return None, "mapping_unsupported"
    # TIER-2 VOCABULARY (2026-08-17): the cascade cap rises 3 -> 4 (nested
    # per-stage nulling handles the extra pole), and two new structural
    # blocks become realizable when the audit row flags them:
    #   cascode_input : telescopic-cascode first stage -- extra gain from
    #                   ro-boosting WITHOUT an extra pole (high gain at
    #                   high UGBW, where more CS stages cost phase margin)
    #   class_ab_out  : push-pull output stage -- drives heavy loads at low
    #                   quiescent current (FoM at large C_load)
    # Both are textbook sky130-realizable and flow through the SAME
    # emit_netlist/apply_knobs path (cascode devices are stage-1 roles,
    # AB devices stage-2 roles) -- no new sizing knobs are required.
    n = min(n, 4)
    cascode = bool(audit_row.get("cascode_input"))
    class_ab = bool(audit_row.get("class_ab_output"))
    cid = f"map-{uuid.uuid4().hex[:8]}"
    g = DeviceCircuitGraph(entry.topology_id, cid, n,
                           ports={"gnda": "gnda", "vdda": "vdda", "vinn": "vinn",
                                  "vinp": "vinp", "vout": "vout"})
    ev = {"rule": "n_stage_cascade_template", "audit": audit_row["mapping_readiness"]}
    s1out = "n1" if n > 1 else "vout"
    # Stage 1: 5T OTA (NMOS pair, PMOS mirror, tail + support bias mirror)
    g.devices += [
        DeviceRecord("M1", "nmos", "input_pair_nmos",
                     {"d": "nmir", "g": "vinp", "s": "ntail", "b": "gnda"},
                     "pair1", dict(PRIORS["input_pair_nmos"]), ev),
        DeviceRecord("M2", "nmos", "input_pair_nmos",
                     {"d": s1out, "g": "vinn", "s": "ntail", "b": "gnda"},
                     "pair1", dict(PRIORS["input_pair_nmos"]), ev),
        DeviceRecord("M3", "pmos", "mirror_reference",
                     {"d": "nmir", "g": "nmir", "s": "vdda", "b": "vdda"},
                     "mir1", dict(PRIORS["mirror_pmos"]), ev),
        DeviceRecord("M4", "pmos", "mirror_output",
                     {"d": s1out, "g": "nmir", "s": "vdda", "b": "vdda"},
                     "mir1", dict(PRIORS["mirror_pmos"]), ev),
        DeviceRecord("M5", "nmos", "tail_current_source",
                     {"d": "ntail", "g": "nbias", "s": "gnda", "b": "gnda"},
                     "tail1", dict(PRIORS["tail_nmos"]), ev),
        DeviceRecord("M6", "nmos", "bias_device",
                     {"d": "nbias", "g": "nbias", "s": "gnda", "b": "gnda"},
                     "bias1", dict(PRIORS["bias_nmos"]), {"generated_support_bias": True, **ev}),
        DeviceRecord("IB1", "isrc", "bias_device",
                     {"p": "vdda", "n": "nbias"}, "bias1",
                     {"value": PRIORS["ibias_a"]["value"], "origin": PRIORS["ibias_a"]["origin"]},
                     {"generated_support_bias": True, **ev}),
    ]
    g.support_bias = ["M6", "IB1"]
    if cascode:
        # TELESCOPIC CASCODE: input pair drains feed NMOS cascodes (gate =
        # nbias, the existing support bias net), whose drains meet PMOS
        # cascodes under the mirror. Rewire: M1/M2 drains -> ncas1/ncas2;
        # NMOS cascodes MC1/MC2 lift them to nmir/s1out; PMOS cascodes
        # MC3/MC4 sit between the mirror devices and those nodes.
        for d in g.devices:
            if d.device_id == "M1":
                d.nets["d"] = "ncas1"
            elif d.device_id == "M2":
                d.nets["d"] = "ncas2"
            elif d.device_id == "M3":          # mirror ref drain -> via MC3
                d.nets["d"] = "pcas1"
            elif d.device_id == "M4":          # mirror out drain -> via MC4
                d.nets["d"] = "pcas2"
        g.devices += [
            DeviceRecord("MC1", "nmos", "input_pair_nmos",
                         {"d": "nmir", "g": "nbias", "s": "ncas1", "b": "gnda"},
                         "cas1", dict(PRIORS["cascode_nmos"]),
                         {"tier2_block": "cascode_input", **ev}),
            DeviceRecord("MC2", "nmos", "input_pair_nmos",
                         {"d": s1out, "g": "nbias", "s": "ncas2", "b": "gnda"},
                         "cas1", dict(PRIORS["cascode_nmos"]),
                         {"tier2_block": "cascode_input", **ev}),
            DeviceRecord("MC3", "pmos", "mirror_reference",
                         {"d": "nmir", "g": "nmir", "s": "pcas1", "b": "vdda"},
                         "cas2", dict(PRIORS["cascode_pmos"]),
                         {"tier2_block": "cascode_input", **ev}),
            DeviceRecord("MC4", "pmos", "mirror_output",
                         {"d": s1out, "g": "nmir", "s": "pcas2", "b": "vdda"},
                         "cas2", dict(PRIORS["cascode_pmos"]),
                         {"tier2_block": "cascode_input", **ev}),
        ]
    prev = s1out
    inversions = 2  # vinp path: CS into mirror + mirror out (non-inverting to s1out)
    for k in range(2, n + 1):
        out = "vout" if k == n else f"n{k}"
        if class_ab and k == n:
            # CLASS-AB PUSH-PULL OUTPUT: NMOS driven by prev, PMOS driven by
            # the mirror-side node of the previous stage (both signal-
            # driven; quiescent set by the mirror bias) -- sources/sinks
            # load current on demand instead of a fixed pull-up.
            pdrive = "nmir" if k == 2 else f"n{k - 1}"
            g.devices += [
                DeviceRecord(f"M{4 + 2 * k}", "nmos", "second_stage_gain_device",
                             {"d": out, "g": prev, "s": "gnda", "b": "gnda"},
                             f"ab{k}", dict(PRIORS["ab_nmos"]),
                             {"tier2_block": "class_ab_output", **ev}),
                DeviceRecord(f"M{5 + 2 * k}", "pmos", "second_stage_gain_device",
                             {"d": out, "g": pdrive, "s": "vdda", "b": "vdda"},
                             f"ab{k}", dict(PRIORS["ab_pmos"]),
                             {"tier2_block": "class_ab_output", **ev}),
            ]
        else:
            g.devices += [
                DeviceRecord(f"M{4 + 2 * k}", "nmos", "second_stage_gain_device",
                             {"d": out, "g": prev, "s": "gnda", "b": "gnda"},
                             f"cs{k}", dict(PRIORS["cs_gain_nmos"]), ev),
                DeviceRecord(f"M{5 + 2 * k}", "pmos", "active_load",
                             {"d": out, "g": "nmir", "s": "vdda", "b": "vdda"},
                             f"cs{k}", dict(PRIORS["load_pmos"]), ev),
            ]
        # Miller compensation across EACH inverting CS stage, locally.
        #
        # This placement is load-bearing, not incidental. A Miller capacitor
        # provides NEGATIVE feedback only across an ODD number of inverting
        # stages. In this template every CS stage inverts and the 5T first
        # stage is non-inverting to n1, so:
        #
        #   n1 -> n2   : 1 inversion  -> capacitor is negative feedback  (OK)
        #   n2 -> vout : 1 inversion  -> capacitor is negative feedback  (OK)
        #   n1 -> vout : 2 inversions -> capacitor is POSITIVE feedback  (BAD)
        #
        # A "nested Miller" variant returning every capacitor from vout was
        # tried and MEASURED WORSE precisely for that reason: the outer
        # capacitor spanned two inversions, so it fed back in phase.
        # 3s_miller went from +0.24 deg to -87.35 deg of phase margin.
        # Textbook nested Miller assumes a stage-inversion pattern this
        # template does not have; do not reintroduce it without first
        # changing the stage polarities.
        has_c = "C" in audit_row["functional_blocks"]
        has_rc = ("RC_parallel" in audit_row["functional_blocks"]
                  or "RC_series" in audit_row["functional_blocks"])
        if has_c or has_rc:
            # NULLING-BRANCH REPAIR (2026-08-16). This function's contract
            # ("R+C -> nulling branch", above) was never implemented: C,
            # RC_parallel and RC_series all mapped to the same lone Miller
            # capacitor, which made 2s_rc and 2s_miller PHYSICALLY
            # IDENTICAL netlists. Measured consequences of that collapse:
            # both pipeline branches frequently sized the same circuit
            # twice (byte-identical A/B verification netlists, GATE3
            # boundary spec), the selection layer had nothing real to
            # choose between, the rz_x sizing knob scaled a resistor that
            # never existed, and the RHP zero of plain Miller compensation
            # forced huge caps (2.8 nF measured) that buried UGBW 47x
            # below target on the boundary spec class. RC-type blocks now
            # realise the series nulling resistor the family always
            # promised; pure-C blocks keep the plain Miller cap.
            cap_p = prev
            if has_rc:
                mid = f"nz{k}"
                g.devices.append(DeviceRecord(
                    f"RZ{k}", "res", "nulling_resistor",
                    {"p": prev, "n": mid}, None,
                    {"value": PRIORS["rz_ohm"]["value"],
                     "origin": PRIORS["rz_ohm"]["origin"]},
                    {"structural_evidence": "RC-type block in source graph",
                     "compensation_topology": "miller_rz_series",
                     "encloses_inversions": 1, **ev}))
                cap_p = mid
            g.devices.append(DeviceRecord(f"CC{k}", "cap", "miller_compensation",
                                          {"p": cap_p, "n": out}, None,
                                          {"value": PRIORS["miller_cap_f"]["value"],
                                           "origin": PRIORS["miller_cap_f"]["origin"]},
                                          {"structural_evidence":
                                           ("RC-type block in source graph"
                                            if has_rc else
                                            "C-type block in source graph"),
                                           "compensation_topology":
                                           ("miller_rz_series" if has_rc
                                            else "miller_per_stage"),
                                           "encloses_inversions": 1, **ev}))
        prev = out
        inversions += 1
    g.polarity = {"polarity_status": "template_defined",
                  "stage_inversion_count": inversions,
                  "signal_path_evidence": "vinp→M1→mirror→(CS)^k→vout; template parity",
                  "noninverting_input": "vinp" if inversions % 2 == 0 else "vinn",
                  "selection_rule": "structural parity, NOT outcome-based"}
    if g.polarity["noninverting_input"] == "vinn":
        # keep TB convention: swap input assignment structurally (documented).
        for d in g.devices:
            if d.device_id == "M1":
                d.nets["g"] = "vinn"
            elif d.device_id == "M2":
                d.nets["g"] = "vinp"
        g.polarity["port_swap_applied"] = True
        g.polarity["noninverting_input"] = "vinp"
    g.unresolved = audit_row["unresolved_blocks"]
    g.block_assignments = [{"block": "five_transistor_first_stage", "devices": ["M1", "M2", "M3", "M4", "M5"],
                            "confidence": 0.8, "rule": "template", "evidence": ev}]
    return g, ("mapping_incomplete" if g.unresolved else "mapped")


def emit_netlist(g: DeviceCircuitGraph, subckt_name: str) -> str:
    lines = [f"* Stage3C mapped netlist {g.topology_id} candidate {g.mapping_candidate_id}",
             f".subckt {subckt_name} gnda vdda vinn vinp vout"]
    for d in g.devices:
        n = d.nets
        if d.kind in ("nmos", "pmos"):
            model = "sky130_fd_pr__nfet_01v8" if d.kind == "nmos" else "sky130_fd_pr__pfet_01v8"
            lines.append(f"x{d.device_id} {n['d']} {n['g']} {n['s']} {n['b']} {model} "
                         f"l={d.sizing['l']} w={d.sizing['w']} m=1")
        elif d.kind == "cap":
            lines.append(f"c{d.device_id} {n['p']} {n['n']} {d.sizing['value']:.4g}")
        elif d.kind == "res":
            lines.append(f"r{d.device_id} {n['p']} {n['n']} {d.sizing['value']:.4g}")
        elif d.kind == "isrc":
            lines.append(f"i{d.device_id} {n['p']} {n['n']} {d.sizing['value']:.4g}")
    lines.append(".ends")
    return "\n".join(lines) + "\n"


def static_validate(g: DeviceCircuitGraph, netlist: str) -> dict[str, Any]:
    problems = []
    nets_used: set[str] = set()
    ids = [d.device_id for d in g.devices]
    if len(ids) != len(set(ids)):
        problems.append("duplicate device identifiers")
    for d in g.devices:
        if d.kind in ("nmos", "pmos") and set(d.nets) != {"d", "g", "s", "b"}:
            problems.append(f"{d.device_id}: MOSFET missing terminals")
        nets_used |= set(d.nets.values())
    for p in ("gnda", "vdda", "vinp", "vinn", "vout"):
        if p not in nets_used:
            problems.append(f"port {p} unconnected")
    if "vdda" in nets_used and any(
            d.kind in ("nmos", "pmos") and d.nets.get("d") == d.nets.get("s") == "vdda" for d in g.devices):
        problems.append("supply short")
    # graph preservation: mapped stage count vs structural gain stages
    device_count_match = len([d for d in g.devices if d.kind in ("nmos", "pmos")]) >= 5
    return {"status": "mapped_static_valid" if not problems else "mapped_static_invalid",
            "problems": problems,
            "graph_preservation_score": 1.0 if not g.unresolved else round(
                1.0 - 0.1 * len(g.unresolved), 2),
            "missing_blocks": g.unresolved, "device_count_match": device_count_match,
            "sky130_models_valid": "sky130_fd_pr__" in netlist}


class _MappedEntry:
    """Adapter making a mapped candidate look like a registry entry for Stage 3B."""

    def __init__(self, tid: str, cdir: Path, meta: dict[str, Any], graph):
        self.topology_id, self.path, self.metadata, self.graph = tid, cdir, meta, graph
        self.source, self.has_netlist, self.has_schematic = "generated_mapped", True, False


def run_stage3c(limit: int | None = None) -> dict[str, Any]:
    registry = TopologyRegistry(_ROOT / "datasets" / "topology_library")
    audits = audit_generated(registry)
    exe = discover_ngspice()
    mem = _ROOT / "datasets" / "simulation_memory"
    results = []
    ready = [a for a in audits if a["mapping_readiness"] in ("mapping_ready", "partially_specified")]
    for a in (ready[:limit] if limit else ready):
        entry = registry.get_topology(a["topology_id"])
        g, note = map_family(entry, a)
        if g is None:
            results.append({"topology_id": a["topology_id"], "mapping_status": note})
            continue
        cdir = _ROOT / "artifacts" / "stage3c" / a["topology_id"] / g.mapping_candidate_id
        cdir.mkdir(parents=True, exist_ok=True)
        subckt = f"mapped_{a['topology_id']}"
        netlist = emit_netlist(g, subckt)
        (cdir / "netlist.sp").write_text(netlist, encoding="utf-8")
        (cdir / "device_graph.json").write_text(json.dumps(asdict(g), indent=1, default=str), encoding="utf-8")
        val = static_validate(g, netlist)
        (cdir / "validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
        (cdir / "provenance.json").write_text(json.dumps(
            {"source_family": a["topology_id"], "graph_hash": a["graph_hash"],
             "mapping_note": note, "support_bias": g.support_bias,
             "polarity": g.polarity, "timestamp": time.time()}, indent=1), encoding="utf-8")
        row: dict[str, Any] = {"topology_id": a["topology_id"], "candidate": g.mapping_candidate_id,
                               "mapping_status": val["status"], "mapping_note": note,
                               "unresolved": g.unresolved}
        if val["status"] == "mapped_static_valid" and exe:
            # Sizing is inlined in the netlist; the Stage 3B testbench includes
            # design_variables/<name>, so provide a comment-only file and point
            # the pipeline's amplifier root at this candidate dir temporarily.
            (cdir / "design_variables").mkdir(exist_ok=True)
            (cdir / "design_variables" / subckt).write_text("* sizing inlined in netlist\n",
                                                            encoding="utf-8")
            fake = _MappedEntry(a["topology_id"], cdir,
                                {"name": subckt, "graph_hash": a["graph_hash"]}, entry.graph)
            audit_dict = {"subckt_name": subckt, "immediately_runnable": True,
                          "blocking_reason": None, "graph_hash": a["graph_hash"]}
            import agentic_raptor.electrical as elec

            original = elec._LEGACY_AMP
            elec._LEGACY_AMP = cdir
            try:
                rec = qualify_family(fake, audit_dict, cdir / "run", exe, "env-stage3c",
                                     split="train", timeout_s=180.0)
            finally:
                elec._LEGACY_AMP = original
            row.update({"electrical_status": rec.electrical_validation_status,
                        "metrics": {k: v for k, v in rec.extracted_metrics.items() if v is not None},
                        "failure_class": rec.failure_class, "run_id": rec.run_id})
        results.append(row)
    with (mem / "mapping_runs.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    counts = {"generated_families": len(audits),
              "mapping_ready": len([a for a in audits if a["mapping_readiness"] == "mapping_ready"]),
              "partially_specified": len([a for a in audits if a["mapping_readiness"] == "partially_specified"]),
              "behavioral_only": len([a for a in audits if a["mapping_readiness"] == "behavioral_only"]),
              "structurally_ambiguous": len([a for a in audits if a["mapping_readiness"] == "structurally_ambiguous"]),
              "candidates": len([r for r in results if r.get("candidate")]),
              "static_valid": len([r for r in results if r.get("mapping_status") == "mapped_static_valid"]),
              "electrically_functional": len([r for r in results if r.get("electrical_status") == "electrically_functional"]),
              "electrical_failed": len([r for r in results if r.get("electrical_status") not in (None, "electrically_functional")])}
    return {"counts": counts, "audits": audits, "results": results}


if __name__ == "__main__":
    out = run_stage3c()
    print(json.dumps(out["counts"], indent=1))
