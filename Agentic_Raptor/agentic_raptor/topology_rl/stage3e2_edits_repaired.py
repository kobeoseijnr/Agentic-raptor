"""EDIT-OPERATOR REPAIR (2026-08-13): EXPERIMENTAL repaired operator
variants. NOTHING live imports this module -- the live EDIT_TEMPLATES are
untouched; these variants are exercised only through the opt-in
`templates=` override in apply_edit()/LLMSeededEditRegistry.

Measured causal evidence behind each repair (single-edit parent->child
pairs, artifacts/publication_v3/az_edit_causal_pairs/):

ADD_VERIFIED_STAGE (live): dz = -1.81 / -1.60, child gain = -173.8 dB on
  BOTH parents -- a fixed, parent-independent collapse. Mechanism: the
  template inserts an NMOS CS driver whose gate sits at the previous
  stage's output DC with an nmir-gated PMOS load; the resulting DC point
  turns the stage off/pins the output. REPAIR: insert the stage using the
  MEASURED-WORKING wiring from replace_stage_with_compatible_block (dz =
  +0.095 / +0.009, gain preserved 89/79 dB on the same parents): PMOS CS
  driver (gate at previous output -- a bias point measured to work) +
  nbias-gated NMOS current-sink load. Same stage-count/polarity
  bookkeeping, same preconditions.

ADD_SUPPORTED_OUTPUT_STAGE (live): dz = -1.81 / -1.60, child gain =
  -39.2 dB on BOTH parents. Mechanism: the follower's "current source"
  pull-up is a PMOS whose gate references the NMOS-mirror net `nmir`
  (~0.56 V => Vsg ~ 1.24 V): polarity-mismatched biasing -- a hard-on
  pull-up fighting the follower, not a current source. This is exactly a
  "structurally valid but electrically nonsensical" wiring. REPAIR: the
  deterministic bias-plausibility rule "a current-source gate must
  reference a matching-polarity mirror net; when none exists, use a diode
  connection" -- the pull-up becomes a diode-connected PMOS (self-biased,
  defined current, no cross-polarity net reference). Follower device
  unchanged.

All other operators are left byte-identical: REPLACE_SUPPORTED_COMPENSATION
measured GENERALLY_HELPFUL (+0.19 fail->pass / +1.04); REPLACE_STAGE
mildly positive; CONNECT_FEEDBACK and REPLACE_LOAD are destructive as sole
edits but each appears in a verified NovelFeasible passing chain
(CONTEXT_DEPENDENT -- Section 5 forbids removal); ADD_COMPENSATION and
REMOVE_OPTIONAL have insufficient data.
"""
from __future__ import annotations

from agentic_raptor.mapping import DeviceRecord
from agentic_raptor.topology_rl.stage3e2_edits import (EDIT_SCHEMA_VERSION,
                                                        EDIT_TEMPLATES,
                                                        EditRejected, PRIORS,
                                                        _clone, _find, _nets,
                                                        _rename_net)

REPAIR_VERSION = "az_edit_repair.1"


def add_verified_stage_repaired(g):
    """Append a gain stage using the MEASURED-WORKING cs_pmos pattern."""
    if g.stage_count >= 3:
        raise EditRejected("max_supported_stage_count_reached")
    if "vout" not in _nets(g):
        raise EditRejected("no_output_net")
    if "nbias" not in _nets(g):
        raise EditRejected("bias_net_missing_for_sink_load")
    ng = _clone(g, "addstageR")
    k = ng.stage_count + 1
    inner = f"nx{k}"
    _rename_net(ng, "vout", inner)
    ng.devices += [
        DeviceRecord(f"ME{2 * k}P", "pmos", "second_stage_gain_device",
                     {"d": "vout", "g": inner, "s": "vdda", "b": "vdda"},
                     f"cse{k}", dict(PRIORS["load_pmos"]),
                     {"edit": "add_verified_stage_repaired",
                      "template": REPAIR_VERSION}),
        DeviceRecord(f"ME{2 * k + 1}N", "nmos", "current_sink_load",
                     {"d": "vout", "g": "nbias", "s": "gnda", "b": "gnda"},
                     f"cse{k}", dict(PRIORS["tail_nmos"]),
                     {"edit": "add_verified_stage_repaired",
                      "template": REPAIR_VERSION}),
    ]
    ng.stage_count = k
    ng.polarity["stage_inversion_count"] = ng.polarity.get("stage_inversion_count", 0) + 1
    ng.polarity["edit_note"] = ("one extra inversion from added PMOS CS stage "
                                "(repaired wiring, measured-working pattern)")
    return ng


def add_output_stage_repaired(g):
    """PMOS source follower with a DIODE-CONNECTED pull-up (bias-polarity
    plausibility rule) instead of the measured-broken nmir-gated PMOS."""
    if _find(g, role="output_follower"):
        raise EditRejected("output_stage_already_present")
    if "vout" not in _nets(g):
        raise EditRejected("no_output_net")
    ng = _clone(g, "addoutR")
    _rename_net(ng, "vout", "nfo")
    ng.devices += [
        DeviceRecord("MOF", "pmos", "output_follower",
                     {"d": "gnda", "g": "nfo", "s": "vout", "b": "vdda"}, "outb",
                     dict(PRIORS["load_pmos"]),
                     {"edit": "add_output_stage_repaired", "template": REPAIR_VERSION}),
        DeviceRecord("MOPD", "pmos", "diode_pullup",
                     {"d": "vout", "g": "vout", "s": "vdda", "b": "vdda"}, "outb",
                     dict(PRIORS["mirror_pmos"]),
                     {"edit": "add_output_stage_repaired", "template": REPAIR_VERSION}),
    ]
    return ng


#: the EXPERIMENTAL template set: live templates with ONLY the two
#: measured-deterministically-broken operators replaced.
REPAIRED_EDIT_TEMPLATES = dict(EDIT_TEMPLATES)
REPAIRED_EDIT_TEMPLATES["ADD_VERIFIED_STAGE"] = {
    **EDIT_TEMPLATES["ADD_VERIFIED_STAGE"],
    "fn": add_verified_stage_repaired,
    "block_family": "cs_pmos_stage",
    "repair_version": REPAIR_VERSION,
    "repair_evidence": "causal pairs: live dz -1.81/-1.60 (gain -173.8 dB both); "
                       "wiring pattern from REPLACE_STAGE measured dz +0.095/+0.009",
}
REPAIRED_EDIT_TEMPLATES["ADD_SUPPORTED_OUTPUT_STAGE"] = {
    **EDIT_TEMPLATES["ADD_SUPPORTED_OUTPUT_STAGE"],
    "fn": add_output_stage_repaired,
    "repair_version": REPAIR_VERSION,
    "repair_evidence": "causal pairs: live dz -1.81/-1.60 (gain -39.2 dB both); "
                       "cross-polarity nmir-gated pull-up replaced by diode "
                       "connection (deterministic bias-plausibility rule)",
}
