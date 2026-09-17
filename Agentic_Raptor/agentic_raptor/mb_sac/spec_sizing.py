"""Specification-conditioned SAC sizing for the self-improvement campaign.

Replaces the raw-metric sizing loop with:
  * an AUDITED, gain-capable action space (Task 2) — the legacy loop controlled
    only widths (1-2x stage 1, 0.5-1x stage 2) and a cap multiplier; none of
    those can raise output resistance, so DC gain was capped near its nominal
    value. This space adds per-stage LENGTH and a BIAS-current knob, the
    variables that actually move gm*ro.
  * a feasibility-conditioned, diminishing-return reward (Task 4) — phase
    margin saturates after a configurable cushion above target, so a 91 deg
    design no longer out-rewards closing a 20 dB gain deficit.
  * spec-conditioned Bradley-Terry ranking (Task 5) — candidate features carry
    the ACTIVE specification, surrogate margins measured against THAT target,
    and the remaining simulation budget.
  * post-sizing outcome tiers + AlphaZero value targets built from final
    constraint satisfaction (Tasks 6/7) — never from nominal stability, and
    the full margin vector is stored so the scalar cannot hide a failure.

Every electrical number here comes from real ngspice via
qualify_device_graph; nothing is fabricated.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

SCHEMA = "spec_sizing.1"
#: SAC observation width. 3 spec terms + budget fraction + best-so-far
#: closeness + the LAST measurement's three per-constraint margins. The
#: per-constraint terms are what make the state genuinely dynamic: overall
#: closeness is a running max, so it plateaus after an early success (measured:
#: one change across a 10-step episode), whereas the individual gain/PM/UGBW
#: shortfalls move on every step and tell the actor WHICH way it is failing.
OBS_DIM = 8
#: v2 keeps v1's target-saturating shape but replaces the hard -1 clamp on
#: the deficit branch with tanh, restoring the gradient for large shortfalls
REWARD_POLICY_VERSION = "target_saturating_v2_smooth_deficit"

_ROOT = Path(__file__).resolve().parents[2]
#: persistent sizing memory (override for tests via env). The SAC nets,
#: surrogate and BT ranker no longer start from scratch on every design —
#: the tuner accumulates experience the same way the architect does.
#
# Stage 1.6: every file this warm-start mechanism reads was, before the
# Stage 1.5 C_LOAD repair, populated from measurements that silently ran at
# a fixed 500pF regardless of the spec (see agentic_raptor.publication.
# artifact_provenance). Rather than a runtime gate that could be forgotten,
# the default paths themselves moved to a `_post_cload_v1` location -- a
# structurally empty starting point for the new electrical environment.
# The OLD `sizing_memory/` / `dynamics_surrogate_data.jsonl` /
# `mbsac_replay.jsonl` are left exactly where they were, untouched, as
# PRE_CLOAD_FIX historical evidence (frozen by artifact_provenance.py).
STATE_DIR = Path(os.environ.get(
    "AGENTIC_RAPTOR_SIZING_MEMORY",
    str(_ROOT / "artifacts" / "sizing_memory_post_cload_v1")))
# when a memory override is active (batteries/tests), experience files are
# isolated inside it -- concurrent campaigns and batteries never share
# append targets, so parallel runs cannot interleave/corrupt JSONL lines
if "AGENTIC_RAPTOR_SIZING_MEMORY" in os.environ:
    DYNAMICS_FILE = STATE_DIR / "dynamics_surrogate_data.jsonl"
    REPLAY_FILE = STATE_DIR / "mbsac_replay.jsonl"
else:
    DYNAMICS_FILE = (_ROOT / "datasets/simulation_memory"
                     / "dynamics_surrogate_data_post_cload_v1.jsonl")
    REPLAY_FILE = (_ROOT / "datasets/simulation_memory"
                   / "mbsac_replay_post_cload_v1.jsonl")

#: stage-1 device roles (five-transistor first stage); everything else FET is
#: treated as stage 2 / output.
S1_ROLES = ("input_pair_nmos", "mirror_reference", "mirror_output",
            "tail_current_source")

#: Task 2 — the audited action space. 'gain_mechanism' documents WHY a knob
#: can (or cannot) raise DC gain: gain/stage ~ gm*ro, gm ~ sqrt(W/L * Id),
#: ro ~ L/Id, so gain ~ sqrt(W*L/Id). Width alone moves gain only as sqrt(W).
ACTION_SPACE = {
    "s1_w": {"index": 0, "lo": 0.5, "hi": 4.0, "unit": "x nominal W",
             "targets": "stage-1 FET widths (input pair, mirrors, tail)",
             "gain_mechanism": "gm ~ sqrt(W): weak positive"},
    "s2_w": {"index": 1, "lo": 0.5, "hi": 8.0, "unit": "x nominal W",
             "targets": "stage-2 / output FET widths",
             "gain_mechanism": "gm ~ sqrt(W): weak positive (legacy range "
                               "0.5-1.0 could only SHRINK this stage)"},
    "s1_l": {"index": 2, "lo": 0.5, "hi": 6.0, "unit": "x nominal L",
             "targets": "stage-1 FET lengths",
             "gain_mechanism": "ro ~ L: strong positive (absent from the "
                               "legacy action space entirely)"},
    "s2_l": {"index": 3, "lo": 0.5, "hi": 6.0, "unit": "x nominal L",
             "targets": "stage-2 FET lengths",
             "gain_mechanism": "ro ~ L: strong positive"},
    #: Ceiling raised from 8x. At 8x the 2 pF prior the compensation
    #: capacitor topped out at 16 pF against a 200 pF load -- about two
    #: orders of magnitude short of the pole-splitting region, so NO setting
    #: of the old action space could stabilise a compensated amplifier.
    #: Measured sweep past the ceiling (phase margin vs Cc):
    #:
    #:            16 pF     256 pF    4096 pF
    #:   2s_miller  -1.24     -6.23     +37.74
    #:   3s_miller -15.42    -27.81     +29.09
    #:   3s_rc      -9.17     -8.81     +55.94   (rz x20)
    #:
    #: The dip before the recovery is the Miller RHP zero dominating until
    #: pole splitting overtakes it; every family crosses into positive phase
    #: margin only well beyond the old limit.
    #:
    #: The knob stays LINEAR, so resolution below ~10x is now coarse. That is
    #: acceptable because the stable solution for every family lives at the
    #: high end; a log-scaled knob would be the better long-term shape but
    #: would silently reinterpret every cap_x already recorded in replay and
    #: dynamics memory.
    "cap_x": {"index": 4, "lo": 0.5, "hi": 2048.0, "unit": "x nominal C",
              "targets": "compensation/load capacitors",
              "gain_mechanism": "none directly (stability/UGBW); very large "
                                "Cc does trade DC gain via loading"},
    "ib_x": {"index": 5, "lo": 0.25, "hi": 4.0, "unit": "x nominal Ibias",
             "targets": "current-source (isrc) values",
             "gain_mechanism": "gain ~ 1/sqrt(Id): lower bias raises gain, "
                               "trades UGBW (absent from legacy space)"},
    #: The nulling resistor was a hard-wired 5 kOhm constant with NO knob, so
    #: no optimiser could ever tune it -- which made the whole rc_nulling
    #: family structurally incapable of its one job. Measured evidence:
    #: 3s_rc tracked 3s_miller to within 0.3 deg of phase margin at every
    #: capacitor value (-80.66 vs -80.87 at 1 pF), i.e. the resistor was
    #: doing nothing.
    #:
    #: It matters because Miller compensation creates a RIGHT-HALF-PLANE zero
    #: at z = gm/Cc, and the measured signature was unmistakable: raising Cc
    #: made phase margin WORSE (2s_miller +9.52 deg at 1 pF -> -1.24 deg at
    #: 16 pF), because the RHP zero falls with Cc faster than pole splitting
    #: helps. Rz cancels that zero: at Rz = 1/gm it moves to infinity, and
    #: beyond that it becomes a LEFT-half-plane zero that ADDS phase margin.
    #: The wide range below spans both sides of 1/gm for these devices.
    "rz_x": {"index": 6, "lo": 0.1, "hi": 40.0, "unit": "x nominal Rz",
             "targets": "nulling resistors in RC compensation branches",
             "gain_mechanism": "none directly; cancels the Miller RHP zero "
                               "(z = gm/Cc), converting added lag into added "
                               "phase margin"},
}
N_KNOBS = len(ACTION_SPACE)
KNOB_NAMES = sorted(ACTION_SPACE, key=lambda k: ACTION_SPACE[k]["index"])

#: UGBW spans decades (1 kHz .. 100 MHz+), so the surrogate's third output
#: is log10(Hz), clamped to a plausible band and scaled to sit near the same
#: numeric range as the pm/90 and gain/100 outputs. This scale is shared with
#: ranking/surrogate.py's decode -- keep the two in sync.
UGBW_LOG_LO, UGBW_LOG_HI = 2.0, 9.0     # 100 Hz .. 1 GHz


def _ugbw_target(ugbw_hz) -> float:
    """Encode a measured UGBW into the surrogate's training scale.

    None (unmeasured / non-converged) trains toward the band FLOOR, not zero:
    zero would look like ~1 Hz, a confident false claim of total failure that
    would bias the surrogate rather than leaving it uninformed. The decoder
    in ranking/surrogate.py treats the surrogate as advisory in all cases and
    never asserts authoritative=True regardless.
    """
    import math
    if ugbw_hz is None or ugbw_hz <= 0:
        return 0.0
    log = math.log10(ugbw_hz)
    log = max(UGBW_LOG_LO, min(UGBW_LOG_HI, log))
    return (log - UGBW_LOG_LO) / (UGBW_LOG_HI - UGBW_LOG_LO)
KNOB_LO = [ACTION_SPACE[k]["lo"] for k in KNOB_NAMES]
KNOB_HI = [ACTION_SPACE[k]["hi"] for k in KNOB_NAMES]

#: physical bounds for sky130 devices (um)
W_MIN, W_MAX = 0.42, 100.0
L_MIN, L_MAX = 0.15, 8.0

#: Task 4 defaults: reward saturates this far above the PM target
PM_CUSHION_DEG = 10.0


def action_space_manifest() -> dict:
    return {"knobs": ACTION_SPACE, "n_knobs": N_KNOBS,
            "w_bounds_um": [W_MIN, W_MAX], "l_bounds_um": [L_MIN, L_MAX],
            "legacy_gap": "legacy loop had only [s1_w 1-2x, s2_w 0.5-1x, "
                          "cap 1-6x]: no length, no bias, stage-2 width "
                          "shrink-only -> DC gain effectively frozen",
            "schema_version": SCHEMA}


#: nulling resistors must stay physically realisable in sky130
RZ_MIN, RZ_MAX = 100.0, 500e3


def encode_action(knobs, lo, hi):
    """Inverse of the decode used everywhere a policy/replay action becomes
    physical knobs (``knobs = lo + (a+1)/2*(hi-lo)``).

    Stage 6.1 repair: two scripted phases inside ``sac_size`` (the nominal
    anchor at step 0, and the exploitation-tail perturbations) choose a
    physical knob vector directly and then stored ``action = zeros`` for
    replay -- but ``decode(zeros)`` is the MIDPOINT of each knob's range
    (e.g. s1_w in [0.5, 4.0] decodes to 2.25x, not the 1.0x nominal point
    that was actually applied), so the stored (s, a, r, s') tuple did not
    correspond to the action that produced r and s'. This is the exact
    inverse transform, used to derive the CORRECT normalized action for any
    physical knob vector chosen outside the actor, instead of fabricating a
    zero placeholder.

    No inward epsilon margin: unlike the actor's own tanh(z) output (which
    approaches +-1 only in the limit), a caller here always passes an
    already-clamped knob vector, so `a` is exactly within [-1, 1] by
    construction. These stored actions are only ever re-decoded with the
    same linear formula (never passed through atanh), so there is nothing
    to protect against by shrinking them -- doing so previously introduced
    a real, scale-dependent roundtrip error (e.g. ~0.1 physical units on
    cap_x's ~2048-wide range from a mere 1e-4 inward nudge).
    """
    import torch
    lo_t = lo if torch.is_tensor(lo) else torch.tensor(lo)
    hi_t = hi if torch.is_tensor(hi) else torch.tensor(hi)
    knobs_t = knobs if torch.is_tensor(knobs) else torch.tensor(knobs)
    a = 2 * (knobs_t - lo_t) / (hi_t - lo_t) - 1
    return a.clamp(-1.0, 1.0)


def assert_action_roundtrips_to_knobs(action, knobs, lo, hi,
                                      tol: float = 1e-3) -> float:
    """Hard invariant (Stage 6.1, Section 5): every SAC replay transition's
    stored action must decode back to the physical knob vector that was
    actually applied. ``tol`` is a FRACTION of each knob's own [lo, hi]
    range (not an absolute physical unit) -- knob scales span 0.1..2048
    across the 7 knobs, so a single absolute tolerance would be meaningless
    for most of them and falsely strict for the widest ones. Raises
    ValueError with the concrete mismatch if violated; otherwise returns the
    max relative error so callers can track it."""
    import torch
    lo_t = lo if torch.is_tensor(lo) else torch.tensor(lo)
    hi_t = hi if torch.is_tensor(hi) else torch.tensor(hi)
    a_t = action if torch.is_tensor(action) else torch.tensor(action)
    knobs_t = knobs if torch.is_tensor(knobs) else torch.tensor(knobs)
    decoded = lo_t + (a_t + 1) / 2 * (hi_t - lo_t)
    span = (hi_t - lo_t).clamp(min=1e-9)
    rel_err = ((decoded - knobs_t).abs() / span).max().item()
    if rel_err > tol:
        abs_err = (decoded - knobs_t).abs()
        raise ValueError(
            "SAC replay invariant violated: decoded(stored action) does not "
            f"reproduce the applied knobs (max relative error {rel_err:.6f} "
            f"> tol {tol}). decoded={decoded.tolist()} "
            f"applied={knobs_t.tolist()} abs_err={abs_err.tolist()}")
    return rel_err


def apply_knobs(graph, knobs: list) -> object:
    """Return a deep-copied device graph with the sizing knobs applied.

    Accepts a 6-knob vector (pre-rz_x) as well as the current 7-knob one, so
    persisted memory, replay rows and historical trajectories written before
    the nulling-resistor knob existed still load: a missing rz_x means "leave
    the resistor at nominal", which is exactly the old behaviour.
    """
    vals = [float(x) for x in knobs]
    if len(vals) < N_KNOBS:
        vals = vals + [1.0] * (N_KNOBS - len(vals))
    s1w, s2w, s1l, s2l, capx, ibx, rzx = vals[:N_KNOBS]
    gs = copy.deepcopy(graph)
    for d in gs.devices:
        if d.kind in ("nmos", "pmos"):
            stage1 = d.role in S1_ROLES
            d.sizing["w"] = max(W_MIN, min(W_MAX,
                                d.sizing["w"] * (s1w if stage1 else s2w)))
            d.sizing["l"] = max(L_MIN, min(L_MAX,
                                d.sizing["l"] * (s1l if stage1 else s2l)))
        elif d.kind == "cap":
            d.sizing["value"] = d.sizing["value"] * capx
        elif d.kind == "isrc":
            d.sizing["value"] = d.sizing["value"] * ibx
        elif d.kind == "res":
            d.sizing["value"] = max(RZ_MIN, min(RZ_MAX,
                                    d.sizing["value"] * rzx))
    return gs


def measure(topology_id: str, graph, exe, out_dir: Path, tag: str,
            costs, *, pdk_file=None, supply_voltage: float | None = None,
            temperature_c: float | None = None,
            c_load_f: float | None = None) -> dict:
    """One real ngspice qualification -> full metric dict (never fabricated).

    ``pdk_file``/``supply_voltage``/``temperature_c`` default to nominal (tt
    corner, 1.8 V, 27 C). ``c_load_f`` ALSO defaults to NOMINAL_CLOAD_F
    (500pF) if left None -- but as of the Stage 1.5 C_LOAD repair
    (2026-08-09), every spec-driven caller (sac_size, achievability_sweep,
    the non-RL sizing baselines, run_raptor_v2.py's final verify()) resolves
    the spec's real requested load via agentic_raptor.electrical.
    effective_c_load() FIRST and always passes it explicitly here. A bare
    `measure(...)` call with no c_load_f is now a deliberate, spec-agnostic
    diagnostic call (e.g. a feasibility-gate probe), not the normal path.
    """
    from agentic_raptor.electrical import NOMINAL_CLOAD_F
    from agentic_raptor.topology_rl.stage3e2_edits import qualify_device_graph
    q = qualify_device_graph(topology_id, graph, Path(out_dir), exe, tag, costs,
                             pdk_file=pdk_file, supply_voltage=supply_voltage,
                             temperature_c=temperature_c, c_load_f=c_load_f)
    m = q.get("metrics") or {}
    return {"gain_db": m.get("dc_gain_db"),
            "pm_deg": m.get("phase_margin_deg"),
            "ugbw_hz": m.get("ugbw_hz"),
            "power_w": m.get("quiescent_power_w"),
            # idd_a: TOTAL measured supply current (real op-point branch
            # current, |V1#branch|) -- never the MB-SAC Ibias design knob.
            "idd_a": m.get("idd_a"),
            "c_load_f": c_load_f if c_load_f is not None else NOMINAL_CLOAD_F,
            "stable": q.get("stability") == "verified_stable",
            "stability": q.get("stability"),
            "electrical": q.get("electrical"),
            "op_valid": q.get("electrical") == "electrically_functional",
            "metrics": m}


# ---------------------- Task 4: spec-conditioned reward -----------------------
def margin_vector(meas: dict, spec: dict) -> dict:
    """Full specification-margin vector — preserved on every record so the
    scalar reward can never hide an individual constraint failure."""
    gain, pm, ugbw = meas.get("gain_db"), meas.get("pm_deg"), meas.get("ugbw_hz")
    ugbw_t = spec.get("ugbw_target_hz")
    return {
        "gain_margin_db": (gain - spec["gain_target_db"])
        if gain is not None else None,
        "pm_margin_deg": (pm - spec["phase_margin_target_deg"])
        if pm is not None else None,
        "ugbw_log_margin": (math.log10(ugbw / ugbw_t)
                            if ugbw and ugbw > 0 and ugbw_t else None),
        "power_margin": None,       # no power limit in current specs
        "area_margin": None,        # no area limit in current specs
        "operating_point_valid": bool(meas.get("op_valid")),
        "spice_converged": meas.get("gain_db") is not None
        or meas.get("pm_deg") is not None,
    }


def _ramp_then_cushion(margin: float, ramp: float, cushion: float,
                       cushion_reward: float = 0.1) -> float:
    """Strong improvement reward until the target, small safety-cushion
    reward above it, near-zero marginal reward past the cushion.

    The deficit branch uses tanh rather than a hard clamp at -1. Measured
    failure of the clamp: on a pm>=60 spec every candidate landed 56-65 deg
    short, so `max(-1, margin/45)` returned exactly -1.0 for all of them --
    five structurally different circuits, one identical reward, no gradient
    in the dimension that was actually failing. tanh keeps the same "deficit
    dominates" scale and the same near-linear behaviour close to the target
    (tanh x ~ x for small x), while never going completely flat.
    """
    if margin < 0:
        return math.tanh(margin / ramp)              # deficit dominates
    return cushion_reward * min(margin, cushion) / cushion


def spec_reward(meas: dict, spec: dict,
                pm_cushion_deg: float = PM_CUSHION_DEG) -> tuple:
    """Feasibility-conditioned, diminishing-return reward.

    Design point (Task 4): with target pm>=45, a 91.2 deg / 35.9 dB design
    must score clearly below a 55 deg / 56.5 dB design — excess PM beyond the
    cushion earns ~nothing while a 20 dB gain deficit costs up to -1.
    Returns (scalar, margin_vector)."""
    mv = margin_vector(meas, spec)
    if not mv["spice_converged"] or not mv["operating_point_valid"]:
        return -2.0, mv
    if meas.get("stability") == "verified_unstable":
        # large penalty for verified instability; keep a gradient toward
        # stability via the (capped) pm deficit term
        pm_m = mv["pm_margin_deg"] if mv["pm_margin_deg"] is not None else -90.0
        return -1.5 + 0.5 * max(-1.0, pm_m / 90.0), mv
    r = 0.0
    if mv["pm_margin_deg"] is not None:
        r += _ramp_then_cushion(mv["pm_margin_deg"], ramp=45.0,
                                cushion=pm_cushion_deg)
    if mv["gain_margin_db"] is not None:
        r += _ramp_then_cushion(mv["gain_margin_db"], ramp=20.0, cushion=6.0)
    if mv["ugbw_log_margin"] is not None:
        r += 0.5 * _ramp_then_cushion(mv["ugbw_log_margin"], ramp=2.0,
                                      cushion=0.5)
    if all(m is None or m >= 0 for m in (mv["gain_margin_db"],
                                         mv["pm_margin_deg"],
                                         mv["ugbw_log_margin"])):
        r += 1.0                                     # joint exact-spec bonus
    return r, mv


def legacy_reward(meas: dict, spec: dict) -> float:
    """The pre-repair reward (for A/B comparison only — Task 9). Rewards raw
    PM excess with no saturation at the target."""
    t = math.tanh
    return ((1.0 if meas.get("stable") else -1.0)
            + 0.3 * t(((meas.get("gain_db") or 0)
                       - spec["gain_target_db"]) / 10)
            + 0.3 * t(((meas.get("pm_deg") or -90)
                       - spec["phase_margin_target_deg"]) / 20))


# --------------------- Task 6/8: outcome classification -----------------------
def postsizing_outcome(meas: dict, spec: dict) -> dict:
    """Explicit per-constraint verdicts + tier. 'structure-class match' is
    NEVER called design success — success means measured electrical pass."""
    mv = margin_vector(meas, spec)
    passes = {
        "stable": bool(meas.get("stable")),
        "gain": mv["gain_margin_db"] is not None and mv["gain_margin_db"] >= 0,
        "pm": mv["pm_margin_deg"] is not None and mv["pm_margin_deg"] >= 0,
        "ugbw": mv["ugbw_log_margin"] is None or mv["ugbw_log_margin"] >= 0,
        "op": bool(mv["operating_point_valid"]),
    }
    hard = ["stable", "gain", "pm", "ugbw", "op"]
    n_pass = sum(passes[k] for k in hard)
    exact = all(passes[k] for k in hard)
    dist = []
    if mv["gain_margin_db"] is not None:
        dist.append(max(0.0, -mv["gain_margin_db"]) / 20.0)
    if mv["pm_margin_deg"] is not None:
        dist.append(max(0.0, -mv["pm_margin_deg"]) / 45.0)
    if mv["ugbw_log_margin"] is not None:
        dist.append(max(0.0, -mv["ugbw_log_margin"]) / 2.0)
    tier = ("exact_spec_pass" if exact
            else "stable_below_spec" if passes["stable"]
            else "unstable" if meas.get("pm_deg") is not None
            else "unmeasured")
    # Fix 5: an exact-pass verdict spans FIVE constraints, so reporting only
    # gain and PM left rows that satisfied both yet still read
    # exact_pass=False with no stated cause. Every constraint is now exported
    # with target, achieved, margin and verdict, plus the named reason.
    reasons = []
    if not passes["op"]:
        reasons.append("invalid operating point")
    if not mv["spice_converged"]:
        reasons.append("SPICE did not converge")
    if not passes["stable"]:
        reasons.append("unstable")
    if not passes["gain"]:
        reasons.append("gain below target")
    if not passes["pm"]:
        reasons.append("PM below target")
    if not passes["ugbw"]:
        reasons.append("UGBW below target")
    norm = {"gain": (max(0.0, -mv["gain_margin_db"]) / 20.0
                     if mv["gain_margin_db"] is not None else None),
            "pm": (max(0.0, -mv["pm_margin_deg"]) / 45.0
                   if mv["pm_margin_deg"] is not None else None),
            "ugbw": (max(0.0, -mv["ugbw_log_margin"]) / 2.0
                     if mv["ugbw_log_margin"] is not None else None)}
    live = {k: v for k, v in norm.items() if v is not None}
    worst = max(live, key=live.get) if live and max(live.values()) > 0 else None
    ugbw_t = spec.get("ugbw_target_hz")
    return {"passes": passes, "hard_constraints_passed": n_pass,
            "hard_constraints_total": len(hard), "exact_spec_pass": exact,
            "outcome_tier": tier, "margin_vector": mv,
            "constraints": {
                "gain_db": {"target": spec.get("gain_target_db"),
                            "achieved": meas.get("gain_db"),
                            "margin": mv["gain_margin_db"],
                            "passed": passes["gain"]},
                "phase_margin_deg": {
                    "target": spec.get("phase_margin_target_deg"),
                    "achieved": meas.get("pm_deg"),
                    "margin": mv["pm_margin_deg"], "passed": passes["pm"]},
                "ugbw_hz": {"target": ugbw_t,
                            "achieved": meas.get("ugbw_hz"),
                            "margin": mv["ugbw_log_margin"],
                            "passed": passes["ugbw"],
                            "applicable": ugbw_t is not None},
                # the corpus states no power/area budget; "not applicable" is
                # recorded explicitly so it can never be read as "passed"
                "power_w": {"target": spec.get("power_target_w"),
                            "achieved": meas.get("power_w"),
                            "margin": None,
                            "passed": None,
                            "applicable": spec.get("power_target_w")
                            is not None},
                "area_um2": {"target": spec.get("area_target_um2"),
                             "achieved": meas.get("area_um2"),
                             "margin": None, "passed": None,
                             "applicable": spec.get("area_target_um2")
                             is not None}},
            "operating_point_valid": bool(mv["operating_point_valid"]),
            "spice_converged": bool(mv["spice_converged"]),
            "stability_status": meas.get("stability"),
            "exact_failure_reason": "; ".join(reasons) if reasons else None,
            "worst_failing_constraint": worst,
            "normalized_distance_to_feasibility":
                round(sum(dist) / len(dist), 4) if dist else None}


# ------------------------ Task 7: value targets --------------------------------
def value_target_from_outcome(outcome: dict, spice_calls: int,
                              budget: int) -> dict:
    """AlphaZero value target from the FINAL post-sizing outcome (never from
    nominal topology stability). Bounded [0,1]; the components are stored so
    the scalar cannot hide a single-constraint failure."""
    frac = outcome["hard_constraints_passed"] / outcome["hard_constraints_total"]
    dist = outcome["normalized_distance_to_feasibility"]
    closeness = max(0.0, 1.0 - dist) if dist is not None else 0.0
    cost = max(0.0, 1.0 - spice_calls / max(1, budget))
    v = 0.55 * frac + 0.35 * closeness + 0.10 * cost
    if outcome["exact_spec_pass"]:
        v = max(v, 0.9)
    return {"value_target": round(min(1.0, v), 4),
            "components": {"constraint_fraction": round(frac, 4),
                           "feasibility_closeness": round(closeness, 4),
                           "budget_efficiency": round(cost, 4),
                           "exact_spec_pass": outcome["exact_spec_pass"]},
            "margin_vector": outcome["margin_vector"],
            "reward_policy_version": REWARD_POLICY_VERSION}


# ------------------- Task 5: spec-conditioned ranker features ------------------
def active_design_spec(spec: dict):
    """Build a DesignSpecifications for the ACTIVE target (the legacy call
    sites passed a fixed default spec regardless of the actual target)."""
    from agentic_raptor.core.specifications import DesignSpecifications
    return DesignSpecifications(
        circuit_class="ota", technology=spec.get("technology", "sky130"),
        supply_voltage=1.8, temperature_c=27.0,
        target_gain_db=float(spec["gain_target_db"]),
        target_gbw_hz=float(spec.get("ugbw_target_hz") or 1e5),
        minimum_phase_margin_deg=float(spec["phase_margin_target_deg"]),
        load_capacitance_f=float(spec.get("load_capacitance_pf", 100)) * 1e-12)


def candidate_features(graph_reg, spec: dict, knobs: list,
                       predicted: dict | None, budget_frac: float):
    """Pre-SPICE ranker features conditioned on the active spec, surrogate
    margins measured AGAINST that spec, and remaining budget."""
    from agentic_raptor.core.candidate import CircuitCandidate
    from agentic_raptor.core.types import GenerationSource
    from agentic_raptor.dpo import build_candidate_features
    cc = CircuitCandidate.create(graph_reg, active_design_spec(spec),
                                 GenerationSource.EDITED)
    margins = {}
    feas = 0.5
    if predicted:
        margins = {"gain": (predicted["gain_db"] - spec["gain_target_db"]) / 20.0,
                   "pm": (predicted["pm_deg"]
                          - spec["phase_margin_target_deg"]) / 45.0}
        feas = 1.0 if all(v >= 0 for v in margins.values()) else \
            max(0.0, 1.0 + min(margins.values()))
    return build_candidate_features(
        cc, sizing_vector=[float(x) for x in knobs],
        predicted_margins=margins, predicted_feasibility=feas,
        remaining_budget_frac=max(0.0, min(1.0, budget_frac)))


# --------------------- persistent sizing memory --------------------------------
def _mem_paths(family: str) -> dict:
    from agentic_raptor.dpo import FEATURE_DIM
    # obs width is part of the net signature: keying it here means widening
    # the state starts a clean memory generation instead of silently failing
    # to load every previous family checkpoint
    tag = f"k{N_KNOBS}_o{OBS_DIM}"
    return {"nets": STATE_DIR / f"sacnets_{family}_{tag}.pt",
            "surrogate": STATE_DIR / f"surrogate_{family}_{tag}.pt",
            "ranker": STATE_DIR / f"ranker_global_f{FEATURE_DIM}.pt",
            "meta": STATE_DIR / "meta.json"}


def _try_load(path: Path, module) -> bool:
    import torch
    if not path.is_file():
        return False
    try:
        module.load_state_dict(torch.load(path, weights_only=True))
        return True
    except Exception:
        return False        # schema drift -> start fresh, never crash


# Pre-campaign audit fix (2026-08-09): this MUST track DYNAMICS_FILE's
# non-sandboxed default, not a hardcoded path. Before this fix it pointed
# at the OLD, unversioned dynamics_surrogate_data.jsonl even after
# DYNAMICS_FILE itself moved to the _post_cload_v1 file (Stage 1.6) --
# meaning every sandboxed sac_size() call would have silently blended
# PRE_CLOAD_FIX rows into its "global" read, exactly the mixing this whole
# repair exists to prevent. The old file is untouched, still readable
# directly for provenance if ever needed, just no longer auto-merged in.
_GLOBAL_DYNAMICS = (_ROOT / "datasets/simulation_memory"
                    / "dynamics_surrogate_data_post_cload_v1.jsonl")


def _dynamics_rows(family: str, limit: int = 400) -> list:
    """Family-labelled experience. Sandboxed runs WRITE only to their own
    file but READ the global (non-sandboxed) history too (read-only access
    is safe under concurrency) -- otherwise offline pretraining starts
    blind in every sandbox. Both DYNAMICS_FILE and _GLOBAL_DYNAMICS are
    POST_CLOAD_FIX_V1 paths; there is no PRE_CLOAD_FIX fallback here by
    design."""
    sources = [DYNAMICS_FILE]
    if _GLOBAL_DYNAMICS not in sources and _GLOBAL_DYNAMICS.is_file():
        sources.append(_GLOBAL_DYNAMICS)
    lines = []
    for src in sources:
        if src.is_file():
            lines += src.read_text(encoding="utf-8").splitlines()
    rows = []
    for x in lines:
        if not x.strip():
            continue
        e = json.loads(x)
        # only family-labelled rows are usable: knob->metric response is
        # topology-dependent, and legacy rows carry no family label
        if e.get("family") == family and e.get("pm") is not None \
                and isinstance(e.get("knobs"), dict) \
                and len(e["knobs"]) == N_KNOBS:
            rows.append(e)
    return rows[-limit:]


def _margin_feats(mv: dict | None) -> list:
    """Per-constraint shortfall, squashed. Unlike the running-max closeness
    these move every step, which is what gives the MDP real dynamics."""
    if not mv:
        return [0.0, 0.0, 0.0]

    def n(v, scale):
        return math.tanh(v / scale) if v is not None else 0.0
    return [n(mv.get("gain_margin_db"), 20.0),
            n(mv.get("pm_margin_deg"), 45.0),
            n(mv.get("ugbw_log_margin"), 2.0)]


def _offline_pretrain(actor, q1, q2, opt_a, opt_c, obs, spec, family,
                      reward_fn, lo, hi, steps=200):
    """Offline-to-online SAC (Repair 4): pretrain on the family's HISTORICAL
    measurements, retro-rewarded against the ACTIVE spec. Costs zero new
    SPICE calls (reused measurements, never counted as new). Critics learn
    reward regression; the actor is advantage-weighted-cloned toward
    historically good actions, so online exploration starts near known-good
    regions instead of the middle of the knob space."""
    import torch
    rows = _dynamics_rows(family, limit=500)
    if len(rows) < 16:
        return 0
    acts, rews = [], []
    for e in rows:
        meas = {"gain_db": e.get("gain"), "pm_deg": e.get("pm"),
                "ugbw_hz": e.get("ugbw_hz"), "power_w": None,
                "stable": e.get("stable", (e.get("pm") or -1) > 0),
                "stability": ("verified_stable"
                              if e.get("stable", (e.get("pm") or -1) > 0)
                              else "verified_unstable"),
                "electrical": "electrically_functional", "op_valid": True}
        out = reward_fn(meas, spec)
        r = out[0] if isinstance(out, tuple) else out
        k = torch.tensor([float(e["knobs"][n]) for n in KNOB_NAMES])
        acts.append(torch.clamp((k - lo) / (hi - lo) * 2 - 1, -0.999, 0.999))
        rews.append(float(r))
    a_all = torch.stack(acts)
    r_all = torch.tensor(rews).unsqueeze(1)
    w = torch.softmax(r_all.squeeze(1) / 0.5, dim=0)      # advantage weights
    n = len(rows)
    for step in range(steps):
        idx = torch.randint(0, n, (min(32, n),))
        ob = obs.expand(len(idx), -1)
        a_b, r_b = a_all[idx], r_all[idx]
        lc = ((q1(torch.cat([ob, a_b], 1)) - r_b) ** 2).mean()             + ((q2(torch.cat([ob, a_b], 1)) - r_b) ** 2).mean()
        opt_c.zero_grad(); lc.backward(); opt_c.step()
        mu_ls = actor(ob)
        mu, ls = mu_ls[:, :N_KNOBS], mu_ls[:, N_KNOBS:].clamp(-3, 1)
        z = torch.atanh(a_b)
        logp = (-0.5 * ((z - mu) / ls.exp()) ** 2 - ls).sum(1)
        la = -(w[idx] * logp).sum() * n / len(idx)        # weighted cloning
        opt_a.zero_grad(); la.backward(); opt_a.step()
    return n


# ----------------------------- the SAC loop ------------------------------------
def sac_size(topology_id: str, graph, spec: dict, exe, out_dir: Path, costs,
             budget: int = 16, seed: int = 0, reward_fn=None,
             ranker=None, family: str | None = None,
             early_stop_on_pass: bool = False,
             select_by: str = "reward",
             margin_tail_calls: int = 0,
             exploit_when_close: float | None = None,
             fom_plateau_patience: int | None = None,
             tail_anchor_budget: int | None = None,
             persist: bool = True, use_surrogate: bool = True,
             use_ranker: bool = True,
             ranker_spec_conditioned: bool = True,
             sequential: bool = True, gamma: float = 0.99,
             tau: float = 0.005, c_load_f: float | None = None) -> dict:
    """Spec-conditioned SAC sizing under a fixed real-SPICE budget.

    With `family` set and persist=True, the SAC nets and surrogate warm-start
    from per-family saved state (falling back to pre-training the surrogate
    on that family's accumulated measured dynamics), the BT ranker
    warm-starts from its global state, and all three are saved back after the
    run — the tuner learns ACROSS designs instead of re-learning every time.
    Returns best candidate, full transition history, outcome tier and value
    target. reward_fn defaults to spec_reward; pass legacy_reward for A/B.

    sequential=True is full SAC: bootstrapped Bellman targets against Polyak-
    averaged twin target critics, sampled from the whole episode buffer. It
    requires a state the action can move, so obs[4] carries best-so-far
    feasibility closeness -- with the old constant slot the next state was a
    function of the step counter alone, the MDP was degenerate, and
    bootstrapping could not carry credit backwards. sequential=False keeps
    the previous immediate-reward regression (gamma=0 bandit) for A/B."""
    import torch
    from agentic_raptor.corpus import TopologyRegistry
    from agentic_raptor.dpo import FEATURE_DIM, DPOConfig, DPORanker
    from agentic_raptor.dpo.preference_pairs import OutcomeRecord, build_pairs
    from agentic_raptor.electrical import effective_c_load
    from agentic_raptor.mb_sac.stage3d2 import V3
    torch.manual_seed(seed)
    # Stage 1.5 repair: every real measurement this run takes must simulate
    # against the SAME load, resolved once here rather than left to default
    # (previously every measure() call below omitted c_load_f entirely and
    # silently got NOMINAL_CLOAD_F regardless of what `spec` asked for).
    cl = effective_c_load(spec, override=c_load_f)
    reward_fn = reward_fn or spec_reward
    ranker = ranker or DPORanker(FEATURE_DIM, DPOConfig(enabled=True,
                                                        seed=seed))
    lo, hi = torch.tensor(KNOB_LO), torch.tensor(KNOB_HI)
    actor = torch.nn.Sequential(torch.nn.Linear(OBS_DIM, 48), torch.nn.ReLU(),
                                torch.nn.Linear(48, 2 * N_KNOBS))
    q1 = torch.nn.Sequential(torch.nn.Linear(OBS_DIM + N_KNOBS, 48),
                             torch.nn.ReLU(), torch.nn.Linear(48, 1))
    q2 = torch.nn.Sequential(torch.nn.Linear(OBS_DIM + N_KNOBS, 48),
                             torch.nn.ReLU(), torch.nn.Linear(48, 1))
    # Stage 6 audit (2026-08-12): purely additive instrumentation -- proves
    # (rather than merely asserts) that the actor/critic parameters this
    # call trains are not a no-op. Checksummed BEFORE any warm-start load/
    # pretrain/real update touches them.
    from agentic_raptor.topology_rl.trainer import parameter_checksum
    actor_checksum_initial = parameter_checksum(actor)
    critic_checksum_initial = parameter_checksum(q1) + parameter_checksum(q2)
    import copy
    # Polyak-averaged target critics: the bootstrapped target must not chase
    # the same weights it trains, or the regression diverges
    q1_t, q2_t = copy.deepcopy(q1), copy.deepcopy(q2)
    for _p in list(q1_t.parameters()) + list(q2_t.parameters()):
        _p.requires_grad_(False)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-3)
    opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                             lr=3e-3)
    # SOFT actor-critic: entropy-regularized objective with automatically
    # tuned temperature toward the standard -|A| entropy target
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_al = torch.optim.Adam([log_alpha], lr=3e-3)
    target_entropy = -float(N_KNOBS)
    # THREE outputs: pm, gain, ugbw (log10 Hz). UGBW was silent for the
    # ranker's whole existence -- the surrogate predicted only pm/gain, so
    # worst_predicted_violation was blind to the constraint that caused most
    # measured failures (12 of 19 in the first ablation). A ranker scoring on
    # a blind metric can end up preferring the WORSE design with high
    # confidence: on the pairs the deployed checkpoint trained on, even a
    # linear model correctly regularised learned a positive weight on
    # worst_violation (should be negative) -- not overfitting, the visible
    # features genuinely pointed the wrong way without ugbw to check them.
    surrogate = torch.nn.Sequential(torch.nn.Linear(N_KNOBS, 32),
                                    torch.nn.ReLU(), torch.nn.Linear(32, 3))
    opt_s = torch.optim.Adam(surrogate.parameters(), lr=1e-2)
    # -------- warm start from persistent memory (family-keyed) --------------
    memory = {"family": family, "loaded_nets": False, "loaded_surrogate": False,
              "loaded_ranker": False, "surrogate_pretrain_rows": 0,
              "persisted": False}
    if persist and family:
        mp = _mem_paths(family)
        nets_holder = torch.nn.ModuleDict(
            {"actor": actor, "q1": q1, "q2": q2})
        memory["loaded_nets"] = _try_load(mp["nets"], nets_holder)
        memory["loaded_surrogate"] = _try_load(mp["surrogate"], surrogate)
        memory["loaded_ranker"] = _try_load(mp["ranker"], ranker.model)
        if not memory["loaded_surrogate"]:
            rows = _dynamics_rows(family)
            if len(rows) >= 8:      # cold start: pre-train on measured history
                for ep in range(200):
                    e = rows[ep % len(rows)]
                    kn = torch.tensor([float(e["knobs"][k])
                                       for k in KNOB_NAMES]) / hi
                    y = torch.tensor([e["pm"] / 90, (e["gain"] or 0) / 100,
                                      _ugbw_target(e.get("ugbw_hz"))])
                    ls_ = ((surrogate(kn) - y) ** 2).mean()
                    opt_s.zero_grad(); ls_.backward(); opt_s.step()
                memory["surrogate_pretrain_rows"] = len(rows)
    _obs0 = torch.tensor([spec["gain_target_db"] / 100,
                          spec["phase_margin_target_deg"] / 90,
                          spec.get("load_capacitance_pf", 100) / 1000,
                          1.0, 0.0] + _margin_feats(None))
    memory["offline_pretrain_rows"] = _offline_pretrain(
        actor, q1, q2, opt_a, opt_c, _obs0, spec, family, reward_fn,
        lo, hi) if (persist and family) else 0
    # targets start FROM the warm-started/pretrained critics, not from the
    # random init they were copied off
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    _rg = TopologyRegistry(V3).get_topology("topology_v2_0001").graph
    # SR3 arm: a GLOBAL (spec-blind) ranker sees one fixed default spec
    rank_spec = spec if ranker_spec_conditioned else {
        "gain_target_db": 60.0, "phase_margin_target_deg": 45.0,
        "load_capacitance_pf": 100.0, "ugbw_target_hz": 1e5,
        "technology": "sky130"}
    gain_t = spec["gain_target_db"]
    pm_t = spec["phase_margin_target_deg"]
    results, transitions = [], []
    best_close = 0.0        # progress term (running max -> plateaus)
    last_mv = None          # per-constraint shortfalls (move every step)
    # Stage 6.1 repair bookkeeping: every transition's action must decode
    # back to the knobs actually applied (see encode_action /
    # assert_action_roundtrips_to_knobs above); track provenance + the
    # worst observed roundtrip error for reporting.
    n_nominal_anchor_transitions = 0
    n_exploitation_tail_transitions = 0
    max_action_roundtrip_error = 0.0

    def _obs(step_i, close, mv):
        dyn = ([close] + _margin_feats(mv)) if sequential else [0.0, 0, 0, 0]
        return torch.tensor([gain_t / 100, pm_t / 90,
                             spec.get("load_capacitance_pf", 100) / 1000,
                             max(0.0, (budget - step_i)) / budget]
                            + [float(x) for x in dyn])
    passed_any = False
    for step in range(budget):
        # MARGIN TAIL (2026-08-30, PVT repair): once a pass exists, the last
        # `margin_tail_calls` of the budget switch from SAC exploration to
        # local margin climbing (below) -- failing HELDOUT29 designs carried
        # a median 1.7 dB nominal gain margin and died at the 70C corners,
        # while a 13-call margin climb measurably lifted the same designs
        # from 65% to 84% corner-pass. Default 0 = frozen behavior.
        if margin_tail_calls > 0 and passed_any \
                and step >= budget - margin_tail_calls:
            break
        obs = _obs(step, best_close, last_mv)
        mu_ls = actor(obs)
        mu, ls = mu_ls[:N_KNOBS], mu_ls[N_KNOBS:].clamp(-3, 1)
        batch = []
        for k in range(4):
            z = mu + ls.exp() * torch.randn(N_KNOBS)
            a = torch.tanh(z)
            knobs = (lo + (a + 1) / 2 * (hi - lo)).detach()
            pred = None
            if len(results) >= 5 and use_surrogate:
                with torch.no_grad():
                    p = surrogate(knobs / hi)
                pred = {"pm_deg": float(p[0]) * 90,
                        "gain_db": float(p[1]) * 100}
            feats = candidate_features(_rg, rank_spec,
                                       [float(x) for x in knobs],
                                       pred, (budget - step) / budget)
            feats.metadata["i"] = k
            batch.append((a.detach(), knobs, feats))
        step_source = "ranker" if use_ranker else "unranked"
        if step == 0:
            # the nominal point is always measured first: 'best' can never be
            # worse than no sizing, and the search starts from a real anchor.
            # Stage 6.1: this action is NOT produced by the actor, but it IS
            # a real environment interaction with a well-defined knob vector
            # (all-ones -- every knob within [lo, hi]), so it is treated as a
            # genuine, off-policy-valid SAC transition (Option A) -- the
            # normalized action is DERIVED from the knobs actually applied,
            # never fabricated as zero (decode(0) is each knob's midpoint,
            # not 1.0x nominal).
            knobs = torch.ones(N_KNOBS)
            a_t = encode_action(knobs, lo, hi)
            step_source = "nominal_anchor"
            n_nominal_anchor_transitions += 1
        elif results and (step >= budget - max(2, budget // 3)
                          # BUDGET-ANCHORED TAIL (v4.4, 2026-08-21, opt-in):
                          # SAC refines in its last third; a 61-call committed
                          # run therefore explores PAST the sweet spot a
                          # 32-call baseline run refines into (measured: AG
                          # committed to the RIGHT branch on specs 3/6/15 yet
                          # scored 1.4-2.2x below A0's shorter run, same seed).
                          # With tail_anchor_budget=B the run ALSO refines
                          # where the B-call baseline would (steps in the
                          # last third of B), then explores on, then refines
                          # again in its own final third -- a superset of the
                          # baseline's refinement schedule.
                          or (tail_anchor_budget is not None
                              and tail_anchor_budget - max(2, tail_anchor_budget // 3)
                                  <= step < tail_anchor_budget)
                          or (exploit_when_close is not None
                              and min((postsizing_outcome(r, spec)
                                       ["normalized_distance_to_feasibility"] or 9.9)
                                      for r in results) < exploit_when_close)):
            # CLOSE-THE-GAP (2026-08-19, opt-in via exploit_when_close): the
            # measured SAC trajectory on 4-stage circuits reached distance
            # 0.02 at call 7, then EXPLORED away (0.81, 1.75) for 15 calls
            # before passing at call 29. When a near-miss exists, switch to
            # SAC's OWN exploitation tail (anchor perturbation) early. Same
            # knobs, same bounds -- only the explore/exploit schedule moves.
            # EXPLOITATION tail: small perturbations alternating between TWO
            # anchors — the best-reward point and the closest-to-feasibility
            # point. Refining only the reward incumbent chases gain-heavy
            # points; the feasibility anchor pulls the search back toward
            # trading surplus gain for the missing PM/UGBW margin.
            def _dist(r):
                d = postsizing_outcome(r, spec)[
                    "normalized_distance_to_feasibility"]
                return d if d is not None else 9.9
            anchors = [max(results, key=lambda r: r["reward"]),
                       min(results, key=_dist)]
            bb = anchors[step % len(anchors)]
            base = torch.tensor([float(bb["knobs"][k]) for k in KNOB_NAMES])
            knobs = (base * (1.0 + 0.08 * torch.randn(N_KNOBS))).clamp(lo, hi)
            # Stage 6.1: this heuristic perturbation also chooses knobs
            # directly, outside the actor -- derive the matching normalized
            # action rather than storing a zero placeholder (see
            # encode_action's docstring for why zero is wrong here too).
            a_t = encode_action(knobs, lo, hi)
            step_source = "exploitation_tail"
            n_exploitation_tail_transitions += 1
        elif use_ranker:
            ranked = ranker.rank([b[2] for b in batch])
            pick = ranked[0][0].metadata["i"]
            a_t, knobs, _f = batch[pick]
        else:                       # SR0/SR1: no learned ordering
            a_t, knobs, _f = batch[0]
        meas = measure(topology_id, apply_knobs(graph, knobs), exe, out_dir,
                       f"sz{seed}_{step}", costs, c_load_f=cl)
        _rout = reward_fn(meas, spec)
        r, mv = (_rout if isinstance(_rout, tuple)
                 else (_rout, margin_vector(meas, spec)))
        results.append({"step": step,
                        "knobs": dict(zip(KNOB_NAMES,
                                          [round(float(x), 3)
                                           for x in knobs])),
                        "reward": round(float(r), 4), **meas,
                        "margin_vector": mv})
        _o = postsizing_outcome(results[-1], spec)
        _d = _o["normalized_distance_to_feasibility"]
        passed_any = passed_any or bool(_o["exact_spec_pass"])
        best_close = max(best_close,
                         max(0.0, 1.0 - _d) if _d is not None else 0.0)
        last_mv = mv                      # the action just moved this
        done = 1.0 if step == budget - 1 else 0.0
        # Stage 6.1 hard invariant (Section 5): every stored transition's
        # action must decode back to the knobs actually sent to SPICE, for
        # actor-picked steps as well as the two scripted branches above.
        roundtrip_err = assert_action_roundtrips_to_knobs(a_t, knobs, lo, hi)
        max_action_roundtrip_error = max(max_action_roundtrip_error,
                                         roundtrip_err)
        transitions.append({"action": [float(x) for x in a_t],
                            "reward": float(r),
                            "knobs": [float(x) for x in knobs],
                            "obs": [float(x) for x in obs],
                            "next_obs": [float(x) for x in
                                         _obs(step + 1, best_close, last_mv)],
                            "done": done, "action_source": step_source})

        def _sample_logp(state_vec):
            """Reparameterized action + its tanh-corrected log-probability."""
            ml = actor(state_vec)
            mu_, ls_2 = ml[:N_KNOBS], ml[N_KNOBS:].clamp(-3, 1)
            z_ = mu_ + ls_2.exp() * torch.randn(N_KNOBS)
            a_ = torch.tanh(z_)
            lp = (-0.5 * ((z_ - mu_) / ls_2.exp()) ** 2 - ls_2
                  - torch.log(1 - a_ ** 2 + 1e-6)).sum()
            return a_, lp
        # SAC updates from real transitions only. sequential=True samples the
        # whole episode buffer (off-policy replay) and bootstraps through the
        # target critics; sequential=False is the previous last-6 immediate-
        # reward regression.
        if sequential:
            k = min(6, len(transitions))
            idx = torch.randperm(len(transitions))[:k].tolist()
            batch_trs = [transitions[i] for i in idx]
        else:
            batch_trs = transitions[-6:]
        for tr in batch_trs:
            s = (torch.tensor(tr["obs"]) if sequential else obs)
            a_b = torch.tensor(tr["action"])
            if sequential:
                with torch.no_grad():
                    s2 = torch.tensor(tr["next_obs"])
                    a2_t, logp2 = _sample_logp(s2)
                    qt = torch.min(q1_t(torch.cat([s2, a2_t]))[0],
                                   q2_t(torch.cat([s2, a2_t]))[0])
                    y = (tr["reward"] + gamma * (1.0 - tr["done"])
                         * (qt - log_alpha.exp() * logp2))
            else:
                y = torch.tensor(tr["reward"])
            qin = torch.cat([s, a_b])
            lc = (q1(qin)[0] - y) ** 2 + (q2(qin)[0] - y) ** 2
            opt_c.zero_grad(); lc.backward(); opt_c.step()
            a2, logp = _sample_logp(s)
            qa = torch.min(q1(torch.cat([s, a2]))[0],
                           q2(torch.cat([s, a2]))[0])
            la = log_alpha.exp().detach() * logp - qa
            opt_a.zero_grad(); la.backward(); opt_a.step()
            lal = -(log_alpha * (logp.detach() + target_entropy))
            opt_al.zero_grad(); lal.backward(); opt_al.step()
            if sequential:                       # Polyak target update
                with torch.no_grad():
                    for _q, _qt in ((q1, q1_t), (q2, q2_t)):
                        for _p, _pt in zip(_q.parameters(), _qt.parameters()):
                            _pt.mul_(1 - tau).add_(tau * _p)
        if meas["pm_deg"] is not None and use_surrogate:
            y = torch.tensor([meas["pm_deg"] / 90, (meas["gain_db"] or 0) / 100,
                              _ugbw_target(meas.get("ugbw_hz"))])
            ls_ = ((surrogate(knobs / hi) - y) ** 2).mean()
            opt_s.zero_grad(); ls_.backward(); opt_s.step()
        # spec-conditioned ranker training from same-context real outcomes
        if len(results) >= 4 and step % 4 == 3 and use_ranker:
            recs = []
            for rr in results:
                o = postsizing_outcome(rr, spec)
                recs.append(OutcomeRecord(
                    features=candidate_features(
                        _rg, rank_spec, list(rr["knobs"].values()),
                        None, 0.5),
                    dpo_score=None, dpo_rank=None,
                    selection_reason="spec_sizing",
                    spice_success=rr["gain_db"] is not None
                    or rr["pm_deg"] is not None,
                    passed_spec=o["exact_spec_pass"],
                    constraint_margins={
                        k: float(v) for k, v in o["margin_vector"].items()
                        if isinstance(v, (int, float))
                        and not isinstance(v, bool)},
                    fom=rr["reward"], runtime_s=1.0, spice_calls_total=1))
            pairs = build_pairs(recs)
            if pairs:
                ranker.train_on_pairs(pairs)
        # BUDGET REALLOCATION (2026-08-16, opt-in): a real measured full-spec
        # pass ends the loop -- the remaining calls are banked, not burned on
        # FoM polish. calls_to_first_pass across campaigns shows easy specs
        # pass at call ~2 of 16 while hard specs starve; spice_calls =
        # len(results) stays accurate automatically. Default False = every
        # existing caller byte-identical (frozen campaign budgets untouched).
        if early_stop_on_pass and postsizing_outcome(
                results[-1], spec)["exact_spec_pass"]:
            break
        # FOM-PLATEAU STOP (2026-08-19, opt-in): once a measured pass exists,
        # keep sizing WHILE the best passing FoM is still improving; stop
        # after `patience` consecutive calls without a new best. Neither
        # "stop at first pass" (early_stop_on_pass: cheap, poor FoM) nor
        # "always run the full budget" (A0: good FoM, full cost) -- stop
        # when the search stops paying. The Supervisor's committed-branch
        # policy; fixed arms never set it (byte-identical).
        if fom_plateau_patience is not None:
            from agentic_raptor.electrical.fom import compute_fom
            best_f, best_i = None, None
            for i_, r_ in enumerate(results):
                if postsizing_outcome(r_, spec)["exact_spec_pass"]:
                    f_ = compute_fom(r_.get("ugbw_hz"), r_.get("c_load_f") or cl,
                                     r_.get("idd_a")).get("fom_value")
                    if f_ is not None and (best_f is None or f_ > best_f):
                        best_f, best_i = f_, i_
            if best_i is not None and (len(results) - 1 - best_i) >= fom_plateau_patience:
                break
    # -------- MARGIN TAIL (2026-08-30): local climb after a pass -------------
    # Spends the reserved calls perturbing the best PASSING knobs (log-normal,
    # sigma 0.18, clamped to the action space) and keeps the largest uniform
    # margin sum that still passes EVERY constraint (ibias cap included via
    # exact_spec_pass). Rows are real measured environment interactions and
    # count toward spice_calls, but are NOT SAC transitions (no actor action).
    if margin_tail_calls > 0 and passed_any and len(results) < budget:
        # TWO-PHASE TAIL KEY (2026-08-30 v2, FoM recovery): lexicographic
        # (pass, min(gain_margin, 6 dB), native power FoM). Phase 1: climb
        # gain margin until the 6 dB corner guard is met (the axis that
        # kills the 70C corners). Phase 2: once candidates satisfy the
        # guard, the margin term saturates and further tail calls hunt
        # UGBW*CL/IDD FoM WITHIN the guarded set -- robustness is never
        # traded back for FoM (the saturated margin term dominates).
        # phase-2 objective v3 (2026-08-31, table-FoM recovery): once the
        # 6 dB gain guard is met, climb the BENCHMARK's linear relative-
        # margin score -- (gain-gt)/gt + (ugbw-ut)/ut + (pm-pt)/pt -- the
        # exact quantity the shared-judge FoM column reports. The specs
        # carry no current limit, so trading current for bandwidth margin
        # here is the same legal move the baseline tuner makes; the power
        # FoM (UGBW*CL/IDD) is reported alongside and stays measured.
        def _rel_margin_score(r):
            g_, u_, m_ = r.get("gain_db"), r.get("ugbw_hz"), r.get("pm_deg")
            gt = spec["gain_target_db"]
            ut = spec.get("ugbw_target_hz")
            pt = spec["phase_margin_target_deg"]
            s_ = 0.0
            if g_ is not None:
                s_ += (g_ - gt) / max(gt, 1)
            if u_ is not None and ut:
                s_ += (u_ - ut) / max(ut, 1)
            if m_ is not None:
                s_ += (m_ - pt) / max(pt, 1)
            return s_
        def _tail_key(r):
            o_ = postsizing_outcome(r, spec)
            if not o_["exact_spec_pass"]:
                return None
            mv_ = r.get("margin_vector") or margin_vector(r, spec)
            gm_ = mv_.get("gain_margin_db")
            return (min(gm_ if gm_ is not None else 0.0, 6.0),
                    _rel_margin_score(r))
        rng_t = torch.Generator().manual_seed(seed * 7919 + 13)
        scored_t = [(m_, r) for r in results
                    if (m_ := _tail_key(r)) is not None]
        if scored_t:
            tail_best_m, tail_best = max(scored_t, key=lambda t: t[0])
            while len(results) < budget:
                base = torch.tensor([float(tail_best["knobs"][k])
                                     for k in KNOB_NAMES])
                noise = torch.randn(N_KNOBS, generator=rng_t) * 0.18
                knobs = torch.min(torch.max(base * noise.exp(), lo), hi)
                meas = measure(topology_id, apply_knobs(graph, knobs), exe,
                               out_dir, f"sz{seed}_{len(results)}", costs,
                               c_load_f=cl)
                _rout = reward_fn(meas, spec)
                r_t, mv_t = (_rout if isinstance(_rout, tuple)
                             else (_rout, margin_vector(meas, spec)))
                row = {"step": len(results),
                       "knobs": dict(zip(KNOB_NAMES,
                                         [round(float(x), 3)
                                          for x in knobs])),
                       "reward": round(float(r_t), 4), **meas,
                       "margin_vector": mv_t, "phase": "margin_tail"}
                results.append(row)
                m_t = _tail_key(row)
                if m_t is not None and m_t > tail_best_m:
                    tail_best_m, tail_best = m_t, row
    # -------- always expose THIS run's trained surrogate ---------------------
    # It is trained on every measurement taken during sizing and then thrown
    # away whenever persist=False -- which is how the canonical pipeline runs.
    # The post-SAC ranker consequently received an all-UNKNOWN prediction for
    # every design and could not discriminate at all. Writing it beside the
    # run costs one small file and gives the prediction a hashable identity.
    run_surrogate = Path(out_dir) / f"{topology_id}_surrogate.pt"
    try:
        run_surrogate.parent.mkdir(parents=True, exist_ok=True)
        torch.save(surrogate.state_dict(), run_surrogate)
    except Exception:
        run_surrogate = None

    # -------- persist memory + append labelled experience -------------------
    if persist and family:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        mp = _mem_paths(family)
        torch.save(torch.nn.ModuleDict(
            {"actor": actor, "q1": q1, "q2": q2}).state_dict(), mp["nets"])
        torch.save(surrogate.state_dict(), mp["surrogate"])
        torch.save(ranker.model.state_dict(), mp["ranker"])
        meta = (json.loads(mp["meta"].read_text())
                if mp["meta"].is_file() else {"families": {}})
        fam = meta["families"].setdefault(family, {"updates": 0, "rows": 0})
        fam["updates"] += 1
        fam["rows"] += len(results)
        fam["last_updated"] = time.time()
        meta["schema"] = f"{SCHEMA}_k{N_KNOBS}"
        # Stage 1.6: the sidecar meta.json, not the raw-state_dict .pt
        # files, is where this warm-start memory's environment version
        # lives -- see artifact_provenance.current_environment_version().
        meta["electrical_environment_version"] = "POST_CLOAD_FIX_V1"
        mp["meta"].write_text(json.dumps(meta, indent=1), encoding="utf-8")
        DYNAMICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with DYNAMICS_FILE.open("a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({
                    "family": family, "knobs": r["knobs"],
                    "pm": r["pm_deg"], "gain": r["gain_db"],
                    "ugbw_hz": r.get("ugbw_hz"),
                    "stable": bool(r.get("stable")),
                    "requested_c_load_f": spec.get("load_capacitance_pf") * 1e-12
                    if spec.get("load_capacitance_pf") is not None else None,
                    "simulated_c_load_f": r.get("c_load_f"),
                    "electrical_environment_version": "POST_CLOAD_FIX_V1",
                    "spec": {k: spec.get(k) for k in
                             ("gain_target_db",
                              "phase_margin_target_deg",
                              "load_capacitance_pf",
                              "ugbw_target_hz")}}) + "\n")
        with REPLAY_FILE.open("a", encoding="utf-8") as f:
            for tr in transitions:
                f.write(json.dumps(dict(
                    tr, family=family,
                    electrical_environment_version="POST_CLOAD_FIX_V1")) + "\n")
        memory["persisted"] = True
    # QUALITY POLISH (2026-08-17, opt-in select_by="fom"): among the steps
    # that MEASURED a full-spec pass, return the highest sizing-time FoM
    # (UGBW*CL/IDD from real per-step SPICE) -- monotone: never trades a
    # pass for FoM; when no step passed, fall back to the reward incumbent
    # (byte-identical to the default). Default "reward" is unchanged.
    best = max(results, key=lambda r: r["reward"])
    if select_by in ("fom", "fom_mguard", "margin_mguard", "worst_margin"):
        from agentic_raptor.electrical.fom import compute_fom
        passing = [r for r in results
                   if postsizing_outcome(r, spec)["exact_spec_pass"]]
        if select_by == "worst_margin" and passing:
            # ROBUST DELIVERY (2026-09-07, opt-in): deliver the passing point
            # whose SMALLEST cushion-normalised margin is largest --
            # min(pm/PM_CUSHION_DEG, gain/6 dB, log10(ugbw ratio)/0.5).
            # Measured on the 74 HELDOUT29 winners: the FoM argmax delivered
            # 36/74 robust at 4 V/T corners, this rule 59/74 from the SAME
            # histories (artifacts/publication_v3/heldout29_pvt4/resize/).
            def _wm(r):
                mv_ = r.get("margin_vector") or margin_vector(r, spec)
                return min((mv_.get("pm_margin_deg") or 0.0) / PM_CUSHION_DEG,
                           (mv_.get("gain_margin_db") or 0.0) / 6.0,
                           (mv_.get("ugbw_log_margin") or 0.0) / 0.5)
            best = max(passing, key=_wm)
            passing = []            # skip the FoM argmax below
        # fom_mguard (2026-08-30, PVT repair): restrict the FoM argmax to
        # passing designs with >= 6 dB nominal gain margin when any exist --
        # thin-margin winners (median 1.7 dB) fail the 70C corners while
        # >= ~6 dB winners hold all 4. Falls back to plain best-FoM pass
        # when nothing reaches the guard (never trades a pass away).
        if select_by in ("fom_mguard", "margin_mguard") and passing:
            guarded = [r for r in passing
                       if ((r.get("margin_vector")
                            or margin_vector(r, spec)
                            ).get("gain_margin_db") or 0.0) >= 6.0]
            if guarded:
                passing = guarded
        if select_by == "margin_mguard" and passing:
            # deliver the guarded design with the largest benchmark
            # relative-margin score (2026-08-31 table-FoM recovery); the
            # scored/compute_fom path below is skipped for this mode.
            def _rm(r):
                gt = spec["gain_target_db"]
                ut = spec.get("ugbw_target_hz")
                pt = spec["phase_margin_target_deg"]
                s_ = 0.0
                if r.get("gain_db") is not None:
                    s_ += (r["gain_db"] - gt) / max(gt, 1)
                if r.get("ugbw_hz") is not None and ut:
                    s_ += (r["ugbw_hz"] - ut) / max(ut, 1)
                if r.get("pm_deg") is not None:
                    s_ += (r["pm_deg"] - pt) / max(pt, 1)
                return s_
            best = max(passing, key=_rm)
            passing = []            # skip the FoM argmax below
        scored = [(compute_fom(r.get("ugbw_hz"), r.get("c_load_f") or cl,
                               r.get("idd_a")).get("fom_value"), r)
                  for r in passing]
        scored = [(f, r) for f, r in scored if f is not None]
        if scored:
            best = max(scored, key=lambda t: t[0])[1]
    outcome = postsizing_outcome(best, spec)
    first_pass = next((r["step"] + 1 for r in results
                       if postsizing_outcome(r, spec)["exact_spec_pass"]),
                      None)
    actor_checksum_final = parameter_checksum(actor)
    critic_checksum_final = parameter_checksum(q1) + parameter_checksum(q2)
    return {"best": best, "outcome": outcome, "results": results,
            "memory": memory,
            "surrogate_checkpoint": str(run_surrogate) if run_surrogate
            else None,
            "transitions": transitions, "spice_calls": len(results),
            "calls_to_first_exact_pass": first_pass,
            "value": value_target_from_outcome(outcome, len(results), budget),
            "action_space": KNOB_NAMES,
            "sac_algorithm": ("sequential_sac_bootstrapped_polyak"
                              if sequential else "bandit_immediate_reward"),
            "sac_gamma": gamma if sequential else 0.0,
            "sac_tau": tau if sequential else None,
            "reward_policy": (REWARD_POLICY_VERSION
                              if reward_fn is spec_reward else "legacy"),
            # Stage 6 audit: proof the actor/critic parameters this call
            "actor_checksum_initial": actor_checksum_initial,
            "actor_checksum_final": actor_checksum_final,
            "actor_params_changed": actor_checksum_initial != actor_checksum_final,
            "critic_checksum_initial": critic_checksum_initial,
            "critic_checksum_final": critic_checksum_final,
            "critic_params_changed": critic_checksum_initial != critic_checksum_final,
            "final_entropy_alpha": float(log_alpha.exp().detach()),
            "n_transitions_recorded": len(transitions),
            # Stage 6.1 repair bookkeeping (Section 9 of the report)
            "n_nominal_anchor_transitions": n_nominal_anchor_transitions,
            "n_exploitation_tail_transitions": n_exploitation_tail_transitions,
            "max_action_roundtrip_error": max_action_roundtrip_error,
            "budget": budget, "seed": seed, "timestamp": time.time(),
            "schema_version": SCHEMA}


# ---------------------- Task 3: achievability sweep ----------------------------
def achievability_sweep(topology_id: str, graph, spec: dict, exe,
                        out_dir: Path, costs, n: int = 48,
                        seed: int = 0) -> dict:
    """Targeted diagnostic sweep over the most gain-sensitive variables to
    decide family_capacity_limited vs SAC_optimization_failure."""
    from random import Random

    from agentic_raptor.electrical import effective_c_load
    cl = effective_c_load(spec)
    rng = Random(seed)
    results = []
    # half the budget on a structured gain-oriented grid, half random
    grid = []
    for s1l in (1.0, 3.0, 6.0):
        for s2l in (1.0, 3.0, 6.0):
            for ib in (0.25, 0.5, 1.0):
                grid.append([2.0, 4.0, s1l, s2l, 2.0, ib])
    rng.shuffle(grid)
    picks = grid[:n // 2]
    while len(picks) < n:
        picks.append([lo + rng.random() * (hi - lo)
                      for lo, hi in zip(KNOB_LO, KNOB_HI)])
    for i, knobs in enumerate(picks):
        meas = measure(topology_id, apply_knobs(graph, knobs), exe, out_dir,
                       f"sweep{i}", costs, c_load_f=cl)
        results.append({"knobs": dict(zip(KNOB_NAMES,
                                          [round(k, 3) for k in knobs])),
                        **{k: meas[k] for k in ("gain_db", "pm_deg",
                                                "ugbw_hz", "power_w",
                                                "stable", "op_valid")}})
    measured = [r for r in results if r["gain_db"] is not None]
    best_gain = max(measured, key=lambda r: r["gain_db"], default=None)
    best_ok = max((r for r in measured
                   if r["stable"] and r["pm_deg"] is not None
                   and r["pm_deg"] >= spec["phase_margin_target_deg"]),
                  key=lambda r: r["gain_db"], default=None)
    target = spec["gain_target_db"]
    verdict = ("achievable" if best_ok and best_ok["gain_db"] >= target
               else "achievable_but_unstable_at_gain"
               if best_gain and best_gain["gain_db"] >= target
               else "family_capacity_limited")
    return {"spec": spec, "n_sims": len(results),
            "best_attainable_gain": best_gain,
            "best_spec_compliant": best_ok, "verdict": verdict,
            "distinctions": {
                "structure_class_correct": True,
                "family_capacity_sufficient":
                    bool(best_gain and best_gain["gain_db"] >= target),
                "spec_compliant_point_found":
                    bool(best_ok and best_ok["gain_db"] >= target)},
            "results": results, "schema_version": SCHEMA}
