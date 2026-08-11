"""Typed separation of PREDICTION from AUTHORITATIVE MEASUREMENT (Parts 4/5).

The defect this exists to make impossible: the previous v2 code took the
outcome dict returned by `sac_size(..., exe=ngspice, ...)` -- a real measured
result -- and assigned it to fields named `predicted_feasible`,
`predicted_margins` and `operating_point_valid`, then handed that to the
post-SAC ranker. The ranker was therefore choosing which design to "verify"
while already holding the verification answer. Every downstream number would
look excellent and mean nothing.

Two distinct types, and a `SurrogatePrediction` that structurally CANNOT
carry a SPICE result: there is no field for one, and `authoritative` is a
frozen False. Any attempt to build a prediction from measured data must go
through `SurrogatePrediction.from_model(...)`, which records a surrogate
checkpoint hash.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


class PredictionLeakage(AssertionError):
    """A prediction object was built from authoritative measurements."""


# ------------------------- Part 5: real checkpoint hashing -------------------
def checkpoint_sha256(path) -> str | None:
    """SHA-256 over the checkpoint FILE BYTES.

    Tensor-sum hashing (the previous approach) collides trivially: permuting
    any two weights, or negating a symmetric pair, leaves the sum unchanged.
    A checkpoint identity that cannot distinguish different models is worse
    than none, because it looks like provenance.
    """
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(state_dict) -> str:
    """SHA-256 over a deterministic serialisation of an ORDERED state dict.

    Used when a model is held in memory rather than on disk. Every parameter
    tensor contributes its full byte content, in key order.
    """
    h = hashlib.sha256()
    for k in sorted(state_dict):
        v = state_dict[k]
        h.update(k.encode())
        try:
            arr = v.detach().cpu().contiguous().numpy()
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        except Exception:
            h.update(repr(v).encode())
    return h.hexdigest()


def directory_sha256(path) -> str | None:
    """SHA-256 over every file in a checkpoint directory, in path order."""
    p = Path(path)
    if not p.is_dir():
        return checkpoint_sha256(p)
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if not f.is_file():
            continue
        h.update(str(f.relative_to(p)).encode())
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


#: hard constraints a specification can state. A constraint is REQUIRED when
#: the spec gives it a target; anything else is not applicable.
_SPEC_CONSTRAINTS = (("gain", "gain_target_db"),
                     ("pm", "phase_margin_target_deg"),
                     ("ugbw", "ugbw_target_hz"),
                     ("power", "power_target_w"),
                     ("area", "area_target_um2"))


def required_constraint_names(spec: dict) -> tuple:
    """Which hard constraints this specification actually demands."""
    return tuple(name for name, key in _SPEC_CONSTRAINTS
                 if spec.get(key) is not None)


@dataclass(frozen=True)
class SurrogatePrediction:
    """Pre-SPICE estimate for ONE sized design. Never authoritative.

    Frozen so a caller cannot mutate a prediction into a measurement after
    construction.
    """
    topology_hash: str
    sizing_manifest_hash: str
    # predicted electricals
    gain_db: float | None = None
    pm_deg: float | None = None
    ugbw_hz: float | None = None
    power_w: float | None = None
    area_um2: float | None = None
    # probabilities in [0, 1] -- None means UNKNOWN, never "fine"
    operating_point_probability: float | None = None
    stability_probability: float | None = None
    # normalised margins: >= 0 satisfied, < 0 violated
    normalized_margins: dict = field(default_factory=dict)
    predictive_uncertainty: float | None = None
    # provenance
    source: str = "surrogate"
    authoritative: bool = False
    spice_result_id: None = None
    surrogate_checkpoint_hash: str | None = None
    feature_schema_version: str = "post_sac_surrogate.v1"
    prediction_timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.source != "surrogate" or self.authoritative is not False:
            raise PredictionLeakage(
                "SurrogatePrediction must be non-authoritative")
        if self.spice_result_id is not None:
            raise PredictionLeakage(
                "SurrogatePrediction carries a SPICE result id -- the ranker "
                "must not see the authoritative answer")

    @property
    def worst_predicted_violation(self) -> float | None:
        """Part 7: >= 0 always; 0 means nothing violated, larger is worse.

        Computed HERE from normalised margins rather than accepted from a
        caller, so sign conventions cannot drift between producers.
        """
        vals = [v for v in self.normalized_margins.values()
                if isinstance(v, (int, float))]
        if not vals:
            return None
        return float(max(0.0, max(-v for v in vals)))

    def predicted_feasible_for(self, spec: dict) -> bool | None:
        """Part 3: feasibility against EVERY constraint the spec requires.

        A missing prediction is UNKNOWN, never a pass. Returning True because
        the two constraints that happen to be predicted are satisfied would
        let a design with unknown UGBW and unknown power outrank one that is
        merely slightly short on gain -- the ranker would systematically
        prefer ignorance.
        """
        required = required_constraint_names(spec)
        if not required:
            return None
        for name in required:
            v = self.normalized_margins.get(name)
            if v is None:
                return None                     # unknown stays unknown
        return all(self.normalized_margins[n] >= 0 for n in required)

    @property
    def predicted_feasible(self) -> bool | None:
        """Spec-free fallback: only for margins already present.

        Prefer `predicted_feasible_for(spec)`; this cannot know which
        constraints the specification demanded.
        """
        if not self.normalized_margins:
            return None
        if any(v is None for v in self.normalized_margins.values()):
            return None
        return all(v >= 0 for v in self.normalized_margins.values())

    def missing_constraints(self, spec: dict) -> tuple:
        return tuple(n for n in required_constraint_names(spec)
                     if self.normalized_margins.get(n) is None)

    def hard_constraints_satisfied(self) -> int:
        return sum(1 for v in self.normalized_margins.values()
                   if isinstance(v, (int, float)) and v >= 0)


@dataclass(frozen=True)
class AuthoritativeSpiceOutcome:
    """A real ngspice measurement. The only thing allowed to decide pass."""
    call_id: str
    topology_hash: str
    sizing_manifest_hash: str
    netlist_hash: str
    mode: str                       # final_verification | backup | sizing
    exact_spec_pass: bool
    operating_point_valid: bool
    spice_converged: bool
    verified_stable: bool           # explicit -- never parsed from a string
    # Part 6: the specification this measurement belongs to. Without these a
    # pair can be assembled from two different specs and look well-formed.
    spec_id: str | None = None
    spec_hash: str | None = None
    stability_status: str | None = None
    gain_db: float | None = None
    pm_deg: float | None = None
    ugbw_hz: float | None = None
    power_w: float | None = None
    # idd_a: TOTAL measured supply current (real ngspice op-point branch
    # current). Never the MB-SAC Ibias design/optimizer knob -- see
    # agentic_raptor.electrical.fom for the distinction this feeds.
    idd_a: float | None = None
    # c_load_f: the load capacitance ACTUALLY applied by the testbench that
    # produced this measurement (not necessarily the spec's `cl=` target --
    # see agentic_raptor.electrical.NOMINAL_CLOAD_F).
    c_load_f: float | None = None
    area_um2: float | None = None
    robustness: float | None = None          # e.g. PVT success fraction
    hard_constraints_passed: int = 0
    hard_constraints_total: int = 0
    normalized_distance_to_feasibility: float | None = None
    exact_failure_reason: str | None = None
    worst_failing_constraint: str | None = None
    constraints: dict = field(default_factory=dict)
    source: str = "ngspice"
    authoritative: bool = True
    measurement_timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.source != "ngspice" or self.authoritative is not True:
            raise AssertionError("AuthoritativeSpiceOutcome must be ngspice")


def netlist_hash_of(graph) -> str:
    """Stable hash of the emitted device graph."""
    try:
        from agentic_raptor.topology_rl.stage3e2_edits import device_graph_hash
        return device_graph_hash(graph)
    except Exception:
        return hashlib.sha256(repr(graph).encode()).hexdigest()[:16]


def outcome_from_sizing(outcome: dict, *, call_id: str, topology_hash: str,
                        sizing_manifest_hash: str, netlist_hash: str,
                        mode: str, best: dict | None = None,
                        spec_id: str | None = None,
                        spec_hash: str | None = None
                        ) -> AuthoritativeSpiceOutcome:
    """Wrap a real measured outcome dict as an authoritative outcome.

    `verified_stable` is taken from an explicit check, not from
    `str.startswith("verified")` -- that idiom accepts "verified_unstable".
    """
    best = best or {}
    stab = str(outcome.get("stability_status")
               or best.get("stability") or "")
    return AuthoritativeSpiceOutcome(
        call_id=call_id, topology_hash=topology_hash,
        sizing_manifest_hash=sizing_manifest_hash, netlist_hash=netlist_hash,
        mode=mode, spec_id=spec_id, spec_hash=spec_hash,
        exact_spec_pass=bool(outcome.get("exact_spec_pass")),
        operating_point_valid=bool(outcome.get("operating_point_valid")),
        spice_converged=bool(outcome.get("spice_converged")),
        verified_stable=(stab == "verified_stable"),
        stability_status=stab or None,
        gain_db=best.get("gain_db"), pm_deg=best.get("pm_deg"),
        ugbw_hz=best.get("ugbw_hz"), power_w=best.get("power_w"),
        idd_a=best.get("idd_a"), c_load_f=best.get("c_load_f"),
        hard_constraints_passed=outcome.get("hard_constraints_passed", 0),
        hard_constraints_total=outcome.get("hard_constraints_total", 0),
        normalized_distance_to_feasibility=outcome.get(
            "normalized_distance_to_feasibility"),
        exact_failure_reason=outcome.get("exact_failure_reason"),
        worst_failing_constraint=outcome.get("worst_failing_constraint"),
        constraints=outcome.get("constraints") or {})


def assert_no_leakage(pred: SurrogatePrediction):
    assert pred.source == "surrogate", "prediction source is not surrogate"
    assert pred.authoritative is False, "prediction marked authoritative"
    assert pred.spice_result_id is None, "prediction carries a SPICE id"
    wv = pred.worst_predicted_violation
    assert wv is None or wv >= 0, f"violation must be >= 0, got {wv}"
    return True


def content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
