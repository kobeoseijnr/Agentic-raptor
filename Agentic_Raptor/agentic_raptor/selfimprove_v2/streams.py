"""Routing one verified A/B comparison into six separate learning streams.

Each stream is written to its own file so no learner can accidentally read
another's data, and so a stream can be regenerated without touching the rest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

#: every stream this package writes; the loop reports counts per generation
STREAMS = ("ranker_pairs", "puct_examples", "rag_memory", "sac_replay",
           "sft_queue", "proposer_dpo_pairs")


@dataclass(frozen=True)
class GenerationPaths:
    """Where one generation's data and checkpoints live."""
    root: Path
    gen: int

    @property
    def gdir(self) -> Path:
        return self.root / f"gen_{self.gen:03d}"

    @property
    def runs(self) -> Path:
        return self.gdir / "runs"

    @property
    def streams(self) -> Path:
        return self.gdir / "streams"

    @property
    def ckpt(self) -> Path:
        return self.gdir / "checkpoints"

    def stream(self, name: str) -> Path:
        return self.streams / f"{name}.jsonl"

    def mkdirs(self):
        for d in (self.runs, self.streams, self.ckpt):
            d.mkdir(parents=True, exist_ok=True)
        return self


def append(path: Path, rows: list) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return len(rows)


def read(path: Path) -> list:
    if not Path(path).is_file():
        return []
    return [json.loads(x) for x in
            Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _obj_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# SFT admission: the strictest stream, because a wrong target teaches the
# proposer to emit something that does not work.
# --------------------------------------------------------------------------
def sft_admission_reasons(hv: dict, label: str, *, split: str,
                          protected_ids: set,
                          quality_threshold: float | None = None) -> list:
    """Every reason this branch may NOT become a positive SFT target.

    Empty list = admissible. Written as reasons rather than a bool so a
    rejected candidate can be audited instead of vanishing.
    """
    bad = []
    br = (hv.get("branches") or {}).get(label) or {}
    auth = br.get("authoritative")
    des = br.get("design") or {}
    spec = hv.get("spec") or {}

    if split != "train":
        bad.append(f"not_train_split:{split}")
    if spec.get("spec_id") in protected_ids:
        bad.append("protected_evaluation_record")
    # Pre-campaign audit hardening: context_id (spec_id) is a display label
    # that happens to be collision-free ACROSS splits today (verified
    # directly against corpus.json), but that's an empirical property, not
    # a structural guarantee. evaluation_context_id is a content hash of
    # the full spec -- collision-free by construction -- checked here too,
    # belt and braces, so this guard doesn't silently depend on the corpus
    # never being regenerated with different naming.
    if spec:
        try:
            from agentic_raptor.llm_dpo.integrity import evaluation_context_id
            from agentic_raptor.publication.eval_sets import \
                excluded_evaluation_context_ids
            if evaluation_context_id(spec) in excluded_evaluation_context_ids():
                if "protected_evaluation_record" not in bad:
                    bad.append("protected_evaluation_record")
        except Exception:
            pass
    if auth is None:
        bad.append("not_authoritatively_measured")
        return bad                      # nothing below is checkable
    if not auth.get("spice_converged"):
        bad.append("simulation_did_not_converge")
    if not auth.get("operating_point_valid"):
        bad.append("operating_point_invalid")
    # provenance: the measurement must belong to THIS topology, sizing, spec
    if auth.get("topology_hash") != des.get("canonical_graph_hash"):
        bad.append("topology_hash_mismatch")
    if auth.get("sizing_manifest_hash") != des.get("sizing_manifest_hash"):
        bad.append("sizing_manifest_mismatch")
    if auth.get("spec_id") != spec.get("spec_id"):
        bad.append("spec_id_mismatch")
    if auth.get("spec_hash") != hv.get("spec_hash"):
        bad.append("spec_hash_mismatch")
    if not auth.get("netlist_hash"):
        bad.append("missing_netlist_hash")
    # quality: a pass, or an explicitly approved near-miss
    if not auth.get("exact_spec_pass"):
        d = auth.get("normalized_distance_to_feasibility")
        if quality_threshold is None or d is None or d > quality_threshold:
            bad.append("failed_spec_and_below_quality_threshold")
    return bad


def structurally_valid(obj) -> bool:
    from agentic_raptor.llm_dpo import proposal_dict_valid
    try:
        ok, _ = proposal_dict_valid(obj)
        return bool(ok)
    except Exception:
        return False


# --------------------------------------------------------------------------
def harvest_run(hv: dict, *, split: str, protected_ids: set,
                quality_threshold: float | None = None) -> dict:
    """Split ONE pipeline run into per-stream rows.

    Returns {stream_name: [rows]}. Callers append them; nothing is written
    here so a harvest can be inspected before it lands on disk.
    """
    out = {k: [] for k in STREAMS}
    spec = hv.get("spec") or {}
    spec_id = spec.get("spec_id")

    # Pre-campaign audit finding (2026-08-09): sft_admission_reasons() below
    # was the ONLY leakage guard inside this function -- ranker_pairs,
    # puct_examples, rag_memory, and sac_replay had NO protected_ids check
    # at all, relying entirely on callers happening to only ever pass
    # split="train" runs here. That held in practice (verified: train and
    # heldout context_id sets are disjoint, and every real caller today
    # does pass split="train") but was not a STRUCTURAL guarantee -- a
    # future adaptive-mode call on split="heldout"/"blindtest" would have
    # harvested frozen evaluation outcomes into every stream with zero
    # protection. Checked here, once, for ALL streams: both the display-
    # name check (protected_ids) and the content-hash check
    # (evaluation_context_id, collision-free by construction) must clear
    # before ANYTHING is harvested from this run.
    is_protected = bool(spec) and spec.get("spec_id") in protected_ids
    if spec and not is_protected:
        try:
            from agentic_raptor.llm_dpo.integrity import evaluation_context_id
            from agentic_raptor.publication.eval_sets import \
                excluded_evaluation_context_ids
            is_protected = (evaluation_context_id(spec)
                           in excluded_evaluation_context_ids())
        except Exception:
            pass
    if is_protected:
        return out          # every stream empty -- nothing harvested at all
    branches = hv.get("branches") or {}
    A, B = branches.get("A") or {}, branches.get("B") or {}
    aA, aB = A.get("authoritative"), B.get("authoritative")
    # Stage 1.6: every stream record this function produces is, by
    # construction, generated by code that resolves load through
    # effective_c_load() -- so it is unconditionally POST_CLOAD_FIX_V1.
    # requested_c_load_f is the spec's own target; simulated is read back
    # per-branch from `authoritative.c_load_f` below (aA/aB), which is
    # what the real measurement actually used -- carried per-record too,
    # since A and B can in principle differ if either used an override.
    from agentic_raptor.publication.artifact_provenance import POST_CLOAD_FIX_V1
    base = {"generation_spec_id": spec_id, "split": split,
            "seed": hv.get("seed"), "spec_hash": hv.get("spec_hash"),
            "spec_index": hv.get("spec_index"),
            "requested_c_load_f": hv.get("requested_c_load_f"),
            "electrical_environment_version": POST_CLOAD_FIX_V1}

    # ---- 1. ranker pairs: need BOTH sides authoritatively measured --------
    if aA and aB and aA.get("call_id") != aB.get("call_id"):
        def key(a):
            return (0 if a.get("exact_spec_pass") else 1,
                    0 if a.get("operating_point_valid") else 1,
                    0 if a.get("verified_stable") else 1,
                    a.get("normalized_distance_to_feasibility")
                    if a.get("normalized_distance_to_feasibility") is not None
                    else 9.9)
        ka, kb = key(aA), key(aB)
        if ka != kb:
            winner = "A" if ka < kb else "B"
            out["ranker_pairs"].append({
                **base, "winner": winner,
                "prediction_A": A.get("prediction"),
                "prediction_B": B.get("prediction"),
                "design_A": A.get("design"), "design_B": B.get("design"),
                "outcome_A": aA, "outcome_B": aB,
                "ranker_choice": (hv.get("ranker") or {}).get(
                    "selected_design"),
                "spec": spec})

    # ---- 2. PUCT policy/value: visits + measured value target ------------
    # candidate OBJECTS are stored so the registry is rebuildable at train
    # time; ids like "a_sel_p00" are meaningless outside this run
    if hv.get("root_visits") and hv.get("candidates"):
        sel_lbl = (hv.get("ranker") or {}).get("selected_design")
        sel_auth = branches.get(sel_lbl, {}).get("authoritative")
        vt = None
        if sel_auth:
            d = sel_auth.get("normalized_distance_to_feasibility")
            vt = (1.0 if sel_auth.get("exact_spec_pass")
                  else max(-1.0, 1.0 - 2.0 * float(d)) if d is not None
                  else None)
        out["puct_examples"].append({
            **base, "value_target": vt, "spec": spec,
            "simulated_c_load_f": sel_auth.get("c_load_f") if sel_auth else None,
            # Pre-campaign audit: SPICE provenance for the measurement that
            # produced value_target -- was recorded implicitly (via
            # candidates[*].canonical_graph_hash + measured_from_design)
            # but not the actual call_id/topology_hash, which is what makes
            # this value_target traceable to one specific real ngspice call.
            "measured_call_id": sel_auth.get("call_id") if sel_auth else None,
            "measured_topology_hash": sel_auth.get("topology_hash") if sel_auth else None,
            "visit_distribution": hv.get("root_visits"),
            "candidate_visits": hv.get("candidate_visits"),
            "root_action_ids": hv.get("root_action_ids"),
            # the REAL search state and per-candidate manifests, so training
            # reconstructs what the search saw rather than approximating it
            "root_state": hv.get("root_state"),
            "candidate_manifests": hv.get("candidate_manifests") or {},
            "candidates": [{"llm_proposal_id": c["llm_proposal_id"],
                            "canonical_graph_hash": c["canonical_graph_hash"],
                            "canonical_family": c["canonical_family"],
                            "obj": c["obj"]}
                           for c in hv.get("candidates") or []],
            "search": hv.get("search"),
            "measured_from_design": sel_lbl})

    # ---- 3. RAG memory: EVERY authoritative outcome, pass or fail ---------
    for lbl, br in (("A", A), ("B", B)):
        a = br.get("authoritative")
        if not a:
            continue
        des = br.get("design") or {}
        out["rag_memory"].append({
            **base, "context_id": spec_id, "branch": lbl,
            "variant": des.get("canonical_graph_hash"),
            "family": des.get("topology_family"),
            "stages": (int(str(des.get("topology_family") or "0")[0])
                       if str(des.get("topology_family") or "")[:1].isdigit()
                       else None),
            "stability": ("verified_stable" if a.get("verified_stable")
                          else "verified_unstable"),
            "pm": a.get("pm_deg"), "gain_db": a.get("gain_db"),
            "ugbw_hz": a.get("ugbw_hz"), "idd_a": a.get("idd_a"),
            "simulated_c_load_f": a.get("c_load_f"),
            "postsizing": bool(a.get("exact_spec_pass")),
            "exact_spec_pass": bool(a.get("exact_spec_pass")),
            "failure_reason": a.get("exact_failure_reason"),
            "call_id": a.get("call_id"), "source": "raptor_v2"})

    # ---- 4. SAC replay (optional persistence) -----------------------------
    for lbl, br in (("A", A), ("B", B)):
        sac = br.get("sac") or {}
        des = br.get("design") or {}
        for step in sac.get("trajectory") or []:
            if step.get("knobs") is None:
                continue
            out["sac_replay"].append({
                **base, "branch": lbl,
                "family": des.get("topology_family"),
                "topology_hash": des.get("canonical_graph_hash"),
                **step})

    # ---- 5. SFT queue: VERIFIED SUCCESSES ONLY ---------------------------
    cand_by_pid = {c["llm_proposal_id"]: c
                   for c in hv.get("candidates") or []}
    for lbl, br in (("A", A), ("B", B)):
        des = br.get("design") or {}
        pid = des.get("llm_proposal_id")
        cand = cand_by_pid.get(pid)
        if not cand:
            continue
        reasons = sft_admission_reasons(
            hv, lbl, split=split, protected_ids=protected_ids,
            quality_threshold=quality_threshold)
        if not structurally_valid(cand["obj"]):
            reasons.append("structurally_invalid")
        if reasons:
            # FALSE / rejected outcomes are not routed to SFT at all -- this
            # stream is TRUE-only. Failures still reach rag_memory below.
            continue
        a = br.get("authoritative") or {}
        row = {**base, "branch": lbl, "spec": spec,
               "canonical_graph_hash": des.get("canonical_graph_hash"),
               "family": des.get("topology_family"),
               "obj": cand["obj"], "obj_hash": _obj_hash(cand["obj"]),
               "exact_spec_pass": bool(a.get("exact_spec_pass")),
               "distance": a.get("normalized_distance_to_feasibility"),
               "budget": hv.get("budget"),
               # 2026-08-11 (SFT self-improvement wiring): sft_admission_
               # reasons() already re-derives all of these from `a`/`des` to
               # decide admission, but previously discarded them once a row
               # was admitted -- an eligibility check or dataset builder
               # downstream of THIS file had no way to see the very
               # provenance that made the row admissible in the first
               # place, nor the measured performance (gain/PM/UGBW/IDD)
               # every quality tier or metadata report needs. Additive only:
               # existing readers use .get() and are unaffected.
               "verification_spice_call_id": a.get("call_id"),
               "verification_mode": a.get("mode"),
               "simulated_c_load_f": a.get("c_load_f"),
               "gain_db": a.get("gain_db"), "pm_deg": a.get("pm_deg"),
               "ugbw_hz": a.get("ugbw_hz"), "idd_a": a.get("idd_a"),
               "admitted": True, "rejected_because": []}
        out["sft_queue"].append(row)

    # ---- 6. proposer DPO preference pairs --------------------------------
    if aA and aB and A.get("design") and B.get("design"):
        ha = A["design"].get("canonical_graph_hash")
        hb = B["design"].get("canonical_graph_hash")
        pa = bool(aA.get("exact_spec_pass"))
        pb = bool(aB.get("exact_spec_pass"))
        if ha != hb and pa != pb:          # only a CLEAR measured preference
            win, lose = ("A", "B") if pa else ("B", "A")
            out["proposer_dpo_pairs"].append({
                **base,
                "preferred_hash": branches[win]["design"][
                    "canonical_graph_hash"],
                "rejected_hash": branches[lose]["design"][
                    "canonical_graph_hash"],
                "preferred_family": branches[win]["design"]["topology_family"],
                "rejected_family": branches[lose]["design"]["topology_family"],
                "preferred_obj": cand_by_pid.get(
                    branches[win]["design"].get("llm_proposal_id"), {}
                ).get("obj"),
                "rejected_obj": cand_by_pid.get(
                    branches[lose]["design"].get("llm_proposal_id"), {}
                ).get("obj"),
                "basis": "authoritative_exact_spec_pass"})
    return out
