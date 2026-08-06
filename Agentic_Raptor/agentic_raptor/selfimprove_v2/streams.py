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
    branches = hv.get("branches") or {}
    A, B = branches.get("A") or {}, branches.get("B") or {}
    aA, aB = A.get("authoritative"), B.get("authoritative")
    base = {"generation_spec_id": spec_id, "split": split,
            "seed": hv.get("seed"), "spec_hash": hv.get("spec_hash")}

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
            "ugbw_hz": a.get("ugbw_hz"),
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
        a = br.get("authoritative") or {}
        row = {**base, "branch": lbl,
               "canonical_graph_hash": des.get("canonical_graph_hash"),
               "family": des.get("topology_family"),
               "obj": cand["obj"], "obj_hash": _obj_hash(cand["obj"]),
               "exact_spec_pass": bool(a.get("exact_spec_pass")),
               "distance": a.get("normalized_distance_to_feasibility"),
               "budget": hv.get("budget"),
               "admitted": not reasons, "rejected_because": reasons}
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
