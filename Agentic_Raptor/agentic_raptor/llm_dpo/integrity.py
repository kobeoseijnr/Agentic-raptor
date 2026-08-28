"""Data-integrity layer for the Qwen3 self-improvement / DPO campaign.

Pure stdlib (no torch) so every rule is unit-testable without a GPU.

Core principle: a DPO pair is valid only when it carries one real
specification, compares two circuits measured under the same complete
evaluation context, has one non-contradictory evidence-backed preference
direction, and cannot cause dataset collapse.

Provides:
  * canonical specification / evaluation-context / circuit identity (B1, B2)
  * spec-conditioned pair building with an explicit hierarchy (D1-D4, A2-A5)
  * dedup + contradiction resolution (A3, E)
  * balancing and collapse guards (F)
  * frozen-exam manifest + leakage checks (J, A9)
  * self-earned example tiering (G, A10)
  * DPO acceptance decision (I, A7)
  * pre-training readiness blockers (L)
  * campaign directories + archive manifests (K, A8)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

SCHEMA = "integrity.1"
PREFERENCE_POLICY_VERSION = "pref_policy_v1"

# Every measurement in this pipeline runs through ONE executable testbench
# (stage3e2 qualify_device_graph on real ngspice). These constants pin that
# evaluation environment; if the testbench ever changes, bump the version so
# old measurements stop being comparable with new ones (A2).
EVAL_ENV = {
    "technology": "sky130", "pdk_version": "sky130A",
    "process_corner": "tt", "temperature_c": 27, "supply_voltage_v": 1.8,
    "testbench_id": "stage3e2_qualify_device_graph",
    "testbench_version": "1", "simulator": "ngspice",
    "simulation_policy": "ac_stability_v1", "sizing_budget": "default",
}
TESTBENCH_HASH = hashlib.sha256(
    json.dumps(EVAL_ENV, sort_keys=True).encode()).hexdigest()[:16]

#: phase-margin differences below this are treated as measurement noise (A4)
PM_NOISE_DEG = 5.0

#: dataset balancing / collapse limits (F) — configurable
DEFAULT_LIMITS = {
    "max_pairs_per_spec": 4,
    "max_pairs_per_evaluation_context": 4,
    "max_pairs_per_topology_family": 12,
    "max_chosen_appearances_per_candidate": 10,
    "max_rejected_appearances_per_candidate": 10,
    "max_chosen_family_fraction": 0.7,
    "max_single_response_fraction": 0.6,
    "minimum_unique_specs": 3,
    "minimum_unique_chosen_candidates": 2,
    "minimum_unique_topology_families": 2,
}

#: DPO acceptance thresholds (I) — configurable
DEFAULT_ACCEPTANCE = {
    "min_valid_rate": 0.85,
    "valid_drop_tol": 0.05,
    "min_unique_structures": 2,
    "spec_match_drop_tol": 0.02,
    "max_single_response_fraction": 0.6,
}


# ---------------------------- hashing helpers --------------------------------
def sha_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()[:16]


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sha_checkpoint(ckpt_dir) -> str:
    """Hash of a saved adapter directory (weights + config, name-ordered)."""
    p = Path(ckpt_dir)
    if not p.is_dir():
        return "missing"
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".json"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ------------------ B1: canonical specification fields -----------------------
_SPEC_RE = re.compile(
    r"### SPEC gain>=([\d.]+)dB pm>=([\d.]+)deg cl=([\d.]+)pF "
    r"ugbw>=([\d.eE+-]+)Hz tech=(\w+)")


def parse_spec(prompt: str) -> dict | None:
    """Structured spec parsed from the REAL prompt. None -> quarantine (A1).

    PROTOCOL NOTE (Stage 1.5 repair, 2026-08-09; supersedes the prior
    "parsed but not applied" note): ``load_capacitance_pf`` IS now the
    authoritative load for every real simulation in a spec-driven run --
    sizing (sac_size/non-RL baselines), final verification, and every PVT
    corner all resolve their load through
    ``agentic_raptor.electrical.effective_c_load(spec)``, which returns
    ``load_capacitance_pf * 1e-12`` when the spec states one, falling back
    to ``NOMINAL_CLOAD_F`` (500pF) only when it doesn't. The 81-run pilot
    (results_20260809_032835.jsonl) measured nominal.c_load_f == 500pF on
    every row regardless of the spec's stated cl -- that was the bug this
    repair fixes, not intended behaviour. A caller may still force a
    DIFFERENT load via an explicit override (e.g. run_pipeline's
    c_load_override_f), which is recorded, never silent.
    """
    if not prompt:
        return None
    m = _SPEC_RE.search(prompt)
    if not m:
        return None
    return {"gain_target_db": float(m.group(1)),
            "phase_margin_target_deg": float(m.group(2)),
            "load_capacitance_pf": float(m.group(3)),
            "ugbw_target_hz": float(m.group(4)),
            "technology": m.group(5), **EVAL_ENV}


def spec_id(spec: dict) -> str:
    return sha_json(spec)


def evaluation_context_id(spec: dict) -> str:
    """One canonical ID for (structured spec + complete evaluation env). Two
    candidates may be compared ONLY when these IDs are equal (A2/D1)."""
    return sha_json({"spec": spec, "env": EVAL_ENV,
                     "testbench_hash": TESTBENCH_HASH})


# --------------------- B2: canonical circuit identity ------------------------
def compensation_class(obj: dict) -> str:
    c = ((obj.get("compensation") or [{}])[0].get("type", "none")
         if obj.get("compensation") else "none")
    base = {"miller_cap": "miller", "rc_nulling": "rc"}.get(c, c)
    # TIER-2 (2026-08-17): family names carry the new structural markers so
    # "2s_rc" and "2s_rc_cas" are different families downstream (planner,
    # critic, bandit features all key on canonical_family). Pre-existing
    # proposals (no tier-2 blocks) keep their exact historical names.
    from agentic_raptor.llm_dpo.stage3e4 import tier2_flags
    cas, ab = tier2_flags(obj)
    if cas:
        base += "_cas"
    if ab:
        base += "_ab"
    return base


def candidate_identity(obj: dict) -> dict:
    """Canonical identity: insensitive to key order / whitespace / formatting,
    sensitive to real structural differences."""
    from agentic_raptor.llm_dpo.stage3e4 import variant_hash
    stages = len(obj.get("stages", []))
    comp = compensation_class(obj)
    graph = variant_hash(obj)
    norm = hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return {"candidate_id": f"cand_{graph}_{norm[:8]}",
            "topology_hash": graph, "canonical_graph_hash": graph,
            "topology_family_id": f"{stages}s_{comp}",
            "structure_class": [stages, comp,
                                bool(obj.get("output_buffer")),
                                bool(obj.get("local_feedback"))],
            "stage_count": stages, "compensation_class": comp,
            "normalized_response_hash": norm}


# --------------------- D2: full measurement vector ----------------------------
def stability_status(meas: dict) -> str:
    s = meas.get("stability")
    if s == "verified_stable":
        return "verified_stable"
    if s == "verified_unstable":
        return "verified_unstable"
    if s:
        return "stability_ambiguous"
    return "unmeasured"


def outcome_vector(ident: dict, meas: dict | None, spec: dict,
                   target: dict) -> dict:
    """Everything the comparison policy may look at; retained on the pair.
    target: {"variant_hash", "stages"} from the originating corpus record."""
    meas = meas or {}
    status = stability_status(meas) if meas else "unmeasured"
    pm = meas.get("pm")
    functional = meas.get("electrical") == "electrically_functional"
    pm_t = spec["phase_margin_target_deg"]
    pm_margin = (pm - pm_t) if pm is not None else None
    # spec-conditioned hard constraints (A5): the structure must support the
    # spec's gain tier; raw metrics never rank candidates on their own
    tier_feasible = ident["stage_count"] == target.get("stages")
    exact = ident["canonical_graph_hash"] == target.get("variant_hash")
    stable = status == "verified_stable"
    pm_pass = pm_margin is not None and pm_margin >= 0.0
    # post-sizing measurements carry a measured gain; when present it is a
    # hard constraint too — a stable circuit below the gain target is not a
    # spec pass (Task 6). Nominal (gain-less) records keep legacy behaviour.
    gain = meas.get("gain_db")
    gain_pass = gain is not None and gain >= spec["gain_target_db"]
    passed = sum([tier_feasible, exact, stable, pm_pass]) \
        + (int(gain_pass) if gain is not None else 0)
    exact_met = bool(exact and stable and pm_pass
                     and (gain_pass if gain is not None else True))
    return {"parser_valid": True, "schema_valid": True, "graph_valid": True,
            "netlist_buildable": bool(meas), "spice_converged": functional,
            "operating_point_valid": functional,
            "stability_status": status,
            "stability_confidence": 1.0 if status.startswith("verified") else 0.0,
            "phase_margin_deg": pm,
            "phase_margin_margin_deg": pm_margin,
            "gain_db": gain,
            "gain_margin_db": (gain - spec["gain_target_db"])
            if gain is not None else None,
            "gain_tier_feasible": tier_feasible,
            "exact_structure_match": exact,
            "specs_passed_count": passed,
            "exact_spec_met": exact_met,
            "measured": bool(meas), "outcome_tier":
                ("pass" if exact_met else
                 "partial" if passed >= 2 else "fail")}


# ----------------- D3: explicit preference hierarchy --------------------------
def compare_candidates(ova: dict, ovb: dict) -> tuple:
    """Returns (direction 'a'|'b'|None, code, reason_text, confidence).

    Hierarchy (spec-conditioned, A5): measured > valid > operating point >
    gain-tier feasibility > verified stability (within the feasible tier) >
    exact spec pass > hard-constraint count > pm margin beyond noise > no pair.
    Rule note: stability is compared only between gain-tier-feasible
    candidates — 'stable but structurally unable to reach the gain target'
    must not beat 'right structure, not yet stabilised' (that is the raw-
    metric fallacy A5 forbids). Ambiguous stability is never a hard loss; it
    simply cannot decide a pair (D3)."""
    def d(cond_a, cond_b, code, why, conf):
        if bool(cond_a) != bool(cond_b):
            return ("a" if cond_a else "b", code, why, conf)
        return None
    for rule in (
        d(ova["measured"], ovb["measured"], "measured_vs_unmeasured",
          "physically measured beats unmeasured", 0.9),
        d(ova["operating_point_valid"], ovb["operating_point_valid"],
          "operating_point", "valid operating point beats invalid", 0.9),
        d(ova["gain_tier_feasible"], ovb["gain_tier_feasible"],
          "gain_tier_feasibility",
          "structure supports the spec's gain tier; the other cannot reach "
          "the gain target regardless of its raw metrics", 0.9),
    ):
        if rule:
            return rule
    # stability: only between candidates that BOTH support the gain tier and
    # only when both sides are VERIFIED (ambiguous can't decide, A4/D3)
    sa, sb = ova["stability_status"], ovb["stability_status"]
    if ova["gain_tier_feasible"] and ovb["gain_tier_feasible"]:
        if {sa, sb} == {"verified_stable", "verified_unstable"}:
            return (("a" if sa == "verified_stable" else "b"),
                    "verified_stability",
                    "verified stable beats verified unstable within the "
                    "spec-feasible tier", 0.9)
        if ("verified" in sa) != ("verified" in sb):
            return (None, "tie_or_insufficient_evidence",
                    "one side has ambiguous/unmeasured stability; policy "
                    "does not resolve verified-vs-ambiguous", 0.0)
    rule = d(ova["exact_spec_met"], ovb["exact_spec_met"], "exact_spec_pass",
             "exact-spec pass beats non-pass", 0.9)
    if rule:
        return rule
    if ova["specs_passed_count"] != ovb["specs_passed_count"]:
        a_wins = ova["specs_passed_count"] > ovb["specs_passed_count"]
        return (("a" if a_wins else "b"), "hard_constraint_count",
                f"satisfies {max(ova['specs_passed_count'], ovb['specs_passed_count'])} "
                f"hard constraints vs "
                f"{min(ova['specs_passed_count'], ovb['specs_passed_count'])}", 0.8)
    ma, mb = ova["phase_margin_margin_deg"], ovb["phase_margin_margin_deg"]
    if ma is not None and mb is not None and abs(ma - mb) > PM_NOISE_DEG:
        return (("a" if ma > mb else "b"), "pm_margin",
                "larger phase-margin headroom against THIS spec's target "
                f"({round(ma, 1)} vs {round(mb, 1)} deg, beyond "
                f"{PM_NOISE_DEG} deg noise)", 0.7)
    return (None, "tie_or_insufficient_evidence",
            "candidates tied or within measurement noise — no pair", 0.0)


# --------------------- D1/D4: spec-conditioned pair building ------------------
def build_context_pairs(rows: list, meas_map: dict, lineage: dict,
                        quarantine_fn=None) -> dict:
    """rows: design rows [{context_id, prompt, target_variant, target_stages,
    topology_family?, candidates:[{valid, graph_hash, obj}, ...]}, ...].
    meas_map: canonical_graph_hash -> measurement record.
    Only same-evaluation-context candidates are ever compared (D1)."""
    pairs, drops = [], {"dropped_missing_prompt": 0, "dropped_ties": 0,
                        "dropped_low_confidence": 0, "dropped_cross_context": 0,
                        "quarantined_missing_lineage": 0, "no_pair_generated": 0}
    examples = {}
    for row in rows:
        prompt = row.get("prompt")
        spec = parse_spec(prompt) if prompt else None
        if not prompt or not spec or not row.get("target_variant"):
            drops["dropped_missing_prompt"] += 1
            if quarantine_fn:
                quarantine_fn({"record": {k: row.get(k) for k in
                                          ("context_id", "prompt")},
                               "reason": "missing_real_prompt_or_spec"})
            continue
        if row.get("synthetic_prompt") or row.get("training_eligible") is False:
            drops["dropped_missing_prompt"] += 1
            if quarantine_fn:
                quarantine_fn({"record": {"context_id": row.get("context_id")},
                               "reason": "synthetic_prompt_not_training_eligible"})
            continue
        ectx = evaluation_context_id(spec)
        target = {"variant_hash": row["target_variant"],
                  "stages": row.get("target_stages")}
        cands = {}
        for c in row.get("candidates", []):
            if not c.get("valid") or not c.get("obj"):
                continue
            ident = candidate_identity(c["obj"])
            m = meas_map.get(ident["canonical_graph_hash"])
            if m is None:
                continue        # objective 2: unmeasured never enters a pair
            # A2: the measurement must come from the same pinned testbench
            if m.get("testbench_hash", TESTBENCH_HASH) != TESTBENCH_HASH:
                drops["dropped_cross_context"] += 1
                examples.setdefault("dropped_cross_context", {
                    "context_id": row["context_id"],
                    "candidate": ident["canonical_graph_hash"],
                    "why": "measurement from a different testbench/env"})
                continue
            cands[ident["canonical_graph_hash"]] = (c, ident, m)
        keys = sorted(cands)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                (ca, ia, ma), (cb, ib, mb) = cands[keys[i]], cands[keys[j]]
                ova = outcome_vector(ia, ma, spec, target)
                ovb = outcome_vector(ib, mb, spec, target)
                direction, code, why, conf = compare_candidates(ova, ovb)
                if direction is None:
                    drops["dropped_ties" if code.startswith("tie")
                          else "dropped_low_confidence"] += 1
                    examples.setdefault("dropped_ties", {
                        "context_id": row["context_id"], "why": why})
                    continue
                (cw, iw, ovw), (cl, il, ovl) = (
                    ((ca, ia, ova), (cb, ib, ovb)) if direction == "a"
                    else ((cb, ib, ovb), (ca, ia, ova)))
                reason = (f"Both candidates were evaluated under context "
                          f"{ectx[:12]} (spec {spec_id(spec)[:12]}). "
                          f"{iw['topology_family_id']} preferred over "
                          f"{il['topology_family_id']}: {why}.")
                chosen_txt = json.dumps(cw["obj"], separators=(",", ":"))
                rej_txt = json.dumps(cl["obj"], separators=(",", ":"))
                pairs.append({
                    "pair_id": f"pair_{sha_text(ectx + iw['canonical_graph_hash'] + il['canonical_graph_hash'])}",
                    "context_id": row["context_id"],
                    "original_prompt": prompt, "prompt": prompt,
                    "structured_spec": spec, "spec_id": spec_id(spec),
                    "evaluation_context_id": ectx,
                    "prompt_hash": sha_text(prompt),
                    "testbench_hash": TESTBENCH_HASH,
                    "chosen": chosen_txt, "rejected": rej_txt,
                    "preferred": chosen_txt,       # legacy loader alias
                    "chosen_metrics": ovw, "rejected_metrics": ovl,
                    "chosen_identity": iw, "rejected_identity": il,
                    "chosen_topology_hash": iw["canonical_graph_hash"],
                    "rejected_topology_hash": il["canonical_graph_hash"],
                    "preference_reason": reason,
                    "preference_rule": code,
                    "preference_policy_version": PREFERENCE_POLICY_VERSION,
                    "measurement_sources": [
                        {k: m.get(k) for k in ("variant", "device_hash",
                                               "generation", "context_id")}
                        for m in (ma, mb)],
                    "confidence": conf, "split": "train",
                    "lineage": dict(lineage, timestamp=time.time()),
                    "schema_version": SCHEMA})
    return {"pairs": pairs, "drops": drops, "drop_examples": examples}


# ------------- A3/E: dedup + contradiction controls ---------------------------
def dedupe_pairs(pairs: list) -> dict:
    """Canonical unordered-pair key = evaluation_context_id + sorted hashes.
    Exact duplicates collapse to one; reverse/contradictory directions are
    resolved only by a clear confidence gap, else dropped. Never keeps both
    directions (A3)."""
    by_key: dict[tuple, list] = {}
    report = {"raw_pairs": len(pairs), "dropped_exact_duplicates": 0,
              "dropped_reverse_duplicates": 0, "dropped_contradictory": 0,
              "dropped_unresolved_conflict": 0}
    examples: dict[str, dict] = {}
    for p in pairs:
        key = (p["evaluation_context_id"],
               tuple(sorted((p["chosen_topology_hash"],
                             p["rejected_topology_hash"]))))
        by_key.setdefault(key, []).append(p)
    kept = []
    for key, group in sorted(by_key.items()):
        directions = {}
        for p in group:
            directions.setdefault(p["chosen_topology_hash"], []).append(p)
        if len(directions) == 1:
            winner = group[0]
            report["dropped_exact_duplicates"] += len(group) - 1
            if len(group) > 1:
                examples.setdefault("dropped_exact_duplicates",
                                    {"pair_id": group[0]["pair_id"],
                                     "count": len(group)})
            kept.append(winner)
            continue
        # contradictory directions for the same context: resolve only on a
        # clear confidence gap between the best evidence of each side
        (ha, ga), (hb, gb) = sorted(directions.items())
        ca = max(p["confidence"] for p in ga)
        cb = max(p["confidence"] for p in gb)
        if abs(ca - cb) >= 0.2:
            side = ga if ca > cb else gb
            kept.append(sorted(side, key=lambda p: -p["confidence"])[0])
            report["dropped_reverse_duplicates"] += len(group) - 1
            examples.setdefault("dropped_reverse_duplicates",
                                {"pair_id": group[0]["pair_id"],
                                 "kept_confidence": max(ca, cb),
                                 "dropped_confidence": min(ca, cb)})
        else:
            report["dropped_contradictory"] += len(group)
            report["dropped_unresolved_conflict"] += 1
            examples.setdefault("dropped_contradictory",
                                {"pair_id": group[0]["pair_id"],
                                 "directions": sorted(directions),
                                 "confidences": [ca, cb]})
    report["retained_pairs"] = len(kept)
    return {"pairs": kept, "report": report, "examples": examples}


# ------------------ F: balancing + collapse guards ----------------------------
def balance_pairs(pairs: list, limits: dict | None = None) -> dict:
    limits = dict(DEFAULT_LIMITS, **(limits or {}))
    kept, dropped_cap = [], 0
    per_spec, per_ctx, per_family = {}, {}, {}
    chosen_uses, rejected_uses = {}, {}
    for p in sorted(pairs, key=lambda p: (-p["confidence"], p["pair_id"])):
        fam = p["chosen_identity"]["topology_family_id"]
        checks = (
            per_spec.get(p["spec_id"], 0) < limits["max_pairs_per_spec"],
            per_ctx.get(p["evaluation_context_id"], 0)
            < limits["max_pairs_per_evaluation_context"],
            per_family.get(fam, 0) < limits["max_pairs_per_topology_family"],
            chosen_uses.get(p["chosen_topology_hash"], 0)
            < limits["max_chosen_appearances_per_candidate"],
            rejected_uses.get(p["rejected_topology_hash"], 0)
            < limits["max_rejected_appearances_per_candidate"])
        if not all(checks):
            dropped_cap += 1
            continue
        kept.append(p)
        per_spec[p["spec_id"]] = per_spec.get(p["spec_id"], 0) + 1
        per_ctx[p["evaluation_context_id"]] = \
            per_ctx.get(p["evaluation_context_id"], 0) + 1
        per_family[fam] = per_family.get(fam, 0) + 1
        chosen_uses[p["chosen_topology_hash"]] = \
            chosen_uses.get(p["chosen_topology_hash"], 0) + 1
        rejected_uses[p["rejected_topology_hash"]] = \
            rejected_uses.get(p["rejected_topology_hash"], 0) + 1
    n = max(1, len(kept))
    chosen_fam = {}
    rej_fam = {}
    resp_counts = {}
    for p in kept:
        cf = p["chosen_identity"]["topology_family_id"]
        rf = p["rejected_identity"]["topology_family_id"]
        chosen_fam[cf] = chosen_fam.get(cf, 0) + 1
        rej_fam[rf] = rej_fam.get(rf, 0) + 1
        rk = p["chosen_identity"]["normalized_response_hash"]
        resp_counts[rk] = resp_counts.get(rk, 0) + 1
    dist = {"unique_prompts": len({p["prompt_hash"] for p in kept}),
            "unique_specs": len({p["spec_id"] for p in kept}),
            "unique_evaluation_contexts":
                len({p["evaluation_context_id"] for p in kept}),
            "unique_chosen_candidates":
                len({p["chosen_topology_hash"] for p in kept}),
            "unique_rejected_candidates":
                len({p["rejected_topology_hash"] for p in kept}),
            "unique_topology_families": len(chosen_fam),
            "chosen_family_distribution": chosen_fam,
            "rejected_family_distribution": rej_fam,
            "stage_count_distribution": _count(
                kept, lambda p: p["chosen_identity"]["stage_count"]),
            "compensation_class_distribution": _count(
                kept, lambda p: p["chosen_identity"]["compensation_class"]),
            "maximum_candidate_reuse": max(
                [chosen_uses.get(h, 0) for h in chosen_uses] or [0]),
            "maximum_family_fraction": round(
                max(chosen_fam.values()) / n, 3) if chosen_fam else 0.0,
            "max_single_response_fraction": round(
                max(resp_counts.values()) / n, 3) if resp_counts else 0.0,
            "dropped_balance_cap": dropped_cap}
    flags = []
    if kept:
        if dist["max_single_response_fraction"] > limits["max_single_response_fraction"]:
            flags.append("one chosen response exceeds dominance threshold")
        if dist["maximum_family_fraction"] > limits["max_chosen_family_fraction"]:
            flags.append("one topology family dominates chosen examples")
        if dist["unique_specs"] < limits["minimum_unique_specs"]:
            flags.append("too few distinct specifications")
        if dist["unique_chosen_candidates"] < limits["minimum_unique_chosen_candidates"]:
            flags.append("dataset has too few distinct chosen circuits")
        if dist["unique_topology_families"] < limits["minimum_unique_topology_families"]:
            flags.append("too few distinct topology families on chosen side")
    return {"pairs": kept, "distribution": dist, "collapse_flags": flags,
            "limits": limits}


def _count(items, key):
    out = {}
    for it in items:
        k = str(key(it))
        out[k] = out.get(k, 0) + 1
    return out


# ------------------- J/A9: frozen-exam leakage guard --------------------------
def build_exam_manifest(corpus: dict) -> dict:
    held = [r for r in corpus["records"] if r["split"] == "heldout"]
    manifest = {
        "frozen_exam_hash": sha_json(sorted(
            (r["prompt"], r["variant_hash"]) for r in held)),
        "frozen_exam_spec_ids": sorted({r["context_id"] for r in held}),
        "frozen_exam_prompt_hashes": sorted({sha_text(r["prompt"])
                                             for r in held}),
        "frozen_exam_spec_line_hashes": sorted({
            sha_text(r["prompt"].splitlines()[0]) for r in held}),
        "frozen_exam_family_ids": sorted({r["topology_id"] for r in held}),
        "contexts": len(held), "schema_version": SCHEMA}
    return manifest


def leakage_check(examples: list, manifest: dict) -> list:
    """examples: [{prompt, context_id?, topology_id?}]. Returns offenders."""
    offenders = []
    ph = set(manifest["frozen_exam_prompt_hashes"])
    sh = set(manifest["frozen_exam_spec_line_hashes"])
    ids = set(manifest["frozen_exam_spec_ids"])
    fams = set(manifest["frozen_exam_family_ids"])
    for i, e in enumerate(examples):
        p = e.get("prompt") or ""
        reasons = []
        if sha_text(p) in ph:
            reasons.append("exact_exam_prompt")
        if p and sha_text(p.splitlines()[0]) in sh:
            reasons.append("exam_spec_line")
        if e.get("context_id") in ids:
            reasons.append("heldout_spec_id")
        if e.get("topology_id") in fams:
            reasons.append("heldout_topology_family")
        if reasons:
            offenders.append({"index": i, "context_id": e.get("context_id"),
                              "reasons": reasons})
    return offenders


# ----------------- G/A10: self-earned example tiers ---------------------------
def classify_earned(row: dict, cand: dict, meas: dict | None,
                    manifest: dict) -> tuple:
    """Returns (tier, reasons). Only 'verified_self_earned' may enter normal
    SFT. Verified requires: real spec, no exam leakage, valid + realised +
    functional + verified stable + exact spec-structure match + lineage."""
    reasons = []
    prompt = row.get("prompt")
    spec = parse_spec(prompt) if prompt else None
    if not prompt or not spec:
        return "quarantined_self_earned", ["missing_real_prompt_or_spec"]
    if leakage_check([{"prompt": prompt, "context_id": row.get("context_id"),
                       "topology_id": row.get("topology_id")}], manifest):
        return "quarantined_self_earned", ["frozen_exam_leakage"]
    if not cand.get("valid") or not cand.get("obj"):
        return "failed_self_earned", ["invalid_candidate"]
    if not meas:
        return "provisional_self_earned", ["unmeasured"]
    if meas.get("electrical") != "electrically_functional":
        return "failed_self_earned", ["not_electrically_functional"]
    status = stability_status(meas)
    if status == "stability_ambiguous":
        return "provisional_self_earned", ["ambiguous_stability"]
    if status != "verified_stable":
        reasons.append("not_verified_stable")
    ident = candidate_identity(cand["obj"])
    if ident["canonical_graph_hash"] != row.get("target_variant"):
        reasons.append("structure_does_not_match_spec_target")
    pm = meas.get("pm")
    if pm is None or pm < spec["phase_margin_target_deg"]:
        reasons.append("pm_below_spec")
    # verification demands the best POST-SIZING result meet the electrical
    # spec (Task 6): nominal, gain-less measurements can never verify
    gain = meas.get("gain_db")
    if gain is None:
        reasons.append("no_post_sizing_gain_measurement")
    elif gain < spec["gain_target_db"]:
        reasons.append("gain_below_spec")
    if reasons:
        return "provisional_self_earned", reasons
    return "verified_self_earned", ["all_criteria_met"]


# ---------------- I/A7: DPO acceptance decision --------------------------------
def decide_acceptance(sft_exam: dict, dpo_exam: dict,
                      cfg: dict | None = None) -> tuple:
    """Returns (accept_dpo: bool, reasons: list). DPO must not be worse than
    the SFT checkpoint on any configured collapse condition (I)."""
    cfg = dict(DEFAULT_ACCEPTANCE, **(cfg or {}))
    reasons = []
    if dpo_exam["valid_rate"] < cfg["min_valid_rate"]:
        reasons.append(f"valid_rate {dpo_exam['valid_rate']} below "
                       f"{cfg['min_valid_rate']}")
    if dpo_exam["valid_rate"] < sft_exam["valid_rate"] - cfg["valid_drop_tol"]:
        reasons.append("valid rate decreased materially vs SFT")
    if dpo_exam["unique_structures"] < cfg["min_unique_structures"]:
        reasons.append(f"unique_structure_count "
                       f"{dpo_exam['unique_structures']} below threshold")
    if dpo_exam["unique_structures"] < sft_exam["unique_structures"] - 1:
        reasons.append("unique structures collapsed vs SFT")
    if dpo_exam["spec_match_rate"] < sft_exam["spec_match_rate"] - \
            cfg["spec_match_drop_tol"]:
        reasons.append("spec_match_rate materially decreased vs SFT")
    if dpo_exam.get("most_common_response_fraction", 0.0) > \
            cfg["max_single_response_fraction"]:
        reasons.append("one response dominates most prompts")
    return (not reasons), reasons


# ------------------- L: hard pre-training blockers ----------------------------
def readiness_report(pairs: list, manifest: dict, distribution: dict,
                     collapse_flags: list, lineage_ok: bool,
                     lineage_note: str, exam_unchanged: bool,
                     limits: dict | None = None) -> dict:
    limits = dict(DEFAULT_LIMITS, **(limits or {}))
    checks = {}
    checks["all_pairs_have_real_prompts"] = all(
        p.get("prompt") and parse_spec(p["prompt"]) for p in pairs)
    checks["all_pairs_have_structured_specs"] = all(
        p.get("structured_spec") for p in pairs)
    checks["all_pairs_same_evaluation_context"] = all(
        p.get("evaluation_context_id") ==
        evaluation_context_id(p["structured_spec"]) for p in pairs
        if p.get("structured_spec"))
    keys = {}
    for p in pairs:
        k = (p["evaluation_context_id"],
             tuple(sorted((p["chosen_topology_hash"],
                           p["rejected_topology_hash"]))))
        keys.setdefault(k, set()).add(p["chosen_topology_hash"])
    checks["no_reverse_contradictions"] = all(len(v) == 1 for v in keys.values())
    checks["no_frozen_exam_leakage"] = not leakage_check(
        [{"prompt": p["prompt"], "context_id": p.get("context_id")}
         for p in pairs], manifest)
    checks["enough_unique_specs"] = (
        distribution.get("unique_specs", 0) >= limits["minimum_unique_specs"])
    checks["enough_unique_chosen_circuits"] = (
        distribution.get("unique_chosen_candidates", 0)
        >= limits["minimum_unique_chosen_candidates"])
    checks["dominance_limits_pass"] = not collapse_flags
    checks["generation_parent_checkpoint_correct"] = lineage_ok
    checks["frozen_exam_hash_unchanged"] = exam_unchanged
    blockers = [k for k, ok in checks.items() if not ok]
    if collapse_flags:
        blockers += [f"collapse: {f}" for f in collapse_flags]
    if not lineage_ok:
        blockers.append(f"lineage: {lineage_note}")
    return {"training_ready": not blockers, "checks": checks,
            "blockers": blockers, "pair_count": len(pairs),
            "schema_version": SCHEMA}


# --------------------- K/A8: campaign management -------------------------------
def new_campaign(si_root, stale_paths: list, reason: str) -> dict:
    """Fresh campaign directory + archive of stale artifacts with manifest.
    Old artifacts are MOVED (preserved for forensics), never reused."""
    si_root = Path(si_root)
    campaign_id = f"camp_{time.strftime('%Y%m%d_%H%M%S')}"
    root = si_root / campaign_id
    dirs = {n: root / n for n in ("corpus", "pairs", "quarantine", "models",
                                  "evaluations", "logs", "manifests",
                                  "reports", "self_earned", "archive")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in stale_paths:
        p = Path(p)
        if not p.is_file():
            continue
        entry = {"original_path": str(p), "file_hash": sha_file(p),
                 "artifact_type": p.suffix.lstrip("."),
                 "source_campaign": "pre_campaign_era",
                 "archive_reason": reason, "timestamp": time.time()}
        dest = dirs["archive"] / p.name
        p.rename(dest)
        entry["archived_path"] = str(dest)
        manifest.append(entry)
    (dirs["manifests"] / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return {"campaign_id": campaign_id, "root": root, "dirs": dirs,
            "archived": manifest}


def quarantine_writer(campaign: dict):
    qfile = campaign["dirs"]["quarantine"] / "quarantined.jsonl"

    def _write(record: dict):
        with qfile.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(record, timestamp=time.time()),
                               default=str) + "\n")
    return _write
