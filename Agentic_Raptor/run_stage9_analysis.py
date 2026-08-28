"""STAGE 9: A0-A8 campaign analysis -- consumes the canonical driver's
results JSONL + per-run trace files and produces every Stage-9 report
input: per-arm metrics, paired comparisons vs A0, AlphaZero search/
checkpoint audit, DPO authority analysis, campaign integrity audit, and
the immutable publication-results store.

Read-only over campaign outputs; writes only its own analysis artifacts.
"""
from __future__ import annotations

import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ABL_DIR = ROOT / "artifacts/publication_v3/ablation_v3"
TRACE_DIR = ROOT / "artifacts/publication_v2/raptor_v2_runs"   # run_pipeline OUT
OUT_DIR = ABL_DIR / "stage9_analysis"
REQUIRED_AZ_SHA = "822a305c81e856f8d5d56d29e62fedaeb6a57614afdde4bba1ad3346fd6a5125"
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
EXPECTED_JOBS = 81


def load_results() -> list[dict]:
    rows = []
    for p in sorted(ABL_DIR.glob("results_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    # de-dup on experiment_id keeping the LAST occurrence (rerun wins)
    by_id = {}
    for r in rows:
        by_id[r["experiment_id"]] = r
    return list(by_id.values())


def find_trace(row: dict) -> dict | None:
    aid = row.get("ablation_id")
    seed = row.get("pipeline_seed")
    idx = row.get("spec_index")
    if aid is None or seed is None or idx is None:
        return None
    pattern = f"ABLv3_{aid}_s{seed}_heldout_{idx:03d}_*.json"
    matches = sorted(TRACE_DIR.glob(pattern))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(st.mean(vals), 4),
           "std": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0,
           "median": round(st.median(vals), 4), "n": len(vals)}


def _z(row):
    nom = row.get("nominal") or {}
    if nom.get("complete_pass"):
        return 1.0
    tr = row.get("_trace")
    if tr:
        d = (tr.get("stage9_verification") or {}).get("distance_to_feasibility")
        if d is not None:
            return max(-1.0, 1.0 - 2.0 * float(d))
    return None


def arm_metrics(rows: list[dict]) -> dict:
    ok = [r for r in rows if not str(r.get("trace_result", "")).startswith("ERROR")]
    n = len(ok)
    passes = [bool((r.get("nominal") or {}).get("complete_pass")) for r in ok]
    n_pass = sum(passes)
    ctfp = [r.get("calls_to_first_pass") for r in ok]
    ctfp_reached = [c for c in ctfp if c is not None]
    zs = [_z(r) for r in ok]
    zs = [z for z in zs if z is not None]
    dists = []
    fom_all, fom_feas = [], []
    pvt_elig = pvt_robust = 0
    topo = set()
    valid_netlist = 0
    for r in ok:
        tr = r.get("_trace")
        if tr:
            d = (tr.get("stage9_verification") or {}).get("distance_to_feasibility")
            if d is not None:
                dists.append(float(d))
            for h in (tr.get("provenance_chain") or {}).get("puct_selected", []):
                topo.add(h)
            if (tr.get("stage9_verification") or {}).get("measured"):
                valid_netlist += 1
        f = (r.get("fom") or {}).get("fom_value")
        if f is not None:
            fom_all.append(f)
            if (r.get("nominal") or {}).get("complete_pass"):
                fom_feas.append(f)
        pv = r.get("pvt") or {}
        if pv.get("robust_complete_pass") is not None:
            pvt_elig += 1
            if pv.get("robust_complete_pass"):
                pvt_robust += 1
    # Pass@K over seeds per spec: fraction of specs passed by >=1 seed
    by_spec = defaultdict(list)
    for r, p in zip(ok, passes):
        by_spec[r["spec_index"]].append(p)
    pass_at_1 = round(st.mean([any(v[:1]) for v in by_spec.values()]), 4) if by_spec else None
    pass_at_k = round(st.mean([any(v) for v in by_spec.values()]), 4) if by_spec else None
    return {
        "n_jobs": len(rows), "n_ok": n,
        "n_errors": len(rows) - n,
        "final_pass": n_pass, "final_pass_rate": round(n_pass / n, 4) if n else None,
        "pass_at_1_per_spec": pass_at_1, "pass_at_seedK_per_spec": pass_at_k,
        "calls_to_first_pass": {**(_stat(ctfp_reached) or {}),
                                "censored_not_reached": sum(1 for c in ctfp if c is None)},
        "z": _stat(zs), "distance": _stat(dists),
        "fom_all_valid": _stat(fom_all), "fom_feasible_only": _stat(fom_feas),
        "optimization_calls": _stat([(r.get("spice") or {}).get("optimization_calls") for r in ok]),
        "verification_calls": _stat([(r.get("spice") or {}).get("final_verification_calls") for r in ok]),
        "pvt_calls": _stat([(r.get("spice") or {}).get("pvt_calls") for r in ok]),
        "total_spice": _stat([(r.get("spice") or {}).get("total_calls") for r in ok]),
        "runtime_s": _stat([r.get("seconds") for r in ok]),
        "pvt_eligible": pvt_elig,
        "unconditional_robust_pass": pvt_robust,
        "unconditional_robust_rate": round(pvt_robust / n, 4) if n else None,
        "conditional_pvt_robustness": round(pvt_robust / n_pass, 4) if n_pass else None,
        "valid_netlist_rate": round(valid_netlist / n, 4) if n else None,
        "unique_selected_topologies": len(topo),
    }


def paired_vs_a0(rows_by_arm: dict) -> dict:
    a0 = {(r["spec_index"], r["pipeline_seed"]): r for r in rows_by_arm.get("A0", [])}
    out = {}
    for aid in ARMS:
        if aid == "A0":
            continue
        wins = losses = both_pass = both_fail = 0
        dz, dsp, drt = [], [], []
        dw = dl = dt = 0
        n_pairs = 0
        for r in rows_by_arm.get(aid, []):
            key = (r["spec_index"], r["pipeline_seed"])
            a = a0.get(key)
            if a is None or str(r.get("trace_result", "")).startswith("ERROR") \
                    or str(a.get("trace_result", "")).startswith("ERROR"):
                continue
            n_pairs += 1
            ap = bool((a.get("nominal") or {}).get("complete_pass"))
            bp = bool((r.get("nominal") or {}).get("complete_pass"))
            both_pass += ap and bp
            both_fail += (not ap) and (not bp)
            wins += ap and not bp        # A0-only win
            losses += bp and not ap      # ablation-only win
            za, zb = _z(a), _z(r)
            if za is not None and zb is not None:
                dz.append(zb - za)
            ta, tb = a.get("_trace"), r.get("_trace")
            if ta and tb:
                da = (ta.get("stage9_verification") or {}).get("distance_to_feasibility")
                db = (tb.get("stage9_verification") or {}).get("distance_to_feasibility")
                if da is not None and db is not None:
                    if db < da:
                        dw += 1
                    elif db > da:
                        dl += 1
                    else:
                        dt += 1
            if a.get("seconds") and r.get("seconds"):
                drt.append(r["seconds"] - a["seconds"])
            sa = (a.get("spice") or {}).get("total_calls")
            sb = (r.get("spice") or {}).get("total_calls")
            if sa is not None and sb is not None:
                dsp.append(sb - sa)
        out[aid] = {"n_pairs": n_pairs, "a0_only_wins": wins,
                   "ablation_only_wins": losses, "both_pass": both_pass,
                   "both_fail": both_fail, "paired_dz": _stat(dz),
                   "distance_wins_losses_ties_for_ablation": [dw, dl, dt],
                   "runtime_delta_s": _stat(drt), "spice_delta": _stat(dsp)}
    return out


def az_and_dpo_audit(rows: list[dict]) -> dict:
    az_stats = defaultdict(lambda: {"ckpt_ok": 0, "ckpt_bad": 0, "nodes": [],
                                    "depth": [], "edited_selected": 0,
                                    "selected": 0, "fingerprints": set()})
    dpo = {"ranker_authority": 0, "hard_gate_only": 0, "dpo_selected_A": 0,
          "dpo_selected_B": 0, "authority_outcome_pass": 0,
          "authority_outcome_fail": 0}
    identity_bad = []
    cload_bad = []
    post_cload_bad = []
    for r in rows:
        tr = r.get("_trace")
        if not tr:
            continue
        aid = r["ablation_id"]
        s5 = tr.get("stage5_alphazero") or {}
        prov = s5.get("checkpoint_provenance") or {}
        if s5.get("search_topology", "").startswith("true_alphazero"):
            a = az_stats[aid]
            if prov.get("checkpoint_sha256") == REQUIRED_AZ_SHA and prov.get("checkpoint_loaded"):
                a["ckpt_ok"] += 1
            else:
                a["ckpt_bad"] += 1
            if prov.get("parameter_fingerprint"):
                a["fingerprints"].add(json.dumps(prov["parameter_fingerprint"], sort_keys=True))
            a["nodes"].append(s5.get("tree_nodes"))
            a["depth"].append(s5.get("max_depth_reached"))
            for c in s5.get("ranking", []):
                a["selected"] += 1
                if c.get("is_edited_descendant"):
                    a["edited_selected"] += 1
        # topology identity
        pc = (tr.get("provenance_chain") or {}).get("puct_selected") or []
        s6 = tr.get("stage6_sizing") or {}
        hashes6 = {s6.get(l, {}).get("topology_hash") for l in ("A", "B")} - {None}
        if pc and hashes6 and set(pc) != hashes6:
            identity_bad.append(r["experiment_id"])
        nom = tr.get("nominal") or {}
        if nom.get("c_load_unexplained_mismatch"):
            cload_bad.append(r["experiment_id"])
        s9 = tr.get("stage9_verification") or {}
        sr = tr.get("stage8_ranker") or {}
        if sr:
            basis = sr.get("decision_basis")
            if basis == "hard_safety_gate":
                dpo["hard_gate_only"] += 1
            elif basis:
                dpo["ranker_authority"] += 1
                if sr.get("selected_design") == "A":
                    dpo["dpo_selected_A"] += 1
                else:
                    dpo["dpo_selected_B"] += 1
                if nom.get("complete_pass"):
                    dpo["authority_outcome_pass"] += 1
                else:
                    dpo["authority_outcome_fail"] += 1
    az_out = {}
    for aid, a in az_stats.items():
        az_out[aid] = {"checkpoint_ok_jobs": a["ckpt_ok"], "checkpoint_bad_jobs": a["ckpt_bad"],
                      "unique_fingerprints": len(a["fingerprints"]),
                      "tree_nodes": _stat(a["nodes"]), "max_depth": _stat(a["depth"]),
                      "edited_selected": a["edited_selected"],
                      "selected_total": a["selected"]}
    return {"alphazero_by_arm": az_out, "dpo_authority": dpo,
           "topology_identity_violations": identity_bad,
           "cload_violations": cload_bad,
           "post_cload_violations": post_cload_bad}


def build_publication_store(rows: list[dict]) -> dict:
    """Item 17: immutable publication-results store -- graph-hash-level
    outcome records, marked by arm/split/seed, NEVER fed to training."""
    records = []
    for r in rows:
        tr = r.get("_trace")
        if not tr:
            continue
        s9 = tr.get("stage9_verification") or {}
        sel_hash = (tr.get("provenance_chain") or {}).get("ranker_selected")
        nom = tr.get("nominal") or {}
        if sel_hash is None:
            continue
        records.append({
            "stage9_arm": r["ablation_id"], "split": r.get("split") or "heldout",
            "pipeline_seed": r["pipeline_seed"], "spec_index": r["spec_index"],
            "spec_hash": r.get("spec_hash"), "graph_hash": sel_hash,
            "exact_spec_pass": nom.get("complete_pass"),
            "distance_to_feasibility": s9.get("distance_to_feasibility"),
            "electricals": {k: nom.get(k) for k in ("gain_db", "pm_deg", "ugbw_hz", "idd_a")},
            "post_cload_version": "POST_CLOAD_FIX_V1",
            "eligible_for_future_value_research": False,   # heldout split: NEVER train
            "eligibility_note": "heldout publication spec -- permanently excluded "
                               "from all training stores (leakage guard)",
        })
    return {"n_records": len(records), "records": records,
           "future_value_model_eligible": 0,
           "note": "all Stage-9 records are heldout-split; eligible-for-training "
                  "count is ZERO by design -- they document performance, never "
                  "train models"}


def main():
    rows = load_results()
    for r in rows:
        r["_trace"] = find_trace(r)
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["ablation_id"]].append(r)

    combos = {(r["ablation_id"], r["spec_index"], r["pipeline_seed"]) for r in rows}
    integrity = {
        "expected_jobs": EXPECTED_JOBS, "completed_jobs": len(rows),
        "missing": EXPECTED_JOBS - len(combos),
        "duplicates_collapsed": True,
        "errors": [r["experiment_id"] for r in rows
                  if str(r.get("trace_result", "")).startswith("ERROR")],
        "arms_present": sorted(by_arm),
        "traces_found": sum(1 for r in rows if r["_trace"] is not None),
        "config_hashes": {aid: sorted({r.get("configuration_hash") for r in rs})
                         for aid, rs in by_arm.items()},
        "budget_hashes": sorted({r.get("budget_hash") for r in rows if r.get("budget_hash")}),
    }
    audit = az_and_dpo_audit(rows)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "integrity": integrity,
        "per_arm": {aid: arm_metrics(by_arm.get(aid, [])) for aid in ARMS},
        "paired_vs_A0": paired_vs_a0(by_arm),
        "az_dpo_audit": audit,
    }
    store = build_publication_store(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "STAGE9_ANALYSIS.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")
    (OUT_DIR / "STAGE9_PUBLICATION_RESULTS_STORE.json").write_text(
        json.dumps(store, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"integrity": integrity,
                     "per_arm_pass": {a: report["per_arm"][a]["final_pass_rate"]
                                     for a in ARMS if report["per_arm"][a]["n_ok"]}},
                    indent=1, default=str))
    print(f"analysis -> {OUT_DIR}")


if __name__ == "__main__":
    main()
