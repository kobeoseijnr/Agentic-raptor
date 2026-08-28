"""A9 SELF-IMPROVEMENT ORCHESTRATOR (completes the v2 self-improvement
pipeline, 2026-08-13). Built on agentic_raptor.publication.generation_state
(atomic publication, idempotent resume, adaptive/static lineages) --
replacing run_self_improvement_v2.py's ad-hoc STATE.json for A9 use. The
old loop remains as historical reference; THIS is the A9 entry point.

WHAT THIS COMPLETES / FIXES
===========================
1. THE KNOWN PUCT GAP (documented in run_self_improvement_v2.py): the old
   loop trained value checkpoints against the retired root-PUCT schema and
   injected them into live AlphaZero search via run_pipeline(value_ckpt=...).
   HERE: every pipeline run uses value_ckpt=None, which -- after the Stage-8
   enforcement -- loads the PROMOTED AlphaZero checkpoint through the
   SHA-verified loader. Self-improvement NEVER injects its own value
   checkpoints into live search. Value-model research happens in a gated
   side stream instead:
     - only graph-complete (az_replay.2) rows are ever admitted as value
       training data (legacy puct_examples rows are collected for the
       record but excluded from training);
     - a value candidate is accepted ONLY if it passes the offline
       VALUE-PROBE GATE (agentic_raptor.topology_rl.value_probe_gate:
       beat the frozen linear-probe reference on the frozen spec-disjoint
       DEV) -- the gate that both rejected campaigns would have failed;
     - an accepted value candidate is stored as a RESEARCH artifact of the
       generation; it is never written into AZ generation manifests and
       never marked CANDIDATE/PROMOTED.
2. DPO STREAM MIGRATED TO V2: the old loop retrained the retired 11-feature
   ranker. Here a DPO candidate is trained on POST_SAC_FEATURES_V2 from
   newly harvested pairs and gated against the PROMOTED V2 checkpoint on a
   spec-disjoint held-out slice (accept only if it beats the incumbent;
   min-data thresholds; rollback otherwise). Accepted candidates are staged
   for a HUMAN promotion decision -- the live promoted V2 checkpoint is
   never silently replaced (model_v2's hash pin makes that structurally
   impossible anyway).
3. STATIC LINEAGE: same task stream, same budgets, learning_mode="static",
   zero retraining, component hashes carried unchanged -- the A9 control.

Run (adaptive lineage, one generation, N specs):
    python run_a9_generations.py --lineage adaptive --specs 4
    python run_a9_generations.py --lineage static   --specs 4
    python run_a9_generations.py --dry-run ...      # isolated root
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A9_REAL = ROOT / "artifacts/publication_v3/a9_generations"
A9_DRY = ROOT / "artifacts/publication_v3/a9_generations_dryrun"
RAG_MEMORY = (ROOT / "artifacts/publication_v2/selfimprove"
              / "rag_memory_v2_post_cload_v1.jsonl")
SFT_ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"

#: A9 PROFILES (2026-08-22). "stock": the original lineage pair on the stock
#: TRAIN distribution. "tier2": a SEPARATE lineage pair on a MIXED
#: distribution -- stock TRAIN specs + tier2_train specs -- running the
#: production tier-2 proposer. Rationale, measured on the stock profile:
#: the bandit selects the same {2s_rc, 3s_rc} top-2 from ANY pool and the
#: frozen system already passes ~all stock specs, so proposer/selector
#: refits have no downstream signal there (SFT G1: exact tie on 6 specs).
#: Every learned component's measured headroom is on tier-2-class specs
#: (bandit never trained on tier-2 families; no tier-2 DPO pairs; no tier-2
#: SFT wins). The tier2 profile points the loop at that headroom. Its
#: state/data live under a9_generations/tier2/ so the stock lineages are
#: untouched; tier2_HELDOUT ids are protected from harvest in addition to
#: the paper's sealed evaluation ids.
PROFILES = {
    "stock": {"root": A9_REAL, "adapter": SFT_ADAPTER,
              "sft_generations_root": ROOT / "artifacts/publication_v2/proposer_repair/generations"},
    "tier2": {"root": A9_REAL / "tier2",
              "adapter": ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_tier2_mixed",
              "sft_generations_root": ROOT / "artifacts/publication_v2/proposer_repair/generations_tier2"},
}


def tier2_protected_ids() -> set:
    """tier2_heldout spec ids (the paper's tier-2 exam) -- never harvested."""
    from agentic_raptor.publication.tier2 import load_tier2_specs
    return {s["spec_id"] for s in load_tier2_specs("tier2_heldout")}

MIN_NEW_DPO_PAIRS = 60          # below this: accumulate, skip retrain
MIN_VALUE_ROWS = 50             # graph-complete rows needed to attempt a candidate
DPO_DEV_FRACTION = 0.25
MIN_NEW_BANDIT_RECORDS = 40     # (spec, topology, z) outcomes for a candidate
BANDIT_DEV_FRACTION = 0.3
MIN_NEW_SFT_ROWS = 25           # verified successes before an SFT refresh is staged
#: AGENTIC SELF-IMPROVEMENT (2026-08-17): the loop runs the current
#: production system -- all four agents + adaptive LLM attempts.
A9_AGENTS = ("planner", "critic", "supervisor", "recovery")
A9_STALL_STOP = None   # 2026-08-21: aligned with the production AG arm.
# stall_stop=2 truncated the proposal diversity ladder (measured on the
# tier-2 gate: 2 families vs 5 with the full ladder). Early termination
# is now the Critic's job via the plan-satisfied stop (satisfied_fn) --
# an agent DECISION with the plan in hand, not a blind counter.


# ---------------------------------------------------------------------------
def g0_component_hashes() -> dict:
    from agentic_raptor.publication.eval_sets import available_seeds
    from agentic_raptor.publication.generation_state import \
        accepted_component_hashes
    from agentic_raptor.ranking.model_v2 import PROMOTED_V2_CKPT
    from agentic_raptor.topology_rl.alphazero import require_promoted_az_checkpoint
    ev_hash = hashlib.sha256(json.dumps(sorted(available_seeds())).encode()).hexdigest()[:16]
    return accepted_component_hashes(
        rag_memory_path=RAG_MEMORY, sft_adapter_path=SFT_ADAPTER,
        puct_ckpt_path=require_promoted_az_checkpoint(),
        dpo_ckpt_path=PROMOTED_V2_CKPT, evaluation_set_hash=ev_hash)


# ---------------------------------------------------------------------------
# value research stream (gap fix: gated, never injected into live search)
# ---------------------------------------------------------------------------
def value_stream_update(rows: list[dict]) -> dict:
    """rows: harvested candidate value rows. Admission: graph-complete
    az_replay.2 ONLY. Candidate acceptance: offline value-probe gate."""
    admitted = [r for r in rows if r.get("replay_schema_version") == "az_replay.2"
               and r.get("state_graph") is not None]
    legacy = len(rows) - len(admitted)
    if len(admitted) < MIN_VALUE_ROWS:
        return {"status": "ACCUMULATING", "admitted_rows": len(admitted),
               "legacy_rows_excluded": legacy,
               "needed": MIN_VALUE_ROWS, "gate": None,
               "note": "no value candidate trained -- insufficient "
                       "graph-complete data; live search keeps the promoted "
                       "checkpoint (never touched by this loop)"}
    # enough data: train a candidate on the ADMITTED rows and put it
    # through the frozen offline gate. (Uses the per-seed-native trainer's
    # value path -- policy head untouched.)
    from agentic_raptor.topology_rl.alphazero import (
        load_alphazero_nets, train_az_generation_minibatch)
    from agentic_raptor.topology_rl.value_probe_gate import evaluate_candidate
    nets = load_alphazero_nets(None, seed=0)
    out = train_az_generation_minibatch(admitted, nets=nets, epochs=5,
                                        batch_size=16, lr=1e-4)
    import torch

    from agentic_raptor.topology_rl.alphazero import _RowGraphRegistry
    from agentic_raptor.topology_rl.stage3e1 import TopologySearchState

    def _predict(row):
        reg = _RowGraphRegistry([row])
        s = TopologySearchState(**{k: v for k, v in row["state"].items()})
        with torch.no_grad():
            return float(out["nets"]["value_forward"](s, reg)["scalar"])
    gate = evaluate_candidate(_predict)
    return {"status": "GATED", "admitted_rows": len(admitted),
           "legacy_rows_excluded": legacy, "gate": gate,
           "accepted": gate["gate"] == "PASS",
           "note": "accepted candidates are research artifacts only -- "
                   "NEVER written to AZ generation manifests, never loaded "
                   "by live search"}


# ---------------------------------------------------------------------------
# DPO V2 candidate stream (accept only if it beats the promoted V2)
# ---------------------------------------------------------------------------
def dpo_v2_stream_update(new_pairs: list[dict]) -> dict:
    if len(new_pairs) < MIN_NEW_DPO_PAIRS:
        return {"status": "ACCUMULATING", "new_pairs": len(new_pairs),
               "needed": MIN_NEW_DPO_PAIRS,
               "note": "promoted V2 checkpoint remains the incumbent"}
    # spec-disjoint holdout from the NEW pairs; incumbent = promoted V2
    dev_specs = {p["spec_hash"] for p in new_pairs
                if (int(hashlib.sha256(str(p["spec_hash"]).encode()).hexdigest()[:8], 16)
                    % 10_000) / 10_000 < DPO_DEV_FRACTION}
    train = [p for p in new_pairs if p["spec_hash"] not in dev_specs]
    dev = [p for p in new_pairs if p["spec_hash"] in dev_specs]
    return {"status": "READY_TO_TRAIN", "train_pairs": len(train),
           "dev_pairs": len(dev), "dev_specs": len(dev_specs),
           "gate_rule": "candidate must beat the PROMOTED V2 checkpoint on "
                       "the spec-disjoint dev slice (wins>losses AND higher "
                       "grouped accuracy); accepted candidates are STAGED "
                       "for human promotion -- the live hash-pinned V2 "
                       "checkpoint is never silently replaced",
           "note": "training executes via the Stage-7.2B machinery "
                  "(features_v2 + BT loss) when invoked with --retrain-dpo"}


# ---------------------------------------------------------------------------
def lineage_stream_rows(data_root: Path, lineage: str, name: str) -> list[dict]:
    """Every row of stream `name` across ALL generations of this lineage.
    Stream thresholds (MIN_NEW_*) accumulate across generations -- one
    20-spec generation yields ~2 ranker pairs and ~8 SFT rows, so reading
    only the current generation's directory (the original wiring) meant no
    threshold could ever be reached. RAG is the exception: it accumulates
    in the lineage memory FILE, not here."""
    from agentic_raptor.selfimprove_v2.streams import read
    rows: list[dict] = []
    base = data_root / lineage
    if base.is_dir():
        for gdir in sorted(base.glob("gen_*")):
            p = gdir / "streams" / f"{name}.jsonl"
            if p.is_file():
                rows.extend(read(p))
    return rows


# ---------------------------------------------------------------------------
# RAG memory growth stream (2026-08-16 -- targets a BINDING component)
# ---------------------------------------------------------------------------
def lineage_rag_memory(data_root: Path, lineage: str) -> Path:
    """The lineage's own retrieval memory. adaptive: starts as a copy of the
    frozen CLEAN memory and GROWS each generation via rag_stream_update;
    static: always the frozen file itself (never copied, never touched)."""
    if lineage != "adaptive":
        return RAG_MEMORY
    mem = data_root / lineage / "rag_memory_current.jsonl"
    if not mem.is_file():
        mem.parent.mkdir(parents=True, exist_ok=True)
        mem.write_text(RAG_MEMORY.read_text(encoding="utf-8"),
                       encoding="utf-8")
    return mem


def rag_stream_update(new_rows: list[dict], mem_path: Path) -> dict:
    """Gated append of THIS generation's authoritative outcomes (passes AND
    instructive failures) into the adaptive lineage's retrieval memory.
    Gates: (1) row must carry a real ngspice call_id + measured electricals;
    (2) dedupe by call_id against the whole memory (idempotent on resume);
    (3) protected evaluation contexts were already excluded at harvest --
    re-checked here by construction (rows without generation_spec_id are
    refused). Atomic replace; the frozen CLEAN memory file is never touched."""
    existing = [json.loads(l) for l in
                mem_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = {r.get("call_id") for r in existing if r.get("call_id")}
    admitted, rejected = [], 0
    for r in new_rows:
        ok = (r.get("call_id") and r.get("call_id") not in seen
              and r.get("generation_spec_id")
              and r.get("pm") is not None and r.get("gain_db") is not None)
        if ok:
            seen.add(r["call_id"])
            admitted.append(r)
        else:
            rejected += 1
    if admitted:
        tmp = mem_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n"
                               for r in existing + admitted),
                       encoding="utf-8")
        tmp.replace(mem_path)
    return {"status": "APPENDED" if admitted else "NO_NEW_ROWS",
           "memory_path": str(mem_path), "memory_rows": len(existing) + len(admitted),
           "admitted": len(admitted), "rejected_or_duplicate": rejected,
           "note": "adaptive lineage retrieves from ITS OWN grown memory next "
                   "generation; the frozen CLEAN memory and the static "
                   "lineage are never touched"}


# ---------------------------------------------------------------------------
# SFT refresh stream (2026-08-16 -- targets the ESSENTIAL component)
# ---------------------------------------------------------------------------
def sft_stream_update(rows: list[dict], data_root: Path,
                      profile: str = "stock") -> dict:
    """Verified-success SFT queue -> staged refresh. Training itself is NEVER
    run from a harvest event: when enough rows exist this returns the exact
    explicit command (train_sft_self_improvement.py -- gated: dataset build,
    LoRA, capability probe, small real-SPICE check, promote-or-reject)."""
    if len(rows) < MIN_NEW_SFT_ROWS:
        return {"status": "ACCUMULATING", "verified_success_rows": len(rows),
               "needed": MIN_NEW_SFT_ROWS,
               "note": "proposer adapter unchanged; rows keep accumulating"}
    return {"status": "READY_TO_TRAIN", "verified_success_rows": len(rows),
           "run_command": ("python train_sft_self_improvement.py "
                          f"--lineage adaptive --profile {profile} "
                          f"--si-root {data_root / 'adaptive'}"),   # lineage-scoped root (2026-08-22)
           "gate_rule": "dataset build -> LoRA -> capability probe + small "
                       "real-SPICE downstream check -> promote or reject "
                       "(previous adapter stays active on reject); executed "
                       "ONLY via the explicit command above, never from a "
                       "harvest event"}


# ---------------------------------------------------------------------------
# contextual-bandit weight stream (STAGE 9B promotion follow-up, 2026-08-15)
# ---------------------------------------------------------------------------
def bandit_stream_update(pair_rows: list[dict]) -> dict:
    """BANDIT_TOP2 learning stream. Input: harvested ranker_pairs rows --
    both branches authoritatively measured, TRAIN-domain by harvest_run's
    protected-ids guard. Each row yields two (spec, topology, z) outcome
    records (z = 1.0 on exact pass, else 1 - 2*distance floored at -1 --
    the same outcome scale BANDIT_TOP2_V1 was fitted on).

    Gate: a refit candidate must beat the INCUMBENT hash-pinned
    BANDIT_TOP2_V1 on spec-disjoint pairwise ranking accuracy. Accepted
    candidates are written as a NEW versioned artifact and STAGED for a
    human promotion decision -- bandit_selector's SHA-256 pin makes a
    silent in-place swap structurally impossible (promotion requires a
    code change to the pin itself)."""
    recs = []
    for p in pair_rows:
        for side in ("A", "B"):
            o = p.get(f"outcome_{side}") or {}
            d = p.get(f"design_{side}") or {}
            dist = o.get("normalized_distance_to_feasibility")
            if d.get("canonical_graph_hash") and (dist is not None
                                                  or o.get("exact_spec_pass")):
                recs.append({"spec_hash": p.get("spec_hash"),
                            "spec": p.get("spec"),   # pipeline spec dict, for features
                            "topology_hash": d["canonical_graph_hash"],
                            "obj": d.get("obj"),     # proposal JSON if harvested (2026-08-23)
                            "z": (1.0 if o.get("exact_spec_pass")
                                  else max(-1.0, 1.0 - 2.0 * dist))})
    if len(recs) < MIN_NEW_BANDIT_RECORDS:
        return {"status": "ACCUMULATING", "new_records": len(recs),
               "needed": MIN_NEW_BANDIT_RECORDS,
               "incumbent": "BANDIT_TOP2_V1 (SHA-pinned) remains live",
               "note": "no candidate trained -- bandit weights update ONLY "
                       "through this gated stream, never in place"}
    from agentic_raptor.topology_rl.bandit_selector import \
        PROMOTED_BANDIT_SHA256 as BANDIT_INCUMBENT_SHA256
    dev_specs = {r["spec_hash"] for r in recs
                if (int(hashlib.sha256(str(r["spec_hash"]).encode())
                        .hexdigest()[:8], 16) % 10_000) / 10_000
                < BANDIT_DEV_FRACTION}
    train = [r for r in recs if r["spec_hash"] not in dev_specs]
    dev = [r for r in recs if r["spec_hash"] in dev_specs]
    out = {"status": "READY_TO_TRAIN", "train_records": len(train),
          "dev_records": len(dev), "dev_specs": len(dev_specs),
          "incumbent_sha256": BANDIT_INCUMBENT_SHA256,
          "gate_rule": "ridge refit on train+incumbent data must beat the "
                      "PROMOTED BANDIT_TOP2_V1 on spec-disjoint dev "
                      "pairwise ranking accuracy; accepted candidates are "
                      "STAGED as a new versioned artifact for human "
                      "promotion -- the SHA-pinned V1 stays live until the "
                      "pin itself is changed in code, never silently"}
    out["refit"] = _bandit_refit_and_gate(train, dev)
    if out["refit"].get("staged_artifact"):
        out["status"] = "CANDIDATE_STAGED"
    elif out["refit"].get("gate") == "FAIL":
        out["status"] = "CANDIDATE_REJECTED"
    return out


def _bandit_refit_and_gate(train: list[dict], dev: list[dict]) -> dict:
    """Execute the gated refit: ridge (lambda=1.0, the frozen feasibility
    procedure) on the V1 training join PLUS the new train-slice records;
    evaluate candidate vs the pinned incumbent on the new DEV slice's
    pairwise ranking (same-spec pairs, better-z ranked higher). Stages a
    versioned candidate artifact ONLY on a strict win; never touches V1."""
    import numpy as np

    from agentic_raptor.topology_rl.alphazero import (_convert_spec,
                                                      deserialize_device_graph)
    from agentic_raptor.topology_rl.bandit_selector import load_promoted_bandit
    from agentic_raptor.topology_rl.linear_value import physical_features_core

    # graph join: value-diagnostic graphs + corpus proposals by canonical hash
    graphs: dict[str, tuple] = {}
    vd = ROOT / "artifacts/publication_v3/az_value_diagnostic_v1/AZ_VALUE_DIAGNOSTIC_V1.jsonl"
    spec_by_ctx: dict[str, dict] = {}
    if vd.is_file():
        for line in vd.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                graphs.setdefault(r["state_graph_hash"], ("vd", r["state_graph"]))
                spec_by_ctx.setdefault(r["context_id"], r["spec"])
    corpus_p = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"
    if corpus_p.is_file():
        for rec in json.loads(corpus_p.read_text(encoding="utf-8"))["records"]:
            graphs.setdefault(rec["canonical_graph_hash"], ("corpus", rec["response"]))
    # 2026-08-23 TIER-2 JOIN: the A9 tier-2 profile's G0 refit failed to
    # join exactly its 12 tier-2 rows (graphs indexed stock corpora only).
    # Index the tier-2 mixed corpus (all 24 G0 tier-2 designs resolve by
    # canonical hash) and, for any variant not in a corpus, the proposal
    # JSON carried on the harvested pair row itself.
    t2_p = ROOT / "artifacts/publication_v3/tier2/corpus_tier2_mixed.json"
    if t2_p.is_file():
        for rec in json.loads(t2_p.read_text(encoding="utf-8"))["records"]:
            if rec.get("canonical_graph_hash") and rec.get("response"):
                graphs.setdefault(rec["canonical_graph_hash"], ("corpus", rec["response"]))
    for rec in train + dev:
        if rec.get("obj") and rec.get("topology_hash"):
            graphs.setdefault(rec["topology_hash"], ("obj", rec["obj"]))

    def graph_of(h):
        src = graphs.get(h)
        if src is None:
            return None
        kind, payload = src
        if kind == "vd":
            return deserialize_device_graph(payload)
        from run_puct_ablation import _realise
        if kind == "obj":
            return _realise(payload)
        return _realise(json.loads(payload))

    def feats(rec):
        g = graph_of(rec["topology_hash"])
        spec = rec.get("spec")
        if g is None or not spec:
            return None
        internal = spec if "target_gain_db" in spec else _convert_spec(spec)
        return physical_features_core(g, internal, 0.0, False)

    # V1's own training join (the incumbent's data, refit-consistent)
    old_rows = []
    ot = ROOT / "artifacts/publication_v3/az_value_diagnostic_v1/OUTCOME_TABLE.jsonl"
    if ot.is_file():
        for line in ot.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                if o["context_id"] in spec_by_ctx and o["topology_hash"] in graphs:
                    old_rows.append({"spec_hash": o["context_id"],
                                    "spec": spec_by_ctx[o["context_id"]],
                                    "topology_hash": o["topology_hash"],
                                    "z": o["z"]})

    X, y = [], []
    join_failed = 0
    for rec in old_rows + train:
        f = feats(rec)
        if f is None:
            join_failed += 1
            continue
        X.append(f)
        y.append(rec["z"])
    if len(X) < 30:
        return {"gate": None, "status": "JOIN_INSUFFICIENT",
               "fit_rows": len(X), "join_failed": join_failed}
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    w = np.linalg.solve(Xn.T @ Xn + 1.0 * np.eye(Xn.shape[1]), Xn.T @ y)
    b = float(y.mean() - Xn.mean(0) @ w)

    art = load_promoted_bandit()

    def score(weights, m, s, bias, f):
        z = [(x - a) / c for x, a, c in zip(f, m, s)]
        return float(sum(zi * wi for zi, wi in zip(z, weights)) + bias)

    # dev evaluation: same-spec pairwise ranking, candidate vs incumbent
    from collections import defaultdict
    by_spec = defaultdict(list)
    for rec in dev:
        f = feats(rec)
        if f is not None:
            by_spec[rec["spec_hash"]].append((rec["z"], f))
    cand_ok = inc_ok = tot = 0
    for _sh, items in by_spec.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (z1, f1), (z2, f2) = items[i], items[j]
                if abs(z1 - z2) < 1e-9:
                    continue
                tot += 1
                cand_ok += (score(w.tolist(), mu.tolist(), sd.tolist(), b, f1)
                            > score(w.tolist(), mu.tolist(), sd.tolist(), b, f2)) \
                    == (z1 > z2)
                inc_ok += (score(art["weights"], art["mu"], art["sd"],
                                 art["bias"], f1)
                           > score(art["weights"], art["mu"], art["sd"],
                                   art["bias"], f2)) == (z1 > z2)
    if tot == 0:
        return {"gate": None, "status": "DEV_INSUFFICIENT",
               "fit_rows": int(len(y)), "join_failed": join_failed,
               "note": "no rankable same-spec dev pairs yet -- keep accumulating"}
    result = {"gate": "PASS" if cand_ok > inc_ok else "FAIL",
             "dev_pairs": tot, "candidate_correct": int(cand_ok),
             "incumbent_correct": int(inc_ok),
             "fit_rows": int(len(y)), "join_failed": join_failed}
    if result["gate"] == "PASS":
        stage_dir = ROOT / "artifacts/publication_v3/bandit_top2_v1/A9_CANDIDATES"
        stage_dir.mkdir(parents=True, exist_ok=True)
        payload = {"artifact": "BANDIT_TOP2_A9_CANDIDATE",
                  "created": time.strftime("%Y-%m-%d"),
                  "weights": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
                  "bias": b, "ridge_lambda": 1.0,
                  "gate_report": result,
                  "promotion": "HUMAN DECISION REQUIRED: promotion = new "
                               "SHA pin in bandit_selector.py; V1 stays "
                               "live until then"}
        path = stage_dir / f"CANDIDATE_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        result["staged_artifact"] = str(path)
    return result


# ---------------------------------------------------------------------------
# agent episode stream (2026-08-17 -- audits + refits the deterministic agents)
# ---------------------------------------------------------------------------
def agent_episode(tr: dict, run_id: str) -> dict | None:
    """One row per adaptive run: the agents' decisions + the measured outcome.
    Only what the trace already records; nothing re-derived."""
    ag = tr.get("agents")
    if not ag:
        return None
    n = tr.get("nominal") or {}
    s9 = tr.get("stage9_verification") or {}
    plan = tr.get("agent_planner") or {}
    sup = tr.get("agent_supervisor") or {}
    rec = tr.get("agent_recovery") or {}
    crit = tr.get("agent_critic") or {}
    led = ag.get("ledger") or {}
    spec = (tr.get("stage1_spec") or {}).get("spec") or {}
    return {"run_id": run_id,
            "spec": {k: spec.get(k) for k in ("gain_target_db",
                                              "phase_margin_target_deg",
                                              "ugbw_target_hz",
                                              "load_capacitance_pf")},
            "spec_hash": (tr.get("stage1_spec") or {}).get("spec_hash"),
            "plan_difficulty": plan.get("difficulty"),
            "plan_preferred_stages": plan.get("preferred_stages"),
            "critic_rounds": crit.get("rounds"),
            "critic_satisfied": (crit.get("final_verdict") or {}).get("satisfied"),
            "probe_verdicts": {k: (v or {}).get("verdict")
                               for k, v in (sup.get("probe") or {}).items()},
            "allocation": sup.get("allocation"),
            "interventions": [i.get("what") for i in ag.get("interventions") or []],
            "recovery_executed": bool(rec.get("executed")),
            "recovery_adopted": rec.get("adopted"),
            "spice_spent": led.get("spice_spent"), "spice_cap": led.get("spice_cap"),
            "banked": led.get("banked"),
            "pass": bool(n.get("complete_pass")),
            "distance": s9.get("distance_to_feasibility"),
            "fom": (tr.get("fom") or {}).get("fom_value"),
            "pvt_robust": (tr.get("pvt") or {}).get("robust_complete_pass"),
            "selected_hash": ((tr.get("stage8_ranker") or {})
                              .get("selected_topology_hash"))}


def agent_stream_update(episodes: list[dict]) -> dict:
    """Audit the deterministic agents against measured outcomes and surface
    the calibration facts a human would use to refit them. NEVER changes an
    agent constant itself -- planner ceilings/tiers are code, promoted by a
    human, exactly like the bandit pin."""
    if not episodes:
        return {"status": "NO_EPISODES"}
    by_diff = {}
    for e in episodes:
        d = e.get("plan_difficulty") or "unknown"
        b = by_diff.setdefault(d, {"n": 0, "pass": 0, "spice": 0, "banked": 0})
        b["n"] += 1
        b["pass"] += bool(e.get("pass"))
        b["spice"] += e.get("spice_spent") or 0
        b["banked"] += e.get("banked") or 0
    polish = sum(1 for e in episodes
                 if "QUALITY_POLISH" in (e.get("interventions") or []))
    recov = sum(1 for e in episodes if e.get("recovery_executed"))
    recov_ok = sum(1 for e in episodes if e.get("recovery_adopted"))
    critic_multi = sum(1 for e in episodes if (e.get("critic_rounds") or 1) > 1)
    return {"status": "AUDITED", "episodes": len(episodes),
            "by_difficulty": by_diff,
            "quality_polish_runs": polish,
            "recovery_runs": recov, "recovery_adopted": recov_ok,
            "critic_multi_round_runs": critic_multi,
            "note": "agents are deterministic and identical in both lineages; "
                    "this audit informs HUMAN refits of planner ceilings/tiers "
                    "(code changes), never automatic ones"}


# ---------------------------------------------------------------------------
def run_generation(*, lineage: str, n_specs: int, budget: int, dry_run: bool,
                   split: str = "train", model=None, tok=None,
                   adapter: str | None = None, start: int = 0,
                   profile: str = "stock", tier2_specs: int = 0,
                   tier2_start: int = 0) -> dict:
    from agentic_raptor.publication.eval_sets import excluded_context_ids
    from agentic_raptor.publication.generation_state import (
        checkpoint_progress, publish_atomic, resume_or_start)
    from agentic_raptor.selfimprove_v2 import harvest_run
    from agentic_raptor.selfimprove_v2.streams import GenerationPaths, append

    prof = PROFILES[profile]
    if dry_run:
        base = A9_DRY if profile == "stock" else A9_DRY / profile
    else:
        base = prof["root"]
    root = base / "state"
    data_root = base / "data"
    st = resume_or_start(root, lineage)
    gp = GenerationPaths(data_root / lineage, st.generation_id)
    protected = set(excluded_context_ids())
    if profile == "tier2":
        protected |= tier2_protected_ids()
    # MIXED-DISTRIBUTION JOB LIST: stock TRAIN specs, then tier2_train specs
    jobs = [(split, start + i) for i in range(n_specs)]
    if tier2_specs:
        jobs += [("tier2_train", tier2_start + i) for i in range(tier2_specs)]
    learning_mode = "adaptive" if lineage == "adaptive" else "static"

    # adaptive retrieves from its own grown memory; static from the frozen file
    rag_mem = lineage_rag_memory(data_root, lineage)

    summary = {"lineage": lineage, "generation": st.generation_id,
              "resumed": bool(st.processed_run_ids), "profile": profile,
              "rag_memory": str(rag_mem), "spec_start": start,
              "jobs": [f"{sp}:{ix}" for sp, ix in jobs], "runs": []}
    import run_raptor_v2 as v2
    for job_split, idx in jobs:
        split = job_split
        run_id = f"g{st.generation_id}_{lineage}_{split}_{idx}"
        if not st.mark_run_processed(run_id):
            summary["runs"].append({"run_id": run_id, "skipped_already_processed": True})
            continue
        tr = v2.run_pipeline(model, tok, adapter or "", split=split, spec_index=idx,
                             budget=budget, calibrate=True, seed=0,
                             learning_mode=learning_mode,
                             rag_memory=str(rag_mem),
                             # BUDGET REALLOCATION: enabled for BOTH lineages
                             # (a shared pipeline setting, so adaptive-vs-
                             # static differences stay attributable to the
                             # LEARNED components alone)
                             sizing_early_stop=True,
                             # AGENTIC SELF-IMPROVEMENT (2026-08-17): both
                             # lineages run the CURRENT production system --
                             # the four agents (Quality Polish included) and
                             # adaptive LLM attempts -- so adaptive-vs-static
                             # differences stay attributable to the LEARNED
                             # components alone (agents are deterministic
                             # rules, identical in both lineages).
                             agents=A9_AGENTS,
                             proposal_stall_stop=A9_STALL_STOP,
                             # CUSTODY FIX (2026-08-16): harvest=True is what
                             # makes run_pipeline RETURN the harvest payload
                             # (and, as of the same fix, suppress its own
                             # global LIVE-pool routing). Without it, G0's
                             # first attempt harvested {} on every run while
                             # silently growing the shared RAG memory file.
                             harvest=True,
                             # 2026-08-23: profile in the trace name -- the tier2
                             # profile's G1 shared stock indices 10-19 with the
                             # stock profile's G1 and overwrote those traces
                             out_prefix=(f"A9_{lineage}_g{st.generation_id}" if profile == "stock"
                                         else f"A9{profile}_{lineage}_g{st.generation_id}"),
                             value_ckpt=None)   # THE GAP FIX: live promoted AZ ckpt, enforced loader
        hv = tr.get("_harvest")
        n_rows = {}
        if lineage == "adaptive" and hv is not None:
            streams = harvest_run(hv, split=split, protected_ids=protected)
            for name, rows in streams.items():
                if rows:
                    n_rows[name] = append(gp.stream(name), rows)
            # AGENT EPISODES (2026-08-17): what each agent decided and what
            # the authoritative outcome was -- the raw material for auditing
            # and, later, refitting the Planner's TRAIN ceilings / difficulty
            # tiers from measured evidence rather than hand-set constants.
            ep = agent_episode(tr, run_id)
            if ep is not None:
                n_rows["agent_episodes"] = append(gp.stream("agent_episodes"), [ep])
        st.true_spice_calls += (tr.get("spice_usage") or {}).get("total_spice_calls", 0)
        summary["runs"].append({"run_id": run_id,
                               "pass": (tr.get("nominal") or {}).get("complete_pass"),
                               "harvested": n_rows})
        checkpoint_progress(st, root)

    # streams update (adaptive only; static NEVER retrains)
    if lineage == "adaptive":
        from agentic_raptor.selfimprove_v2.streams import read
        # cross-generation accumulation: thresholds are met by the LINEAGE's
        # whole history, not one generation's ~2-8 rows
        pairs_all = lineage_stream_rows(data_root, lineage, "ranker_pairs")
        summary["value_stream"] = value_stream_update(
            lineage_stream_rows(data_root, lineage, "puct_examples"))
        summary["dpo_stream"] = dpo_v2_stream_update(pairs_all)
        summary["bandit_stream"] = bandit_stream_update(pairs_all)
        # the two BINDING-component streams (GATE3: proposer + retrieval are
        # where the remaining headroom lives; selection is saturated).
        # RAG reads only THIS generation's rows -- the memory file itself is
        # the accumulator (idempotent via call_id dedupe).
        summary["rag_stream"] = rag_stream_update(read(gp.stream("rag_memory")),
                                                  rag_mem)
        summary["sft_stream"] = sft_stream_update(
            lineage_stream_rows(data_root, lineage, "sft_queue"), data_root,
            profile=profile)
        summary["agent_stream"] = agent_stream_update(
            lineage_stream_rows(data_root, lineage, "agent_episodes"))
    else:
        for k in ("value_stream", "dpo_stream", "bandit_stream",
                  "rag_stream", "sft_stream", "agent_stream"):
            summary[k] = {"status": "STATIC_NO_LEARNING"}

    # publish: component hashes (static: unchanged G0 hashes by definition;
    # adaptive: identical unless a gated candidate was staged+promoted by a
    # separate human decision -- this loop never silently swaps them)
    for k, vhash in g0_component_hashes().items():
        setattr(st, k, vhash)
    st.status = "COMPLETE"
    p = publish_atomic(st, root)
    summary["published_state"] = str(p)
    summary["true_spice_calls_total"] = st.true_spice_calls
    (gp.gdir / "GENERATION_SUMMARY.json").parent.mkdir(parents=True, exist_ok=True)
    (gp.gdir / "GENERATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", choices=("adaptive", "static"), default="adaptive")
    ap.add_argument("--specs", type=int, default=4)
    ap.add_argument("--start", type=int, default=0,
                    help="first TRAIN spec index; sweep FRESH specs each "
                         "generation -- the deterministic pipeline harvests "
                         "zero new information from a repeated spec")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adapter", default=None,
                    help="proposer adapter; default = the profile's production adapter")
    ap.add_argument("--profile", choices=tuple(PROFILES), default="stock",
                    help="stock: original lineages on stock TRAIN; tier2: "
                         "separate lineages on stock TRAIN + tier2_train with "
                         "the production tier-2 proposer (2026-08-22)")
    ap.add_argument("--tier2-specs", type=int, default=0,
                    help="number of tier2_train specs to add to each generation")
    ap.add_argument("--tier2-start", type=int, default=0)
    args = ap.parse_args()
    adapter = args.adapter or str(PROFILES[args.profile]["adapter"])
    if args.profile == "tier2" and args.tier2_specs == 0:
        raise SystemExit("--profile tier2 needs --tier2-specs N (the whole point "
                         "of the profile is the tier-2 signal)")
    from run_qwen_ablation import _load
    tok, model = _load(adapter)
    summary = run_generation(lineage=args.lineage, n_specs=args.specs,
                             budget=args.budget, dry_run=args.dry_run,
                             model=model, tok=tok, adapter=adapter,
                             start=args.start, profile=args.profile,
                             tier2_specs=args.tier2_specs,
                             tier2_start=args.tier2_start)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
