"""Stage 3E.2 campaign driver: versioned target sets, executable-edit
realisation, LLM-proposal path, mixed-action AlphaZero MCTS (3 seeds), shared
Phase-D graph-conditioned SAC (3 seeds), dynamics/surrogate calibration,
ranker comparison, repair curriculum sample, held-out evaluation, MCTS
baselines, PVT verification, exact cost accounting.

All electrical results come from real ngspice; bounded ENGINEERING scale
(documented per part) — not publication scale.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Any

from agentic_raptor.corpus import TopologyRegistry
from agentic_raptor.electrical import discover_ngspice
from agentic_raptor.mb_sac import load_pools
from agentic_raptor.mb_sac.stage3d1 import PostSizingTopologyScore, _sim_family
from agentic_raptor.mb_sac.stage3d2 import V3, build_mp_conditioner
from agentic_raptor.mapping import map_family
from agentic_raptor.topology_rl.stage3e2_edits import (
    EDIT_TEMPLATES, EditRejected, FixtureProposalProvider, apply_edit,
    build_manifest, device_graph_hash, qualify_device_graph, realise_proposal)
from agentic_raptor.utils.seeding import apply_torch_omp_workaround

_ROOT = Path(__file__).resolve().parents[2]
OUT = _ROOT / "artifacts" / "stage3e2"
SCHEMA = "3e2.1"

COST_KEYS = ("validator_calls", "mapping_attempts", "rejected_edits", "policy_calls",
             "value_calls", "mcts_simulations", "mbsac_real_transitions",
             "mbsac_model_transitions", "surrogate_calls", "ranker_calls",
             "real_spice_calls", "pvt_spice_calls", "cache_hits", "cache_misses",
             "simulator_failures", "failed_calls", "wall_clock_s")


def new_costs() -> dict[str, float]:
    return {k: 0 for k in COST_KEYS}


def map_root(reg: TopologyRegistry, tid: str, costs: dict):
    e = reg.get_topology(tid)
    blocks = {n.block_role for n in e.graph.nodes.values() if n.block_role}
    row = {"topology_id": tid,
           "gain_stages": max(1, sum(1 for n in e.graph.nodes.values()
                                     if n.block_role == "gain_stage")),
           "functional_blocks": sorted(blocks) + ["C"], "unresolved_blocks": [],
           "mapping_readiness": "mapping_ready",
           "graph_hash": e.metadata.get("graph_hash")}
    costs["mapping_attempts"] += 1
    g, _ = map_family(e, row)
    return g


# ====================== Part C: versioned target sets ========================
TARGET_TIERS = {"easy": (0.6, 45.0), "boundary": (0.95, 45.0), "hard": (1.15, 60.0),
                "validation": (0.85, 45.0), "heldout": (1.05, 50.0)}


def build_target_sets(seed: int = 42) -> dict[str, Any]:
    reg = TopologyRegistry(V3)
    pools = load_pools()
    d1 = json.loads((_ROOT / "artifacts/stage3d1/SUMMARY.json").read_text())
    anchors: dict[str, dict] = {}
    for line in (_ROOT / "datasets/simulation_memory/stage3c2b_runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("metrics", {}).get("dc_gain_db") is not None:
            anchors[r["topology_id"]] = {"gain": r["metrics"]["dc_gain_db"], "src": "stage3c2b"}
    for line in (_ROOT / "datasets/simulation_memory/topology_summaries.jsonl").read_text().splitlines():
        r = json.loads(line)
        g = (r.get("verified_metrics") or {}).get("dc_gain_db") or r.get("dc_gain_db")
        if g:
            anchors.setdefault(r["topology_id"], {"gain": g, "src": "topology_summaries"})
    records = []
    for r in pools["A1"] + pools["A2"]:
        tid = r["topology_id"]
        anchor = anchors.get(tid, {"gain": 40.0, "src": "default_anchor"})
        pm_base = d1["env_validation"].get(tid, {}).get("phase_margin_deg")
        h = reg.get_topology(tid).graph.structural_hash()
        for tier, (gf, pm_t) in TARGET_TIERS.items():
            records.append({
                "topology_id": tid, "graph_hash": h,
                "target_id": f"t_{tier}_{tid}", "source": anchor["src"],
                "gain_target_db": round(anchor["gain"] * gf, 2),
                "ugbw_target_hz": 1e4, "phase_margin_target_deg": pm_t,
                "power_limit_w": 5e-3, "area_limit": None, "slew_rate_target": None,
                "output_swing_target": None, "topology_specific": {},
                "difficulty": tier,
                "split": "test" if tier == "heldout" else
                         "validation" if tier == "validation" else "train",
                "feasibility_provenance": f"anchored to measured baseline "
                                          f"(pm={pm_base}, gain_src={anchor['src']})",
                "generation_seed": seed, "schema_version": SCHEMA})
    OUT.mkdir(parents=True, exist_ok=True)
    tdir = _ROOT / "datasets" / "target_sets_v1"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "targets.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8")
    (tdir / "META.json").write_text(json.dumps(
        {"seed": seed, "tiers": {k: list(v) for k, v in TARGET_TIERS.items()},
         "families": 16, "records": len(records), "schema_version": SCHEMA,
         "heldout_topology_note": "all 16 stable families are needed for training; "
         "held-out TOPOLOGY split is technically unsupported at n=16 stable — "
         "held-out targets + unseen edited/proposed topologies serve instead"},
        indent=1), encoding="utf-8")
    return {"records": len(records), "heldout": sum(1 for r in records
                                                    if r["split"] == "test")}


def load_targets(split: str | None = None) -> list[dict]:
    recs = [json.loads(x) for x in
            (_ROOT / "datasets/target_sets_v1/targets.jsonl").read_text().splitlines()]
    return [r for r in recs if split is None or r["split"] == split]


# ============== Parts D-K: executable-edit realisation demo ==================
EXECUTABLE_EDITS = ["ADD_VERIFIED_STAGE", "REPLACE_STAGE_WITH_COMPATIBLE_BLOCK",
                    "REPLACE_LOAD_WITH_COMPATIBLE_BLOCK",
                    "ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE",
                    "REPLACE_SUPPORTED_COMPENSATION_STRUCTURE",
                    "ADD_SUPPORTED_OUTPUT_STAGE", "CONNECT_VERIFIED_FEEDBACK_PATH"]


def run_edit_demo(root_tid: str = "topology_v2_0001") -> dict[str, Any]:
    reg = TopologyRegistry(V3)
    exe = discover_ngspice()
    costs = new_costs()
    g0 = map_root(reg, root_tid, costs)
    lineage_note = None
    if g0.stage_count < 2:      # CS-stage edits need a second stage: honest lineage
        g0, pre_audit = apply_edit(g0, "ADD_VERIFIED_STAGE")
        lineage_note = {"base_derivation": "root+ADD_VERIFIED_STAGE", "audit": pre_audit}
    base_hash = device_graph_hash(g0)
    results, rejected = {}, {}
    for et in EXECUTABLE_EDITS:
        try:
            ng, audit = apply_edit(g0, et)
        except EditRejected as exc:
            rejected[et] = str(exc)
            costs["rejected_edits"] += 1
            continue
        costs["validator_calls"] += 1
        q = qualify_device_graph(root_tid, ng, OUT / "edits", exe, et.lower()[:24], costs)
        results[et] = {"audit": audit, "qualification": q}
    # round-trip reversibility: add stage then remove -> parent hash recovered
    g_add, _ = apply_edit(g0, "ADD_VERIFIED_STAGE")
    g_rt, _ = apply_edit(g_add, "REMOVE_OPTIONAL_SUPPORTED_STAGE")
    roundtrip = device_graph_hash(g_rt) == base_hash
    parent_immutable = device_graph_hash(g0) == base_hash
    out = {"root": root_tid, "base_hash": base_hash, "base_lineage": lineage_note,
           "executable": results,
           "rejected": rejected, "roundtrip_hash_recovered": roundtrip,
           "parent_immutable": parent_immutable, "costs": costs}
    (OUT / "edit_demo.json").write_text(json.dumps(out, indent=1, default=str),
                                        encoding="utf-8")
    return out


def run_llm_proposal_demo() -> dict[str, Any]:
    exe = discover_ngspice()
    costs = new_costs()
    provider = FixtureProposalProvider()
    good = realise_proposal(provider.propose({}), OUT / "llm", exe, costs)
    from agentic_raptor.topology_rl.stage3e2_edits import TopologyProposal, validate_proposal
    bad_cases = {
        "malformed_no_stages": TopologyProposal("bad1", [], {"gnda": "g"}, [], {}, [], [], [], "x"),
        "unsupported_block": TopologyProposal(
            "bad2", [{"block": "quantum_stage", "role": "gain_stage", "outputs": ["o"]}],
            {"gnda": "g", "vdda": "v", "vinp": "p", "vinn": "n", "vout": "o"},
            [], {}, ["bias_mirror"], [], [], "x"),
        "missing_bias": TopologyProposal(
            "bad3", [{"block": "cs_gain_stage", "role": "gain_stage", "outputs": ["vout"]}],
            {"gnda": "g", "vdda": "v", "vinp": "p", "vinn": "n", "vout": "o"},
            [], {}, [], [], [], "x"),
        "illegal_feedback": TopologyProposal(
            "bad4", [{"block": "cs_gain_stage", "role": "gain_stage", "outputs": ["vout"]}],
            {"gnda": "g", "vdda": "v", "vinp": "p", "vinn": "n", "vout": "o"},
            [], {}, ["bias_mirror"], [], [{"from": "vout", "to": "vinp", "sign": "positive"}], "x"),
    }
    rejections = {k: validate_proposal(p)[1] for k, p in bad_cases.items()}
    out = {"valid_proposal": good, "rejections": rejections, "costs": costs}
    (OUT / "llm_demo.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


# ============ Parts M/N: mixed-action AlphaZero campaign (3 seeds) ===========
def run_alphazero_campaign(seeds=(0, 1, 2), episodes_per_seed: int = 1,
                           spice_per_episode: int = 4) -> dict[str, Any]:
    apply_torch_omp_workaround()
    import torch
    from agentic_raptor.topology_rl import stage3e1 as s1
    from agentic_raptor.topology_rl.trainer import parameter_checksum

    reg = TopologyRegistry(V3)
    pools = load_pools()
    pool_ids = [r["topology_id"] for r in pools["A1"] + pools["A2"]]
    root_tid = "topology_v2_0001"
    exe = discover_ngspice()
    train_targets = [t for t in load_targets("train") if t["topology_id"] == root_tid]
    per_seed = {}
    for seed in seeds:
        torch.manual_seed(seed)
        nets = s1.build_policy_value(seed)
        # extended action embedding: numeric edit features appended (Part M)
        edit_feats = {et: torch.tensor([EDIT_TEMPLATES[et]["edit_cost"] / 2.0,
                                        len(EDIT_TEMPLATES[et].get("template_params", {})) / 2.0,
                                        1.0])
                      for et in EXECUTABLE_EDITS}
        costs = new_costs()
        cfg = s1.SearchConfig(num_simulations=6, max_depth=1, max_children=2,
                              max_real_spice_calls=spice_per_episode,
                              leaf_mode="hybrid_visit_threshold",
                              spice_visit_threshold=1, leaf_spice_budget=2, seed=seed)
        ep_stats = []
        for ep in range(episodes_per_seed):
            state = s1.make_root_state(root_tid, reg, dict(
                train_targets[ep % len(train_targets)]) if train_targets else s1.DEFAULT_SPEC)
            sel_actions, rej = s1.generate_actions(state, reg, pool_ids, cfg.max_children)
            costs["validator_calls"] += 1
            g0 = map_root(reg, root_tid, costs)
            edit_actions, edited = [], {}
            for et in ("ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE", "ADD_VERIFIED_STAGE"):
                try:
                    ng, audit = apply_edit(g0, et)
                    aid = f"a_edit_{et.lower()[:16]}"
                    edit_actions.append(s1.Stage3E1Action(
                        aid, s1.Stage3E1ActionType(et), source_ref=root_tid,
                        target_location=audit["child_hash"], provenance="edit_template"))
                    edited[aid] = (ng, audit)
                except EditRejected as exc:
                    rej.append({"action_id": et, "reason": str(exc)})
                    costs["rejected_edits"] += 1
            actions = sorted(sel_actions + edit_actions, key=lambda a: a.action_id)
            with torch.no_grad():
                # priors: base policy + edit-feature bonus head (documented v2 embedding)
                acts_o, logits, probs = nets["policy_forward"](state, actions, reg)
                costs["policy_calls"] += 1
                pri = probs.clone()
                for i, a in enumerate(acts_o):
                    if a.action_id in edited:
                        pri[i] = pri[i] * (1.0 + 0.1 * float(edit_feats[a.action_type.value][0]))
                pri = pri / pri.sum()
            # one-ply MCTS over mixed actions with SPICE leaf evals
            N = {a.action_id: 0 for a in acts_o}
            W = dict.fromkeys(N, 0.0)
            cache: dict[str, float] = {}
            leaf_scores: dict[str, dict] = {}
            for sim in range(cfg.num_simulations):
                costs["mcts_simulations"] += 1
                tot = sum(N.values()) or 1
                scores = {a.action_id: (W[a.action_id] / max(N[a.action_id], 1))
                          + cfg.c_puct * float(pri[i]) * math.sqrt(tot) / (1 + N[a.action_id])
                          for i, a in enumerate(acts_o)}
                aid = max(scores, key=lambda k: (scores[k], k))
                a = next(x for x in acts_o if x.action_id == aid)
                key = a.target_location or a.source_ref or aid
                if key in cache:
                    costs["cache_hits"] += 1
                    v = cache[key]
                elif (a.action_type == s1.Stage3E1ActionType.TERMINATE_SEARCH
                      or costs["real_spice_calls"] + 1 > spice_per_episode):
                    with torch.no_grad():
                        v = float(nets["value_forward"](state, reg)["scalar"])
                    costs["value_calls"] += 1
                elif aid in edited:
                    q = qualify_device_graph(root_tid, edited[aid][0], OUT / f"az_s{seed}",
                                             exe, f"{aid[7:23]}_e{ep}", costs)
                    pm = (q.get("metrics") or {}).get("phase_margin_deg")
                    sc = PostSizingTopologyScore(
                        root_tid, "az", q["electrical"] if q.get("metrics") else "simulator_failure",
                        q.get("stability") == "verified_stable",
                        bool(q.get("stability") == "verified_stable"
                             and (q["metrics"].get("dc_gain_db") or 0) > 40),
                        q.get("metrics", {}), {"pm": ((pm or -90) - 45) / 45},
                        q["spice_calls"], None, 0)
                    v = sc.compute_scalar()
                    leaf_scores[aid] = asdict(sc)
                    cache[key] = v
                    costs["cache_misses"] += 1
                else:
                    from agentic_raptor.mb_sac.stage3d2 import evaluate_topology_for_mcts
                    tgt = a.source_ref if a.action_type.value == "SELECT_EXISTING_TOPOLOGY" else root_tid
                    r = evaluate_topology_for_mcts(tgt, real_spice_budget=2, seed=seed)
                    costs["real_spice_calls"] += r["budget"]["real_spice_calls"]
                    v = r["scalar_leaf_value"]
                    leaf_scores[aid] = r["score"]
                    cache[key] = v
                    costs["cache_misses"] += 1
                N[aid] += 1
                W[aid] += v
            tot = sum(N.values())
            visit_dist = {k: n / tot for k, n in N.items()}
            best = max(leaf_scores.values(), key=lambda x: x.get("scalar_value", -9),
                       default=None)
            vt = best["scalar_value"] if best else None
            # training update from MCTS-derived targets (policy CE + value MSE)
            report = None
            if vt is not None:
                acts_o2, logits2, _ = nets["policy_forward"](state, actions, reg)
                target = torch.tensor([visit_dist[a.action_id] for a in acts_o2])
                pl = -(target / target.sum() * torch.log_softmax(logits2, 0)).sum()
                vl = (nets["value_forward"](state, reg)["scalar"] - vt) ** 2
                opt = torch.optim.Adam(nets["params"], lr=1e-3, weight_decay=1e-4)
                ck0 = parameter_checksum(nets["heads"])
                enc0 = parameter_checksum(nets["encoder"])
                opt.zero_grad()
                (pl + vl).backward()
                gn = float(torch.nn.utils.clip_grad_norm_(nets["params"], 5.0))
                opt.step()
                ent = float(-(torch.softmax(logits2, 0)
                              * torch.log_softmax(logits2, 0)).sum().detach())
                report = {"policy_loss": round(float(pl.detach()), 4),
                          "value_loss": round(float(vl.detach()), 4),
                          "grad_norm": round(gn, 3), "policy_entropy": round(ent, 3),
                          "heads_changed": parameter_checksum(nets["heads"]) != ck0,
                          "encoder_changed": parameter_checksum(nets["encoder"]) != enc0}
            ep_stats.append({"episode": ep, "visit_distribution": visit_dist,
                             "action_categories": {a.action_id: a.action_type.value
                                                   for a in acts_o},
                             "pv": [max(N, key=lambda k: N[k])],
                             "value_target": vt, "train": report,
                             "validator_rejections": len(rej)})
        per_seed[seed] = {"episodes": ep_stats, "costs": costs}
    out = {"root": root_tid, "seeds": list(seeds), "per_seed": per_seed,
           "schema_version": SCHEMA}
    (OUT / "alphazero_campaign.json").write_text(json.dumps(out, indent=1, default=str),
                                                 encoding="utf-8")
    return out


# ============== Part O: shared Phase-D graph-conditioned SAC =================
def run_phase_d(seeds=(0, 1, 2), steps_per_family: int = 2) -> dict[str, Any]:
    apply_torch_omp_workaround()
    import torch
    from agentic_raptor.topology_rl.trainer import parameter_checksum

    reg = TopologyRegistry(V3)
    pools = load_pools()
    fams = [(r["topology_id"], "A1") for r in pools["A1"]] + \
           [(r["topology_id"], "A2") for r in pools["A2"]]
    exe = discover_ngspice()
    train_targets = {t["topology_id"]: t for t in load_targets("train")
                     if t["difficulty"] == "boundary"}
    transitions: list[dict] = []
    per_seed = {}
    for seed in seeds:
        torch.manual_seed(seed)
        enc, embed = build_mp_conditioner()
        actor = torch.nn.Sequential(torch.nn.Linear(20, 32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 2))
        q1 = torch.nn.Sequential(torch.nn.Linear(21, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        q2 = torch.nn.Sequential(torch.nn.Linear(21, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        log_alpha = torch.zeros(1, requires_grad=True)
        opt_a = torch.optim.Adam(list(actor.parameters()) + list(enc.parameters()), lr=3e-4)
        opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters())
                                 + list(enc.parameters()), lr=3e-4)
        opt_al = torch.optim.Adam([log_alpha], lr=3e-4)
        costs = new_costs()
        grads = {"actor_to_encoder": [], "critic_to_encoder": []}
        buf: list[tuple] = []
        stable_final = {"A1": 0, "A2": 0}
        enc0 = parameter_checksum(enc)
        for tid, src in fams:
            e = reg.get_topology(tid)
            prev = torch.zeros(4)
            for step in range(steps_per_family):
                emb = embed(e.graph)[1]
                obs = torch.cat([emb.detach(), prev])
                mu, logstd = actor(obs)
                std = torch.exp(logstd.clamp(-3, 1))
                a_raw = torch.tanh(mu + std * torch.randn(()))
                scale = float(2 ** (0.4 * a_raw.detach()))
                rec = _sim_family(tid, reg, exe, costs, OUT / f"pd_s{seed}", {"w": scale})
                costs["mbsac_real_transitions"] += 1
                m = rec.extracted_metrics if rec else {}
                pm, gain = m.get("phase_margin_deg"), m.get("dc_gain_db")
                stable = bool(pm and pm > 0)
                tgt = train_targets.get(tid, {"gain_target_db": 40.0})
                r = (1.0 if stable else -1.0) + 0.2 * math.tanh(
                    ((gain or 0) - tgt["gain_target_db"]) / 10)
                buf.append((e.graph, prev.clone(), float(a_raw.detach()), r))
                transitions.append({"seed": seed, "tid": tid, "source": src,
                                    "action": float(a_raw.detach()), "scale": scale,
                                    "pm": pm, "gain": gain, "reward": r,
                                    "emb": [round(float(x), 4) for x in emb.detach()],
                                    "run_ref": f"pd_s{seed}/{tid}"})
                if step == steps_per_family - 1 and stable:
                    stable_final[src] += 1
                prev = torch.tensor([(pm or -90) / 90, (gain or 0) / 100,
                                     step / steps_per_family, 0.5])
                # SAC updates on a minibatch (encoder INSIDE both objectives)
                batch = buf[-4:]
                al = log_alpha.exp().detach()
                for gph, pv, act, rew in batch:
                    em = embed(gph)[1]
                    ob = torch.cat([em, pv])
                    qin = torch.cat([ob, torch.tensor([act])])
                    tq = torch.tensor(rew)
                    lc = (q1(qin)[0] - tq) ** 2 + (q2(qin)[0] - tq) ** 2
                    opt_c.zero_grad()
                    lc.backward()
                    grads["critic_to_encoder"].append(
                        float(sum(p.grad.abs().sum() for p in enc.parameters()
                                  if p.grad is not None)))
                    opt_c.step()
                    em2 = embed(gph)[1]
                    ob2 = torch.cat([em2, pv])
                    mu2, ls2 = actor(ob2)
                    st2 = torch.exp(ls2.clamp(-3, 1))
                    z = mu2 + st2 * torch.randn(())
                    a2 = torch.tanh(z)
                    logp = (-0.5 * ((z - mu2) / st2) ** 2 - ls2
                            - torch.log(1 - a2 ** 2 + 1e-6))
                    qa = torch.min(q1(torch.cat([ob2, a2.reshape(1)]))[0],
                                   q2(torch.cat([ob2, a2.reshape(1)]))[0])
                    la = (al * logp - qa)
                    opt_a.zero_grad()
                    la.backward()
                    grads["actor_to_encoder"].append(
                        float(sum(p.grad.abs().sum() for p in enc.parameters()
                                  if p.grad is not None)))
                    opt_a.step()
                    lal = -(log_alpha * (logp.detach() + 1.0))
                    opt_al.zero_grad()
                    lal.backward()
                    opt_al.step()
        torch.save({"actor": actor.state_dict(), "encoder": enc.state_dict(),
                    "q1": q1.state_dict(), "q2": q2.state_dict()},
                   OUT / f"phase_d_seed{seed}.pt")
        per_seed[seed] = {
            "stable_final": stable_final, "families": len(fams),
            "real_spice_calls": costs["real_spice_calls"],
            "encoder_changed": parameter_checksum(enc) != enc0,
            "actor_to_encoder_grad_nonzero": all(g > 0 for g in grads["actor_to_encoder"]),
            "critic_to_encoder_grad_nonzero": all(g > 0 for g in grads["critic_to_encoder"]),
            "mean_actor_enc_grad": round(sum(grads["actor_to_encoder"])
                                         / len(grads["actor_to_encoder"]), 4),
            "mean_critic_enc_grad": round(sum(grads["critic_to_encoder"])
                                          / len(grads["critic_to_encoder"]), 4),
            "alpha_final": round(float(log_alpha.exp().detach()), 4)}
    rates = [sum(per_seed[s]["stable_final"].values()) / 16 for s in seeds]
    out = {"per_seed": per_seed,
           "stable_rate_mean": round(sum(rates) / len(rates), 3),
           "stable_rate_std": round((sum((x - sum(rates) / len(rates)) ** 2
                                         for x in rates) / len(rates)) ** 0.5, 3),
           "total_real_spice": sum(per_seed[s]["real_spice_calls"] for s in seeds),
           "scale_note": f"bounded engineering scale: {steps_per_family} real "
                         f"transitions x 16 families x {len(seeds)} seeds"}
    (OUT / "phase_d.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (OUT / "phase_d_transitions.jsonl").write_text(
        "\n".join(json.dumps(t) for t in transitions), encoding="utf-8")
    return out


# ====== Parts P/Q: dynamics + surrogate calibration from real transitions ====
def run_calibration() -> dict[str, Any]:
    apply_torch_omp_workaround()
    import torch
    trans = [json.loads(x) for x in
             (OUT / "phase_d_transitions.jsonl").read_text().splitlines()]
    train = [t for t in trans if t["seed"] in (0, 1) and t["pm"] is not None]
    test = [t for t in trans if t["seed"] == 2 and t["pm"] is not None]

    def fit_eval(outkey):
        torch.manual_seed(0)
        members = []
        for m in range(2):
            net = torch.nn.Sequential(torch.nn.Linear(17, 24), torch.nn.ReLU(),
                                      torch.nn.Linear(24, 1))
            opt = torch.optim.Adam(net.parameters(), lr=1e-2)
            for _ in range(200):
                i = torch.randint(0, len(train), (8,))
                x = torch.stack([torch.tensor(train[j]["emb"] + [train[j]["action"]])
                                 for j in i])
                y = torch.tensor([[train[j][outkey] / 90.0] for j in i])
                loss = ((net(x) - y) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            members.append(net)
        errs, dis = {}, []
        for t in test:
            x = torch.tensor(t["emb"] + [t["action"]])
            with torch.no_grad():
                ps = [float(m(x)) * 90 for m in members]
            dis.append(abs(ps[0] - ps[1]))
            errs.setdefault(t["source"], []).append(abs(sum(ps) / 2 - t[outkey]))
        return {src: round(sum(v) / len(v), 2) for src, v in errs.items()}, \
               round(sum(dis) / len(dis), 2)

    pm_err, disagreement = fit_eval("pm")
    glob = round(sum(pm_err.values()) / len(pm_err), 2)
    gates = {"global": glob < 15.0,
             **{f"source_{s}": e < 15.0 for s, e in pm_err.items()},
             "topology_level": "insufficient_data_gate_disabled"}
    out = {"pm_abs_error_deg_by_source": pm_err, "global_pm_error": glob,
           "ensemble_disagreement_deg": disagreement, "rollout_gates": gates,
           "gate_rule": "enable model rollouts only if held-out one-step |pm error| < 15 deg; "
                        "disabled contexts keep real-only learning (not simulator failures)",
           "surrogate_note": "same held-out split (seed 2); predictions never stored "
                             "as verified measurements"}
    (OUT / "calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ================= Part S: ranker comparison (equal budget) ==================
def run_ranker_comparison(budget_per_family: int = 2) -> dict[str, Any]:
    from agentic_raptor.dpo import (FEATURE_DIM, DPOConfig, DPORanker,
                                    build_candidate_features)
    from agentic_raptor.core.candidate import CircuitCandidate
    from agentic_raptor.core.types import GenerationSource
    from agentic_raptor.mb_sac.stage3d2 import _spec
    reg = TopologyRegistry(V3)
    exe = discover_ngspice()
    fams = ["topology_0002", "topology_v2_0001"]
    scales = [0.7, 0.9, 1.1, 1.4]
    modes = {}
    for mode in ("bt_ranker", "random", "scalar_heuristic"):
        costs = new_costs()
        stable_sel, first_pass = 0, []
        for tid in fams:
            rng = Random(7)
            if mode == "bt_ranker":
                ranker = DPORanker(FEATURE_DIM, DPOConfig(enabled=True, seed=0))
                feats = []
                for sc in scales:
                    c = CircuitCandidate.create(reg.get_topology(tid).graph, _spec(),
                                                GenerationSource.EDITED)
                    f = build_candidate_features(c, sizing_vector=[sc],
                                                 predicted_margins={"w": sc - 1},
                                                 predicted_feasibility=0.5)
                    f.metadata["scale"] = sc
                    feats.append(f)
                ranked = ranker.rank(feats)
                costs["ranker_calls"] += 1
                chosen = [ranked[0][0].metadata["scale"], ranked[-1][0].metadata["scale"]]
            elif mode == "random":
                chosen = rng.sample(scales, 2)
            else:
                chosen = sorted(scales, key=lambda s: abs(s - 1))[:2]
            found = False
            for i, sc in enumerate(chosen[:budget_per_family]):
                rec = _sim_family(tid, reg, exe, costs, OUT / f"rk_{mode}", {"w": sc})
                pm = rec.extracted_metrics.get("phase_margin_deg") if rec else None
                if pm and pm > 0:
                    stable_sel += 1
                    if not found:
                        first_pass.append(i + 1)
                        found = True
            if not found:
                first_pass.append(None)
        modes[mode] = {"stable_selections": stable_sel,
                       "total": len(fams) * budget_per_family,
                       "calls_to_first_stable": first_pass,
                       "real_spice_calls": costs["real_spice_calls"],
                       "exploration_retained": mode == "bt_ranker"}
    out = {"modes": modes, "equal_budget": budget_per_family * len(fams),
           "note": "bounded comparison (2 families); NDCG/pairwise-accuracy need "
                   "larger candidate sets — deferred with exact blocker recorded"}
    (OUT / "ranker_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ==================== Part V: repair-curriculum sample =======================
def run_repair_sample(n: int = 5) -> dict[str, Any]:
    reg = TopologyRegistry(V3)
    exe = discover_ngspice()
    costs = new_costs()
    audit = [json.loads(x) for x in
             (_ROOT / "datasets/simulation_memory/stage3d_repair_audit.jsonl")
             .read_text().splitlines()]
    comp = [r for r in audit if "comp" in json.dumps(r).lower()
            and str(r.get("topology_id", "")).startswith("topology_v2")][:n]
    results = []
    for r in comp:
        tid = r["topology_id"]
        try:
            g = map_root(reg, tid, costs)
        except Exception as exc:
            results.append({"topology_id": tid, "status": f"map_failed:{exc}"})
            continue
        caps = [d for d in g.devices if d.kind == "cap"]
        if not caps:
            results.append({"topology_id": tid, "status": "no_compensation_cap_present"})
            continue
        for d in caps:
            d.sizing["value"] = d.sizing["value"] * 4.0   # compensation-value repair
        q = qualify_device_graph(tid, g, OUT / "repair", exe, f"rep_{tid[-8:]}", costs)
        results.append({"topology_id": tid, "repair_class": "compensation_value",
                        "stability_after": q.get("stability"),
                        "pm_after": (q.get("metrics") or {}).get("phase_margin_deg"),
                        "electrical": q.get("electrical")})
    restored = sum(1 for r in results if r.get("stability_after") == "verified_stable")
    out = {"attempted": len(results), "stability_restored": restored,
           "results": results, "real_spice_calls": costs["real_spice_calls"],
           "note": "repair metrics SEPARATE from normal sizing; full 91-family "
                   "curriculum deferred (compute) — same code path scales by n"}
    (OUT / "repair_sample.json").write_text(json.dumps(out, indent=1, default=str),
                                            encoding="utf-8")
    return out


# =================== Part Y: held-out target evaluation ======================
def run_heldout_eval() -> dict[str, Any]:
    apply_torch_omp_workaround()
    import torch
    reg = TopologyRegistry(V3)
    exe = discover_ngspice()
    costs = new_costs()
    heldout = {t["topology_id"]: t for t in load_targets("test")}
    ck = torch.load(OUT / "phase_d_seed0.pt", map_location="cpu", weights_only=False)
    enc, embed = build_mp_conditioner()
    enc.load_state_dict(ck["encoder"])
    actor = torch.nn.Sequential(torch.nn.Linear(20, 32), torch.nn.ReLU(),
                                torch.nn.Linear(32, 2))
    actor.load_state_dict(ck["actor"])
    rows = []
    for tid, tgt in heldout.items():
        e = reg.get_topology(tid)
        with torch.no_grad():
            emb = embed(e.graph)[1]
            mu, _ = actor(torch.cat([emb, torch.zeros(4)]))
            scale = float(2 ** (0.4 * torch.tanh(mu)))       # deterministic, no noise
        rec = _sim_family(tid, reg, exe, costs, OUT / "heldout", {"w": scale})
        m = rec.extracted_metrics if rec else {}
        pm, gain = m.get("phase_margin_deg"), m.get("dc_gain_db")
        rows.append({"topology_id": tid, "target_id": tgt["target_id"],
                     "scale": round(scale, 3), "pm": pm, "gain": gain,
                     "stable": bool(pm and pm > 0),
                     "pass": bool(pm and pm > 0 and (gain or 0) >= tgt["gain_target_db"])})
    out = {"families": len(rows),
           "stable_rate": round(sum(r["stable"] for r in rows) / len(rows), 3),
           "full_pass_rate": round(sum(r["pass"] for r in rows) / len(rows), 3),
           "real_spice_calls": costs["real_spice_calls"], "rows": rows,
           "rag_retrieval_disabled_during_heldout": True}
    (OUT / "heldout.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# =================== Part W: equal-budget MCTS baselines =====================
def run_mcts_baselines(seeds=(0, 1, 2)) -> dict[str, Any]:
    from agentic_raptor.topology_rl import stage3e1 as s1
    reg = TopologyRegistry(V3)
    pools = load_pools()
    pool_ids = [r["topology_id"] for r in pools["A1"] + pools["A2"]]
    modes = {"full_az": {}, "no_policy_prior": {"use_policy_prior": False},
             "random": {"use_policy_prior": False, "use_value_net": False}}
    table = {}
    for name, over in modes.items():
        rows = []
        for seed in seeds:
            cfg = s1.SearchConfig(num_simulations=6, max_depth=1, max_children=2,
                                  max_real_spice_calls=2, spice_visit_threshold=1,
                                  leaf_spice_budget=2, training_mode=False,
                                  seed=seed, **over)
            nets = s1.build_policy_value(seed)
            m = s1.TopologyMCTS(nets, reg, pool_ids, cfg)
            root = m.run(s1.make_root_state(pool_ids[0], reg, s1.DEFAULT_SPEC))
            best = max((r["scalar_leaf_value"] for r in m.leaf_cache.values()),
                       default=None)
            rows.append({"seed": seed, "best_scalar": best,
                         "spice": m.costs.real_spice_calls,
                         "nodes": len(m.nodes)})
        vals = [r["best_scalar"] for r in rows if r["best_scalar"] is not None]
        table[name] = {"rows": rows,
                       "best_scalar_mean": round(sum(vals) / len(vals), 3) if vals else None}
    (OUT / "mcts_baselines.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    return table


# ========================= Part AA: PVT verification =========================
PVT_MATRIX = {"corners": ["tt", "ss", "ff"], "temps_c": [27, 85],
              "supplies_v": [1.8, 1.62], "version": SCHEMA}


def run_pvt(tid: str = "topology_0002") -> dict[str, Any]:
    import agentic_raptor.electrical as elec
    reg = TopologyRegistry(V3)
    exe = discover_ngspice()
    corners_dir = elec._PDK_TT.parent
    orig_pdk, orig_tb = elec._PDK_TT, elec.build_testbench
    rows = []
    costs = new_costs()
    try:
        for corner in PVT_MATRIX["corners"]:
            for temp in PVT_MATRIX["temps_c"]:
                for vdd in PVT_MATRIX["supplies_v"]:
                    if (temp, vdd) == (85, 1.62) and corner != "tt":
                        continue   # documented reduced matrix (10 points)
                    elec._PDK_TT = corners_dir / f"{corner}.spice"

                    def tb(entry, audit, _v=vdd, _t=temp):
                        t = orig_tb(entry, audit)
                        return t.replace("supply_voltage = 1.8",
                                         f"supply_voltage = {_v}").replace(
                            "V1 vdd 0", f".temp {_t}\nV1 vdd 0")
                    elec.build_testbench = tb
                    rec = _sim_family(tid, reg, exe, costs, OUT / "pvt",
                                      None)
                    m = rec.extracted_metrics if rec else {}
                    rows.append({"corner": corner, "temp_c": temp, "vdd": vdd,
                                 "pm": m.get("phase_margin_deg"),
                                 "gain": m.get("dc_gain_db"),
                                 "ugbw": m.get("ugbw_hz"),
                                 "stable": bool((m.get("phase_margin_deg") or 0) > 0)})
    finally:
        elec._PDK_TT, elec.build_testbench = orig_pdk, orig_tb
    ok = [r for r in rows if r["pm"] is not None]
    worst = min(ok, key=lambda r: r["pm"], default=None)
    out = {"design": tid, "matrix": PVT_MATRIX, "points": len(rows),
           "pvt_pass_rate": round(sum(r["stable"] for r in rows) / len(rows), 3),
           "worst_corner": worst, "rows": rows,
           "pvt_spice_calls": costs["real_spice_calls"],
           "note": "PVT calls counted separately from nominal training"}
    (OUT / "pvt.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ============================ driver =========================================
def run_all() -> dict[str, Any]:
    t0 = time.time()
    summary = {"targets": build_target_sets()}
    summary["edits"] = {k: {"electrical": v["qualification"]["electrical"],
                            "stability": v["qualification"].get("stability")}
                        for k, v in run_edit_demo()["executable"].items()}
    summary["llm"] = run_llm_proposal_demo()["valid_proposal"]["status"]
    summary["alphazero"] = {s: d["episodes"][0]["train"]
                            for s, d in run_alphazero_campaign()["per_seed"].items()}
    summary["phase_d"] = {k: v for k, v in run_phase_d().items() if k != "per_seed"}
    summary["calibration"] = run_calibration()["rollout_gates"]
    summary["ranker"] = {k: v["stable_selections"]
                         for k, v in run_ranker_comparison()["modes"].items()}
    summary["repair"] = run_repair_sample()["stability_restored"]
    summary["heldout"] = run_heldout_eval()["stable_rate"]
    summary["baselines"] = {k: v["best_scalar_mean"]
                            for k, v in run_mcts_baselines().items()}
    summary["pvt"] = run_pvt()["pvt_pass_rate"]
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str),
                                      encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=1, default=str))
