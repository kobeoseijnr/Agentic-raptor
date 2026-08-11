"""Stage 5 training: the FIRST real AlphaZero topology-RL campaign.

Orchestrates: freeze non-AlphaZero components -> build disjoint AZ_TRAIN/
AZ_VALIDATION spec sets -> collect real (state, pi, z) episodes on
AZ_TRAIN -> audit replay for degeneracy -> train a candidate generation ->
validate it PAIRED against its parent on AZ_VALIDATION (never trained on)
-> promote or reject -> repeat for up to 3 generations total.

Every stage before "collect real episodes" is cheap (file hashing, spec
selection) and safe to run directly. Episode collection is genuinely
expensive (real LLM generation + real MCTS + real MB-SAC + real SPICE per
episode) -- this script is built and mechanically verified with --use-llm
false / a tiny --specs/--seeds scale, then handed off for the real,
full-scale run. See the module docstring's example commands at the bottom.

Run (small mechanical/dry verification, no GPU LLM needed):
  python run_alphazero_campaign.py --specs 2 --val-specs 2 --seeds 0 \\
      --use-llm false --simulations 32 --generations 1

Run (the real campaign):
  python run_alphazero_campaign.py --specs 16 --val-specs 6 --seeds 0,1 \\
      --use-llm true --simulations 128 --generations 3
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAMPAIGN_ROOT = ROOT / "artifacts/publication_v3/alphazero_campaign_01"


# ---------------------------------------------------------------------------
# Section 0/1: restore point + frozen non-AlphaZero components
# ---------------------------------------------------------------------------
def git_commit_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def freeze_campaign_components(campaign_root: Path | None = None) -> dict:
    """Everything except AlphaZero policy/value weights must be identical
    across every generation this campaign trains -- recorded once, up
    front, so a later diff can prove nothing else silently drifted."""
    import hashlib

    from agentic_raptor.publication.preflight import ROOT as PUB_ROOT
    from agentic_raptor.ranking.types import directory_sha256
    from agentic_raptor.mb_sac.spec_sizing import KNOB_NAMES
    from agentic_raptor.topology_rl import stage3e1 as s1

    campaign_root = campaign_root or CAMPAIGN_ROOT
    rag_path = PUB_ROOT / "artifacts/publication_v2/selfimprove/rag_memory_v2_post_cload_v1_clean.jsonl"
    sft_path = PUB_ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"
    dpo_path = PUB_ROOT / "artifacts/publication_v2/post_sac_ranker/ranker.pt"
    corpus_path = PUB_ROOT / "artifacts/stage3e4/corpus.json"

    def _hash_file(p: Path):
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None

    corpus = json.loads(corpus_path.read_text(encoding="utf-8")) if corpus_path.is_file() else {}
    split_manifest = corpus.get("split_manifest") or {}

    frozen = {
        "git_commit": git_commit_hash(),
        "rag_snapshot_path": str(rag_path), "rag_snapshot_hash": _hash_file(rag_path),
        "sft_checkpoint_path": str(sft_path),
        "sft_checkpoint_hash": directory_sha256(str(sft_path)) if sft_path.is_dir() else None,
        "dpo_checkpoint_path": str(dpo_path), "dpo_checkpoint_hash": _hash_file(dpo_path),
        "corpus_hash": split_manifest.get("corpus_hash"),
        "split_hash": split_manifest.get("split_hash"),
        "surrogate": "fresh per-call (sac_size persist=False in this campaign's "
                    "terminal_evaluation) -- no persisted surrogate checkpoint is "
                    "read or written, so there is nothing to freeze/hash here; "
                    "recorded explicitly rather than silently omitted",
        "electrical_environment_version": "POST_CLOAD_FIX_V1",
        "cload_policy": "effective_c_load(spec) resolved once per episode from the "
                        "spec's own requested load -- never NOMINAL_CLOAD_F fallback",
        "reward_definition": "feasibility-first scalar: 1.0 for exact_spec_pass, "
                             "else max(-1.0, 1.0 - 2*normalized_distance_to_feasibility) "
                             "-- same formula as agentic_raptor.selfimprove_v2.streams' "
                             "puct_examples value_target, not a new AlphaZero-specific "
                             "objective (Section 4/14)",
        "edit_action_vocabulary": sorted(t.value for t in
                                        __import__("agentic_raptor.topology_rl.alphazero",
                                                   fromlist=["AZ_EDIT_ACTION_TYPES"]
                                                   ).AZ_EDIT_ACTION_TYPES),
        "action_schema_version": s1.SCHEMA_VERSION,
        "topology_representation": "DeviceCircuitGraph (LLM/mapping realisation) -> "
                                   "CircuitGraph via value_refresh.device_graph_to_"
                                   "circuit_graph -- canonical, role-aware "
                                   "structural_hash() used for all novelty/dedup/"
                                   "lineage checks",
        "validator_config": "agentic_raptor.topology_rl.alphazero."
                            "validate_alphazero_candidate -- structural edits gated "
                            "through registry.derive_edited() (real apply_edit(), "
                            "not an allow-list), semantic validity via "
                            "_has_gain_device role markers, ancestor-cycle rejection "
                            "via structural_hash lineage membership",
        "action_state_machine": "two-phase (Stage 5 Campaign 01B Section 1): "
                                "SELECT_EXISTING_TOPOLOGY legal ONLY at the virtual "
                                "super-root (exactly one seed commit); structural "
                                "edits + TERMINATE legal only post-selection; "
                                "KEEP_TOPOLOGY removed",
        "mb_sac_knob_vector": list(KNOB_NAMES),
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    campaign_root.mkdir(parents=True, exist_ok=True)
    (campaign_root / "frozen_components.json").write_text(
        json.dumps(frozen, indent=1, default=str), encoding="utf-8")
    return frozen


# ---------------------------------------------------------------------------
# Section 2/3: leakage-checked, disjoint AZ_TRAIN / AZ_VALIDATION
# ---------------------------------------------------------------------------
def build_az_splits(n_train: int, n_val: int, *, split: str = "train",
                    campaign_root: Path | None = None) -> dict:
    campaign_root = campaign_root or CAMPAIGN_ROOT
    from agentic_raptor.publication.eval_sets import (
        excluded_context_ids, excluded_evaluation_context_ids)
    from agentic_raptor.publication.spec_registry import build as build_registry

    protected_ids = excluded_context_ids()
    protected_eval_ids = excluded_evaluation_context_ids()
    reg = build_registry()

    def _clean(e):
        if e["split"] != split or not e["parsed_spec"]:
            return False
        if e["context_id"] in protected_ids:
            return False
        try:
            from agentic_raptor.llm_dpo.integrity import evaluation_context_id
            if evaluation_context_id(e["parsed_spec"]) in protected_eval_ids:
                return False
        except Exception:
            pass
        return True

    entries = [e for e in reg["entries"] if _clean(e)]
    by_tier: dict = {}
    for e in entries:
        by_tier.setdefault(e["difficulty_tier"], []).append(e)
    for tier in by_tier:
        by_tier[tier] = sorted(by_tier[tier], key=lambda e: e["spec_index"])

    def _draw(n, used_hashes, used_ctx):
        # Dedup on BOTH spec_hash AND context_id -- a context_id can host
        # more than one spec_hash variant, so hash-only dedup can still
        # leak the same context_id into both AZ_TRAIN and AZ_VALIDATION
        # (caught by build_az_splits' own context_id-disjointness assert
        # during Campaign 01 retry mechanical verification).
        out, tiers = [], ("easy", "medium", "hard")
        per_tier = max(1, n // len(tiers))
        for tier in tiers:
            for e in by_tier.get(tier, []):
                if e["spec_hash"] in used_hashes or e["context_id"] in used_ctx:
                    continue
                out.append(e)
                used_hashes.add(e["spec_hash"])
                used_ctx.add(e["context_id"])
                if sum(1 for o in out if o["difficulty_tier"] == tier) >= per_tier:
                    break
        if len(out) < n:
            for e in entries:
                if len(out) >= n:
                    break
                if e["spec_hash"] not in used_hashes and e["context_id"] not in used_ctx:
                    out.append(e)
                    used_hashes.add(e["spec_hash"])
                    used_ctx.add(e["context_id"])
        return out[:n]

    used_hashes: set = set()
    used_ctx: set = set()
    az_train = _draw(n_train, used_hashes, used_ctx)
    az_validation = _draw(n_val, used_hashes, used_ctx)

    # Section 3 (Campaign 01 retry): disjointness verified on BOTH spec
    # identity axes (context_id AND content/spec_hash), and protected-set
    # exclusion checked for BOTH splits, not just AZ_TRAIN -- abort
    # immediately (assert, not a soft warning) on any leakage.
    train_hashes = {e["spec_hash"] for e in az_train}
    val_hashes = {e["spec_hash"] for e in az_validation}
    train_ctx = {e["context_id"] for e in az_train}
    val_ctx = {e["context_id"] for e in az_validation}
    assert train_hashes.isdisjoint(val_hashes), "AZ_TRAIN/AZ_VALIDATION spec_hash must be disjoint"
    assert train_ctx.isdisjoint(val_ctx), "AZ_TRAIN/AZ_VALIDATION context_id must be disjoint"
    assert train_ctx.isdisjoint(protected_ids), "AZ_TRAIN leaks a protected context_id"
    assert val_ctx.isdisjoint(protected_ids), "AZ_VALIDATION leaks a protected context_id"
    assert train_hashes.isdisjoint(protected_ids | {
        e["spec_hash"] for e in entries if e["context_id"] in protected_ids})
    assert val_hashes.isdisjoint(protected_ids | {
        e["spec_hash"] for e in entries if e["context_id"] in protected_ids})

    result = {
        "az_train": [{"spec_index": e["spec_index"], "spec_hash": e["spec_hash"],
                     "context_id": e["context_id"], "difficulty_tier": e["difficulty_tier"]}
                    for e in az_train],
        "az_validation": [{"spec_index": e["spec_index"], "spec_hash": e["spec_hash"],
                          "context_id": e["context_id"], "difficulty_tier": e["difficulty_tier"]}
                         for e in az_validation],
        "az_train_tier_distribution": dict(Counter(e["difficulty_tier"] for e in az_train)),
        "az_validation_tier_distribution": dict(Counter(e["difficulty_tier"] for e in az_validation)),
        "n_train_requested": n_train, "n_train_actual": len(az_train),
        "n_val_requested": n_val, "n_val_actual": len(az_validation),
        "disjoint_verified": True, "split": split,
    }
    campaign_root.mkdir(parents=True, exist_ok=True)
    (campaign_root / "az_splits.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result, {e["spec_index"]: e for e in az_train}, {e["spec_index"]: e for e in az_validation}


# ---------------------------------------------------------------------------
# Section 5/6: real episode collection
# ---------------------------------------------------------------------------
def get_llm_candidates(spec_entry: dict, rec: dict, *, model, tok, adapter,
                       rag_memory_path, use_llm: bool, seed: int,
                       target_k: int = 4) -> tuple:
    from run_raptor_v2 import propose_and_validate, rag_stage
    spec = dict(spec_entry["parsed_spec"], spec_id=spec_entry["context_id"],
               spec_hash=spec_entry["spec_hash"], topology_id=rec.get("topology_id"))
    rag = rag_stage(spec, rec["prompt"], use_rag=True, memory_path=rag_memory_path)
    prop = propose_and_validate(model, tok, rag["prompt"], target_k=target_k,
                                conditioning="exclusion", seed0=seed,
                                use_llm=use_llm, spec=spec)
    return spec, prop["candidates"]


class ActionSpaceInvariantViolation(Exception):
    """Section 6 (Campaign 01 retry): hard-fail, never silently continue,
    if a real episode ever violates the Section 1 two-phase action-space
    contract -- these are exactly the shapes CAMPAIGN_01_ATTEMPT_1's bugs
    produced, now asserted mechanically on every real episode instead of
    only checked after the fact by the replay auditor."""


def _assert_action_space_invariants(ep: dict) -> None:
    traj = [s["selected_action_id"] for s in ep["steps"]]
    n_selects = sum(1 for a in traj if a.startswith("a_sel_"))
    if n_selects > 1:
        raise ActionSpaceInvariantViolation(
            f"more than one SELECT action in trajectory {traj}")
    if traj and traj[0].startswith("a_sel_") and any(a.startswith("a_sel_") for a in traj[1:]):
        raise ActionSpaceInvariantViolation(
            f"SELECT occurred after seed commitment in trajectory {traj}")
    if any(s["selected_action_type"] == "KEEP_TOPOLOGY" for s in ep["steps"]):
        raise ActionSpaceInvariantViolation(
            f"KEEP_TOPOLOGY action present -- removed from the action space, "
            f"trajectory {traj}")
    for i, s in enumerate(ep["steps"]):
        at_super_root = s["state_topology_id"] == "proposal_root"
        is_select = s["selected_action_type"] == "SELECT_EXISTING_TOPOLOGY"
        if at_super_root and not is_select:
            raise ActionSpaceInvariantViolation(
                f"step {i}: non-SELECT action taken at the super-root "
                f"(only SELECT is ever legal there): {s['selected_action_type']}")
        if not at_super_root and is_select:
            raise ActionSpaceInvariantViolation(
                f"step {i}: SELECT action taken at a non-super-root state")
    if ep["terminal_topology_hash"] is None:
        raise ActionSpaceInvariantViolation("terminal_topology_hash missing")


def collect_episode(spec_entry: dict, rec: dict, seed: int, *, model, tok, adapter,
                    rag_memory_path, value_ckpt, az_config, sizing_budget: int,
                    use_llm: bool, generation_id: str, campaign_seed: int = 0,
                    deterministic: bool = False) -> dict:
    """`deterministic` (Section 5): False for TRAIN collection (stochastic
    pi-sampling at az_config.alphazero_temperature, exploring); True for
    AZ_VALIDATION/paired-promotion episodes (max-visit, reproducible --
    never sampled, so a promotion decision is never itself a coin flip).
    `campaign_seed` (Section 3) flows into run_alphazero_episode's stable
    per-episode RNG derivation."""
    from agentic_raptor.topology_rl.alphazero import (
        build_replay_rows, run_alphazero_episode, terminal_evaluation)

    spec, candidates = get_llm_candidates(
        spec_entry, rec, model=model, tok=tok, adapter=adapter,
        rag_memory_path=rag_memory_path, use_llm=use_llm, seed=seed)
    if len(candidates) < 2:
        return {"spec_index": spec_entry["spec_index"], "seed": seed,
               "aborted": "insufficient_llm_candidates", "n_candidates": len(candidates)}

    seed_hashes_before = [c["canonical_graph_hash"] for c in candidates]
    ep = run_alphazero_episode(
        candidates, spec, spec_entry["context_id"], spec_entry["spec_hash"],
        spec_index=spec_entry["spec_index"], value_ckpt=value_ckpt, seed=seed,
        campaign_seed=campaign_seed, config=az_config,
        max_episode_depth=az_config.alphazero_max_edit_depth,
        deterministic=deterministic, generation_id=generation_id)
    _assert_action_space_invariants(ep)   # Section 6: hard-fail on violation
    term = terminal_evaluation(ep, spec, budget=sizing_budget, seed=seed)
    rows = build_replay_rows(ep, term, checkpoint_hash=value_ckpt)

    return {
        "spec_index": spec_entry["spec_index"], "spec_hash": spec_entry["spec_hash"],
        "seed": seed, "initial_llm_seed_hashes": seed_hashes_before,
        "n_candidates": len(candidates),
        "seed_ids": ep["seed_ids"], "seed_topology_hashes": ep["seed_topology_hashes"],
        "n_steps": len(ep["steps"]), "steps": ep["steps"],
        "edit_trajectory": [s["selected_action_id"] for s in ep["steps"]],
        "raw_visit_counts_by_step": [s["raw_visit_counts"] for s in ep["steps"]],
        "pi_by_step": [s["pi"] for s in ep["steps"]],
        "terminal_topology_id": ep["terminal_topology_id"],
        "terminal_topology_hash": ep["terminal_topology_hash"],
        "terminal_came_from_edit": ep["terminal_came_from_edit"],
        "seed_novel_hashes": ep["seed_novel_hashes"],
        "seed_novel_count": ep["seed_novel_count"],
        "episode_rng_seed": ep["episode_rng_seed"], "campaign_seed": campaign_seed,
        "dirichlet_epsilon": ep["dirichlet_epsilon"], "dirichlet_alpha": ep["dirichlet_alpha"],
        "deterministic": deterministic,
        "terminal": term, "replay_rows": rows, "registry": ep["registry"],
        "aborted": None,
    }


def novelty_accounting(episodes: list, known_corpus_hashes: set) -> dict:
    """Section 13: SeedNovel/CorpusNovel plus the electrically-meaningful
    extension -- NovelSized (a novel state was realised/sized at all,
    true for every terminal state since sizing always runs) and
    NovelFeasible (novel AND exact_spec_pass)."""
    seed_novel = sum(1 for e in episodes if not e.get("aborted") and e["seed_novel_count"] > 0)
    corpus_novel_terminal = sum(
        1 for e in episodes if not e.get("aborted")
        and e["terminal_topology_hash"] not in known_corpus_hashes
        and e["terminal_came_from_edit"])
    novel_sized = sum(1 for e in episodes if not e.get("aborted") and e["terminal_came_from_edit"])
    novel_feasible = sum(1 for e in episodes if not e.get("aborted") and e["terminal_came_from_edit"]
                         and e["terminal"].get("exact_spec_pass"))
    n = max(1, sum(1 for e in episodes if not e.get("aborted")))
    return {"seed_novel_episodes": seed_novel, "corpus_novel_terminal_episodes": corpus_novel_terminal,
           "novel_sized": novel_sized, "novel_feasible": novel_feasible,
           "novel_feasible_yield": round(novel_feasible / n, 4), "n_episodes": n}


# ---------------------------------------------------------------------------
# Section 7 (Campaign 01B): replay quality audit -- revised degeneracy
# policy. CAMPAIGN_01_ATTEMPT_1 treated "one terminal topology dominates"
# as automatic corruption; post-mortem (see Section 1 of the 01B spec)
# found the REAL cause was two mechanical bugs (SELECT legal at every
# depth; identical RNG seed reused across all episodes), now fixed. With
# those fixed, low diversity in an EARLY, still-mostly-untrained
# generation is expected and legitimate -- different specs can genuinely
# route to the same structural recipe while still producing very
# different, real z (CAMPAIGN_01_ATTEMPT_1's own raw numbers proved this:
# identical 4-action trajectories, z ranging -1.0 to +0.912). So "same
# terminal/action dominates" is now a WARNING (campaign-quality signal),
# not a hard reject -- only genuine corruption, or a TRUE pathological
# collapse (single trajectory AND single terminal AND single action
# across MULTIPLE distinct specs, despite exploration being enabled) is
# still a hard reject.
def _semantic_action_type(action_id: str) -> str:
    """Section 7: recover the SEMANTIC action type from an action id --
    e.g. "a_sel_p00"/"a_sel_p01" both collapse to SELECT_EXISTING_TOPOLOGY,
    while "a_edit_ADD_VERIFIED_STAGE" keeps its own real edit-type name.
    Pure string parsing of the SAME naming convention generate_alphazero_
    actions()/apply_alphazero_action() already use everywhere -- no new
    field needed on the replay row schema."""
    if action_id.startswith("a_sel_"):
        return "SELECT_EXISTING_TOPOLOGY"
    if action_id == "a_term":
        return "TERMINATE_SEARCH"
    if action_id.startswith("a_edit_"):
        return action_id[len("a_edit_"):]
    return action_id


def _entropy(counter: Counter) -> float:
    import math
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c)


def _generation_health(episodes: list, max_edit_depth: int) -> dict:
    """Section 19: per-generation policy-collapse monitoring, computed
    directly on the collected episode list (not just the flattened replay
    rows) so trajectory-level and episode-level rates are exact, not
    reconstructed. Purely diagnostic/reporting here -- the actual abort
    gate is audit_replay()'s pathological-collapse hard reject, which
    runs BEFORE training on this same data."""
    ok = [e for e in episodes if not e.get("aborted")]
    n = len(ok) or 1
    immediate_term = sum(1 for e in ok if e["n_steps"] == 2
                         and e["edit_trajectory"] and e["edit_trajectory"][-1] == "a_term")
    max_depth_hit = sum(1 for e in ok if e["n_steps"] >= max_edit_depth)
    trajectories = Counter(tuple(e["edit_trajectory"]) for e in ok)
    terminals = Counter(e["terminal_topology_hash"] for e in ok)
    seeds = Counter(e["edit_trajectory"][0] for e in ok if e["edit_trajectory"])
    all_actions = Counter(a for e in ok for a in e["edit_trajectory"])
    edit_only = Counter(_semantic_action_type(a) for e in ok for a in e["edit_trajectory"]
                        if a.startswith("a_edit_"))
    return {
        "n_episodes": len(ok),
        "immediate_termination_rate": round(immediate_term / n, 4),
        "max_depth_rate": round(max_depth_hit / n, 4),
        "seed_selection_entropy_bits": round(_entropy(seeds), 4),
        "structural_action_entropy_bits": round(_entropy(edit_only), 4),
        "one_action_dominance": (round(max(all_actions.values()) / sum(all_actions.values()), 4)
                                 if all_actions else 0.0),
        "repeated_trajectory_rate": round(max(trajectories.values()) / n, 4) if trajectories else 0.0,
        "repeated_terminal_rate": round(max(terminals.values()) / n, 4) if terminals else 0.0,
        "distinct_trajectories": len(trajectories), "distinct_terminals": len(terminals),
    }


# ---------------------------------------------------------------------------
def audit_replay(rows: list, *, exploration_enabled: bool = True) -> dict:
    import math

    if not rows:
        return {"episode_count": 0, "state_count": 0, "degenerate": True,
               "hard_reject_reasons": ["no replay rows collected"], "warnings": []}
    z_values = [r["z"] for r in rows]
    action_counts = Counter(r["selected_action_id"] for r in rows)
    depth_counts = Counter(r["edit_depth"] for r in rows)
    terminal_hashes = Counter(r["terminal_topology_hash"] for r in rows)
    spec_hashes = {r["spec_hash"] for r in rows}
    pi_sums = [sum(r["pi"].values()) for r in rows]
    # one episode = one (spec_hash, seed) pair; its trajectory is the
    # ordered selected_action_id sequence across that episode's steps.
    episodes: dict = {}
    for r in rows:
        episodes.setdefault((r["spec_hash"], r["seed"]), []).append(r)
    trajectories = {k: tuple(rr["selected_action_id"] for rr in
                             sorted(v, key=lambda rr: rr["step"]))
                   for k, v in episodes.items()}
    episode_rng_seeds = {k: v[0].get("episode_rng_seed") for k, v in episodes.items()}

    hard_reject_reasons: list[str] = []
    warnings: list[str] = []

    bad_pi = [s for s in pi_sums if not math.isfinite(s) or abs(s - 1.0) > 1e-4]
    if bad_pi:
        hard_reject_reasons.append(f"{len(bad_pi)} rows have a corrupted (non-normalised) pi")
    nan_rows = sum(1 for z in z_values if not math.isfinite(z))
    if nan_rows:
        hard_reject_reasons.append(f"{nan_rows} rows have NaN/Inf z")
    missing_prov = sum(1 for r in rows if not r.get("terminal_authoritative_call_id"))
    if missing_prov:
        hard_reject_reasons.append(f"{missing_prov} rows missing authoritative call_id provenance")
    # identical-RNG-seed-caused-by-a-bug: two DISTINCT episodes (different
    # spec_hash or different rollout seed) must never share the same
    # derived episode_rng_seed -- if they do, stable_episode_rng_seed()
    # (or its wiring) regressed, not legitimate self-play variance.
    seed_values = [v for v in episode_rng_seeds.values() if v is not None]
    if len(seed_values) > 1 and len(set(seed_values)) < len(seed_values):
        hard_reject_reasons.append(
            "duplicate episode_rng_seed across distinct (spec_hash, seed) "
            "episodes -- RNG derivation bug, not legitimate variance")
    # impossible action sequence: more than one SELECT_EXISTING_TOPOLOGY
    # id in a single trajectory is exactly the CAMPAIGN_01_ATTEMPT_1 bug.
    for key, traj in trajectories.items():
        n_selects = sum(1 for a in traj if a.startswith("a_sel_"))
        if n_selects > 1:
            hard_reject_reasons.append(
                f"impossible action sequence for {key}: {n_selects} SELECT "
                f"actions in one trajectory {traj}")
        if traj and traj[0].startswith("a_sel_") and any(
                a.startswith("a_sel_") for a in traj[1:]):
            hard_reject_reasons.append(
                f"SELECT occurred after seed commitment for {key}: {traj}")
    # zero legal-action diversity due to implementation error: every single
    # step across every episode offered only one legal action.
    if rows and all(len(r["legal_action_ids"]) <= 1 for r in rows):
        hard_reject_reasons.append(
            "zero legal-action diversity: every step offered <= 1 legal "
            "action across the whole replay -- likely an action-generation bug")
    if len(spec_hashes) > 1 and len(set(round(z, 6) for z in z_values)) == 1:
        hard_reject_reasons.append(
            f"all z identical ({z_values[0]}) across {len(spec_hashes)} distinct "
            "specs -- real SPICE sizing should vary by spec even when the "
            "same structural recipe is applied; this is more consistent with "
            "a stale/cached measurement than legitimate self-play data")

    top_action_frac = max(action_counts.values()) / len(rows)
    if top_action_frac > 0.9:
        warnings.append(f"one action dominates {top_action_frac:.0%} of rows: "
                        f"{action_counts.most_common(1)}")
    top_terminal_frac = max(terminal_hashes.values()) / len(rows)
    if top_terminal_frac > 0.9:
        warnings.append(f"one terminal topology dominates {top_terminal_frac:.0%}")
    n_distinct_traj = len(set(trajectories.values()))
    if n_distinct_traj <= 1 and len(trajectories) > 1:
        warnings.append(f"only {n_distinct_traj} distinct trajectory across "
                        f"{len(trajectories)} episodes")

    # TRUE pathological collapse (Section 11/7): zero trajectory diversity
    # AND zero terminal diversity AND zero action diversity across MULTIPLE
    # distinct specs, even though real exploration (Dirichlet noise +
    # stochastic training-temperature sampling) was enabled for this
    # collection -- this is the one "low diversity" shape still treated as
    # a hard abort, not a warning, because exploration having been on
    # removes the two previously-known mechanical explanations.
    pathological = (exploration_enabled and len(spec_hashes) > 1
                    and n_distinct_traj <= 1 and len(terminal_hashes) <= 1
                    and len(action_counts) <= 1)
    if pathological:
        hard_reject_reasons.append(
            "pathological collapse: a single trajectory/terminal/action was "
            "used across multiple distinct specs even with real exploration "
            "(Dirichlet root noise + stochastic training sampling) enabled")

    # Section 7 (Campaign 01 retry): action-ID diversity conflates
    # "which seed was selected" with "what structural edit happened" --
    # report all three views separately so a wide spread of SELECT(p00)/
    # SELECT(p01)/... is never misread as varied EDITING behavior.
    semantic_counts = Counter(_semantic_action_type(a) for a in
                              (r["selected_action_id"] for r in rows))
    edit_only_counts = Counter(_semantic_action_type(a) for a in
                               (r["selected_action_id"] for r in rows)
                               if a.startswith("a_edit_"))
    seed_selection_counts = Counter(r["selected_action_id"] for r in rows
                                    if r["selected_action_id"].startswith("a_sel_"))

    degenerate = bool(hard_reject_reasons)
    return {
        "state_count": len(rows), "episode_count": len(trajectories),
        "z_distribution": {"min": min(z_values), "max": max(z_values),
                          "mean": round(sum(z_values) / len(z_values), 4),
                          "pass_fraction": round(sum(1 for z in z_values if z == 1.0) / len(rows), 4)},
        "action_id_distribution": dict(action_counts.most_common(15)),
        "semantic_action_type_distribution": dict(semantic_counts.most_common(15)),
        "structural_edit_type_distribution": dict(edit_only_counts.most_common(15)),
        "seed_selection_distribution": dict(seed_selection_counts),
        "action_id_entropy_bits": round(_entropy(action_counts), 4),
        "semantic_action_type_entropy_bits": round(_entropy(semantic_counts), 4),
        "structural_edit_type_entropy_bits": round(_entropy(edit_only_counts), 4),
        "seed_selection_entropy_bits": round(_entropy(seed_selection_counts), 4),
        # kept for callers/dashboards still reading the old field name
        "action_distribution": dict(action_counts.most_common(10)),
        "edit_depth_distribution": dict(sorted(depth_counts.items())),
        "unique_terminal_topologies": len(terminal_hashes),
        "unique_trajectories": n_distinct_traj,
        "unique_actions_used": len(action_counts),
        "exploration_enabled": exploration_enabled,
        "degenerate": degenerate,
        "hard_reject_reasons": hard_reject_reasons, "warnings": warnings,
        # kept for callers/dashboards still reading the old field name
        "reasons": hard_reject_reasons,
    }


# ---------------------------------------------------------------------------
# Section 8-10: train, validate, promote
# ---------------------------------------------------------------------------
def validate_paired(parent_ckpt, candidate_ckpt, az_validation_entries: dict,
                    pool_recs: list, *, model, tok, adapter, rag_memory_path,
                    az_config, sizing_budget, use_llm, seed: int,
                    campaign_seed: int = 0) -> dict:
    """Section 9: paired G(parent) vs G(candidate) on AZ_VALIDATION,
    identical everything except the checkpoint. Never adds to replay.
    `az_config` here must be a VALIDATION config (training_mode=False --
    no Dirichlet root noise) and every episode runs deterministic=True
    (max-visit, never sampled) per Section 4/5's IMPORTANT note: a
    promotion decision must never itself depend on a random draw."""
    def _run_all(ckpt):
        results = []
        for idx, e in az_validation_entries.items():
            rec = pool_recs[idx]
            r = collect_episode(e, rec, seed, model=model, tok=tok, adapter=adapter,
                               rag_memory_path=rag_memory_path, value_ckpt=ckpt,
                               az_config=az_config, sizing_budget=sizing_budget,
                               use_llm=use_llm, generation_id="VALIDATION",
                               campaign_seed=campaign_seed, deterministic=True)
            results.append(r)
        return results

    parent_results = _run_all(parent_ckpt)
    cand_results = _run_all(candidate_ckpt)

    def _summ(results, max_edit_depth):
        """Sections 11/13: search-learning + electrical-outcome metrics,
        computed per checkpoint on the SAME frozen AZ_VALIDATION specs."""
        ok = [r for r in results if not r.get("aborted")]
        n = max(1, len(ok))
        depths = [r["n_steps"] for r in ok]
        novel = sum(1 for r in ok if r["seed_novel_count"] > 0)
        corpus_novel = sum(1 for r in ok if r["terminal_came_from_edit"])
        passes = sum(1 for r in ok if r["terminal"].get("exact_spec_pass"))
        calls = [r["terminal"]["sizing_spice_calls"] for r in ok]
        root_term = sum(1 for r in ok if r["n_steps"] == 2
                        and r["edit_trajectory"] and r["edit_trajectory"][-1] == "a_term")
        seeds = Counter(r["edit_trajectory"][0] for r in ok if r["edit_trajectory"])
        struct = Counter(_semantic_action_type(a) for r in ok for a in r["edit_trajectory"]
                         if a.startswith("a_edit_"))
        unique_terminals = len({r["terminal_topology_hash"] for r in ok})
        z_vals = [r["terminal"]["z"] for r in ok if r["terminal"].get("z") is not None]
        dist_vals = [r["terminal"]["normalized_distance_to_feasibility"] for r in ok
                    if r["terminal"].get("normalized_distance_to_feasibility") is not None]
        pi_ent = [_entropy(Counter({k: round(v * 1000) for k, v in pi.items()}))
                 for r in ok for pi in r.get("pi_by_step", []) if pi]

        def _mean(xs):
            return round(sum(xs) / len(xs), 4) if xs else None

        return {
            "n": len(ok), "mean_steps": round(sum(depths) / n, 2),
            "max_edit_depth_rate": round(sum(1 for d in depths if d >= max_edit_depth) / n, 4),
            "root_termination_rate": round(root_term / n, 4),
            "seed_novel_episodes": novel, "corpus_novel_episodes": corpus_novel,
            "exact_spec_pass_count": passes, "exact_spec_pass_rate": round(passes / n, 4),
            "unique_terminal_topologies": unique_terminals,
            "selected_seed_distribution": dict(seeds),
            "structural_edit_distribution": dict(struct),
            "mean_pi_entropy_bits": _mean(pi_ent),
            "mean_sizing_spice_calls": round(sum(calls) / n, 2) if calls else None,
            "total_sizing_spice_calls": sum(calls) if calls else 0,
            "z_min": min(z_vals) if z_vals else None, "z_max": max(z_vals) if z_vals else None,
            "z_mean": _mean(z_vals),
            "mean_distance_to_feasibility": _mean(dist_vals),
            "mean_gain_db": _mean([r["terminal"]["gain_db"] for r in ok
                                  if r["terminal"].get("gain_db") is not None]),
            "mean_pm_deg": _mean([r["terminal"]["pm_deg"] for r in ok
                                 if r["terminal"].get("pm_deg") is not None]),
            "mean_ugbw_hz": _mean([r["terminal"]["ugbw_hz"] for r in ok
                                  if r["terminal"].get("ugbw_hz") is not None]),
            "mean_idd_a": _mean([r["terminal"]["idd_a"] for r in ok
                                if r["terminal"].get("idd_a") is not None]),
        }

    def _policy_vs_mcts(ckpt, results):
        """Section 11: does the TRAINED policy agree with what MCTS itself
        found, or has it learned nothing (~uniform) / memorised one path
        (degenerate agreement everywhere including states MCTS disagreed
        with)? Loads `ckpt` once and re-runs policy_forward on every
        recorded validation state (never re-runs search)."""
        from agentic_raptor.topology_rl.alphazero import (
            generate_alphazero_actions, load_alphazero_nets)
        from agentic_raptor.topology_rl.stage3e1 import TopologySearchState
        import math as _math

        nets = load_alphazero_nets(ckpt, seed=0)
        policy_ent, mcts_ent, cross_ent, agree = [], [], [], []
        for r in results:
            if r.get("aborted"):
                continue
            reg = r["registry"]
            for s in r.get("steps", []):
                try:
                    st = TopologySearchState(**s["state"])
                except Exception:
                    continue
                legal, _ = generate_alphazero_actions(st, reg, reg.seed_ids)
                if not legal:
                    continue
                import torch
                with torch.no_grad():
                    _, _lg, probs = nets["policy_forward"](st, legal, reg)
                policy_dist = {a.action_id: p.item() for a, p in
                               zip(sorted(legal, key=lambda a: a.action_id), probs)}
                mcts_pi = s["pi"]
                common = sorted(set(policy_dist) & set(mcts_pi))
                if not common:
                    continue
                p_vec = [max(policy_dist[a], 1e-12) for a in common]
                m_vec = [max(mcts_pi.get(a, 0.0), 1e-12) for a in common]
                p_sum, m_sum = sum(p_vec), sum(m_vec)
                p_vec = [v / p_sum for v in p_vec]
                m_vec = [v / m_sum for v in m_vec]
                policy_ent.append(-sum(p * _math.log2(p) for p in p_vec))
                mcts_ent.append(-sum(m * _math.log2(m) for m in m_vec))
                cross_ent.append(-sum(m * _math.log2(p) for m, p in zip(m_vec, p_vec)))
                policy_top1 = common[p_vec.index(max(p_vec))]
                mcts_top1 = common[m_vec.index(max(m_vec))]
                agree.append(1.0 if policy_top1 == mcts_top1 else 0.0)

        def _mean(xs):
            return round(sum(xs) / len(xs), 4) if xs else None
        return {"n_states_compared": len(agree),
               "mean_policy_entropy_bits": _mean(policy_ent),
               "mean_mcts_pi_entropy_bits": _mean(mcts_ent),
               "mean_policy_vs_mcts_cross_entropy_bits": _mean(cross_ent),
               "policy_top1_vs_mcts_top_visit_agreement_rate": _mean(agree)}

    max_depth = az_config.alphazero_max_edit_depth
    parent_summ = _summ(parent_results, max_depth)
    cand_summ = _summ(cand_results, max_depth)
    parent_policy = _policy_vs_mcts(parent_ckpt, parent_results)
    cand_policy = _policy_vs_mcts(candidate_ckpt, cand_results)
    return {"parent": parent_summ, "candidate": cand_summ,
           "parent_policy_vs_mcts": parent_policy, "candidate_policy_vs_mcts": cand_policy,
           "parent_episodes": parent_results, "candidate_episodes": cand_results}


def promotion_decision(mech: dict, val: dict) -> dict:
    """Section 10: mechanical validity is a hard gate; downstream evidence
    must be directionally sensible (not required to be significant)."""
    failures = list(mech.get("failures") or [])
    p, c = val["parent"], val["candidate"]
    if c["exact_spec_pass_rate"] < p["exact_spec_pass_rate"] - 1e-9 and c["n"] >= 2:
        # allow tiny-sample noise but flag a clear regression
        if p["exact_spec_pass_rate"] - c["exact_spec_pass_rate"] > 0.25:
            failures.append(f"exact_spec_pass_rate regressed clearly: "
                            f"{p['exact_spec_pass_rate']} -> {c['exact_spec_pass_rate']}")
    if c["mean_steps"] <= 1.0 and p["mean_steps"] > 1.0:
        failures.append("candidate collapsed to near-zero search depth")
    return {"passed": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# One full generation cycle: collect -> audit -> train -> validate -> promote
# ---------------------------------------------------------------------------
def run_generation(parent_gen_id: str, output_gen_id: str, *,
                   az_train_entries: dict, az_validation_entries: dict,
                   pool_recs: list, model, tok, adapter, rag_memory_path,
                   seeds: list, train_az_config, val_az_config,
                   sizing_budget: int, use_llm: bool,
                   train_steps: int, train_lr: float, known_corpus_hashes: set,
                   campaign_seed: int = 0) -> dict:
    from agentic_raptor.topology_rl.alphazero import (
        LLMSeededEditRegistry, az_generation_dir, read_az_generation_manifest,
        require_promoted_az_checkpoint, train_az_generation,
        validate_az_candidate, write_az_generation_manifest)
    from agentic_raptor.topology_rl import stage3e1 as s1

    parent = read_az_generation_manifest(parent_gen_id)
    if parent is None:
        raise SystemExit(f"parent generation {parent_gen_id!r} has no manifest")
    parent_ckpt = parent["checkpoint_path"]

    # ---- collect real episodes on AZ_TRAIN ------------------------------
    t0 = time.time()
    episodes = []
    all_rows = []
    episode_train_batches = []   # [(rows, registry), ...] -- kept SEPARATE per
    # episode because topology_ids (e.g. "p00") are only locally unique
    # within one episode's own LLMSeededEditRegistry and WOULD collide if
    # merged across episodes from different specs.
    for idx, e in az_train_entries.items():
        rec = pool_recs[idx]
        for seed in seeds:
            r = collect_episode(e, rec, seed, model=model, tok=tok, adapter=adapter,
                               rag_memory_path=rag_memory_path, value_ckpt=parent_ckpt,
                               az_config=train_az_config, sizing_budget=sizing_budget,
                               use_llm=use_llm, generation_id=output_gen_id,
                               campaign_seed=campaign_seed, deterministic=False)
            episodes.append(r)
            if not r.get("aborted"):
                all_rows.extend(r["replay_rows"])
                if r["replay_rows"]:
                    episode_train_batches.append((r["replay_rows"], r["registry"]))
                print(f"  episode spec={idx} seed={seed}: steps={r['n_steps']} "
                     f"terminal_from_edit={r['terminal_came_from_edit']} "
                     f"pass={r['terminal'].get('exact_spec_pass')} z={r['terminal'].get('z')}",
                     flush=True)
            else:
                print(f"  episode spec={idx} seed={seed}: ABORTED {r['aborted']}", flush=True)
    collection_wall_s = round(time.time() - t0, 1)

    novelty = novelty_accounting(episodes, known_corpus_hashes)
    replay_audit = audit_replay(all_rows, exploration_enabled=train_az_config.training_mode)
    health = _generation_health(episodes, train_az_config.alphazero_max_edit_depth)
    print(f"generation health: {json.dumps(health, default=str)}", flush=True)
    print(f"collected {len(all_rows)} replay rows from {len(episodes)} episodes "
         f"in {collection_wall_s}s", flush=True)
    print(json.dumps(replay_audit, indent=1, default=str), flush=True)

    gen_dir = az_generation_dir(output_gen_id)
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps({k: v for k, v in e.items() if k != "replay_rows"},
                             default=str) for e in episodes), encoding="utf-8")
    from agentic_raptor.topology_rl.alphazero import write_replay_rows
    replay_path = write_replay_rows(all_rows, output_gen_id, out_root=gen_dir)

    if replay_audit["degenerate"]:
        manifest = {"generation_id": output_gen_id, "parent_generation": parent_gen_id,
                   "rejected": True, "checkpoint_status": None,
                   "rejection_reasons": ["degenerate replay: " +
                                        "; ".join(replay_audit["hard_reject_reasons"])],
                   "replay_audit": replay_audit, "novelty": novelty, "health": health}
        write_az_generation_manifest(output_gen_id, manifest)
        return {"promoted": False, "manifest": manifest, "stage": "replay_audit"}

    # ---- train candidate: one train_az_generation() call PER EPISODE,
    # each with its OWN registry, threading the SAME nets/opt through so
    # gradients accumulate across the whole batch (see train_az_
    # generation's docstring) -----------------------------------------
    if not episode_train_batches:
        manifest = {"generation_id": output_gen_id, "parent_generation": parent_gen_id,
                   "rejected": True, "checkpoint_status": None,
                   "rejection_reasons": ["no usable episodes collected"],
                   "replay_audit": replay_audit, "novelty": novelty, "health": health}
        write_az_generation_manifest(output_gen_id, manifest)
        return {"promoted": False, "manifest": manifest, "stage": "no_episodes"}

    running_nets, running_opt = None, None
    epoch_reports = []
    for rows, reg in episode_train_batches:
        train_rec = train_az_generation(rows, reg, parent_checkpoint=parent_ckpt,
                                        lr=train_lr, epochs=train_steps, seed=0,
                                        nets=running_nets, opt=running_opt)
        running_nets, running_opt = train_rec["nets"], train_rec["opt"]
        epoch_reports.extend(train_rec["epoch_reports"])
    ck_path = gen_dir / "checkpoint" / "policy_value.pt"
    s1.save_checkpoint(running_nets, ck_path, {
        "generation_id": output_gen_id, "parent_generation": parent_gen_id,
        "replay_rows": len(all_rows), "episodes_trained_on": len(episode_train_batches),
        "train_steps_per_episode": train_steps, "lr": train_lr})
    import hashlib
    ck_hash = hashlib.sha256(ck_path.read_bytes()).hexdigest()[:16]

    # ---- validate paired on AZ_VALIDATION ---------------------------------
    val = validate_paired(parent_ckpt, str(ck_path), az_validation_entries, pool_recs,
                          model=model, tok=tok, adapter=adapter,
                          rag_memory_path=rag_memory_path, az_config=val_az_config,
                          sizing_budget=sizing_budget, use_llm=use_llm, seed=99,
                          campaign_seed=campaign_seed)
    mech_spec_entry = next(iter(az_validation_entries.values()))
    mech_rec = pool_recs[mech_spec_entry["spec_index"]]
    _mech_spec, mech_candidates = get_llm_candidates(
        mech_spec_entry, mech_rec, model=model, tok=tok, adapter=adapter,
        rag_memory_path=rag_memory_path, use_llm=use_llm, seed=0)
    mech = (validate_az_candidate(running_nets, mech_candidates, _mech_spec,
                                  mech_spec_entry["context_id"], seed=0)
           if len(mech_candidates) >= 2 else
           {"passed": False, "failures": ["fewer than 2 candidates for mechanical check"]})
    promo = promotion_decision(mech, val)

    manifest = {
        "generation_id": output_gen_id, "parent_generation": parent_gen_id,
        "checkpoint_path": str(ck_path) if promo["passed"] else None,
        "candidate_checkpoint_path": str(ck_path), "checkpoint_hash": ck_hash,
        "checkpoint_status": "CANDIDATE" if not promo["passed"] else "PROMOTED",
        "replay_row_count": len(all_rows), "replay_audit": replay_audit,
        "novelty": novelty, "health": health,
        "replay_window_policy": "newest_generation_only -- this generation "
                                "trains ONLY on replay collected in THIS "
                                "run_generation() call (all_rows above); no "
                                "prior generation's replay rows are mixed "
                                "in or reused. Every generation (G1, G2, "
                                "G3, ...) collects fresh (state, pi, z) "
                                "episodes from its own parent checkpoint "
                                "(Section 18, documented not silently "
                                "changed between generations).",
        "training_record": {
            "episodes_trained_on": len(episode_train_batches),
            "policy_loss_curve": [r["policy_loss"] for r in epoch_reports],
            "value_loss_curve": [r["value_loss"] for r in epoch_reports],
            "grad_norm_curve": [r["grad_norm"] for r in epoch_reports],
            "final_policy_loss": epoch_reports[-1]["policy_loss"] if epoch_reports else None,
            "final_value_loss": epoch_reports[-1]["value_loss"] if epoch_reports else None,
            "lr": train_lr, "optimizer": "Adam", "seed": 0,
            "epoch_reports": epoch_reports},
        "mechanical_validation": mech, "paired_validation": val,
        "promotion": promo, "rejected": not promo["passed"],
        "collection_wall_clock_s": collection_wall_s,
        "exploration_config": {
            "campaign_seed": campaign_seed,
            "train_dirichlet_epsilon": train_az_config.dirichlet_epsilon,
            "train_dirichlet_alpha": train_az_config.dirichlet_alpha,
            "train_training_mode": train_az_config.training_mode,
            "train_temperature": train_az_config.alphazero_temperature,
            "train_deterministic_sampling": False,
            "validation_training_mode": val_az_config.training_mode,
            "validation_deterministic_sampling": True},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_az_generation_manifest(output_gen_id, manifest)
    print(f"{output_gen_id}: {'PROMOTED' if promo['passed'] else 'REJECTED'} "
         f"{promo['failures']}", flush=True)
    return {"promoted": promo["passed"], "manifest": manifest, "stage": "complete"}


def _spec_conditioning_audit(ok_episodes: list) -> dict:
    """Section 6 (Campaign 01B): prove -- against the REAL G0 network, on a
    REAL registry produced by this gate's own episodes -- that the policy/
    value outputs actually change when the specification changes on an
    otherwise-IDENTICAL topology state. Does not require an UNTRAINED net
    to make good decisions, only that the spec-conditioning pathway is
    live (not accidentally disconnected)."""
    from agentic_raptor.topology_rl import alphazero as az
    if not ok_episodes:
        return {"passed": False, "reason": "no successful episodes to audit against"}
    import torch
    reg = ok_episodes[0]["registry"]
    nets = az.load_alphazero_nets(value_ckpt=None, seed=0)
    base_spec = {"spec_id": "gate_spec_cond", "gain_target_db": 60.0,
                "phase_margin_target_deg": 55.0, "load_capacitance_pf": 100.0,
                "ugbw_target_hz": 1e5}
    alt_spec = {**base_spec, "gain_target_db": base_spec["gain_target_db"] + 40.0}
    state_a = az.build_root_state(base_spec, "gate_spec_cond", reg.seed_ids)
    state_b = az.build_root_state(alt_spec, "gate_spec_cond", reg.seed_ids)
    legal_a, _ = az.generate_alphazero_actions(state_a, reg, reg.seed_ids)
    _, logits_a, _probs_a = nets["policy_forward"](state_a, legal_a, reg)
    v_a = nets["value_forward"](state_a, reg)["scalar"].item()
    _, logits_b, _probs_b = nets["policy_forward"](state_b, legal_a, reg)
    v_b = nets["value_forward"](state_b, reg)["scalar"].item()
    logits_differ = not torch.allclose(logits_a, logits_b, atol=1e-6)
    value_differs = abs(v_a - v_b) > 1e-6
    return {"passed": bool(logits_differ or value_differs),
           "logits_differ": logits_differ, "value_differs": value_differs,
           "v_gain60": v_a, "v_gain100": v_b}


# ---------------------------------------------------------------------------
# Section 8/9/10: small real bootstrap diversity gate -- diagnostic only,
# trains and promotes nothing. Gates whether a real Campaign 01 retry is
# even worth running after the Section 1-5 fixes.
# ---------------------------------------------------------------------------
def run_diversity_gate(*, az_train_entries: dict, pool_recs: list, model, tok, adapter,
                       rag_memory_path, seeds: list, train_az_config, sizing_budget: int,
                       use_llm: bool, campaign_seed: int, frozen: dict,
                       value_ckpt=None, n_specs: int = 4) -> dict:
    gate_specs = list(az_train_entries.items())[:n_specs]
    episodes = []
    for idx, e in gate_specs:
        rec = pool_recs[idx]
        for seed in seeds:
            r = collect_episode(e, rec, seed, model=model, tok=tok, adapter=adapter,
                               rag_memory_path=rag_memory_path, value_ckpt=value_ckpt,
                               az_config=train_az_config, sizing_budget=sizing_budget,
                               use_llm=use_llm, generation_id="DIVERSITY_GATE",
                               campaign_seed=campaign_seed, deterministic=False)
            episodes.append(r)
            if r.get("aborted"):
                print(f"  gate episode spec={idx} seed={seed}: ABORTED {r['aborted']}",
                     flush=True)
            else:
                print(f"  gate episode spec={idx} seed={seed}: "
                     f"rng_seed={r['episode_rng_seed']} steps={r['n_steps']} "
                     f"trajectory={r['edit_trajectory']} z={r['terminal'].get('z')}",
                     flush=True)

    ok_episodes = [e for e in episodes if not e.get("aborted")]
    all_rows = [row for e in ok_episodes for row in e["replay_rows"]]
    replay_audit_result = audit_replay(all_rows, exploration_enabled=train_az_config.training_mode)

    per_episode = []
    for e in episodes:
        if e.get("aborted"):
            per_episode.append({"spec_index": e["spec_index"], "seed": e["seed"],
                               "aborted": e["aborted"]})
            continue
        per_episode.append({
            "episode_rng_seed": e["episode_rng_seed"], "spec_hash": e["spec_hash"],
            "spec_index": e["spec_index"], "rollout_seed": e["seed"],
            "selected_llm_seed": (e["edit_trajectory"][0].replace("a_sel_", "")
                                  if e["edit_trajectory"] and
                                  e["edit_trajectory"][0].startswith("a_sel_") else None),
            "action_trajectory": e["edit_trajectory"], "edit_depth": e["n_steps"],
            "root_terminated_before_max_depth": (bool(e["edit_trajectory"])
                and e["edit_trajectory"][-1] == "a_term"
                and e["n_steps"] < train_az_config.alphazero_max_edit_depth),
            "raw_visit_counts_by_step": e["raw_visit_counts_by_step"],
            "pi_by_step": e["pi_by_step"],
            "terminal_topology_hash": e["terminal_topology_hash"],
            "z": e["terminal"].get("z"),
            "seed_novel": e["seed_novel_count"] > 0,
            "corpus_novel": e["terminal_came_from_edit"]})

    trajectories = [tuple(e["edit_trajectory"]) for e in ok_episodes]
    terminal_hashes = [e["terminal_topology_hash"] for e in ok_episodes]
    selected_seeds = [t[0] for t in trajectories if t]
    rng_seeds = [e["episode_rng_seed"] for e in ok_episodes]
    depths = [e["n_steps"] for e in ok_episodes]
    root_term_count = sum(1 for e in ok_episodes if e["edit_trajectory"]
                          and e["edit_trajectory"][-1] == "a_term"
                          and e["n_steps"] < train_az_config.alphazero_max_edit_depth)

    action_counter = Counter(a for e in ok_episodes for a in e["edit_trajectory"])
    total_actions = sum(action_counter.values()) or 1
    import math as _math
    action_entropy = -sum((c / total_actions) * _math.log2(c / total_actions)
                          for c in action_counter.values() if c)
    step0_pi_signatures = {json.dumps(e["pi_by_step"][0], sort_keys=True)
                           for e in ok_episodes if e.get("pi_by_step")}

    spec_cond = _spec_conditioning_audit(ok_episodes)

    aggregate = {
        "n_episodes": len(episodes), "n_aborted": len(episodes) - len(ok_episodes),
        "distinct_trajectories": len(set(trajectories)),
        "distinct_terminal_topologies": len(set(terminal_hashes)),
        "distinct_selected_seeds": len(set(selected_seeds)),
        "all_rng_seeds_unique": len(set(rng_seeds)) == len(rng_seeds) if rng_seeds else False,
        "action_entropy_bits": round(action_entropy, 4),
        "distinct_pi_signatures_at_step0": len(step0_pi_signatures),
        "root_termination_before_max_depth_rate": (
            round(root_term_count / len(ok_episodes), 4) if ok_episodes else 0.0),
        "mean_edit_depth": round(sum(depths) / len(depths), 4) if depths else 0.0,
        "max_edit_depth_observed": max(depths) if depths else 0,
        "z_min": min((e["terminal"]["z"] for e in ok_episodes), default=None),
        "z_max": max((e["terminal"]["z"] for e in ok_episodes), default=None),
    }

    checks = {
        "all_episode_rng_seeds_unique": aggregate["all_rng_seeds_unique"],
        "select_at_most_once_per_trajectory": all(
            sum(1 for a in t if a.startswith("a_sel_")) <= 1 for t in trajectories),
        "no_select_after_seed_selection": all(
            not any(a.startswith("a_sel_") for a in t[1:]) for t in trajectories),
        "structural_editing_remained_valid": len(ok_episodes) == len(episodes),
        "more_than_one_distinct_trajectory": aggregate["distinct_trajectories"] > 1,
        "more_than_one_distinct_terminal_topology": aggregate["distinct_terminal_topologies"] > 1,
        "visit_distributions_differ_across_episodes": aggregate["distinct_pi_signatures_at_step0"] > 1,
        "spec_conditioning_audit_passes": spec_cond["passed"],
        "replay_provenance_valid": not replay_audit_result.get("hard_reject_reasons"),
    }
    passed = all(checks.values())

    return {
        "frozen_components": frozen,
        "gate_config": {
            "n_specs": len(gate_specs), "seeds": seeds,
            "n_episodes_target": len(gate_specs) * len(seeds),
            "campaign_seed": campaign_seed,
            "dirichlet_epsilon": train_az_config.dirichlet_epsilon,
            "dirichlet_alpha": train_az_config.dirichlet_alpha,
            "training_mode": train_az_config.training_mode,
            "training_temperature": train_az_config.alphazero_temperature,
            "max_edit_depth": train_az_config.alphazero_max_edit_depth,
            "sizing_budget": sizing_budget, "value_ckpt": value_ckpt},
        "per_episode": per_episode, "aggregate": aggregate,
        "replay_audit": replay_audit_result, "spec_conditioning_audit": spec_cond,
        "gate": {"checks": checks, "passed": passed,
                "note": ("TERMINATE-before-max-depth as a MECHANISM is proven "
                        "separately by tests/test_alphazero.py::"
                        "test_terminate_can_occur_before_max_edit_depth (a "
                        "controlled-policy unit test) -- it is reported here "
                        "(root_termination_before_max_depth_rate) but not "
                        "required to be > 0 in this specific 8-episode sample: "
                        "a lightly-explored G0 legitimately choosing to run "
                        "to full depth on every gate episode would not, by "
                        "itself, indicate the mechanism is broken.")},
    }


# ---------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", type=int, default=16, help="AZ_TRAIN spec count")
    ap.add_argument("--val-specs", type=int, default=6, help="AZ_VALIDATION spec count")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--adapter", default=str(
        ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"))
    ap.add_argument("--use-llm", choices=("true", "false"), default="true")
    ap.add_argument("--simulations", type=int, default=128)
    ap.add_argument("--max-edit-depth", type=int, default=4)
    ap.add_argument("--sizing-budget", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=1)
    ap.add_argument("--train-lr", type=float, default=1e-3)
    ap.add_argument("--generations", type=int, default=3,
                    help="max generations to attempt (stops early on rejection)")
    ap.add_argument("--campaign-seed", type=int, default=0,
                    help="Section 3: mixed into every episode's stable RNG "
                         "seed derivation, so a full campaign rerun under a "
                         "different --campaign-seed is a genuinely distinct "
                         "sample, not a silent replay of the same draws")
    ap.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                    help="Section 4: root exploration noise weight during "
                         "TRAIN collection only")
    ap.add_argument("--dirichlet-alpha", type=float, default=0.30,
                    help="Section 4: root exploration Dirichlet concentration")
    ap.add_argument("--diversity-gate", action="store_true",
                    help="Section 8: run ONLY the small bootstrap diversity "
                         "gate (--specs specs x --seeds seeds, no training, "
                         "no promotion) and stop -- does not run the full "
                         "campaign")
    ap.add_argument("--campaign-tag", default="",
                    help="Campaign 01 retry Section 1: appended to the "
                         "artifact namespace (artifacts/publication_v3/"
                         "alphazero_campaign_01<_TAG>/) and to generation "
                         "ids (AZ_G1<_TAG>, AZ_G2<_TAG>, ...) so this run's "
                         "output is never confused with or silently "
                         "overwrites CAMPAIGN_01_ATTEMPT_1 or the "
                         "Campaign 01B diversity-gate run in the shared "
                         "alphazero_campaign_01/ directory. AZ_G0 is exempt "
                         "(shared, immutable, identical starting point "
                         "regardless of campaign attempt).")
    args = ap.parse_args()

    from agentic_raptor.topology_rl.alphazero import AlphaZeroConfig, ensure_az_g0_manifest

    tag_suffix = f"_{args.campaign_tag}" if args.campaign_tag else ""
    campaign_root = ROOT / f"artifacts/publication_v3/alphazero_campaign_01{tag_suffix}"

    print("=== Section 0/1: restore point + frozen components ===", flush=True)
    print(f"campaign artifact namespace: {campaign_root}", flush=True)
    frozen = freeze_campaign_components(campaign_root)
    print(json.dumps(frozen, indent=1, default=str), flush=True)

    print("\n=== Section 2/3: AZ_TRAIN / AZ_VALIDATION ===", flush=True)
    splits, az_train_entries, az_validation_entries = build_az_splits(
        args.specs, args.val_specs, campaign_root=campaign_root)
    print(json.dumps({k: v for k, v in splits.items()
                      if k not in ("az_train", "az_validation")}, indent=1), flush=True)
    print(f"AZ_TRAIN: {[e['spec_index'] for e in splits['az_train']]}", flush=True)
    print(f"AZ_VALIDATION: {[e['spec_index'] for e in splits['az_validation']]}", flush=True)

    print("\n=== Section 4: G0 ===", flush=True)
    g0 = ensure_az_g0_manifest()
    print(json.dumps(g0, indent=1, default=str), flush=True)

    corpus = json.loads((ROOT / "artifacts/stage3e4/corpus.json").read_text(encoding="utf-8"))
    pool_recs = [r for r in corpus["records"] if r["split"] == "train"]
    known_corpus_hashes = {r.get("canonical_graph_hash") for r in
                           json.loads((ROOT / "artifacts/publication_v2/proposer_repair/"
                                      "corpus_diverse.json").read_text(encoding="utf-8"))["records"]}

    use_llm = args.use_llm == "true"
    model = tok = None
    if use_llm:
        from run_qwen_ablation import _load
        # _load() returns (tok, model) -- confirmed against its own source
        # and every OTHER call site in this codebase (run_self_improvement_
        # v2.py, run_stage4_diversity_diagnostic.py, sft_self_improvement.py
        # all correctly unpack tok, model = _load(...)). This was the one
        # place that had it backwards, which is why --use-llm true crashed
        # deep inside model.generate() with a string reaching torch.
        # embedding() -- `tok` was actually the model being called as if it
        # were the tokenizer.
        tok, model = _load(args.adapter)

    # Section 4/5: TRAIN collection gets Dirichlet root exploration ON and
    # stochastic training-temperature pi-sampling (deterministic=False,
    # threaded via collect_episode); VALIDATION/paper-inference-style
    # configs get training_mode=False (no noise) and deterministic=True
    # sampling -- never the same config object for both, so there is no
    # way for exploration noise to silently leak into a promotion decision.
    train_az_config = AlphaZeroConfig(
        alphazero_simulations_per_move=args.simulations,
        alphazero_max_edit_depth=args.max_edit_depth, seed=0,
        training_mode=True, dirichlet_epsilon=args.dirichlet_epsilon,
        dirichlet_alpha=args.dirichlet_alpha)
    val_az_config = AlphaZeroConfig(
        alphazero_simulations_per_move=args.simulations,
        alphazero_max_edit_depth=args.max_edit_depth, seed=0,
        training_mode=False)
    seeds = [int(s) for s in args.seeds.split(",")]

    from agentic_raptor.publication.preflight import ROOT as PUB_ROOT
    rag_memory_path = (PUB_ROOT / "artifacts/publication_v2/selfimprove/"
                       "rag_memory_v2_post_cload_v1_clean.jsonl")

    if args.diversity_gate:
        print("\n=== Section 8: bootstrap diversity gate (no training) ===", flush=True)
        gate_report = run_diversity_gate(
            az_train_entries=az_train_entries, pool_recs=pool_recs,
            model=model, tok=tok, adapter=args.adapter,
            rag_memory_path=str(rag_memory_path), seeds=seeds,
            train_az_config=train_az_config, sizing_budget=args.sizing_budget,
            use_llm=use_llm, campaign_seed=args.campaign_seed, frozen=frozen,
            value_ckpt=g0["checkpoint_path"], n_specs=args.specs)
        gate_path = campaign_root / "DIVERSITY_GATE_REPORT.json"
        gate_path.write_text(json.dumps(gate_report, indent=1, default=str),
                             encoding="utf-8")
        print(json.dumps(gate_report["gate"], indent=1, default=str), flush=True)
        print(f"\ndiversity gate report written to {gate_path}", flush=True)
        print(f"GATE RESULT: {'PASS' if gate_report['gate']['passed'] else 'FAIL'}",
             flush=True)
        return

    parent_gen = "AZ_G0"
    results = []
    for gen_num in range(1, args.generations + 1):
        output_gen = f"AZ_G{gen_num}{tag_suffix}"
        print(f"\n=== Generation {output_gen} (parent {parent_gen}) ===", flush=True)
        r = run_generation(parent_gen, output_gen, az_train_entries=az_train_entries,
                           az_validation_entries=az_validation_entries, pool_recs=pool_recs,
                           model=model, tok=tok, adapter=args.adapter,
                           rag_memory_path=str(rag_memory_path), seeds=seeds,
                           train_az_config=train_az_config, val_az_config=val_az_config,
                           sizing_budget=args.sizing_budget,
                           use_llm=use_llm, train_steps=args.train_steps,
                           train_lr=args.train_lr, known_corpus_hashes=known_corpus_hashes,
                           campaign_seed=args.campaign_seed)
        results.append({"generation": output_gen, **r})
        if not r["promoted"]:
            print(f"{output_gen} not promoted -- stopping generational chain here.",
                 flush=True)
            break
        parent_gen = output_gen

    report_path = campaign_root / "FINAL_REPORT.json"
    report_path.write_text(json.dumps({
        "campaign_tag": args.campaign_tag, "campaign_root": str(campaign_root),
        "frozen_components": frozen, "splits_summary":
            {k: v for k, v in splits.items() if k not in ("az_train", "az_validation")},
        "generations": results,
        "final_promoted_generation": next(
            (r["generation"] for r in reversed(results) if r["promoted"]), "AZ_G0"),
    }, indent=1, default=str), encoding="utf-8")
    print(f"\ncampaign report written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
