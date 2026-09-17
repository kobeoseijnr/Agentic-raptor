"""LINEAR CONTEXTUAL-BANDIT TOP-2 TOPOLOGY SELECTOR (2026-08-14).

Chooses which 2 validated LLM proposals advance to MB-SAC sizing by a
learned linear scoresheet over the 24 physical features
(linear_value.physical_features_core) -- the same feature implementation
the frozen value probe uses, so offline training data and live scoring
cannot drift.

WHY THIS EXISTS: the Stage-9B offline feasibility gate (2026-08-14)
measured, spec-disjoint on held-out TRAIN-domain contexts, that this
scoresheet ranks measured topology outcomes at 12/12 pairwise accuracy
where the frozen deterministic rule (run_puct_ablation.topology_priors)
scores 1/12 -- the prior rule assigns identical scores to same-family
topologies and therefore cannot rank within a family at all, which is
exactly where the measured regret concentrated (mean 0.25 z per context;
1.37 z per wrong pair). Full numbers: BANDIT_TOP2_V1.json
["feasibility_gate_2026_08_14"].

EXPERIMENTAL, OPT-IN ONLY: nothing live imports this by default; the live
FULL defaults (search="one_root", ranker_mode="dpo") are untouched.
Pipeline modes:
  search="bandit_top2"     -- bandit ranks the ORIGINAL validated LLM
                              proposals, top-2 to sizing (no AlphaZero
                              search runs at all);
  search="bandit_top2_az"  -- OPTION C HYBRID: AlphaZero runs as a
                              CANDIDATE GENERATOR (its top-2, possibly
                              edited descendants, join the pool) and the
                              bandit scores the union and picks the final
                              top-2.

WEIGHTS ARE HASH-PINNED: load_bandit_v1() verifies the artifact's SHA-256
against the pin below and hard-fails on any mismatch -- a silently edited
or regenerated weights file must never masquerade as the gated V1.
Updates happen ONLY through A9 adaptive TRAIN generations (spec-disjoint
gate vs this incumbent, accept-or-rollback), producing a NEW versioned
artifact + a NEW pin, never a mutation of this one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
BANDIT_V1_PATH = _ROOT / "artifacts/publication_v3/bandit_top2_v1/BANDIT_TOP2_V1.json"
BANDIT_V1_SHA256 = "703bd887a3d1f5e0a4e65251ba61866a5c622d9fb320fc9bd380cbf624de2a61"

#: V2 PROMOTION (2026-08-16, human decision): refit on the post-nulling-fix
#: TRAIN outcome matrix (36 stratified specs x 5 families, real ngspice,
#: gain range 42-156 dB). Spec-disjoint gate vs the V1 incumbent:
#: top-2-contains-passing-family 10/10 vs 6/10, pairwise 80/88 vs 41/88.
#: V1 stays on disk as the frozen historical incumbent.
BANDIT_V2_PATH = _ROOT / "artifacts/publication_v3/bandit_top2_v1/BANDIT_TOP2_V2.json"
BANDIT_V2_SHA256 = "0b6aad71d20f7ecf1d57c3100564fb674644bff086b0ba4b98a235cdcfc4aece"
PROMOTED_BANDIT_PATH = BANDIT_V2_PATH
PROMOTED_BANDIT_SHA256 = BANDIT_V2_SHA256


class BanditSelectorError(RuntimeError):
    """Hard failure: missing/tampered weights or an unusable candidate
    pool. No fallback to priors or arbitrary candidates by design."""


def _load_pinned(path: Path, sha_pin: str, name: str) -> dict:
    if not path.is_file():
        raise BanditSelectorError(f"{name} weights not found at {path}")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != sha_pin:
        raise BanditSelectorError(
            f"{name} hash mismatch: file {sha[:16]}... != pinned "
            f"{sha_pin[:16]}... -- weights file was modified or "
            f"regenerated; refusing to score with unverified weights")
    art = json.loads(raw.decode("utf-8"))
    from agentic_raptor.topology_rl.linear_value import FEATURE_NAMES
    if tuple(art["feature_names"]) != tuple(FEATURE_NAMES):
        raise BanditSelectorError(
            f"{name} feature names no longer match "
            "linear_value.FEATURE_NAMES -- feature implementation drifted "
            "since the weights were frozen")
    return art


def load_bandit_v1() -> dict:
    """The frozen V1 incumbent (historical; superseded by V2)."""
    return _load_pinned(BANDIT_V1_PATH, BANDIT_V1_SHA256, "BANDIT_TOP2_V1")


def load_promoted_bandit() -> dict:
    """The PROMOTED weights the live selector scores with (V2)."""
    return _load_pinned(PROMOTED_BANDIT_PATH, PROMOTED_BANDIT_SHA256,
                        "BANDIT_TOP2_V2")


def _score(art: dict, feats: list[float]) -> float:
    z = [(f - m) / s for f, m, s in zip(feats, art["mu"], art["sd"])]
    return float(sum(zi * wi for zi, wi in zip(z, art["weights"])) + art["bias"])


def bandit_select_two(candidates: list[dict], spec: dict, ctx_id: str, *,
                      az_selected: list[dict] | None = None,
                      art: dict | None = None) -> dict:
    """Validated LLM candidates in, exactly two topology objects out --
    the same external contract as alphazero_select_two() /
    direct_prior_select_two(), ready for size_and_predict().

    `az_selected`: optional AlphaZero top-2 (Option C hybrid). Entries may
    be edited descendants carrying their own `device_graph`; they enter
    the scoring pool alongside the originals, deduplicated by canonical
    hash (an AZ pick that IS an original seed does not double-enter).
    """
    from agentic_raptor.topology_rl.alphazero import _convert_spec
    from agentic_raptor.topology_rl.linear_value import physical_features_core
    from run_puct_ablation import _realise
    if len(candidates) < 2:
        raise BanditSelectorError(
            f"bandit top-2 needs >= 2 LLM candidates, got {len(candidates)}")
    art = art or load_promoted_bandit()   # V2 since 2026-08-16
    internal_spec = _convert_spec(spec)

    pool: dict[str, dict] = {}
    for c in candidates:
        h = c["canonical_graph_hash"]
        if h in pool:
            continue
        dg = _realise(c["obj"])
        entry = dict(c)
        entry["device_graph"] = None      # original proposal: sizing re-realises
        entry["source"] = c.get("source") or "llm"
        entry["edit_depth"] = 0
        entry["is_edited_descendant"] = False
        entry["edit_history"] = []
        entry["originating_seed_id"] = c["llm_proposal_id"]
        entry["originating_seed_hash"] = h
        entry["bandit_features"] = physical_features_core(
            dg, internal_spec, 0.0, False)
        pool[h] = entry
    for c in az_selected or []:
        h = c["canonical_graph_hash"]
        if h in pool:
            pool[h]["also_alphazero_selected"] = True
            continue
        dg = c.get("device_graph")
        if dg is None:
            dg = _realise(c["obj"])
        entry = dict(c)
        entry["source"] = "alphazero_candidate"
        entry["bandit_features"] = physical_features_core(
            dg, internal_spec, float(c.get("edit_depth") or 0),
            bool(c.get("is_edited_descendant")))
        pool[h] = entry

    for h, entry in pool.items():
        entry["bandit_score"] = _score(art, entry["bandit_features"])
    ranked = sorted(pool.values(),
                    key=lambda c: (-c["bandit_score"], c["canonical_graph_hash"]))
    for rank, c in enumerate(ranked):
        c["rank"] = rank
        c["visit_count"] = 0
        c["selected_top2"] = rank < 2
    if len(ranked) < 2:
        raise BanditSelectorError(
            f"bandit pool collapsed to {len(ranked)} distinct canonical "
            "topology(ies) -- cannot select two DISTINCT outputs")
    # DIVERSE TOP-2 (2026-08-30, HELDOUT29 failure analysis): 16/18 AG_FULL
    # 3-target failures picked 2s_none into BOTH slots' capability class
    # while the excluded 3-stage candidate passed the same spec under A0.
    # The learned weights systematically score 3-stage families lowest, so
    # a pure score top-2 can fill both slots with structurally incapable
    # families on high-gain specs. Slot 1 stays the score winner; slot 2
    # becomes the best-scoring candidate with a DIFFERENT stage count
    # (else different compensation, else the plain runner-up). Selection
    # between them remains the Supervisor's measured decision.
    def _sig(c):
        fam = c.get("canonical_family") or ""
        parts = fam.split("_", 1)
        return (parts[0], parts[1] if len(parts) > 1 else "")
    first = ranked[0]
    alt = next((c for c in ranked[1:] if _sig(c)[0] != _sig(first)[0]), None)
    if alt is None:
        alt = next((c for c in ranked[1:] if _sig(c)[1] != _sig(first)[1]),
                   None)
    diversity_promoted = alt is not None and alt is not ranked[1]
    if alt is None:
        alt = ranked[1]
    selected = [first, alt]
    for c in ranked:
        c["selected_top2"] = c is first or c is alt
        c["diversity_promoted"] = diversity_promoted and c is alt
    assert selected[0]["canonical_graph_hash"] != selected[1]["canonical_graph_hash"]
    return {"selected": selected, "ranked": ranked,
           "search": ("bandit_top2_az_hybrid" if az_selected
                      else "bandit_top2"),
           "weights_sha256": PROMOTED_BANDIT_SHA256,
           "pool_size": len(ranked),
           "diversity_promoted": diversity_promoted,
           "az_contributed": sum(1 for c in ranked
                                 if c.get("source") == "alphazero_candidate")}
