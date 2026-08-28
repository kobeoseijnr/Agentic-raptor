"""Stage 7.2B: POST_SAC_FEATURES_V2 -- a richer, versioned post-SAC ranker
feature schema. Does NOT replace agentic_raptor.ranking.model's original
11-feature schema (still used by Stage 7/7.1/7.2A reproductions); this is
an additive, separately-versioned representation.

Temporal information boundary (Section 2): every feature here must be
reconstructable from information that exists BEFORE Level 2 selects a
winner -- i.e. from a branch's OWN completed MB-SAC sizing trajectory
(agentic_raptor.mb_sac.spec_sizing.sac_size()'s `results`, persisted as
sac_replay.jsonl rows). Nothing here may read a final-verification result
(that happens strictly AFTER Level 2 decides).

Provenance taxonomy (Section 10) -- every feature is exactly one of:
  PREDICTED     from the per-run surrogate (agentic_raptor.ranking.surrogate,
                the original 11-feature schema)
  DIRECT        read straight from a real sizing-loop measurement row
  RECONSTRUCTED approximated via the SAME documented precedent already used
                in spec_sizing._offline_pretrain for historical rows
                missing 'stable'/'op_valid' (PM sign / measurement presence)
  COMPUTED      deterministically derived from other real fields (margins,
                trajectory statistics, family-string parsing)
  MISSING       genuinely unavailable for this candidate (e.g. every
                Stage 7.1-era "original" pair, which never persisted a raw
                sizing trajectory) -- filled with 0.0 AND an explicit
                availability flag, never silently treated as a real zero.
"""
from __future__ import annotations

import math
import re
import statistics as st

# ---------------------------------------------------------------------------
# Section 5: retained original 11 surrogate features (PREDICTED)
# ---------------------------------------------------------------------------
_V1_NAMES = ("worst_violation", "n_satisfied", "n_missing", "uncertainty",
            "stability_p", "margin_gain", "margin_pm", "margin_ugbw",
            "has_gain", "has_pm", "has_ugbw")

# ---------------------------------------------------------------------------
# Section 6/7: candidate-own measured margins (DIRECT/RECONSTRUCTED/COMPUTED)
# ---------------------------------------------------------------------------
_CANDIDATE_MEASURED_NAMES = (
    "measured_gain_margin", "measured_pm_margin", "measured_ugbw_margin",
    "measured_distance_to_feasibility", "measured_stable_reconstructed",
    "measured_op_valid", "has_candidate_measurement")

# ---------------------------------------------------------------------------
# Section 6: branch-trajectory summary (DIRECT/COMPUTED)
# ---------------------------------------------------------------------------
_TRAJECTORY_NAMES = (
    "traj_best_distance", "traj_start_distance", "traj_final_distance",
    "traj_absolute_improvement", "traj_relative_improvement",
    "traj_best_reward", "traj_final_reward", "traj_reward_improvement",
    "traj_n_evaluations", "traj_valid_spice_fraction",
    "traj_stable_fraction_reconstructed", "traj_constraint_satisfaction_frequency",
    "traj_gain_mean", "traj_gain_std", "traj_pm_mean", "traj_pm_std",
    "traj_log_ugbw_mean", "traj_log_ugbw_std", "traj_reward_mean", "traj_reward_std",
    "traj_distance_mean", "traj_distance_std", "has_trajectory")

# ---------------------------------------------------------------------------
# Section 9: small, principled topology structural descriptors (COMPUTED)
# ---------------------------------------------------------------------------
_TOPOLOGY_NAMES = ("stage_count", "comp_none", "comp_miller", "comp_rc",
                   "has_topology_family")

FEATURE_NAMES_V2 = _V1_NAMES + _CANDIDATE_MEASURED_NAMES + _TRAJECTORY_NAMES + _TOPOLOGY_NAMES
FEATURE_DIM_V2 = len(FEATURE_NAMES_V2)

FEATURE_PROVENANCE_V2 = (
    tuple("predicted" for _ in _V1_NAMES)
    + ("direct", "direct", "direct", "computed", "reconstructed", "reconstructed", "computed")
    + tuple("computed" if n not in ("traj_n_evaluations", "has_trajectory") else
           ("direct" if n == "traj_n_evaluations" else "computed")
           for n in _TRAJECTORY_NAMES)
    + ("computed", "computed", "computed", "computed", "computed"))
assert len(FEATURE_PROVENANCE_V2) == FEATURE_DIM_V2

_FAMILY_RE = re.compile(r"^(\d+)s_(none|miller|rc)$")


# ---------------------------------------------------------------------------
# topology structural features (Section 9)
# ---------------------------------------------------------------------------
def parse_family(family: str | None) -> dict:
    """Real, already-recorded structural facts (stage count + compensation
    topology) parsed from the family label -- never an arbitrary topology-ID
    embedding that would just memorize the 5 observed families."""
    if not family:
        return {"stage_count": 0.0, "comp_none": 0.0, "comp_miller": 0.0,
                "comp_rc": 0.0, "has_topology_family": 0.0}
    m = _FAMILY_RE.match(family)
    if not m:
        return {"stage_count": 0.0, "comp_none": 0.0, "comp_miller": 0.0,
                "comp_rc": 0.0, "has_topology_family": 0.0}
    stages, comp = m.groups()
    return {"stage_count": float(stages) / 5.0,   # normalized, matches other small scales
           "comp_none": 1.0 if comp == "none" else 0.0,
           "comp_miller": 1.0 if comp == "miller" else 0.0,
           "comp_rc": 1.0 if comp == "rc" else 0.0,
           "has_topology_family": 1.0}


# ---------------------------------------------------------------------------
# candidate-own measured margins (Section 6/7) -- from a REAL pre-selection
# sizing-loop row (mined candidates only; NEVER a final-verification result)
# ---------------------------------------------------------------------------
def candidate_measured_features(row: dict | None, spec: dict) -> dict:
    if row is None:
        return {n: 0.0 for n in _CANDIDATE_MEASURED_NAMES}
    gain, pm, ugbw = row.get("gain_db"), row.get("pm_deg"), row.get("ugbw_hz")
    from agentic_raptor.mb_sac.spec_sizing import postsizing_outcome
    stable = pm is not None and pm > 0     # RECONSTRUCTED, same precedent as elsewhere
    meas = {"gain_db": gain, "pm_deg": pm, "ugbw_hz": ugbw, "stable": stable,
           "stability": "verified_stable" if stable else "verified_unstable",
           "op_valid": gain is not None or pm is not None}
    outcome = postsizing_outcome(meas, spec)
    mv = outcome["margin_vector"]
    dist = outcome["normalized_distance_to_feasibility"]
    return {
        "measured_gain_margin": mv.get("gain_margin_db") / 20.0
        if mv.get("gain_margin_db") is not None else 0.0,
        "measured_pm_margin": mv.get("pm_margin_deg") / 45.0
        if mv.get("pm_margin_deg") is not None else 0.0,
        "measured_ugbw_margin": mv.get("ugbw_log_margin") / 2.0
        if mv.get("ugbw_log_margin") is not None else 0.0,
        "measured_distance_to_feasibility": dist if dist is not None else 1.0,
        "measured_stable_reconstructed": 1.0 if stable else 0.0,
        "measured_op_valid": 1.0 if meas["op_valid"] else 0.0,
        "has_candidate_measurement": 1.0,
    }


# ---------------------------------------------------------------------------
# branch-trajectory summary (Section 6) -- computed over the WHOLE completed
# branch run, exactly what would be available to Level 2 for that branch's
# winner (all sizing-loop measurements for the branch already exist by then)
# ---------------------------------------------------------------------------
def trajectory_summary(branch_rows: list | None, spec: dict) -> dict:
    if not branch_rows:
        return {n: 0.0 for n in _TRAJECTORY_NAMES}
    from agentic_raptor.mb_sac.spec_sizing import postsizing_outcome

    def _dist(r):
        gain, pm, ugbw = r.get("gain_db"), r.get("pm_deg"), r.get("ugbw_hz")
        stable = pm is not None and pm > 0
        meas = {"gain_db": gain, "pm_deg": pm, "ugbw_hz": ugbw, "stable": stable,
               "stability": "verified_stable" if stable else "verified_unstable",
               "op_valid": gain is not None or pm is not None}
        o = postsizing_outcome(meas, spec)
        return o["normalized_distance_to_feasibility"], o["exact_spec_pass"], meas

    dists, passes, valid_rows = [], [], []
    for r in branch_rows:
        d, ok, meas = _dist(r)
        dists.append(d if d is not None else 9.9)
        passes.append(ok)
        if meas["op_valid"]:
            valid_rows.append(r)

    ordered = sorted(branch_rows, key=lambda r: r.get("step", 0))
    start_dist = dists[branch_rows.index(ordered[0])] if ordered else 9.9
    final_dist = dists[branch_rows.index(ordered[-1])] if ordered else 9.9
    best_dist = min(dists) if dists else 9.9
    rewards = [r.get("reward") for r in branch_rows if r.get("reward") is not None]
    start_reward = ordered[0].get("reward") if ordered else None
    final_reward = ordered[-1].get("reward") if ordered else None
    stable_flags = [(r.get("pm_deg") is not None and r.get("pm_deg") > 0) for r in branch_rows]

    def _mean_std(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return 0.0, 0.0
        if len(vals) == 1:
            return vals[0], 0.0
        return st.mean(vals), st.pstdev(vals)

    gain_mean, gain_std = _mean_std([r.get("gain_db") for r in branch_rows])
    pm_mean, pm_std = _mean_std([r.get("pm_deg") for r in branch_rows])
    log_ugbw_vals = [math.log10(r["ugbw_hz"]) for r in branch_rows
                     if r.get("ugbw_hz") and r["ugbw_hz"] > 0]
    log_ugbw_mean, log_ugbw_std = _mean_std(log_ugbw_vals) if log_ugbw_vals else (0.0, 0.0)
    reward_mean, reward_std = _mean_std(rewards)
    dist_mean, dist_std = _mean_std(dists)

    n = len(branch_rows)
    return {
        "traj_best_distance": min(1.0, best_dist), "traj_start_distance": min(1.0, start_dist),
        "traj_final_distance": min(1.0, final_dist),
        "traj_absolute_improvement": max(-1.0, min(1.0, start_dist - final_dist)),
        "traj_relative_improvement": max(-1.0, min(1.0, (start_dist - final_dist)
                                                   / max(start_dist, 1e-3))),
        "traj_best_reward": max(rewards) if rewards else 0.0,
        "traj_final_reward": final_reward if final_reward is not None else 0.0,
        "traj_reward_improvement": ((final_reward - start_reward)
                                    if (final_reward is not None and start_reward is not None)
                                    else 0.0),
        "traj_n_evaluations": n / 32.0,   # normalized against the common budget
        "traj_valid_spice_fraction": len(valid_rows) / n if n else 0.0,
        "traj_stable_fraction_reconstructed": sum(stable_flags) / n if n else 0.0,
        "traj_constraint_satisfaction_frequency": sum(passes) / n if n else 0.0,
        "traj_gain_mean": gain_mean / 100.0, "traj_gain_std": gain_std / 100.0,
        "traj_pm_mean": pm_mean / 90.0, "traj_pm_std": pm_std / 90.0,
        "traj_log_ugbw_mean": log_ugbw_mean / 9.0, "traj_log_ugbw_std": log_ugbw_std / 9.0,
        "traj_reward_mean": reward_mean, "traj_reward_std": reward_std,
        "traj_distance_mean": min(1.0, dist_mean), "traj_distance_std": min(1.0, dist_std),
        "has_trajectory": 1.0,
    }


# ---------------------------------------------------------------------------
# unified builder
# ---------------------------------------------------------------------------
def features_v2(spec: dict, pred, candidate_row: dict | None,
                branch_rows: list | None, topology_family: str | None,
                arm: str = "combined") -> list:
    """arm: "predicted_only" (== old 11, zero-padded to V2 width) |
    "measured_only" (candidate + trajectory + topology, old 11 zeroed) |
    "combined" (everything)."""
    from agentic_raptor.ranking.model import features as features_v1

    v1 = dict(zip(_V1_NAMES, features_v1(spec, pred)))
    cand = candidate_measured_features(candidate_row, spec)
    traj = trajectory_summary(branch_rows, spec)
    topo = parse_family(topology_family)

    if arm == "predicted_only":
        cand = {n: 0.0 for n in _CANDIDATE_MEASURED_NAMES}
        traj = {n: 0.0 for n in _TRAJECTORY_NAMES}
        topo = {n: 0.0 for n in _TOPOLOGY_NAMES}
    elif arm == "measured_only":
        v1 = {n: 0.0 for n in _V1_NAMES}

    row = {**v1, **cand, **traj, **topo}
    return [row[n] for n in FEATURE_NAMES_V2]
