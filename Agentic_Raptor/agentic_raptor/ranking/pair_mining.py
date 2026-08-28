"""Stage 7.2A: mine additional DPO training pairs from candidates MB-SAC's
sizing loop already measured with real ngspice, instead of relying only on
each pipeline run's single winner-vs-winner comparison (record_pair(), one
pair per run despite up to ~32 real measurements per branch).

Not wired into the live pipeline by default -- see `harvest_ranker_pairs`
below. When it IS wired in, it must run AFTER both MB-SAC branches finish,
using their two completed sizing histories, exactly like the historical
offline mining path in this module.

Feature/leakage semantics (Section 12/13): a sizing-loop candidate is never
"future" information relative to the Level-2 decision -- MB-SAC finishes
BOTH branches' full real-SPICE budgets before Level 2 ever runs, so every
candidate in this module already existed before any selection happened.
What must NOT be used as a feature is the candidate's own MEASURED
electricals (that is exactly the leakage SurrogatePrediction/
AuthoritativeSpiceOutcome separation exists to prevent). Instead, a fresh
surrogate is trained on a branch-run's own measured trajectory (same
architecture/encoding as agentic_raptor.mb_sac.spec_sizing's internal
surrogate) and used to score each candidate -- reproducing exactly the
"predict from knobs alone" semantics the live selector's surrogate uses,
without needing the specific historical run's own saved checkpoint path
(not recorded in the persisted replay stream).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Section 10: canonical, sized-design-level identity
# ---------------------------------------------------------------------------
def knob_hash(knobs: dict) -> str:
    """Distinct knob settings on the SAME topology must never collapse into
    one candidate identity -- topology hash alone is insufficient."""
    canon = json.dumps({k: round(float(v), 6) for k, v in sorted(knobs.items())},
                       sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def candidate_id(topology_hash: str, knobs: dict) -> str:
    return f"{topology_hash}:{knob_hash(knobs)}"


def canonical_pair_id(spec_hash: str, candidate_id_a: str, candidate_id_b: str) -> tuple:
    return (spec_hash, *sorted((candidate_id_a, candidate_id_b)))


# ---------------------------------------------------------------------------
# Section 1: candidate provenance
# ---------------------------------------------------------------------------
REQUIRED_ENV_VERSION = "POST_CLOAD_FIX_V1"


def validate_candidate_provenance(row: dict) -> list:
    """Every reason this row is NOT mineable. Empty list = OK."""
    problems = []
    if row.get("electrical_environment_version") != REQUIRED_ENV_VERSION:
        problems.append("not_post_cload_fix_v1")
    if not row.get("knobs"):
        problems.append("missing_knobs")
    if row.get("gain_db") is None and row.get("pm_deg") is None:
        problems.append("no_real_measurement")
    if not row.get("topology_hash"):
        problems.append("missing_topology_hash")
    if not row.get("spec_hash"):
        problems.append("missing_spec_hash")
    if "requested_c_load_f" not in row:
        problems.append("missing_requested_c_load_f")
    return problems


def to_measurement_dict(row: dict) -> dict:
    """Shape expected by spec_sizing.postsizing_outcome/margin_vector.
    'stable'/'op_valid' are not present in the persisted replay stream, so
    they are approximated using the SAME precedent already established in
    agentic_raptor.mb_sac.spec_sizing._offline_pretrain for historical rows
    lacking these fields: a real measured PM/gain pair is treated as a
    convergent, functional operating point, and PM sign stands in for the
    explicit stability flag."""
    pm = row.get("pm_deg")
    gain = row.get("gain_db")
    stable = pm is not None and pm > 0
    return {"gain_db": gain, "pm_deg": pm, "ugbw_hz": row.get("ugbw_hz"),
           "power_w": None, "idd_a": None,
           "c_load_f": row.get("requested_c_load_f"),
           "stable": stable,
           "stability": "verified_stable" if stable else "verified_unstable",
           "electrical": "electrically_functional",
           "op_valid": gain is not None or pm is not None}


# ---------------------------------------------------------------------------
# Section 1: split an appended replay stream into TRUE per-branch runs
# ---------------------------------------------------------------------------
def load_runs_from_replay(replay_path: Path) -> list:
    """One real sac_size() call per returned run. A NEW run starts every
    time step==0 recurs within the same (spec_hash, seed, branch) key --
    the stream is append-only and the SAME nominal (spec_hash, seed) can be
    (and was, in this session) reused across multiple separate real
    pipeline invocations, so that key alone is not a unique run identity."""
    lines = replay_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["spec_hash"], r["seed"], r["branch"])].append(r)
    runs = []
    for key, seq in buckets.items():
        cur = []
        for r in seq:
            if r.get("step") == 0 and cur:
                runs.append({"spec_hash": key[0], "seed": key[1], "branch": key[2],
                            "rows": cur})
                cur = []
            cur.append(r)
        if cur:
            runs.append({"spec_hash": key[0], "seed": key[1], "branch": key[2],
                        "rows": cur})
    return runs


def pair_runs_by_spec_seed(runs: list) -> list:
    """Zips branch-A run i with branch-B run i (temporal/file order) within
    each (spec_hash, seed): both branches of ONE pipeline call are appended
    together (agentic_raptor.selfimprove_v2.streams.harvest_run builds both
    branches' trajectories from one `hv` and writes them in the same
    append() batch), so same-index A/B runs under one (spec_hash, seed)
    originate from the same real pipeline call."""
    by_key = defaultdict(lambda: {"A": [], "B": []})
    for run in runs:
        by_key[(run["spec_hash"], run["seed"])][run["branch"]].append(run)
    paired = []
    for (spec_hash, seed), d in by_key.items():
        a_runs, b_runs = d["A"], d["B"]
        for i in range(min(len(a_runs), len(b_runs))):
            paired.append({"originating_run_id": f"{spec_hash}:{seed}:{i}",
                           "spec_hash": spec_hash, "seed": seed, "run_index": i,
                           "A": a_runs[i], "B": b_runs[i]})
    return paired


# ---------------------------------------------------------------------------
# Section 7: representative candidate selection
# ---------------------------------------------------------------------------
def select_representative_candidates(branch_rows: list, spec: dict,
                                     max_candidates: int = 5) -> list:
    """Roles A-E (Section 7), deduplicated by candidate identity. A branch
    may contribute fewer than max_candidates if roles collapse onto the
    same underlying measured candidate."""
    from agentic_raptor.mb_sac.spec_sizing import postsizing_outcome

    valid = [r for r in branch_rows if not validate_candidate_provenance(r)]
    if not valid:
        return []
    scored = []
    for r in valid:
        meas = to_measurement_dict(r)
        outcome = postsizing_outcome(meas, spec)
        scored.append((r, outcome))

    def _cid(r):
        return candidate_id(r["topology_hash"], r["knobs"])

    picks = {}
    # A. winner (max reward)
    winner = max(scored, key=lambda t: t[0].get("reward", float("-inf")))
    picks.setdefault(_cid(winner[0]), ("winner", winner))
    # B/C. best feasible / closest to feasible (both = min distance; same
    # role by construction under one metric, listed separately per the
    # spec's naming, deduplicated automatically if identical)
    def _dist(t):
        d = t[1]["normalized_distance_to_feasibility"]
        return d if d is not None else 9.9
    closest = min(scored, key=_dist)
    picks.setdefault(_cid(closest[0]), ("closest_to_feasible", closest))
    # D. hard negative: clearly worse (max distance)
    worst = max(scored, key=_dist)
    picks.setdefault(_cid(worst[0]), ("hard_negative", worst))
    # E. electrically diverse: maximize min margin-vector distance from
    # already-picked candidates (real diversity metric over real margins,
    # not a fabricated one)
    def _margin_vec(t):
        mv = t[1]["margin_vector"]
        return [mv.get("gain_margin_db") or 0.0, mv.get("pm_margin_deg") or 0.0,
               mv.get("ugbw_log_margin") or 0.0]
    remaining = [t for t in scored if _cid(t[0]) not in picks]
    while remaining and len(picks) < max_candidates:
        chosen_vecs = [_margin_vec(v[1]) for v in picks.values()]
        def _min_dist_to_chosen(t):
            v = _margin_vec(t)
            return min(sum((a - b) ** 2 for a, b in zip(v, cv)) ** 0.5
                      for cv in chosen_vecs) if chosen_vecs else 0.0
        diverse = max(remaining, key=_min_dist_to_chosen)
        picks[_cid(diverse[0])] = ("diverse", diverse)
        remaining = [t for t in remaining if _cid(t[0]) != _cid(diverse[0])]
    return [{"role": role, "row": r, "outcome": outcome, "candidate_id": cid}
           for cid, (role, (r, outcome)) in picks.items()]


# ---------------------------------------------------------------------------
# Section 12: reconstruct a surrogate faithfully from a branch-run's own
# measured trajectory (no new SPICE; reproduces the live selector's own
# "surrogate trained on this run's real measurements" semantics)
# ---------------------------------------------------------------------------
def train_run_surrogate(branch_rows: list, seed: int = 0):
    """Returns a torch surrogate net trained on ONE branch-run's own real
    (knobs -> pm/gain/ugbw) measurements -- same architecture and encoding
    as agentic_raptor.mb_sac.spec_sizing's internal surrogate."""
    import torch

    from agentic_raptor.mb_sac.spec_sizing import (KNOB_HI, KNOB_NAMES,
                                                    N_KNOBS, _ugbw_target)
    valid = [r for r in branch_rows if not validate_candidate_provenance(r)
            and r.get("pm_deg") is not None]
    if len(valid) < 4:
        return None
    torch.manual_seed(seed)
    hi = torch.tensor(KNOB_HI)
    net = torch.nn.Sequential(torch.nn.Linear(N_KNOBS, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 3))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    xs, ys = [], []
    for r in valid:
        kn = torch.tensor([float(r["knobs"][k]) for k in KNOB_NAMES]) / hi
        y = torch.tensor([r["pm_deg"] / 90, (r.get("gain_db") or 0) / 100,
                          _ugbw_target(r.get("ugbw_hz"))])
        xs.append(kn); ys.append(y)
    X, Y = torch.stack(xs), torch.stack(ys)
    for _ in range(300):
        loss = ((net(X) - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def predict_with_run_surrogate(net, knobs: dict, spec: dict, topology_hash: str,
                               sizing_manifest_hash: str, tmp_dir: Path):
    """Saves `net` to a temp checkpoint and calls the SAME predict_post_sac()
    the live selector uses, so mined-pair predictions and live predictions
    go through identical code -- not a parallel reimplementation."""
    import torch

    from agentic_raptor.ranking.surrogate import predict_post_sac
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_dir / f"mined_surrogate_{knob_hash(knobs)}.pt"
    torch.save(net.state_dict(), ckpt)
    return predict_post_sac(spec, topology_hash=topology_hash,
                            topology_family="mined", sizing_vector=knobs,
                            sizing_manifest_hash=sizing_manifest_hash,
                            surrogate_path=str(ckpt))


# ---------------------------------------------------------------------------
# Section 4/9: pair classification (frozen preference hierarchy, unchanged)
# ---------------------------------------------------------------------------
BOTH_INFEASIBLE_CLOSE_THRESHOLD = 0.2


def classify_pair(outcome_a: dict, outcome_b: dict) -> str:
    fa, fb = outcome_a["exact_spec_pass"], outcome_b["exact_spec_pass"]
    if fa and fb:
        return "both_feasible"
    if fa != fb:
        return "one_feasible_one_infeasible"
    da = outcome_a["normalized_distance_to_feasibility"]
    db = outcome_b["normalized_distance_to_feasibility"]
    close = (da is not None and db is not None
            and max(da, db) < BOTH_INFEASIBLE_CLOSE_THRESHOLD)
    return "both_infeasible_close" if close else "both_infeasible_far"


def classify_ranker_authority(pred_a, pred_b) -> bool:
    from agentic_raptor.ranking.post_sac import hard_safety_tier
    return hard_safety_tier(pred_a) == hard_safety_tier(pred_b)


def measured_preference_from_outcomes(meas_a: dict, meas_b: dict, spec: dict) -> tuple:
    """Same frozen hierarchy as agentic_raptor.ranking.post_sac.
    measured_preference, applied directly to two measurement dicts (mined
    candidates don't have AuthoritativeSpiceOutcome objects, only the
    postsizing_outcome() shape)."""
    def k(o):
        mv = o["margin_vector"]
        return (0 if o["exact_spec_pass"] else 1,
                0 if mv["operating_point_valid"] else 1,
                0 if o["passes"]["stable"] else 1,
                o["normalized_distance_to_feasibility"]
                if o["normalized_distance_to_feasibility"] is not None else 9.9,
                -(o["hard_constraints_passed"] or 0))
    from agentic_raptor.mb_sac.spec_sizing import postsizing_outcome
    oa = postsizing_outcome(meas_a, spec)
    ob = postsizing_outcome(meas_b, spec)
    ka, kb = k(oa), k(ob)
    if ka == kb:
        return None, "tie"
    return ("A" if ka < kb else "B"), "measured_hierarchy"


# ---------------------------------------------------------------------------
# Section 9: cross-branch pair mining (primary mining entry point)
# ---------------------------------------------------------------------------
def mine_cross_branch_pairs(paired_run: dict, spec: dict, tmp_dir: Path,
                            excluded_context_ids: set,
                            max_candidates_per_branch: int = 5) -> list:
    """One paired_run (from pair_runs_by_spec_seed) -> a list of mined pair
    dicts with full provenance (Section 16). Returns [] and does not mine
    anything if the spec is protected (Section 4/26)."""
    spec_id = spec.get("spec_id")
    if spec_id in excluded_context_ids:
        return []

    a_reps = select_representative_candidates(
        paired_run["A"]["rows"], spec, max_candidates_per_branch)
    b_reps = select_representative_candidates(
        paired_run["B"]["rows"], spec, max_candidates_per_branch)
    if not a_reps or not b_reps:
        return []

    net_a = train_run_surrogate(paired_run["A"]["rows"])
    net_b = train_run_surrogate(paired_run["B"]["rows"])
    if net_a is None or net_b is None:
        return []

    # predictions computed ONCE per representative candidate, not once per
    # pair -- the same candidate (e.g. branch A's "winner") can appear in
    # several cross-branch combinations. Stored as asdict() dicts, not the
    # frozen SurrogatePrediction instances, so this pair dict round-trips
    # through json.dumps/loads without silently degrading to str(obj).
    from dataclasses import asdict
    pred_cache_a = {ra["candidate_id"]: asdict(predict_with_run_surrogate(
        net_a, ra["row"]["knobs"], spec, ra["row"]["topology_hash"],
        sizing_manifest_hash=f"mined:{paired_run['originating_run_id']}:A",
        tmp_dir=tmp_dir)) for ra in a_reps}
    pred_cache_b = {rb["candidate_id"]: asdict(predict_with_run_surrogate(
        net_b, rb["row"]["knobs"], spec, rb["row"]["topology_hash"],
        sizing_manifest_hash=f"mined:{paired_run['originating_run_id']}:B",
        tmp_dir=tmp_dir)) for rb in b_reps}

    pairs = []
    seen_ids = set()
    for ra in a_reps:
        for rb in b_reps:
            row_a, row_b = ra["row"], rb["row"]
            cid_a, cid_b = ra["candidate_id"], rb["candidate_id"]
            if row_a["topology_hash"] == row_b["topology_hash"]:
                continue     # never a valid DPO pair (compare() forbids it)
            pair_id = canonical_pair_id(paired_run["spec_hash"], cid_a, cid_b)
            if pair_id in seen_ids:
                continue
            seen_ids.add(pair_id)

            pred_a, pred_b = pred_cache_a[cid_a], pred_cache_b[cid_b]
            meas_a, meas_b = to_measurement_dict(row_a), to_measurement_dict(row_b)
            winner, reason = measured_preference_from_outcomes(meas_a, meas_b, spec)
            if winner is None:
                continue     # tie -- not a usable training pair

            pairs.append({
                "pair_id": pair_id, "originating_run_id": paired_run["originating_run_id"],
                "spec_hash": paired_run["spec_hash"], "spec_id": spec_id,
                "spec": spec,
                "candidate_id_a": cid_a, "candidate_id_b": cid_b,
                "topology_hash_a": row_a["topology_hash"],
                "topology_hash_b": row_b["topology_hash"],
                "knobs_a": row_a["knobs"], "knobs_b": row_b["knobs"],
                "knob_hash_a": knob_hash(row_a["knobs"]),
                "knob_hash_b": knob_hash(row_b["knobs"]),
                "family_a": row_a.get("family"), "family_b": row_b.get("family"),
                "role_a": ra["role"], "role_b": rb["role"],
                "outcome_a": ra["outcome"], "outcome_b": rb["outcome"],
                "prediction_a": pred_a, "prediction_b": pred_b,
                "authoritative_winner": winner, "authoritative_reason": reason,
                "electrical_environment_version": REQUIRED_ENV_VERSION,
                "requested_c_load_f": row_a.get("requested_c_load_f"),
                "cload_equal_ab": (row_a.get("requested_c_load_f")
                                  == row_b.get("requested_c_load_f")),
                "source": "mined_cross_branch",
            })
    return pairs


def deduplicate_pairs(pairs: list) -> tuple:
    """Order-independent dedup by canonical_pair_id. Returns (kept, n_dupes)."""
    seen = {}
    for p in pairs:
        pid = p["pair_id"]
        if pid not in seen:
            seen[pid] = p
    return list(seen.values()), len(pairs) - len(seen)
