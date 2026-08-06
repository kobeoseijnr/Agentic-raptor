"""Frozen per-seed ELECTRICAL evaluation sets (publication v2, Fix 1).

Why this exists. The campaign design sampler seeds on the generation index:

    random.Random(seed * 1000 + 100 + generation)

so every generation designs a DIFFERENT set of specs. Measured consequence
across the 8-campaign run: generation 2 drew specs averaging 86.8 dB gain and
54.8 deg PM with 65% demanding PM >= 55 deg, against 65.0 dB / 47.6 deg and
16% at generation 0. Generation 2 returned zero exact passes -- but that is
confounded with task difficulty and CANNOT be read as a regression.

The repair keeps training contexts fresh (they are training data, and reuse
would encourage memorisation) while adding a SEPARATE evaluation set that is
frozen per seed:

    eval_rng = random.Random(seed * 1000 + 900)

The same specs are then evaluated at every generation, for every checkpoint,
under identical sizing configuration -- so a generational claim finally has a
like-for-like basis.

Leakage discipline: evaluation specs are excluded from SFT examples, DPO
pairs, RAG evidence, L4 memory, surrogate/ranker training, value targets used
for checkpoint selection, and self-earned examples. `excluded_context_ids()`
is the single source of truth; `assert_no_leakage()` enforces it.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from agentic_raptor.llm_dpo import integrity as ig

_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = _ROOT / "artifacts/publication_v2/evaluation_sets"
CORPUS = _ROOT / "artifacts/stage3e4/corpus.json"

#: evaluation specs per seed. Small enough to size every generation for every
#: checkpoint inside a campaign, large enough to separate arms.
EVAL_N = 12
#: offset that separates the evaluation stream from the training stream
EVAL_SEED_OFFSET = 900
TRAIN_SEED_OFFSET = 100


def _content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def feasibility_split(spec: dict) -> str:
    """A / B / C by how hard the binding constraints are.

    Phase margin leads because it is the constraint that actually fails: in
    the 40-circuit stage-rule battery every 3-stage candidate landed between
    -20 and -0.5 deg of margin.
    """
    pm = spec.get("phase_margin_target_deg") or 0.0
    gain = spec.get("gain_target_db") or 0.0
    cl = spec.get("load_capacitance_pf") or 0.0
    hard = (pm >= 55.0) + (gain >= 90.0) + (cl <= 200.0)
    return "C" if hard >= 2 else "B" if hard == 1 else "A"


def difficulty(spec: dict) -> dict:
    return {"pm_target_deg": spec.get("phase_margin_target_deg"),
            "gain_target_db": spec.get("gain_target_db"),
            "load_capacitance_pf": spec.get("load_capacitance_pf"),
            "ugbw_target_hz": spec.get("ugbw_target_hz"),
            "pm_is_tight": bool((spec.get("phase_margin_target_deg") or 0)
                                >= 55.0),
            "gain_is_high": bool((spec.get("gain_target_db") or 0) >= 90.0),
            "load_is_light": bool((spec.get("load_capacitance_pf") or 0)
                                  <= 200.0)}


def build(seed: int, n: int = EVAL_N, overwrite: bool = False) -> dict:
    """Generate (once) and persist the frozen evaluation set for `seed`."""
    out = EVAL_DIR / f"seed_{seed}.json"
    if out.is_file() and not overwrite:
        return load(seed)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    # Drawn from HELDOUT, not train. Sampling from train would have cost 44 of
    # 85 training records once context-id collisions are accounted for -- a
    # 52% cut to the SFT corpus, which is a training-policy change this repair
    # stage is explicitly not allowed to make. The heldout split is already
    # excluded from SFT, DPO, RAG evidence and self-earned examples by
    # construction, so the electrical evaluation set inherits every existing
    # leakage guard at zero cost to training. `blindtest` stays untouched for
    # a final one-shot report.
    pool = [r for r in corpus["records"] if r["split"] == "heldout"]
    rng = random.Random(seed * 1000 + EVAL_SEED_OFFSET)
    picked = rng.sample(pool, min(n, len(pool)))
    specs = []
    for i, r in enumerate(picked):
        sp = ig.parse_spec(r["prompt"])
        if not sp:
            continue
        rec = {
            "eval_spec_id": f"E{seed}_{i:03d}",
            "context_id": r["context_id"],
            "topology_id": r["topology_id"],
            "prompt": r["prompt"],
            "gain_target_db": sp["gain_target_db"],
            "phase_margin_target_deg": sp["phase_margin_target_deg"],
            "ugbw_target_hz": sp["ugbw_target_hz"],
            "load_capacitance_pf": sp["load_capacitance_pf"],
            # the corpus specifies no power or area budget; recorded
            # explicitly as null rather than silently omitted so the
            # constraint export can say "not applicable" instead of "passed"
            "power_target_w": None,
            "area_target_um2": None,
            "supply_voltage": ig.EVAL_ENV.get("supply_voltage", 1.8),
            "technology": sp.get("technology"),
            "topology_constraints": {
                "allowed_families": None,   # soft priors only (Fix 2)
                "forbidden": ["raw_netlist", "feedback_to_input"]},
            "feasibility_split": feasibility_split(sp),
            "difficulty": difficulty(sp),
            "evaluation_context_id": ig.evaluation_context_id(sp)}
        rec["content_hash"] = _content_hash(
            {k: v for k, v in rec.items() if k != "content_hash"})
        specs.append(rec)
    doc = {"seed": seed, "n": len(specs),
           "sampler": f"random.Random({seed}*1000+{EVAL_SEED_OFFSET})",
           "source_split": "heldout",
           "corpus_hash": corpus.get("split_manifest", {}).get("corpus_hash"),
           "testbench_hash": ig.TESTBENCH_HASH,
           "eval_env": ig.EVAL_ENV,
           "specs": specs,
           "split_counts": {s: sum(1 for x in specs
                                   if x["feasibility_split"] == s)
                            for s in ("A", "B", "C")}}
    doc["set_hash"] = _content_hash(doc["specs"])
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def load(seed: int) -> dict:
    """Load and VERIFY the frozen set; a changed hash is a hard error."""
    out = EVAL_DIR / f"seed_{seed}.json"
    if not out.is_file():
        raise FileNotFoundError(
            f"no frozen evaluation set for seed {seed}; run "
            f"python -m agentic_raptor.publication.eval_sets --build")
    doc = json.loads(out.read_text(encoding="utf-8"))
    live = _content_hash(doc["specs"])
    if live != doc["set_hash"]:
        raise ValueError(f"evaluation set seed {seed} changed on disk: "
                         f"{doc['set_hash']} -> {live}")
    for s in doc["specs"]:
        h = _content_hash({k: v for k, v in s.items() if k != "content_hash"})
        if h != s["content_hash"]:
            raise ValueError(f"spec {s['eval_spec_id']} mutated: "
                             f"{s['content_hash']} -> {h}")
    return doc


def available_seeds() -> list:
    if not EVAL_DIR.is_dir():
        return []
    return sorted(int(p.stem.split("_")[1])
                  for p in EVAL_DIR.glob("seed_*.json"))


def excluded_context_ids(seeds=None) -> set:
    """THE single source of truth for what training must never touch.

    Union across seeds: a campaign at seed 11 must not train on seed 23's
    evaluation specs either, or cross-seed comparisons stop being clean.
    """
    out = set()
    for s in (available_seeds() if seeds is None else seeds):
        try:
            doc = load(s)
        except (FileNotFoundError, ValueError):
            continue
        for spec in doc["specs"]:
            out.add(spec["context_id"])
    return out


def assert_no_leakage(context_ids, where: str, seeds=None):
    """Raise if any evaluation spec reached a training consumer."""
    bad = set(context_ids) & excluded_context_ids(seeds)
    if bad:
        raise AssertionError(
            f"EVALUATION LEAKAGE in {where}: {len(bad)} frozen evaluation "
            f"context(s) entered training data -- {sorted(bad)[:5]}")
    return True


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seeds", default="11,23,47")
    ap.add_argument("--n", type=int, default=EVAL_N)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    for s in seeds:
        doc = build(s, n=a.n, overwrite=a.overwrite) if a.build else load(s)
        print(f"seed {s}: n={doc['n']} set_hash={doc['set_hash']} "
              f"splits={doc['split_counts']} -> "
              f"{(EVAL_DIR / f'seed_{s}.json').relative_to(_ROOT)}")
    ex = excluded_context_ids(seeds)
    print(f"excluded context ids (union): {len(ex)}")


if __name__ == "__main__":
    main()
