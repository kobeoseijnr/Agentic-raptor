"""Case C, Steps 3-6: rebuild the proposer corpus so five candidates exist.

Measured root cause (baseline_diversity.json): every one of the 85 training
prompts has exactly ONE target, and 85/85 match the deterministic rule

    stages = 1 if gain < 30 else 2 if gain < 70 else 3
    comp   = none / rc / miller by load and phase margin

The training distribution is therefore a delta per specification. The model
reproduced it perfectly (100% spec-match) and consequently cannot propose
alternatives -- 20 attempts at T=1.5 yielded at most 2 distinct graphs. No
decoding change recovers diversity the data never contained.

This writes a NEW corpus beside the original; it does not mutate
artifacts/stage3e4/corpus.json. The frozen exam, its hash and every historical
result therefore stay reproducible, and the repaired corpus can be compared
against the baseline rather than replacing it silently.

Steps implemented here:
  3  drop the stage-tier rule as a hard target generator
  4  multiple distinct realizable targets per specification
  5  balance by family and by canonical graph, with a dominance cap
  6  exclusion-conditioned examples: "given these, propose a different one"

Run:  python build_diverse_corpus.py [--targets 5] [--max-family-share 0.30]
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.llm_dpo.stage3e4 import variant_hash, variant_text

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/publication_v2/proposer_repair"
SRC = ROOT / "artifacts/stage3e4/corpus.json"
#: the realizable structure space. Step 2's family-reference gate decides
#: which of these may actually be used as targets.
ALL_FAMILIES = ["2s_none", "2s_miller", "2s_rc", "3s_miller", "3s_rc"]
EXCLUDE_LINE = "### EXCLUDE "
#: run_family_spec_gate_bufsfb.py's output -- a SECOND structural axis
#: (output_buffer, local_feedback) that target_for() used to hard-code to
#: (False, False) for every family. Optional: if this file doesn't exist
#: yet (the gate hasn't been run), realizable_variants() below falls back
#: to exactly today's behaviour -- one (family, False, False) target per
#: realizable family -- so this script keeps working unchanged until the
#: gate is actually run.
EXTENDED_GATE = (ROOT / "artifacts/publication_v2/family_spec_gate"
                 / "SUMMARY_bufsfb.json")


def realizable_families() -> tuple:
    """Families with a verified known-feasible reference (Step 2 gate).

    Falls back to ALL_FAMILIES when the gate has not run, but says so: a
    proposer must never be trained to emit structures the downstream mapper
    or testbench cannot realise.

    Source of truth is the per-(family, spec) gate. The older median-spec
    gate is over-strict -- it marks a family unrealizable when it cannot meet
    the MIDDLE of the specs the corpus pairs it with, which excluded families
    that legitimately serve the easier half of their range -- and its stored
    report predates the 3s_none -> 2s_rc swap. Read it only as a fallback.
    """
    gate = ROOT / "artifacts/publication_v2/family_spec_gate/SUMMARY.json"
    if gate.is_file():
        d = json.loads(gate.read_text(encoding="utf-8"))
        ok = d.get("realizable_for_some_spec") or []
        # a family the gate never probed must not be silently admitted
        ok = [f for f in ok if f in ALL_FAMILIES]
        if ok:
            return tuple(ok), f"verified by family_spec_gate ({len(ok)}/5)"
    f = OUT / "family_reference_report.json"
    if not f.is_file():
        alt = ROOT / "artifacts/publication_v2/family_reference/SUMMARY.json"
        f = alt if alt.is_file() else None
    if f is None:
        return tuple(ALL_FAMILIES), "UNVERIFIED (family gate not run)"
    d = json.loads(f.read_text(encoding="utf-8"))
    ok = d.get("resolved") or [k for k, v in (d.get("families") or {}).items()
                               if v.get("feasible")]
    ok = [x for x in ok if x in ALL_FAMILIES]
    return tuple(ok or ALL_FAMILIES), ("verified (median-spec gate, "
                                       "fallback)" if ok else "gate empty")


def family_of(stages: int, comp: str) -> str:
    comp = {"miller_cap": "miller", "rc_nulling": "rc"}.get(comp, comp)
    return f"{stages}s_{comp}"


def target_for(fam: str, buf: bool = False, fb: bool = False) -> tuple:
    """Canonical response text + full-graph hash for a (family, buf, fb)."""
    stages, comp = int(fam[0]), fam.split("_", 1)[1]
    text = variant_text(stages, comp, buf, fb)
    return text, variant_hash(json.loads(text))


def realizable_variants(fams) -> list:
    """(family, buf, fb) triples this corpus may target.

    Always includes (fam, False, False) for every realizable family -- the
    proven baseline every prior run and comparison used. Adds (fam, buf, fb)
    for combos run_family_spec_gate_bufsfb.py has verified realizable, if
    that gate has been run; otherwise this is identical to the old
    family-only target space.
    """
    variants = [(f, False, False) for f in fams]
    if EXTENDED_GATE.is_file():
        d = json.loads(EXTENDED_GATE.read_text(encoding="utf-8"))
        for key in d.get("realizable_combos", []):
            combo = (d.get("combos") or {}).get(key) or {}
            f, buf, fb = combo.get("family"), combo.get("buf"), combo.get("fb")
            if f in fams and (f, buf, fb) not in variants:
                variants.append((f, buf, fb))
    return variants


def soft_family_order(spec: dict, variants) -> list:
    """Step 3: specification INFLUENCES order, it never determines one answer.

    The old rule emitted a single family. This ranks plausibility -- more
    stages for higher gain, compensation for tighter phase margin -- but
    every realizable family stays a legitimate target, because the stage-rule
    battery measured the hard version false (its only exact pass was a
    2s_none design the rule rejected).

    variants are now (family, buf, fb) triples, not bare family strings.
    Primary sort key is variant_rank (0 = proven family baseline, 1 = one
    structural addition, 2 = both) so at the current default of 5 targets,
    every family's (buf=False, fb=False) baseline is chosen BEFORE any
    buf/fb bonus variant -- identical output to before this axis existed.
    Bonus variants only get chosen once --targets is raised past 5.
    """
    gain = float(spec.get("gain_target_db") or 0.0)
    pm = float(spec.get("phase_margin_target_deg") or 45.0)

    def family_score(fam):
        stages, comp = int(fam[0]), fam.split("_", 1)[1]
        s = 0.0
        s += 0.25 * (stages - 2) * max(-1.0, min(1.0, (gain - 70.0) / 40.0))
        if comp != "none":
            s += 0.25 * max(-1.0, min(1.0, (pm - 45.0) / 20.0))
        return -s
    return sorted(variants, key=lambda v: (int(v[1]) + int(v[2]),
                                          family_score(v[0]), v[0], v[1], v[2]))


def build(n_targets: int, max_share: float, with_exclusions: bool) -> dict:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    fams, fam_status = realizable_families()
    fams = [f for f in ALL_FAMILIES if f in fams]
    variants = realizable_variants(fams)
    n_targets = min(n_targets, len(variants))

    train = [r for r in src["records"] if r["split"] == "train"]
    before = Counter(f"{r['stages']}s_{r['comp']}" for r in train)

    # global usage counter drives the dominance cap (Step 5)
    used = Counter()
    cap = None
    records, excl_records = [], []
    for r in train:
        spec = ig.parse_spec(r["prompt"])
        if not spec:
            continue
        order = soft_family_order(spec, variants)
        chosen, seen_hashes = [], set()
        for fam, buf, fb in order:
            if len(chosen) >= n_targets:
                break
            text, h = target_for(fam, buf, fb)
            if h in seen_hashes:            # full-graph identity, not signature
                continue
            seen_hashes.add(h)
            chosen.append((fam, buf, fb, text, h))
        # rebalance: if a family is over its cap, prefer the least-used ones
        if cap is not None:
            chosen.sort(key=lambda c: used[c[0]])
        for rank, (fam, buf, fb, text, h) in enumerate(chosen):
            used[fam] += 1
            records.append({
                "context_id": r["context_id"],
                "topology_id": r["topology_id"],
                "prompt": r["prompt"], "response": text,
                "variant_hash": h,
                "topology_signature": fam,
                "canonical_graph_hash": h,
                "stages": int(fam[0]), "comp": fam.split("_", 1)[1],
                "buffer": buf, "fb": fb,
                "split": "train", "target_rank": rank,
                "target_source": "multi_target_soft_prior"})
        # Step 6: exclusion-conditioned examples -- the model must learn
        # "given these, give me a DIFFERENT one", which is what makes five
        # sequential draws distinct rather than lucky
        if with_exclusions:
            for k in range(1, len(chosen)):
                excluded = chosen[:k]
                nxt_fam, nxt_buf, nxt_fb, nxt_text, nxt_h = chosen[k]
                ex = ", ".join(f"{f}/{hh[:12]}" for f, _b, _fb, _t, hh in excluded)
                prompt = r["prompt"].replace(
                    "### PROPOSAL",
                    f"{EXCLUDE_LINE}{ex}\n### PROPOSAL")
                excl_records.append({
                    "context_id": r["context_id"],
                    "topology_id": r["topology_id"],
                    "prompt": prompt, "response": nxt_text,
                    "variant_hash": nxt_h,
                    "topology_signature": nxt_fam,
                    "canonical_graph_hash": nxt_h,
                    "stages": int(nxt_fam[0]),
                    "comp": nxt_fam.split("_", 1)[1],
                    "buffer": nxt_buf, "fb": nxt_fb,
                    "split": "train", "excluded_count": k,
                    "target_source": "exclusion_conditioned"})

    all_records = records + excl_records
    after = Counter(r["topology_signature"] for r in all_records)
    total = sum(after.values()) or 1
    shares = {k: v / total for k, v in after.items()}
    # Stage 1.6: this doc does no NEW electrical measurement itself -- it
    # only reads family_spec_gate's verdicts -- so it inherits THAT gate's
    # environment version rather than claiming its own. A corpus built from
    # a still-PRE_CLOAD_FIX gate must say so, not silently look current.
    gate_path = ROOT / "artifacts/publication_v2/family_spec_gate/SUMMARY.json"
    gate_env = "UNKNOWN"
    if gate_path.is_file():
        gate_env = json.loads(gate_path.read_text(encoding="utf-8")).get(
            "electrical_environment_version", "PRE_CLOAD_FIX")
    doc = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_corpus": str(SRC.relative_to(ROOT)),
        "source_corpus_hash": src.get("split_manifest", {}).get("corpus_hash"),
        "family_gate_environment_version": gate_env,
        "realizable_families": list(fams),
        "family_gate_status": fam_status,
        "realizable_variants": [f"{f}_buf{int(b)}_fb{int(k)}" for f, b, k in variants],
        "extended_gate_used": EXTENDED_GATE.is_file(),
        "targets_per_spec": n_targets,
        "exclusion_conditioned": with_exclusions,
        "counts": {"specs": len({r['context_id'] for r in train}),
                   "base_targets": len(records),
                   "exclusion_examples": len(excl_records),
                   "total": len(all_records)},
        "distribution_before": dict(before),
        "distribution_after": dict(after),
        "family_share_after": {k: round(v, 4) for k, v in shares.items()},
        "max_family_share": round(max(shares.values()), 4) if shares else 0,
        "dominance_cap": max_share,
        "dominance_ok": (max(shares.values()) <= max_share
                         if shares else False),
        "distinct_canonical_graphs": len(
            {r["canonical_graph_hash"] for r in all_records}),
        "records": all_records}
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=5,
                    help="raise past 5 to actually draw on buf/fb bonus "
                         "variants once run_family_spec_gate_bufsfb.py has "
                         "been run -- at 5 (the default) only each family's "
                         "proven buf=False,fb=False baseline is used, "
                         "identical to before that gate existed")
    ap.add_argument("--max-family-share", type=float, default=0.30)
    ap.add_argument("--no-exclusions", action="store_true")
    args = ap.parse_args()
    doc = build(args.targets, args.max_family_share, not args.no_exclusions)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "corpus_diverse.json").write_text(
        json.dumps(doc, indent=1), encoding="utf-8")
    print(f"realizable families : {doc['realizable_families']} "
          f"({doc['family_gate_status']})")
    print(f"realizable variants : {len(doc['realizable_variants'])} "
          f"(extended buf/fb gate {'FOUND' if doc['extended_gate_used'] else 'not run yet -- family-only'})")
    print(f"targets per spec    : {doc['targets_per_spec']}")
    print(f"specs               : {doc['counts']['specs']}")
    print(f"base targets        : {doc['counts']['base_targets']}")
    print(f"exclusion examples  : {doc['counts']['exclusion_examples']}")
    print(f"total records       : {doc['counts']['total']}")
    print(f"distinct graphs     : {doc['distinct_canonical_graphs']}")
    print("\nfamily distribution BEFORE (1 target/spec):")
    for k, v in sorted(doc["distribution_before"].items()):
        print(f"  {k:12} {v:4}")
    print("\nfamily distribution AFTER:")
    for k, v in sorted(doc["distribution_after"].items()):
        print(f"  {k:12} {v:4}  ({doc['family_share_after'][k]:.0%})")
    print(f"\nmax family share    : {doc['max_family_share']:.0%} "
          f"(cap {args.max_family_share:.0%}) -> "
          f"{'OK' if doc['dominance_ok'] else 'EXCEEDS CAP'}")
    print(f"written -> "
          f"{(OUT / 'corpus_diverse.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
