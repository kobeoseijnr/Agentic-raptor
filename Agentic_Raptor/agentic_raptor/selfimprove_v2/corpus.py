"""The measurement-grounded v2 proposer corpus.

    canonical template examples
  + VERIFIED successful post-ngspice topology graphs
  + exclusion-conditioned diversity examples

The template half teaches what is structurally VALID. The measured half
teaches what actually WORKED. Before this, the proposer only ever saw
templates, so it was a faithful imitator of a rulebook with no idea which
entry survived ngspice.

Two rules keep the measured half honest:

1. A single A/B result is NOT a target. One run is one sample of a stochastic
   sizer -- `sac_size` is not reproducible at a fixed seed (measured: three
   identical calls gave UGBW 6.9k / 20.0k / 8.2k). Targets are aggregated
   across repeated equal-budget results and admitted only on a consistent
   record.

2. Failures never become positive targets. They are already routed to RAG and
   to the preference queue, where "this did not work" is the useful signal.
   Training the proposer to emit a structure that failed would be teaching it
   the wrong lesson from the right data.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from agentic_raptor.llm_dpo.stage3e4 import variant_hash, variant_text

EXCLUDE_LINE = "### EXCLUDE "


def _canon_text(obj) -> str:
    """Canonical response text for the FAMILY this object realises.

    The proposer is trained to emit canonical structures, so a measured
    success is credited to its family rather than to the exact byte string
    the model happened to produce.
    """
    from agentic_raptor.llm_dpo import integrity as ig
    stages = len(obj.get("stages") or [])
    comp = ig.compensation_class(obj)
    return variant_text(stages, comp, bool(obj.get("output_buffer")),
                        bool(obj.get("local_feedback")))


def aggregate_sft_targets(rows: list, *, min_runs: int = 2,
                          min_pass_rate: float = 1.0,
                          equal_budget: bool = True) -> dict:
    """Aggregate admitted SFT-queue rows into (spec, family) targets.

    A (spec, family) pair becomes a target only when it was measured at least
    `min_runs` times and passed at least `min_pass_rate` of them. With
    equal_budget, runs at differing sizing budgets are grouped separately --
    a structure that passes at budget 64 has not demonstrated anything about
    budget 16, and pooling them would credit the wrong configuration.
    """
    groups = defaultdict(list)
    for r in rows:
        if not r.get("admitted"):
            continue
        key = (r.get("generation_spec_id"), r.get("family"),
               r.get("budget") if equal_budget else None)
        groups[key].append(r)

    targets, rejected = [], []
    for (spec_id, family, budget), rs in sorted(
            groups.items(), key=lambda kv: str(kv[0])):
        n = len(rs)
        n_pass = sum(1 for r in rs if r.get("exact_spec_pass"))
        rate = n_pass / n if n else 0.0
        rec = {"spec_id": spec_id, "family": family, "budget": budget,
               "runs": n, "passes": n_pass, "pass_rate": round(rate, 3),
               "seeds": sorted({r.get("seed") for r in rs}),
               "graph_hashes": sorted({r.get("canonical_graph_hash")
                                       for r in rs})}
        if n < min_runs:
            rec["rejected_because"] = f"insufficient_runs:{n}<{min_runs}"
            rejected.append(rec)
        elif rate < min_pass_rate:
            rec["rejected_because"] = (f"pass_rate_below_threshold:"
                                       f"{rate:.2f}<{min_pass_rate}")
            rejected.append(rec)
        else:
            rec["obj"] = rs[0]["obj"]
            targets.append(rec)
    return {"targets": targets, "rejected": rejected,
            "groups": len(groups),
            "policy": {"min_runs": min_runs, "min_pass_rate": min_pass_rate,
                       "equal_budget": equal_budget}}


def build_corpus_v2(base_corpus: dict, targets: list, *,
                    prompts_by_spec: dict,
                    exclusion: bool = True) -> dict:
    """Template corpus + measured targets + exclusion-conditioned examples.

    `base_corpus` is the existing canonical corpus (corpus_diverse.json). Its
    records are carried through unchanged so the structural-validity training
    the proposer already has is not lost when measured targets are added.
    """
    recs = list(base_corpus.get("records") or [])
    n_base = len(recs)
    added, per_spec = [], defaultdict(list)
    # a target whose spec has no prompt cannot become a training example.
    # Counted and returned rather than skipped in silence: "0 measured
    # targets added" from a run that verified dozens of designs is a failure
    # that otherwise looks like a successful corpus build.
    skipped = []

    for t in targets:
        prompt = prompts_by_spec.get(t["spec_id"])
        if not prompt:
            skipped.append({"spec_id": t["spec_id"], "family": t["family"],
                            "reason": "no_prompt_for_spec_id"})
            continue
        text = _canon_text(t["obj"])
        vh = variant_hash(json.loads(text))
        row = {"context_id": t["spec_id"], "prompt": prompt,
               "response": text, "variant_hash": vh,
               "canonical_graph_hash": vh,
               "topology_signature": t["family"],
               "stages": int(str(t["family"])[0]),
               "comp": str(t["family"]).split("_", 1)[1],
               "split": "train", "target_source": "measured_ngspice",
               "evidence": {"runs": t["runs"], "passes": t["passes"],
                            "pass_rate": t["pass_rate"],
                            "seeds": t["seeds"], "budget": t["budget"]}}
        recs.append(row)
        added.append(row)
        per_spec[t["spec_id"]].append((t["family"], vh))

    # exclusion-conditioned: "given these MEASURED structures, propose a
    # different one" -- the conditioning the serve path actually uses
    n_excl = 0
    if exclusion:
        for spec_id, fams in per_spec.items():
            if len(fams) < 2:
                continue
            prompt = prompts_by_spec.get(spec_id)
            for k in range(1, len(fams)):
                ex = ", ".join(f"{f}/{h[:12]}" for f, h in fams[:k])
                nxt_fam, _nxt_h = fams[k]
                nxt = next(r for r in added
                           if r["context_id"] == spec_id
                           and r["topology_signature"] == nxt_fam)
                recs.append({**nxt,
                             "prompt": prompt.replace(
                                 "### PROPOSAL",
                                 f"{EXCLUDE_LINE}{ex}\n### PROPOSAL"),
                             "target_source": "measured_ngspice_exclusion",
                             "excluded_count": k})
                n_excl += 1

    graphs = {r.get("canonical_graph_hash") for r in recs}
    return {"records": recs,
            "counts": {"base": n_base, "measured_targets": len(added),
                       "measured_exclusion": n_excl, "total": len(recs),
                       "skipped_targets": len(skipped)},
            "skipped": skipped,
            "distinct_canonical_graphs": len(graphs),
            "corpus_sha256": hashlib.sha256(
                json.dumps([r["response"] for r in recs],
                           sort_keys=True).encode()).hexdigest()[:16]}
