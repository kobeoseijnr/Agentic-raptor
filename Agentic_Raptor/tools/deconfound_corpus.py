"""Step 2 of the DATE plan: de-confound the SFT corpus (2026-09-07).

THE CONFOUND (measured, see date-plan memory / HELDOUT29 13-failure diagnostic)
  corpus_tier2_extension.json = 765 stock records (5 stock families at every
  gain 42-156 dB, 48 train contexts) + 152 tier-2 records whose classes
  {3s_rc_ab, 4s_rc, 4s_rc_ab, 4s_rc_cas} appear ONLY at 140-175 dB (38
  tier2_train contexts). Gain alone therefore predicts "tier-2 class or not"
  with ~100% accuracy, and the proposer never emits a 4-stage / cascode /
  class-AB structure below 140 dB -- exactly where the heldout specs live
  (42.85-108.29 dB) and where A1 retrieval wins with 4-stage on 65/87.

THE FIX (this tool; three sub-commands)
  --qualify   SPICE-qualify the tier-2 classes on the 42 stock TRAIN contexts
              whose gain lies in 40-130 dB, under the SAME regime that gated
              the existing tier-2 records (forecast matrix: production
              sac_size, budget 16, persist=False, seed 0, spec-stated load).
              Resumable; writes QUAL_MATRIX.jsonl one (context, class) row at
              a time.
  --build     corpus_deconfound.json = stock (765) + tier-2 (152) + NEW
              records for every (train context, class) pair that PASSED
              qualification, exclusion-conditioned in the stock corpus's own
              `family/hash12` format. The ### BLOCKS line is rewritten to the
              extended vocabulary on EVERY record so that the BLOCKS line
              carries no family information (otherwise old-BLOCKS<->stock,
              new-BLOCKS<->tier-2 would be a second confound). Responses are
              byte-identical to their sources.
  --audit     the confound metric on any corpus: best single-threshold
              accuracy of "gain -> is tier-2 class", plus class x gain-band
              table. Run before and after.

GUARDRAILS (asserted in --build, not just documented)
  * only stock TRAIN contexts (split == "train" on every record);
  * no context_id equals any HELDOUT29 spec; no SPEC line appears in the
    sealed blind_test.json (read-only regex over the raw file text);
  * corpus_tier2_extension.json and corpus_diverse.json are never modified;
  * no production checkpoint is touched -- this tool writes only under
    artifacts/publication_v3/deconfound/.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\kobeo\OneDrive\Desktop\raptor1\Agentic_Raptor")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from agentic_raptor.electrical import discover_ngspice, effective_c_load           # noqa: E402
from agentic_raptor.llm_dpo.integrity import candidate_identity, parse_spec        # noqa: E402
from agentic_raptor.llm_dpo.stage3e4 import variant_hash                           # noqa: E402
from agentic_raptor.publication.tier2 import (STOCK_CORPUS, TIER2_CLASSES,          # noqa: E402
                                              TIER2_CORPUS_EXT, class_to_proposal)

OUTD = ROOT / "artifacts/publication_v3/deconfound"
QUAL = OUTD / "QUAL_MATRIX.jsonl"
CORPUS_OUT = OUTD / "corpus_deconfound.json"
REPORT = OUTD / "DECONFOUND_REPORT.md"
BLIND = ROOT / "artifacts/stage3e4/blind_test.json"
HELDOUT_GLOB = "artifacts/publication_v2/raptor_v2_runs/ABLv3HELDOUT29R2_AG_FULL_s*_heldout_*.json"

#: classes to qualify below 140 dB: the four that passed the tier-2 forecast
#: plus the two cascode-input classes step 3 makes proposable.
QUAL_CLASSES = ("4s_rc", "4s_rc_ab", "4s_rc_cas", "3s_rc_ab", "2s_rc_cas", "3s_rc_cas")
#: 40-130 dB: must cover the heldout gain range (42.85-108.29 dB); 45 would
#: drop the four 41.96/42.85 dB train contexts, exactly where 3/13 heldout
#: failures sit. 130 excludes the two 142/155 dB stock contexts (already
#: tier-2 territory).
BAND = (40.0, 130.0)
BUDGET, SEED = 16, 0                       # forecast-matrix regime (sz dirs 0..15)
EXT_BLOCKS = ("### BLOCKS five_transistor_first_stage,cascode_input_stage,"
              "cs_gain_stage,class_ab_output_stage,miller_cap,bias_mirror\n")
TARGETS_PER_SPEC = 5
SPEC_RE = re.compile(r"### SPEC gain>=([\d.]+)dB pm>=([\d.]+)deg cl=([\d.]+)pF ugbw>=([\d.e+]+)Hz")


def P(s: str) -> None:
    print(str(s).encode("ascii", "replace").decode(), flush=True)


def spec_line(prompt: str) -> str:
    return prompt.splitlines()[0]


def gain_of(prompt: str) -> float:
    return float(SPEC_RE.search(prompt).group(1))


def load_stock() -> dict:
    return json.loads(STOCK_CORPUS.read_text(encoding="utf-8"))


def train_contexts(stock: dict) -> list[dict]:
    """One entry per stock train context inside BAND, carrying the rank-0
    (EXCLUDE-free) prompt the new records will be built from."""
    ctx: dict[str, dict] = {}
    for r in stock["records"]:
        if r.get("split") != "train" or "### EXCLUDE" in r["prompt"]:
            continue
        ctx.setdefault(r["context_id"], {"context_id": r["context_id"],
                                         "topology_id": r["topology_id"],
                                         "prompt": r["prompt"],
                                         "gain": gain_of(r["prompt"])})
    rows = [c for c in ctx.values() if BAND[0] <= c["gain"] <= BAND[1]]
    return sorted(rows, key=lambda c: (c["gain"], c["context_id"]))


# --------------------------------------------------------------------------- qualify
def qualify(limit: int, classes: tuple[str, ...]) -> None:
    from agentic_raptor.mb_sac.spec_sizing import sac_size
    from agentic_raptor.topology_rl.stage3e2 import new_costs
    from run_puct_ablation import _realise

    exe = discover_ngspice()
    OUTD.mkdir(parents=True, exist_ok=True)
    ctxs = train_contexts(load_stock())
    if limit:
        ctxs = ctxs[:limit]
    done: set[tuple[str, str]] = set()
    if QUAL.exists():
        for l in QUAL.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done.add((r["context_id"], r["cls"]))
    todo = [(c, cls) for c in ctxs for cls in classes if (c["context_id"], cls) not in done]
    P(f"[qualify] contexts={len(ctxs)} classes={list(classes)} cells={len(ctxs)*len(classes)} "
      f"done={len(done)} todo={len(todo)} budget={BUDGET} seed={SEED} ngspice={exe}")
    t0 = time.time()
    with QUAL.open("a", encoding="utf-8") as fh:
        for i, (c, cls) in enumerate(todo, 1):
            spec = parse_spec(c["prompt"])
            spec["spec_id"] = c["context_id"]
            spec["topology_id"] = c["topology_id"]
            cl_f = effective_c_load(spec)
            row = {"context_id": c["context_id"], "cls": cls, "spec_line": spec_line(c["prompt"]),
                   "gain_target_db": spec["gain_target_db"],
                   "pm_target_deg": spec["phase_margin_target_deg"],
                   "cl_pf": spec["load_capacitance_pf"], "ugbw_hz": spec["ugbw_target_hz"],
                   "budget": BUDGET, "seed": SEED, "protocol": "forecast_matrix_regime"}
            try:
                g = _realise(class_to_proposal(cls))
                if g is None:
                    row.update(pass_=False, status="realise_fail", spice_calls=0)
                else:
                    r = sac_size(f"dc_{cls}_{c['context_id']}", g, spec, exe, OUTD / "sizing",
                                 new_costs(), budget=BUDGET, seed=SEED, persist=False,
                                 early_stop_on_pass=False, use_surrogate=True, use_ranker=True,
                                 c_load_f=cl_f)
                    o, best = r["outcome"], r["best"]
                    row.update(pass_=bool(o.get("exact_spec_pass")),
                               distance=o.get("normalized_distance_to_feasibility"),
                               status="ok", spice_calls=len(r["results"]),
                               best={k: v for k, v in best.items()
                                     if isinstance(v, (int, float, str, bool)) or v is None},
                               outcome={k: v for k, v in o.items()
                                        if isinstance(v, (int, float, str, bool)) or v is None})
            except Exception as e:                     # one bad cell must not kill the matrix
                row.update(pass_=False, status="error", error=f"{type(e).__name__}: {e}"[:300],
                           trace=traceback.format_exc()[-800:], spice_calls=None)
            row["pass"] = row.pop("pass_")
            fh.write(json.dumps(row) + "\n"); fh.flush()
            P(f"  [{len(done)+i:3d}/{len(ctxs)*len(classes)}] {c['context_id']:28} gain={row['gain_target_db']:6.1f} "
              f"{cls:10} pass={row['pass']!s:5} dist={row.get('distance')!s:8.8} calls={row.get('spice_calls')} "
              f"{row['status']}  ({(time.time()-t0)/60:.1f} min)")
    _matrix_summary()


def _matrix_summary() -> dict:
    rows = [json.loads(l) for l in QUAL.open(encoding="utf-8") if l.strip()] if QUAL.exists() else []
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["cls"]][0] += int(r["pass"]); per[r["cls"]][1] += 1
    s = {"cells": len(rows), "errors": sum(r["status"] != "ok" for r in rows),
         "per_class_pass": {k: f"{p}/{n}" for k, (p, n) in sorted(per.items())},
         "contexts_with_any_pass": len({r["context_id"] for r in rows if r["pass"]})}
    P(f"[matrix] {json.dumps(s)}")
    return s


# --------------------------------------------------------------------------- audit
def audit(corpus_path: Path, label: str) -> dict:
    d = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    recs = d["records"]
    pts = [(gain_of(r["prompt"]), (r.get("topology_signature") or f"{r['stages']}s_{r['comp']}") in TIER2_CLASSES)
           for r in recs]
    # best single threshold on gain predicting "is tier-2 class" (either polarity)
    gs = sorted({g for g, _ in pts})
    best = 0.0
    for th in gs + [gs[-1] + 1]:
        acc = sum((g >= th) == y for g, y in pts) / len(pts)
        best = max(best, acc, 1 - acc)
    base = max(sum(y for _, y in pts), sum(not y for _, y in pts)) / len(pts)
    bands = [(0, 80), (80, 130), (130, 200)]
    tab: dict[str, Counter] = defaultdict(Counter)
    for r, (g, _) in zip(recs, pts):
        fam = r.get("topology_signature") or f"{r['stages']}s_{r['comp']}"
        for lo, hi in bands:
            if lo <= g < hi:
                tab[fam][f"{lo}-{hi}"] += 1
    blocks = Counter(next((l for l in r["prompt"].splitlines() if l.startswith("### BLOCKS")), "") for r in recs)
    out = {"label": label, "records": len(recs), "gain_stump_accuracy": round(best, 4),
           "majority_baseline": round(base, 4),
           "class_x_gainband": {k: dict(v) for k, v in sorted(tab.items())},
           "distinct_BLOCKS_lines": len(blocks)}
    P(f"\n=== AUDIT {label} ({corpus_path.name}) ===")
    P(f"  records={len(recs)}  gain->tier2 stump accuracy={best:.3f}  (majority baseline {base:.3f})")
    P(f"  distinct ### BLOCKS lines: {len(blocks)}")
    P(f"  {'class':14}" + "".join(f"{b[0]}-{b[1]:<9}" for b in bands))
    for fam, cnt in sorted(tab.items()):
        P(f"  {fam:14}" + "".join(f"{cnt.get(f'{lo}-{hi}', 0):<12d}" for lo, hi in bands))
    return out


# --------------------------------------------------------------------------- build
def _exclude_format(stock: dict) -> tuple[str, int]:
    """Learn (and verify) the stock corpus's EXCLUDE token format instead of
    assuming it: 'family/hash12', comma-joined."""
    by_hash = {}
    for r in stock["records"]:
        by_hash.setdefault(variant_hash(json.loads(r["response"])), r)
    sep, plen, checked = None, None, 0
    for r in stock["records"]:
        for l in r["prompt"].splitlines():
            if l.startswith("### EXCLUDE "):
                body = l[len("### EXCLUDE "):]
                if sep is None and "," in body:
                    sep = ", " if ", " in body else ","
                toks = [t.strip() for t in re.split(r",\s*", body)]
                for t in toks:
                    fam, h = t.split("/")
                    plen = plen or len(h)
                    assert len(h) == plen, f"inconsistent EXCLUDE hash length in {l!r}"
                    cand = [v for v in by_hash if v.startswith(h)]
                    assert cand, f"EXCLUDE token {t} matches no stock variant_hash"
                    src = by_hash[cand[0]]
                    assert (src.get("topology_signature") or f"{src['stages']}s_{src['comp']}") == fam, t
                    checked += 1
    assert checked > 0 and plen, "no EXCLUDE lines found in stock corpus"
    return (sep or ", "), plen


def build() -> None:
    stock = load_stock()
    ext = json.loads(TIER2_CORPUS_EXT.read_text(encoding="utf-8"))
    t2 = [r for r in ext["records"] if r.get("tier2")]
    assert len(ext["records"]) == len(stock["records"]) + len(t2), "ext != stock + tier2"
    for a, b in zip(stock["records"], ext["records"]):
        assert a["prompt"] == b["prompt"] and a["response"] == b["response"], "stock records drifted"
    sep, plen = _exclude_format(stock)
    rows = [json.loads(l) for l in QUAL.open(encoding="utf-8") if l.strip()]
    passing: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["pass"] and r["status"] == "ok":
            passing[r["context_id"]].append((r.get("distance") if r.get("distance") is not None else 1e9, r["cls"]))
    ctxs = {c["context_id"]: c for c in train_contexts(stock)}

    # ---- guardrails
    held = {os.path.basename(f).split("_heldout_", 1)[1].rsplit(".json", 1)[0][4:]
            for f in glob.glob(HELDOUT_GLOB)}
    blind_text = BLIND.read_text(encoding="utf-8")
    blind_specs = set(re.findall(r"### SPEC [^\\\n\"]+", blind_text))
    leak_ctx = sorted(set(passing) & held)
    leak_spec = sorted(spec_line(ctxs[c]["prompt"]) for c in passing if spec_line(ctxs[c]["prompt"]) in blind_specs)
    assert not leak_ctx, f"heldout context in new records: {leak_ctx}"
    assert not leak_spec, f"blindtest SPEC line in new records: {leak_spec}"

    new = []
    for cid in sorted(passing, key=lambda c: ctxs[c]["gain"]):
        base = ctxs[cid]["prompt"]
        targets = [cls for _, cls in sorted(passing[cid])][:TARGETS_PER_SPEC]
        seen: list[str] = []
        for rank, cls in enumerate(targets):
            obj = class_to_proposal(cls)
            resp = json.dumps(obj, separators=(",", ":"))
            vh = variant_hash(obj)
            excl = (f"### EXCLUDE {sep.join(seen)}\n" if seen else "")
            prompt = base.replace("### PROPOSAL\n", f"{excl}### PROPOSAL\n")
            new.append({"context_id": cid, "split": "train", "topology_id": f"deconf_{cls}",
                        "prompt": prompt, "response": resp,
                        "stages": int(cls[0]), "comp": cls.split("_")[1], "buffer": False, "fb": False,
                        "variant_hash": vh,
                        "canonical_graph_hash": candidate_identity(obj)["canonical_graph_hash"],
                        "topology_signature": cls, "target_rank": rank,
                        "target_source": "deconfound_spice_qualified", "tier2": True,
                        "qualified": {"budget": BUDGET, "seed": SEED,
                                      "distance": next(d for d, k in passing[cid] if k == cls)}})
            seen.append(f"{cls}/{vh[:plen]}")

    def reblock(r: dict) -> dict:
        r = dict(r)
        r["prompt"] = re.sub(r"### BLOCKS [^\n]*\n", EXT_BLOCKS, r["prompt"], count=1)
        assert EXT_BLOCKS in r["prompt"], r["prompt"]
        return r

    merged_recs = [reblock(r) for r in stock["records"]] + [reblock(r) for r in t2] + [reblock(r) for r in new]
    assert all(r["split"] == "train" for r in merged_recs)
    assert all(a["response"] == b["response"] for a, b in zip(merged_recs, stock["records"] + t2 + new))
    merged = {k: v for k, v in ext.items() if k != "records"}
    merged["records"] = merged_recs
    merged["deconfound"] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_stock": str(STOCK_CORPUS.relative_to(ROOT)),
        "source_stock_sha256": hashlib.sha256(STOCK_CORPUS.read_bytes()).hexdigest()[:16],
        "source_tier2_ext": str(TIER2_CORPUS_EXT.relative_to(ROOT)),
        "source_tier2_ext_sha256": hashlib.sha256(TIER2_CORPUS_EXT.read_bytes()).hexdigest()[:16],
        "qualification": {"protocol": "production sac_size, budget 16, seed 0, persist=False, spec-stated load "
                                      "(same regime as the tier-2 forecast matrix)",
                          "classes": list(QUAL_CLASSES), "band_db": list(BAND), "cells": len(rows),
                          "per_class_pass": _matrix_summary()["per_class_pass"]},
        "counts": {"stock": len(stock["records"]), "tier2_140_175": len(t2), "new_45_130": len(new),
                   "total": len(merged_recs), "new_contexts": len(passing)},
        "blocks_line_rewritten_on_all_records": EXT_BLOCKS.strip(),
        "exclude_format": f"family/hash{plen} joined by '{sep}' (stock format, verified)",
        "guardrails": {"heldout_context_overlap": leak_ctx, "blindtest_spec_overlap": leak_spec,
                       "all_split_train": True, "responses_byte_identical_to_sources": True}}
    OUTD.mkdir(parents=True, exist_ok=True)
    CORPUS_OUT.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    P(f"[build] wrote {CORPUS_OUT}  total={len(merged_recs)}  new={len(new)} over {len(passing)} contexts")

    before = audit(TIER2_CORPUS_EXT, "BEFORE (corpus_tier2_extension)")
    after = audit(CORPUS_OUT, "AFTER (corpus_deconfound)")
    md = ["# Corpus de-confound report (DATE plan step 2)\n",
          f"Source: stock {len(stock['records'])} + tier-2 {len(t2)} + **new {len(new)}** SPICE-qualified records "
          f"over {len(passing)} train contexts in {BAND[0]:.0f}-{BAND[1]:.0f} dB.\n",
          "Qualification: production `sac_size`, budget 16, seed 0, persist=False, spec-stated load "
          "(the regime that gated the existing tier-2 records).\n",
          f"Per-class pass in band: `{merged['deconfound']['qualification']['per_class_pass']}`\n",
          "| | before | after |", "|---|---:|---:|",
          f"| records | {before['records']} | {after['records']} |",
          f"| gain-only stump accuracy predicting tier-2 class | {before['gain_stump_accuracy']:.3f} | {after['gain_stump_accuracy']:.3f} |",
          f"| majority baseline | {before['majority_baseline']:.3f} | {after['majority_baseline']:.3f} |",
          f"| distinct BLOCKS lines | {before['distinct_BLOCKS_lines']} | {after['distinct_BLOCKS_lines']} |\n",
          "## class x gain band (after)\n", "| class | 0-80 | 80-130 | 130-200 |", "|---|---:|---:|---:|"]
    for fam, cnt in sorted(after["class_x_gainband"].items()):
        md.append(f"| {fam} | {cnt.get('0-80', 0)} | {cnt.get('80-130', 0)} | {cnt.get('130-200', 0)} |")
    md += ["\n## Guardrails\n",
           f"- heldout context overlap: {leak_ctx} ; blindtest SPEC overlap: {leak_spec}",
           "- every record split == train; responses byte-identical to sources; sources untouched",
           "- residual: tier-2 records at 140-175 dB carry `### RAG none` while stock/new carry `### RAG rag_l2_*` "
           "(no retrieval evidence exists for tier-2 specs); documented, not fixed here."]
    REPORT.write_text("\n".join(md), encoding="utf-8")
    P(f"[build] wrote {REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qualify", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--audit", action="store_true", help="confound metric on the current tier-2 corpus (+ deconfound if built)")
    ap.add_argument("--limit", type=int, default=0, help="qualify: first N contexts only")
    ap.add_argument("--classes", default=",".join(QUAL_CLASSES))
    a = ap.parse_args()
    if a.audit:
        audit(TIER2_CORPUS_EXT, "corpus_tier2_extension (current SFT corpus)")
        if CORPUS_OUT.exists():
            audit(CORPUS_OUT, "corpus_deconfound")
    if a.qualify:
        qualify(a.limit, tuple(x for x in a.classes.split(",") if x))
    if a.build:
        build()
    if not (a.audit or a.qualify or a.build):
        P("nothing to do: --audit | --qualify [--limit N --classes a,b] | --build")


if __name__ == "__main__":
    main()
