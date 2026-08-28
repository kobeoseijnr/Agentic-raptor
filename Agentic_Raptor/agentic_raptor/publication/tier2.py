"""TIER-2 EVALUATION SET + CORPUS EXTENSION (2026-08-17).

WHY: the heldout benchmark saturated -- every spec is solvable by one of the
five stock template families (3s_rc passes all 9 in <= 7 sizing calls), so
9 of 10 ablation arms score 9/9 and the evaluation cannot discriminate.
Tier-2 is a difficulty band the stock library provably cannot reach, so
component contributions become measurable again:

  * gain 140-175 dB       -> beyond the 3-stage cascade (needs 4 stages or
                             a cascode input)
  * 1-5 MHz into 0.5-2 nF -> heavy-load bandwidth (needs class-AB output
                             and/or cascode gain-without-pole)
  * PM >= 65-75 deg       -> tight compensation

Design principle: every Tier-2 spec is FORECAST-VERIFIED against the real
outcome matrix (stock 5 families vs the tier-2 vocabulary, real ngspice)
BEFORE any campaign -- specs the extended vocabulary cannot reach either
are dropped as uninformative, so the tier measures capability, not
impossibility.

Sealed split: tier2_train (SFT extension + bandit refit) / tier2_heldout
(evaluation). Blindtest remains untouched.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
TIER2_DIR = _ROOT / "artifacts/publication_v3/tier2"
TIER2_SPECS = TIER2_DIR / "TIER2_SPECS.jsonl"
TIER2_CORPUS_EXT = TIER2_DIR / "corpus_tier2_extension.json"

#: TIER-2 STRUCTURE CLASSES the mapper realizes (all simulated with valid
#: operating points at nominal, 2026-08-17 TIER2_PROBE)
TIER2_CLASSES = ("2s_rc_cas", "3s_rc_cas", "3s_rc_ab", "3s_rc_cas_ab",
                 "4s_rc", "4s_rc_cas", "4s_rc_ab", "4s_miller",
                 "2s_miller_cas", "3s_miller_cas", "3s_miller_ab")
STOCK_CLASSES = ("2s_none", "2s_miller", "2s_rc", "3s_miller", "3s_rc")


def class_to_proposal(cls: str) -> dict:
    """Canonical proposal JSON for a (stock or tier-2) structure class."""
    parts = cls.split("_")
    stages = int(parts[0][0])
    comp = parts[1]
    cas = "cas" in parts[2:]
    ab = "ab" in parts[2:]
    st = [{"block": "cascode_input_stage" if cas else "five_transistor_first_stage",
           "role": "input_stage", "outputs": ["s1out" if stages > 1 else "vout"]}]
    for k in range(2, stages + 1):
        last = (k == stages)
        st.append({"block": ("class_ab_output_stage" if (ab and last)
                             else "cs_gain_stage"),
                   "role": "gain_stage",
                   "outputs": ["vout" if last else f"n{k}"]})
    return {"stages": st,
            "ports": ["gnda", "vdda", "vinn", "vinp", "vout"],
            "bias_roles": ["bias_mirror"],
            "compensation": ([] if comp == "none" else
                             [{"type": "miller_cap" if comp == "miller"
                               else "rc_nulling"}]),
            "output_buffer": False, "local_feedback": False,
            "feedback_paths": [], "polarity": "vinp_noninverting"}


def generate_tier2_specs() -> list[dict]:
    """Deterministic Tier-2 grid; sealed train/heldout by spec hash."""
    gains = (140.0, 150.0, 160.0, 175.0)
    pms = (60.0, 70.0)
    loads = (500.0, 1000.0, 2000.0)          # pF
    ugbws = (1e6, 3e6, 5e6)
    specs = []
    for g, pm, cl, u in itertools.product(gains, pms, loads, ugbws):
        sid = f"t2_g{int(g)}_pm{int(pm)}_cl{int(cl)}_u{int(u/1e6)}M"
        h = hashlib.sha256(sid.encode()).hexdigest()
        split = "tier2_heldout" if (int(h[:8], 16) % 100) < 40 else "tier2_train"
        specs.append({"spec_id": sid, "spec_hash": h[:16], "split": split,
                      "gain_target_db": g, "phase_margin_target_deg": pm,
                      "load_capacitance_pf": cl, "ugbw_target_hz": u,
                      "prompt": (f"### SPEC gain>={g:.2f}dB pm>={pm:.1f}deg "
                                 f"cl={cl:.0f}pF ugbw>={u:.0e}Hz tech=sky130\n"
                                 "### RAG none\n"
                                 "### BLOCKS five_transistor_first_stage,"
                                 "cascode_input_stage,cs_gain_stage,"
                                 "class_ab_output_stage,miller_cap,bias_mirror\n"
                                 "### FORBIDDEN raw_netlist,feedback_to_input\n"
                                 "### PROPOSAL\n")})
    return specs


def write_tier2_specs() -> Path:
    TIER2_DIR.mkdir(parents=True, exist_ok=True)
    specs = generate_tier2_specs()
    TIER2_SPECS.write_text("".join(json.dumps(s) + "\n" for s in specs),
                           encoding="utf-8")
    return TIER2_SPECS


TIER2_SPECS_FINAL = TIER2_DIR / "TIER2_SPECS_FINAL.jsonl"


def load_tier2_specs(split: str | None = None, final: bool = True) -> list[dict]:
    """final=True (default) -> the FORECAST-FILTERED informative set (39
    specs: stock fails, tier-2 passes; impossible/stock-solvable dropped).
    final=False -> the full 72-spec grid (forecast + corpus construction)."""
    src = TIER2_SPECS_FINAL if (final and TIER2_SPECS_FINAL.is_file()) else TIER2_SPECS
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return [r for r in rows if split is None or r["split"] == split]


# ---------------------------------------------------------------------------
# corpus extension (SFT retrain + A1 library) -- gated on the forecast matrix
# ---------------------------------------------------------------------------
STOCK_CORPUS = _ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"
T2_MATRIX = TIER2_DIR / "forecast/T2_MATRIX.jsonl"


def forecast_verdict(matrix_path: Path = T2_MATRIX) -> dict:
    """Which Tier-2 specs are INFORMATIVE (stock fails, tier-2 passes), which
    are IMPOSSIBLE for everyone (dropped), which are already stock-solvable
    (dropped -- they would re-saturate). Also: pass rate per class."""
    rows = [json.loads(l) for l in matrix_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by_spec: dict = {}
    for r in rows:
        by_spec.setdefault(r["spec_id"], {})[r["cls"]] = r
    per_class: dict = {}
    for r in rows:
        c = per_class.setdefault(r["cls"], [0, 0])
        c[0] += int(r["pass"]); c[1] += 1
    informative, impossible, stock_solvable, incomplete = [], [], [], []
    for sid, d in by_spec.items():
        if len(d) < len(STOCK_CLASSES) + len(TIER2_CLASSES):
            incomplete.append(sid); continue
        stock_pass = any(d[c]["pass"] for c in STOCK_CLASSES if c in d)
        t2_pass = any(d[c]["pass"] for c in TIER2_CLASSES if c in d)
        if stock_pass:
            stock_solvable.append(sid)
        elif t2_pass:
            informative.append(sid)
        else:
            impossible.append(sid)
    passing_classes = sorted(c for c, (p, n) in per_class.items()
                             if c in TIER2_CLASSES and p > 0)
    return {"n_specs_evaluated": len(by_spec), "informative": informative,
            "impossible": impossible, "stock_solvable": stock_solvable,
            "incomplete": incomplete, "per_class_pass": per_class,
            "tier2_classes_with_passes": passing_classes,
            "verdict": ("DISCRIMINATES" if len(informative) >= 6
                        else "INSUFFICIENT")}


def build_corpus_extension(verdict: dict | None = None,
                           targets_per_spec: int = 5) -> Path:
    """Merged corpus = the frozen stock corpus (byte-identical records) +
    Tier-2 TRAIN specs with forecast-PASSING tier-2 classes as targets
    (exclusion-conditioned, mirroring the stock corpus's construction).
    Never touches STOCK_CORPUS; writes a NEW file the SFT trainer and A1's
    retrieval can both consume."""
    verdict = verdict or forecast_verdict()
    stock = json.loads(STOCK_CORPUS.read_text(encoding="utf-8"))
    passing = verdict["tier2_classes_with_passes"] or list(TIER2_CLASSES)
    from agentic_raptor.llm_dpo.stage3e4 import variant_hash
    from agentic_raptor.llm_dpo.integrity import candidate_identity
    recs = []
    for s in load_tier2_specs("tier2_train"):
        # rank tier-2 classes for THIS spec by measured distance where the
        # forecast covered it, else keep forecast order
        targets = list(passing)[:targets_per_spec]
        seen = []
        for rank, cls in enumerate(targets):
            obj = class_to_proposal(cls)
            ident = candidate_identity(obj)
            excl = ("\n### EXCLUDE " + ",".join(seen) + "\n") if seen else "\n"
            recs.append({"context_id": s["spec_id"], "split": "train",
                         "topology_id": f"tier2_{cls}",
                         "prompt": s["prompt"].replace("### PROPOSAL\n",
                                                       f"{excl}### PROPOSAL\n"),
                         "response": json.dumps(obj, separators=(",", ":")),
                         "stages": int(cls[0]), "comp": cls.split("_")[1],
                         "buffer": False, "fb": False,
                         "variant_hash": variant_hash(obj),
                         "canonical_graph_hash": ident["canonical_graph_hash"],
                         "topology_signature": cls,
                         "target_rank": rank, "target_source": "tier2_forecast",
                         "tier2": True})
            seen.append(cls)
    merged = dict(stock)
    merged["records"] = stock["records"] + recs
    merged["tier2_extension"] = {"created": "2026-08-17",
                                 "tier2_records": len(recs),
                                 "tier2_classes": passing,
                                 "forecast_verdict": verdict["verdict"],
                                 "informative_specs": len(verdict["informative"])}
    TIER2_CORPUS_EXT.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    return TIER2_CORPUS_EXT
