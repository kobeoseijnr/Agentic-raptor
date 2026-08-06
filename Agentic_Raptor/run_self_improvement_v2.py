"""Agentic RAPTOR v2 self-improvement loop.

Built entirely around run_raptor_v2.run_pipeline. Shares no code, no
checkpoints and no datasets with the old loop, whose artifacts are archived.

Each GENERATION:
  1. RUN      train specs through the v2 pipeline in CALIBRATION mode, so both
              post-SAC designs are authoritatively measured by ngspice.
  2. HARVEST  route each verified comparison into six separate streams.
  3. RETRAIN  candidate post-SAC ranker + candidate PUCT policy/value.
              The proposer retrains only when enough NEW verified topology
              targets have accumulated -- never after every run.
  4. GATE     evaluate candidates on frozen held-out data; accept an
              improvement, otherwise roll back to the incumbent.
  5. PUBLISH  write STATE.json so generation N+1 loads exactly what
              generation N accepted.

Generation N+1 reads its ranker, PUCT and RAG memory from STATE.json, so the
lineage is explicit rather than implied by file mtimes.

Run:
  python run_self_improvement_v2.py --generations 2 --specs 4
  python run_self_improvement_v2.py --generations 2 --specs 2 --dry-run
"""
import argparse
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
#: real (measured) lineage. --dry-run redirects EVERYTHING here to a separate
#: root: synthetic outcomes carry fabricated pm/gain values, and a measured
#: generation that loaded them would be training on invented physics while
#: looking entirely healthy.
SI_REAL = ROOT / "artifacts/publication_v2/selfimprove"
SI_DRY = ROOT / "artifacts/publication_v2/selfimprove_dryrun"
SI = SI_REAL
STATE = SI / "STATE.json"
RAG_MEMORY = SI / "rag_memory_v2.jsonl"
BASE_ADAPTER = ROOT / "artifacts/publication_v2/proposer_repair/sft_adapter_diverse"
BASE_CORPUS = ROOT / "artifacts/publication_v2/proposer_repair/corpus_diverse.json"

from agentic_raptor.selfimprove_v2 import (STREAMS, GenerationPaths,  # noqa: E402
                                           harvest_run, proposer_gates,
                                           puct_gate, ranker_gate)
from agentic_raptor.selfimprove_v2.corpus import (aggregate_sft_targets,  # noqa: E402
                                                  build_corpus_v2)
from agentic_raptor.selfimprove_v2.streams import append, read  # noqa: E402


# --------------------------------------------------------------------------
def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"generation": -1, "accepted": {
        "ranker_ckpt": None, "value_ckpt": None,
        "proposer_adapter": str(BASE_ADAPTER),
        "rag_memory": str(RAG_MEMORY), "corpus": str(BASE_CORPUS)},
        "metrics": {}, "history": []}


def save_state(st: dict):
    SI.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")


def sha(path) -> str | None:
    from agentic_raptor.ranking import checkpoint_sha256, directory_sha256
    p = Path(path) if path else None
    if not p or not p.exists():
        return None
    return directory_sha256(str(p)) if p.is_dir() else checkpoint_sha256(p)


# --------------------------------------------------------------------------
def train_ranker_candidate(pairs: list, out: Path, epochs=200, lr=1e-2,
                           seed=0):
    """Fit a candidate ranker on measured pairs. Returns (path, n) or None."""
    import torch

    from agentic_raptor.ranking.model import PostSACRanker, features
    from agentic_raptor.ranking.types import SurrogatePrediction
    usable = [p for p in pairs if p.get("winner") and p.get("prediction_A")
              and p.get("prediction_B")]
    if len(usable) < 2:
        return None, len(usable)

    def pred(d):
        keep = {k: v for k, v in (d or {}).items()
                if k in SurrogatePrediction.__dataclass_fields__
                and k != "prediction_timestamp"}
        keep.setdefault("topology_hash", "x")
        keep.setdefault("sizing_manifest_hash", "x")
        return SurrogatePrediction(**keep)

    xw, xl = [], []
    for p in usable:
        fa = features(p["spec"], pred(p["prediction_A"]))
        fb = features(p["spec"], pred(p["prediction_B"]))
        if p["winner"] == "A":
            xw.append(fa); xl.append(fb)
        else:
            xw.append(fb); xl.append(fa)
    torch.manual_seed(seed)
    net = PostSACRanker.build()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tw = torch.tensor(xw, dtype=torch.float32)
    tl = torch.tensor(xl, dtype=torch.float32)
    for _ in range(epochs):
        loss = -torch.nn.functional.logsigmoid(net(tw) - net(tl)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)
    return out, len(usable)


def eval_ranker(ckpt, pairs: list) -> float | None:
    """Pairwise accuracy on held-out measured pairs."""
    from agentic_raptor.ranking.model import PostSACRanker, features
    from agentic_raptor.ranking.types import SurrogatePrediction
    usable = [p for p in pairs if p.get("winner") and p.get("prediction_A")
              and p.get("prediction_B")]
    if not usable:
        return None
    r = PostSACRanker.load(ckpt) if ckpt else None

    def pred(d):
        keep = {k: v for k, v in (d or {}).items()
                if k in SurrogatePrediction.__dataclass_fields__
                and k != "prediction_timestamp"}
        keep.setdefault("topology_hash", "x")
        keep.setdefault("sizing_manifest_hash", "x")
        return SurrogatePrediction(**keep)
    ok = 0
    for p in usable:
        pa, pb = pred(p["prediction_A"]), pred(p["prediction_B"])
        if r is not None:
            import torch
            with torch.no_grad():
                sa = float(r.model(torch.tensor(
                    [features(p["spec"], pa)], dtype=torch.float32)))
                sb = float(r.model(torch.tensor(
                    [features(p["spec"], pb)], dtype=torch.float32)))
        else:                       # incumbent absent -> deterministic scorer
            from agentic_raptor.ranking.post_sac import _deterministic_score
            sa, sb = _deterministic_score(pa), _deterministic_score(pb)
        ok += (sa > sb) == (p["winner"] == "A")
    return round(ok / len(usable), 4)


# --------------------------------------------------------------------------
def puct_examples_to_training(rows: list, verify: bool = True):
    """Rebuild the EXACT (state, pi, value) examples the search produced.

    Earlier this fabricated the state -- graph_hash="root", n_nodes=10.0 --
    so the value net trained on a flattened input that never existed. Now the
    real root state is persisted at search time and reconstructed verbatim,
    and each candidate graph is re-realised from its stored proposal object
    and CHECKED against the structural hash recorded then. A graph that no
    longer hashes the same is dropped with a reason rather than silently
    training the value net on a different circuit than the one measured.

    Returns (examples, registry, report).
    """
    from agentic_raptor.topology_rl.value_refresh import \
        device_graph_to_circuit_graph
    from run_puct_ablation import _realise
    examples, graphs = [], {}
    rep = {"rows": len(rows), "no_value_target": 0, "no_root_state": 0,
           "graphs_rebuilt": 0, "hash_mismatch": [], "rebuild_failed": [],
           "usable": 0}
    for r in rows:
        if r.get("value_target") is None:
            rep["no_value_target"] += 1
            continue
        if not r.get("root_state"):
            rep["no_root_state"] += 1
            continue
        man = r.get("candidate_manifests") or {}
        ok = True
        for c in r.get("candidates") or []:
            pid = c["llm_proposal_id"]
            if pid in graphs:
                continue
            try:
                g = device_graph_to_circuit_graph(_realise(c["obj"]), pid)
            except Exception as exc:
                rep["rebuild_failed"].append(f"{pid}:{type(exc).__name__}")
                ok = False
                continue
            want = (man.get(pid) or {}).get("structural_hash")
            if verify and want and g.structural_hash() != want:
                rep["hash_mismatch"].append(
                    f"{pid}:{g.structural_hash()[:8]}!={want[:8]}")
                ok = False
                continue
            graphs[pid] = g
            rep["graphs_rebuilt"] += 1
        if ok:
            examples.append(r)
    rep["usable"] = len(examples)

    class _E:
        def __init__(self, g):
            self.graph = g

    class _Reg:
        def get_topology(self, tid):
            if tid in graphs:
                return _E(graphs[tid])
            return _E(next(iter(graphs.values())))

        def list_topologies(self):
            return sorted(graphs)
    return examples, (_Reg() if graphs else None), rep


def _state_from_row(r: dict):
    """TopologySearchState reconstructed from the PERSISTED root state."""
    from agentic_raptor.topology_rl.stage3e1 import TopologySearchState
    rs = dict(r["root_state"])
    fields = set(TopologySearchState.__dataclass_fields__)
    return TopologySearchState(**{k: v for k, v in rs.items() if k in fields})


def train_puct_candidate(rows: list, out: Path, epochs=3, seed=0):
    """Fit a candidate policy/value net on RECONSTRUCTED search states."""
    from agentic_raptor.topology_rl import stage3e1 as s1
    ex, reg, rep = puct_examples_to_training(rows)
    if not ex or reg is None:
        return None, 0, rep
    nets = s1.build_policy_value(seed)
    tr = [{"state": r["root_state"],
           "legal_action_ids": r.get("root_action_ids") or [],
           "visit_distribution": r.get("visit_distribution") or {},
           "value_target": r["value_target"]} for r in ex]
    try:
        import torch
        opt = torch.optim.Adam(nets["params"], lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            s1.train_step(nets, tr, reg, opt=opt)
        out.parent.mkdir(parents=True, exist_ok=True)
        s1.save_checkpoint(nets, out, {"examples": len(tr), "epochs": epochs,
                                       "reconstruction": rep})
        return out, len(tr), rep
    except Exception:
        traceback.print_exc()
        return None, len(tr), rep


def eval_puct(ckpt, rows: list):
    """Mean squared value error on held-out RECONSTRUCTED examples."""
    from agentic_raptor.topology_rl import stage3e1 as s1
    ex, reg, rep = puct_examples_to_training(rows)
    if not ex or reg is None:
        return None, rep
    nets = s1.build_policy_value(0)
    if ckpt and Path(ckpt).is_file():
        try:
            s1.load_checkpoint(nets, Path(ckpt))
        except Exception:
            return None, rep
    import torch
    losses = []
    for r in ex:
        st = _state_from_row(r)
        with torch.no_grad():
            v = nets["value_forward"](st, reg)["scalar"]
        losses.append(float((v - torch.tensor(float(r["value_target"]))) ** 2))
    return (round(sum(losses) / len(losses), 4) if losses else None), rep


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--specs", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--eval-specs", type=int, default=2,
                    help="frozen held-out specs for the proposer gate")
    ap.add_argument("--min-ranker-pairs", type=int, default=16,
                    help="accumulate at least this many measured pairs before "
                         "fitting a ranker; a net fitted to a handful of "
                         "examples is noise wearing a checkpoint's name")
    ap.add_argument("--min-puct-examples", type=int, default=16,
                    help="same policy for the policy/value net")
    ap.add_argument("--freeze-proposer", action="store_true", default=True,
                    help="collect SFT/DPO evidence but do NOT retrain the "
                         "proposer (default). Evidence is provisional until "
                         "enough equal-budget multi-seed results exist.")
    ap.add_argument("--sft-min-seeds", type=int, default=2,
                    help="distinct seeds required before a measured topology "
                         "may become an SFT target")
    ap.add_argument("--sft-threshold", type=int, default=8,
                    help="new verified targets required before the proposer "
                         "is retrained; it is NOT retrained per run")
    ap.add_argument("--quality-threshold", type=float, default=None,
                    help="admit a near-miss as an SFT target when its "
                         "normalised distance is <= this (default: only "
                         "exact passes qualify)")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise harvest/gates/corpus with NO GPU and NO "
                         "ngspice, using synthetic runs")
    args = ap.parse_args()

    global SI, STATE, RAG_MEMORY
    SI = SI_DRY if args.dry_run else SI_REAL
    STATE = SI / "STATE.json"
    RAG_MEMORY = SI / "rag_memory_v2.jsonl"
    SI.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print("DRY RUN: synthetic outcomes, isolated root ->", SI)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    idxs = [args.start + i * args.step for i in range(args.specs)]
    st = load_state()
    if not args.dry_run and st.get("synthetic"):
        raise SystemExit(
            "REFUSING TO RUN: this lineage was produced by --dry-run "
            "(synthetic outcomes). Start a measured lineage from a clean "
            f"root, or archive {SI}.")
    if args.dry_run:
        st["synthetic"] = True
    from agentic_raptor.publication.eval_sets import excluded_context_ids
    protected = set(excluded_context_ids())

    model = tok = None
    if not args.dry_run:
        from run_qwen_ablation import _load
        model_adapter = st["accepted"]["proposer_adapter"]
        tok, model = _load(str(model_adapter))

    report = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
              "pipeline": "run_raptor_v2.run_pipeline (canonical v2)",
              "generations": [], "dry_run": args.dry_run}

    for g in range(st["generation"] + 1, st["generation"] + 1 + args.generations):
        gp = GenerationPaths(SI, g).mkdirs()
        acc = st["accepted"]
        print(f"\n{'='*66}\nGENERATION {g}")
        print(f"  loads ranker   : {acc['ranker_ckpt']}")
        print(f"  loads PUCT     : {acc['value_ckpt']}")
        print(f"  loads proposer : {acc['proposer_adapter']}")
        print(f"  loads RAG      : {acc['rag_memory']} "
              f"({len(read(Path(acc['rag_memory']))) if acc['rag_memory'] else 0} rows)")
        print("="*66, flush=True)

        gen = {"generation": g, "loaded": dict(acc), "runs": [],
               "stream_counts": {}, "gates": {}, "accepted_changes": []}

        # ---------------- 1. RUN + 2. HARVEST ---------------------------
        counts = {k: 0 for k in STREAMS}
        for seed in seeds:
            for idx in idxs:
                t0 = time.time()
                if args.dry_run:
                    hv = _synthetic_harvest(g, idx, seed)
                    ok, err = True, None
                else:
                    import run_raptor_v2 as v2
                    try:
                        tr = v2.run_pipeline(
                            model, tok, str(acc["proposer_adapter"]),
                            split="train", spec_index=idx,
                            budget=args.budget, calibrate=True, seed=seed,
                            ranker_ckpt=acc["ranker_ckpt"],
                            value_ckpt=acc["value_ckpt"],
                            rag_memory=acc["rag_memory"],
                            harvest=True, out_prefix=f"G{g:03d}")
                        hv, ok, err = tr.get("_harvest"), True, None
                    except Exception as exc:
                        hv, ok = None, False
                        err = f"{type(exc).__name__}: {str(exc)[:120]}"
                        (gp.gdir / "errors.log").open(
                            "a", encoding="utf-8").write(
                            f"\n=== idx={idx} seed={seed}\n"
                            + traceback.format_exc())
                if hv:
                    (gp.runs / f"harvest_{idx:03d}_s{seed}.json").write_text(
                        json.dumps(hv, indent=1, default=str),
                        encoding="utf-8")
                    streams = harvest_run(
                        hv, split="train", protected_ids=protected,
                        quality_threshold=args.quality_threshold)
                    for k, rows in streams.items():
                        counts[k] += append(gp.stream(k), rows)
                    # RAG is updated IMMEDIATELY: every authoritative success
                    # AND failure is available to the very next run, which is
                    # the only part of the loop that needs no training and no
                    # gate to start paying off
                    append(RAG_MEMORY, streams.get("rag_memory") or [])
                gen["runs"].append({"spec_index": idx, "seed": seed,
                                    "ok": ok, "error": err,
                                    "seconds": round(time.time() - t0, 1)})
                print(f"  run idx={idx} seed={seed} ok={ok} "
                      f"{gen['runs'][-1]['seconds']:>5.0f}s "
                      f"{err or ''}", flush=True)
        gen["stream_counts"] = counts
        print(f"  streams: {counts}", flush=True)

        # ---------------- 3. RETRAIN + 4. GATE: ranker -------------------
        all_pairs = []
        for gg in range(0, g + 1):
            all_pairs += read(GenerationPaths(SI, gg).stream("ranker_pairs"))
        # hold out the newest generation's pairs from fitting
        held = read(gp.stream("ranker_pairs"))
        fit = [p for p in all_pairs if p not in held] or all_pairs
        if len(fit) < args.min_ranker_pairs:
            cand_ck, n_fit = None, len(fit)
            print(f"  ranker       : SKIP retrain -- {len(fit)} pairs < "
                  f"{args.min_ranker_pairs} (accumulating)", flush=True)
        else:
            cand_ck, n_fit = train_ranker_candidate(
                fit, gp.ckpt / "ranker_candidate.pt")
        r_cand = eval_ranker(cand_ck, held) if cand_ck else None
        r_acc = eval_ranker(acc["ranker_ckpt"], held)
        rg = ranker_gate(r_cand, r_acc, len(held))
        gen["gates"]["ranker"] = rg.as_dict()
        gen["gates"]["ranker"]["metrics"]["fit_pairs"] = n_fit
        if rg.passed and cand_ck:
            final = gp.ckpt / "ranker.pt"
            final.write_bytes(Path(cand_ck).read_bytes())
            acc["ranker_ckpt"] = str(final)
            gen["accepted_changes"].append("ranker")
        print(f"  ranker gate  : {'PASS' if rg.passed else 'ROLLBACK'} "
              f"cand={r_cand} accepted={r_acc} {rg.failures}", flush=True)

        # ---------------- 3. RETRAIN + 4. GATE: PUCT ---------------------
        all_px = []
        for gg in range(0, g + 1):
            all_px += read(GenerationPaths(SI, gg).stream("puct_examples"))
        held_px = read(gp.stream("puct_examples"))
        fit_px = [p for p in all_px if p not in held_px] or all_px
        if len(fit_px) < args.min_puct_examples:
            pck, n_px, rec_rep = None, len(fit_px), {"skipped": True}
            print(f"  puct         : SKIP retrain -- {len(fit_px)} examples < "
                  f"{args.min_puct_examples} (accumulating)", flush=True)
        else:
            pck, n_px, rec_rep = train_puct_candidate(
                fit_px, gp.ckpt / "policy_value_candidate.pt")
        p_cand, _ = eval_puct(pck, held_px) if pck else (None, {})
        p_acc, held_rep = eval_puct(acc["value_ckpt"], held_px)
        pg = puct_gate(p_cand, p_acc, len(held_px))
        gen["gates"]["puct"] = pg.as_dict()
        gen["gates"]["puct"]["metrics"]["fit_examples"] = n_px
        gen["gates"]["puct"]["metrics"]["reconstruction"] = rec_rep
        gen["gates"]["puct"]["metrics"]["heldout_reconstruction"] = held_rep
        if pg.passed and pck:
            final = gp.ckpt / "policy_value.pt"
            final.write_bytes(Path(pck).read_bytes())
            acc["value_ckpt"] = str(final)
            gen["accepted_changes"].append("puct_policy_value")
        print(f"  puct gate    : {'PASS' if pg.passed else 'ROLLBACK'} "
              f"cand={p_cand} accepted={p_acc} {pg.failures}", flush=True)

        # ---------------- RAG memory (always published) ------------------
        acc["rag_memory"] = str(RAG_MEMORY)   # appended per-run, above
        gen["rag_memory_rows_total"] = len(read(RAG_MEMORY))

        # ---------------- SFT corpus growth (periodic, gated) ------------
        sft_rows = []
        for gg in range(0, g + 1):
            sft_rows += read(GenerationPaths(SI, gg).stream("sft_queue"))
        aggr = aggregate_sft_targets(sft_rows)
        gen["sft"] = {"queue_rows": len(sft_rows),
                      "admitted_rows": sum(1 for r in sft_rows
                                           if r.get("admitted")),
                      "aggregated_targets": len(aggr["targets"]),
                      "rejected_groups": len(aggr["rejected"]),
                      "policy": aggr["policy"],
                      "threshold": args.sft_threshold}
        multi_seed = [t for t in aggr["targets"]
                      if len(t.get("seeds") or []) >= args.sft_min_seeds]
        gen["sft"]["multi_seed_targets"] = len(multi_seed)
        gen["sft"]["min_seeds"] = args.sft_min_seeds
        gen["sft"]["proposer_frozen"] = bool(args.freeze_proposer)
        gen["sft"]["dpo_pairs_provisional"] = len(
            read(gp.stream("proposer_dpo_pairs")))
        if args.freeze_proposer:
            gen["sft"]["retrain_required"] = False
            gen["sft"]["note"] = (
                f"proposer FROZEN by policy. {len(aggr['targets'])} aggregated "
                f"targets, {len(multi_seed)} with >={args.sft_min_seeds} "
                f"seeds. Evidence saved as PROVISIONAL; no retrain until "
                f"enough equal-budget multi-seed results exist.")
        elif len(multi_seed) >= args.sft_threshold:
            base = json.loads(Path(acc["corpus"]).read_text(encoding="utf-8"))
            prompts = {r["context_id"]: r["prompt"]
                       for r in base.get("records", [])}
            newc = build_corpus_v2(base, multi_seed,
                                   prompts_by_spec=prompts)
            cpath = gp.gdir / "corpus_v2.json"
            cpath.write_text(json.dumps(newc, indent=1, default=str),
                             encoding="utf-8")
            gen["sft"]["corpus_written"] = str(cpath)
            gen["sft"]["corpus_counts"] = newc["counts"]
            gen["sft"]["retrain_required"] = True
            gen["sft"]["note"] = ("corpus built; proposer retraining is a GPU "
                                  "step run separately, then gated by "
                                  "proposer_gates before acceptance")
        else:
            gen["sft"]["retrain_required"] = False
            gen["sft"]["note"] = (
                f"{len(aggr['targets'])} verified targets < threshold "
                f"{args.sft_threshold}; proposer NOT retrained this "
                f"generation (by design -- not per-run)")
        print(f"  sft          : {gen['sft']['aggregated_targets']} targets "
              f"(threshold {args.sft_threshold}) "
              f"retrain={gen['sft']['retrain_required']}", flush=True)

        # ---------------- 5. PUBLISH -------------------------------------
        acc["checkpoint_hashes"] = {
            "ranker": sha(acc["ranker_ckpt"]),
            "value": sha(acc["value_ckpt"]),
            "proposer": sha(acc["proposer_adapter"])}
        st["generation"] = g
        st["accepted"] = acc
        st.setdefault("history", []).append(
            {"generation": g, "accepted_changes": gen["accepted_changes"],
             "hashes": acc["checkpoint_hashes"]})
        save_state(st)
        (gp.gdir / "GENERATION.json").write_text(
            json.dumps(gen, indent=1, default=str), encoding="utf-8")
        report["generations"].append(gen)
        print(f"  published    : {acc['checkpoint_hashes']}", flush=True)

    (SI / "SELF_IMPROVEMENT_REPORT.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nwritten -> {SI}")
    return report


# --------------------------------------------------------------------------
def _synthetic_graph(obj):
    """Real CircuitGraph for a synthetic proposal, via the production path."""
    from agentic_raptor.topology_rl.value_refresh import \
        device_graph_to_circuit_graph
    from run_puct_ablation import _realise
    return device_graph_to_circuit_graph(_realise(obj), "synthetic")


def _synthetic_root_state(obj) -> dict:
    """Root state DERIVED from a real graph, never hardcoded.

    The dry run must exercise the same reconstruction and structural-hash
    verification a measured run does; a fabricated state would test nothing
    and would hide exactly the defect this rework removed.
    """
    g = _synthetic_graph(obj)
    return {"topology_id": "proposal_root", "graph_hash": g.structural_hash(),
            "lineage": [], "spec": {"target_gain_db": 60.0,
                                    "target_gbw_hz": 1e4,
                                    "minimum_phase_margin_deg": 45.0,
                                    "load_capacitance_f": 100e-12,
                                    "supply_voltage": 1.8},
            "rag_context_ids": [], "available_blocks": [],
            "legal_action_ids": [], "edit_history": [],
            "validation_status": "validated",
            "structural_features": {"n_nodes": float(len(g.nodes))},
            "previous_evidence_ref": None, "remaining_search_budget": 6,
            "remaining_spice_budget": 0, "depth": 0}


def _synthetic_manifests(by_pid: dict) -> dict:
    out = {}
    for pid, (obj, fam) in by_pid.items():
        g = _synthetic_graph(obj)
        out[pid] = {"canonical_graph_hash": f"h_{fam}",
                    "canonical_family": fam,
                    "structural_hash": g.structural_hash(),
                    "n_nodes": float(len(g.nodes)),
                    "n_edges": float(len(g.edges)),
                    "policy_prior": 0.2,
                    "visits": 29 if pid == "p00" else 20}
    return out


def _synthetic_harvest(g: int, idx: int, seed: int) -> dict:
    """Dry-run harvest: exercises routing/gates with no GPU and no ngspice.

    Deterministic and clearly labelled so a dry-run artifact can never be
    mistaken for a measured one.
    """
    import json as _json

    from agentic_raptor.llm_dpo.stage3e4 import variant_text
    fams = ["2s_none", "2s_rc", "3s_miller", "3s_rc"]
    fa, fb = fams[idx % 4], fams[(idx + 1 + g) % 4]
    # REAL corpus context ids, so the SFT corpus-growth path is genuinely
    # exercised: build_corpus_v2 needs a prompt for the spec, and a synthetic
    # id would be skipped, making a broken build look like a clean one
    sid = f"synth_{idx:03d}"
    try:
        base = _json.loads(BASE_CORPUS.read_text(encoding="utf-8"))
        ids = sorted({r["context_id"] for r in base.get("records", [])})
        if ids:
            sid = ids[idx % len(ids)]
    except Exception:
        pass

    def mk(f, ok, pid):
        obj = _json.loads(variant_text(int(f[0]), f.split("_", 1)[1],
                                       False, False))
        return obj, {
            "design": {"label": pid, "spec_id": sid,
                       "llm_proposal_id": pid,
                       "canonical_graph_hash": f"h_{f}",
                       "topology_family": f, "sizing_vector": {"s1_w": 1.0},
                       "sizing_manifest_hash": "spec_sizing.1",
                       "sizing_spice_calls": 4, "puct_rank": 0,
                       "final_netlist_hash": f"n_{f}"},
            "prediction": {"gain_db": 80.0, "pm_deg": 60.0 if ok else 10.0,
                           "normalized_margins": {"gain": 1.0,
                                                  "pm": 0.3 if ok else -0.8},
                           "predictive_uncertainty": 0.1,
                           "stability_probability": 0.9 if ok else 0.1,
                           "surrogate_checkpoint_hash": "synth"},
            "sac": {"spice_calls": 4, "algorithm": "synthetic",
                    "reward_policy": "synthetic",
                    "trajectory": [{"step": i, "knobs": {"s1_w": 1.0 + i},
                                    "reward": 0.1 * i, "pm_deg": 60.0,
                                    "gain_db": 80.0, "ugbw_hz": 1e5}
                                   for i in range(4)]},
            "authoritative": {
                "call_id": f"synth:{g}:{idx}:{seed}:{pid}",
                "topology_hash": f"h_{f}",
                "sizing_manifest_hash": "spec_sizing.1",
                "netlist_hash": f"n_{f}", "mode": "final_verification",
                "exact_spec_pass": ok, "operating_point_valid": True,
                "spice_converged": True, "verified_stable": ok,
                "spec_id": sid, "spec_hash": f"sh_{idx:03d}",
                "gain_db": 80.0, "pm_deg": 60.0 if ok else 10.0,
                "ugbw_hz": 1e5, "power_w": 1e-4,
                "normalized_distance_to_feasibility": 0.0 if ok else 0.4,
                "exact_failure_reason": None if ok else "PM below target"}}
    oa, ba = mk(fa, True, "p00")
    ob, bb = mk(fb, False, "p01")
    return {"spec": {"spec_id": sid, "gain_target_db": 60.0,
                     "phase_margin_target_deg": 45.0,
                     "ugbw_target_hz": 1e4, "load_capacitance_pf": 100.0},
            "spec_hash": f"sh_{idx:03d}", "split": "train",
            "spec_index": idx, "seed": seed, "budget": 12,
            "candidates": [
                {"llm_proposal_id": "p00", "canonical_graph_hash": f"h_{fa}",
                 "canonical_family": fa, "obj": oa, "rank": 0,
                 "visit_count": 29, "selected_top2": True},
                {"llm_proposal_id": "p01", "canonical_graph_hash": f"h_{fb}",
                 "canonical_family": fb, "obj": ob, "rank": 1,
                 "visit_count": 20, "selected_top2": True}],
            "root_visits": {"a_sel_p00": 29, "a_sel_p01": 20, "a_keep": 5},
            "candidate_visits": {"p00": 29, "p01": 20},
            "root_action_ids": ["a_keep", "a_sel_p00", "a_sel_p01"],
            "root_state": _synthetic_root_state(oa),
            "candidate_manifests": _synthetic_manifests(
                {"p00": (oa, fa), "p01": (ob, fb)}),
            "search": "single_root_over_llm_candidates",
            "branches": {"A": ba, "B": bb},
            "ranker": {"arm": "dpo_ranker", "selected_design": "A",
                       "backup_design": "B",
                       "decision_basis": "dpo_ranker",
                       "deciding_level": "learned", "low_confidence": False,
                       "score_A": 1.0, "score_B": 0.0,
                       "checkpoint_hash": "synth"},
            "proposer_checkpoint": "synthetic", "synthetic": True}


if __name__ == "__main__":
    main()
