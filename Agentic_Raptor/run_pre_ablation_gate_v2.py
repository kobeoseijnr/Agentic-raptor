"""Pre-ablation gate (publication v2, Part 13).

Eighteen checks that must all hold before any publication ablation runs.
Static checks run offline; checks that require a real runtime trace read the
artefacts a canonical run leaves behind, and report NOT VERIFIED when no
trace exists rather than passing on the strength of the code being present.

The distinction matters: the architecture audit failed precisely because
modules existed while the runtime did something else. A gate that inspects
only source would repeat that error.

Run:  python run_pre_ablation_gate_v2.py [--json]
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "artifacts/publication_v2"
TRACES = V2 / "raptor_v2_runs"

PASS, FAIL, NOTVER = "PASS", "FAIL", "NOT VERIFIED"


class Check:
    def __init__(self, n, title):
        self.n, self.title = n, title
        self.status, self.detail = NOTVER, ""

    def set(self, status, detail=""):
        self.status, self.detail = status, detail
        return self

    def row(self):
        return {"n": self.n, "check": self.title, "status": self.status,
                "detail": self.detail}


def _latest_trace():
    if not TRACES.is_dir():
        return None
    files = sorted(TRACES.glob("TRACE_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("result") == "OK":
                return d
        except Exception:
            continue
    return None


def run_checks() -> list:
    t = _latest_trace()
    cks = []

    # 1 accepted SFT-only proposer checkpoint
    c = Check(1, "Accepted SFT-only proposer checkpoint loaded")
    if t is None:
        c.set(NOTVER, "no canonical runtime trace present")
    else:
        m = (t.get("models") or {}).get("proposer") or {}
        c.set(PASS if m.get("training_method") == "SFT" and m.get("hash")
              else FAIL,
              f"training_method={m.get('training_method')} "
              f"hash={str(m.get('hash'))[:12]}")
    cks.append(c)

    # 2 five candidates generated directly by the LLM
    c = Check(2, "Five candidates generated directly by the LLM")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        p = t.get("stage3_propose") or {}
        srcs = set(p.get("sources") or [])
        c.set(PASS if p.get("distinct") == 5 and srcs <= {"llm"} else FAIL,
              f"distinct={p.get('distinct')} sources={sorted(srcs)}")
    cks.append(c)

    # 3 unique by FULL canonical graph hash
    c = Check(3, "Five unique by full canonical graph hash")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        h = (t.get("stage3_propose") or {}).get("canonical_graph_hashes") or []
        c.set(PASS if len(h) == 5 and len(set(h)) == 5 else FAIL,
              f"{len(set(h))} unique of {len(h)}")
    cks.append(c)

    # 4 no enumeration in the v2 path (static: source inspection)
    c = Check(4, "No enumeration in the v2 path")
    try:
        src = (ROOT / "run_raptor_v2.py").read_text(encoding="utf-8")
        bad = [k for k in ("variant_text(", "mcts_enumerated",
                           "CORPUS_CLASSES") if k in src]
        c.set(PASS if not bad else FAIL,
              "clean" if not bad else f"found {bad}")
    except FileNotFoundError:
        c.set(FAIL, "run_raptor_v2.py missing")
    cks.append(c)

    # 5 one PUCT root with exactly five LLM actions
    # 2026-08-11: this check is specific to the RETIRED root-level PUCT
    # architecture (exactly one "a_sel_*" action per LLM candidate at a
    # single root). TRUE_ALPHAZERO's root legal-action set is richer
    # (keep/term/select-per-seed/real-edit actions) and has no reason to
    # equal 5 -- reported NOT VERIFIED for post-migration traces rather
    # than silently reinterpreted or left to report a misleading FAIL.
    c = Check(5, "One PUCT root has exactly five LLM actions (retired-"
             "architecture check)")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    elif "stage5_alphazero" in t:
        c.set(NOTVER, "root-level PUCT retired -- see TRUE_ALPHAZERO's own "
             "tree_nodes/max_depth_reached in stage5_alphazero instead")
    else:
        s5 = t.get("stage5_puct") or {}
        c.set(PASS if (s5.get("root_action_count") == 5
                       and s5.get("single_root") is True) else FAIL,
              f"actions={s5.get('root_action_count')} "
              f"single_root={s5.get('single_root')}")
    cks.append(c)

    # 6 topology search returns exactly two
    c = Check(6, "Topology search returns exactly two")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        s5 = t.get("stage5_alphazero") or t.get("stage5_puct") or {}
        c.set(PASS if s5.get("selected_count") == 2 else FAIL,
              f"selected={s5.get('selected_count')}")
    cks.append(c)

    # 7 two independent sizing branches
    c = Check(7, "SAC sizes two independent branches")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        s6 = t.get("stage6_sizing") or {}
        ok = (set(s6) == {"A", "B"}
              and s6.get("A", {}).get("topology_hash")
              != s6.get("B", {}).get("topology_hash"))
        c.set(PASS if ok else FAIL, f"branches={sorted(s6)}")
    cks.append(c)

    # 8 branch hashes unchanged PUCT -> SAC
    c = Check(8, "Branch hashes unchanged between PUCT and SAC")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        sel = set((t.get("provenance_chain") or {}).get("puct_selected") or [])
        sized = {v.get("topology_hash")
                 for v in (t.get("stage6_sizing") or {}).values()}
        c.set(PASS if sel and sel == sized else FAIL,
              f"selected={len(sel)} sized={len(sized)} equal={sel == sized}")
    cks.append(c)

    # 9 surrogate predictions carry no authoritative ids
    c = Check(9, "Surrogate predictions have no authoritative SPICE ids")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        s7 = (t.get("stage7_surrogate") or {})
        bad = [k for k, v in s7.items()
               if v.get("authoritative") or v.get("spice_result_id")
               or v.get("source") != "surrogate"]
        c.set(PASS if s7 and not bad else FAIL,
              "clean" if s7 and not bad else f"leaking={bad or 'no data'}")
    cks.append(c)

    # 10 ranker checkpoint DPO-trained and frozen
    c = Check(10, "Post-SAC ranker checkpoint is DPO-trained and frozen")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        m = (t.get("models") or {}).get("ranker") or {}
        c.set(PASS if (m.get("training_method") == "DPO"
                       and m.get("frozen") is True and m.get("hash"))
              else FAIL,
              f"method={m.get('training_method')} frozen={m.get('frozen')}")
    cks.append(c)

    # 11 ranker receives two fully sized designs
    c = Check(11, "Ranker receives two fully sized designs")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        s8 = t.get("stage8_ranker") or {}
        c.set(PASS if s8.get("input_count") == 2
              and s8.get("both_sized") is True else FAIL,
              f"inputs={s8.get('input_count')} sized={s8.get('both_sized')}")
    cks.append(c)

    # 12 learned ranker participates within equal safety tiers
    c = Check(12, "Learned ranker participates inside equal safety tiers")
    try:
        from agentic_raptor.ranking import post_sac
        src = inspect.getsource(post_sac.compare)
        static = ("model.score(" in src
                  and "RankerCheckpointMissing" in src
                  and "except Exception as exc" in src)
    except Exception as exc:
        static, src = False, str(exc)
    if t is None:
        c.set(PASS if static else FAIL,
              "static: learned path reachable, missing-checkpoint fails loudly"
              if static else "static check failed")
    else:
        basis = (t.get("stage8_ranker") or {}).get("decision_basis")
        c.set(PASS if static and basis in ("dpo_ranker", "hard_safety_gate")
              else FAIL, f"basis={basis}")
    cks.append(c)

    # 13 one selected + one backup
    c = Check(13, "Ranker outputs one selected and one backup")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        s8 = t.get("stage8_ranker") or {}
        c.set(PASS if s8.get("selected_design") in ("A", "B")
              and s8.get("backup_design") in ("A", "B")
              and s8.get("selected_design") != s8.get("backup_design")
              else FAIL,
              f"selected={s8.get('selected_design')} "
              f"backup={s8.get('backup_design')}")
    cks.append(c)

    # 14 a NEW final ngspice call after selection
    c = Check(14, "New authoritative ngspice call after ranker selection")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        s9 = t.get("stage9_verification") or {}
        fid = s9.get("final_spice_call_id")
        sizing_ids = set(s9.get("sizing_spice_call_ids") or [])
        c.set(PASS if (fid and fid not in sizing_ids
                       and s9.get("final_verification_source") == "ngspice")
              else FAIL,
              f"final_call_id={fid} reused_sizing_id={fid in sizing_ids}")
    cks.append(c)

    # 15 backup verified only when policy requires
    c = Check(15, "Backup receives a final SPICE call only when triggered")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        s9 = t.get("stage9_verification") or {}
        sim = bool(s9.get("second_design_simulated"))
        reason = s9.get("second_design_reason")
        c.set(PASS if (sim == bool(reason)) else FAIL,
              f"simulated={sim} reason={reason}")
    cks.append(c)

    # 16 trusted pairs need two authoritative same-spec outcomes
    c = Check(16, "Trusted DPO pairs require two authoritative outcomes")
    try:
        from agentic_raptor.ranking import post_sac
        src = inspect.getsource(post_sac._provenance_ok)
        need = ("not_authoritative", "topology_hash_mismatch",
                "sizing_manifest_mismatch", "missing_netlist_hash",
                "spec_id_mismatch")
        miss = [k for k in need if k not in src]
        c.set(PASS if not miss else FAIL,
              "full provenance enforced" if not miss else f"missing {miss}")
    except Exception as exc:
        c.set(FAIL, str(exc)[:120])
    cks.append(c)

    # 17 branch-specific feedback written
    c = Check(17, "Complete branch-specific feedback written")
    if t is None:
        c.set(NOTVER, "no runtime trace")
    else:
        fb = t.get("stage11_feedback") or {}
        c.set(PASS if fb.get("branch_assertions_passed") else FAIL,
              f"routing={fb.get('routed')} "
              f"assertions={fb.get('branch_assertions_passed')}")
    cks.append(c)

    # 18 evaluation / blind-test leakage blocked
    c = Check(18, "Evaluation and blind-test leakage blocked")
    try:
        from agentic_raptor.publication.eval_sets import (available_seeds,
                                                          excluded_context_ids,
                                                          load)
        seeds = available_seeds()
        ids = excluded_context_ids()
        for s in seeds:
            load(s)                      # re-verifies content hashes
        c.set(PASS if seeds and ids else FAIL,
              f"seeds={seeds} excluded_ids={len(ids)} hashes verified")
    except Exception as exc:
        c.set(FAIL, f"{type(exc).__name__}: {exc}"[:140])
    cks.append(c)

    return cks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    checks = run_checks()
    rows = [c.row() for c in checks]
    n_pass = sum(1 for r in rows if r["status"] == PASS)
    n_fail = sum(1 for r in rows if r["status"] == FAIL)
    n_nv = sum(1 for r in rows if r["status"] == NOTVER)
    overall = (n_fail == 0 and n_nv == 0)
    if args.json:
        print(json.dumps({"checks": rows, "pass": n_pass, "fail": n_fail,
                          "not_verified": n_nv,
                          "pre_ablation_pass": overall}, indent=1))
    else:
        for r in rows:
            mark = {PASS: "[PASS]", FAIL: "[FAIL]",
                    NOTVER: "[----]"}[r["status"]]
            print(f"{mark} {r['n']:2}. {r['check']}")
            if r["detail"]:
                print(f"        {r['detail']}")
        print(f"\n{n_pass} pass, {n_fail} fail, {n_nv} not verified")
        if n_nv:
            print("NOT VERIFIED means no canonical runtime trace exists yet; "
                  "run run_raptor_v2.py to produce one.")
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / "pre_ablation_gate.json").write_text(
        json.dumps({"checks": rows, "pre_ablation_pass": overall}, indent=1),
        encoding="utf-8")
    print(f"\nPRE-ABLATION PASS: {overall}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
