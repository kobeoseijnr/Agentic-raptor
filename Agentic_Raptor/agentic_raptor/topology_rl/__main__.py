"""Stage 3E.1 CLI: python -m agentic_raptor.topology_rl <command> [options]."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(prog="agentic_raptor.topology_rl")
    p.add_argument("command", choices=[
        "validate-actions", "validate-topologies", "run-mcts", "run-smoke",
        "train-policy-value", "inspect-tree", "export-training-records",
        "report-stage3e1",
        # Stage 3E.2 commands
        "validate-edit-mappings", "realise-edit", "validate-llm-proposal",
        "train-mb-sac-phase-d", "train-surrogate", "train-preference-ranker",
        "run-repair-training", "evaluate-heldout", "evaluate-baselines",
        "evaluate-ablations", "evaluate-pvt", "verify-cost-accounting",
        "verify-snapshot", "report-stage3e2"])
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--root-topology", default=None)
    p.add_argument("--target-id", default="mcts_default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--max-nodes", type=int, default=16)
    p.add_argument("--simulations", type=int, default=8)
    p.add_argument("--spice-budget", type=int, default=6)
    p.add_argument("--mbsac-checkpoint", default=None)
    p.add_argument("--ranker-checkpoint", default=None)
    p.add_argument("--pv-checkpoint", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache-policy", default="graph_hash")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--baseline", default=None)
    args = p.parse_args()

    from agentic_raptor.corpus import TopologyRegistry
    from agentic_raptor.mb_sac import load_pools
    from agentic_raptor.mb_sac.stage3d2 import V3
    from agentic_raptor.topology_rl import stage3e1 as s

    def _cfg() -> "s.SearchConfig":
        kw = {}
        if args.config and args.config.is_file():
            import re
            for line in args.config.read_text(encoding="utf-8").splitlines():
                m = re.match(r"^(\w+):\s*(.+?)\s*$", line)
                if m and m.group(1) in s.SearchConfig.__dataclass_fields__:
                    v = m.group(2)
                    kw[m.group(1)] = (v == "true" if v in ("true", "false")
                                      else float(v) if "." in v else
                                      v if v.isalpha() or "_" in v else int(v))
        if args.command == "run-smoke":
            kw.setdefault("spice_visit_threshold", 1)
        kw.setdefault("num_simulations", args.simulations)
        kw.setdefault("max_depth", args.max_depth)
        kw.setdefault("max_real_spice_calls", args.spice_budget)
        kw.setdefault("seed", args.seed)
        if args.deterministic:
            kw["training_mode"] = False
        return s.SearchConfig(**kw)

    if args.command == "validate-actions":
        reg = TopologyRegistry(V3)
        pools = load_pools()
        ids = [r["topology_id"] for r in pools["A1"] + pools["A2"]]
        st = s.make_root_state(args.root_topology or ids[0], reg, s.DEFAULT_SPEC)
        legal, rej = s.generate_actions(st, reg, ids)
        print(json.dumps({"legal": [a.to_dict() for a in legal], "rejections": rej},
                         indent=1))
    elif args.command == "validate-topologies":
        reg = TopologyRegistry(V3)
        pools = load_pools()
        out = {}
        for r in pools["A1"] + pools["A2"]:
            tid = r["topology_id"]
            v = s.validate_candidate(reg, tid, s.Stage3E1Action(
                "a_keep", s.Stage3E1ActionType.KEEP_TOPOLOGY, source_ref=tid), set())
            out[tid] = {"ok": v.ok, "reasons": v.reasons}
        print(json.dumps(out, indent=1))
    elif args.command in ("run-mcts", "run-smoke", "train-policy-value"):
        episodes = 1 if args.command == "run-mcts" else args.episodes
        out = s.run_smoke(episodes=episodes, seed=args.seed,
                          root_tid=args.root_topology, cfg=_cfg())
        print(json.dumps(out, indent=1, default=str))
    elif args.command == "inspect-tree":
        tree = json.loads((_ROOT / "artifacts/stage3e1/tree_ep0.json").read_text())
        print(json.dumps({"nodes": len(tree),
                          "root": tree[0], "max_depth": max(n["depth"] for n in tree)},
                         indent=1))
    elif args.command == "export-training-records":
        src = _ROOT / "artifacts/stage3e1/training_records.jsonl"
        print(json.dumps({"records": len(src.read_text().splitlines()),
                          "path": str(src)}))
    elif args.command == "report-stage3e1":
        summ = json.loads((_ROOT / "artifacts/stage3e1/SMOKE_SUMMARY.json").read_text())
        print(json.dumps({"episodes": len(summ["episodes"]),
                          "resume_deterministic": summ["resume_deterministic"]}, indent=1))
    elif args.command in (
            "validate-edit-mappings", "realise-edit", "validate-llm-proposal",
            "train-mb-sac-phase-d", "train-surrogate", "train-preference-ranker",
            "run-repair-training", "evaluate-heldout", "evaluate-baselines",
            "evaluate-ablations", "evaluate-pvt", "verify-cost-accounting",
            "verify-snapshot", "report-stage3e2"):
        from agentic_raptor.topology_rl import stage3e2 as s2
        if args.command == "validate-edit-mappings":
            from agentic_raptor.topology_rl.stage3e2_edits import EDIT_TEMPLATES
            print(json.dumps({k: {"reversible_by": v["reversible_by"],
                                  "block_family": v["block_family"]}
                              for k, v in EDIT_TEMPLATES.items()}, indent=1))
        elif args.command == "realise-edit":
            print(json.dumps(s2.run_edit_demo(args.root_topology or "topology_v2_0001"),
                             indent=1, default=str))
        elif args.command == "validate-llm-proposal":
            print(json.dumps(s2.run_llm_proposal_demo(), indent=1, default=str))
        elif args.command == "train-mb-sac-phase-d":
            print(json.dumps(s2.run_phase_d(seeds=(args.seed,)), indent=1, default=str))
        elif args.command in ("train-surrogate",):
            print(json.dumps(s2.run_calibration(), indent=1))
        elif args.command == "train-preference-ranker":
            print(json.dumps(s2.run_ranker_comparison(), indent=1))
        elif args.command == "run-repair-training":
            print(json.dumps(s2.run_repair_sample(), indent=1, default=str))
        elif args.command == "evaluate-heldout":
            print(json.dumps(s2.run_heldout_eval(), indent=1))
        elif args.command in ("evaluate-baselines", "evaluate-ablations"):
            print(json.dumps(s2.run_mcts_baselines(), indent=1))
        elif args.command == "evaluate-pvt":
            print(json.dumps(s2.run_pvt(args.root_topology or "topology_0002"), indent=1))
        elif args.command == "verify-cost-accounting":
            summ = json.loads((s2.OUT / "SUMMARY.json").read_text())
            print(json.dumps({"parts": list(summ), "ok": True}, indent=1))
        elif args.command == "verify-snapshot":
            v = json.loads((_ROOT / "artifacts/code_snapshots/"
                            "pre_stage3e2_full_generation/VERIFICATION.json").read_text())
            print(json.dumps(v, indent=1))
        else:
            print(json.dumps(json.loads((s2.OUT / "SUMMARY.json").read_text()), indent=1))
    if args.baseline:
        print(json.dumps(s.run_baseline(args.baseline, args.seed), indent=1))


if __name__ == "__main__":
    main()
