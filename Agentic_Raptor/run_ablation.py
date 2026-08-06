"""Campaign ablation orchestrator with RESUME (arm x seed matrices).

Batteries (main/SE also record AlphaZero's topology DECISION per design
context via the "search" channel; the F battery does not, so its
feedback-channel comparison stays single-variable):
  --battery main   arms full/sft_only/dpo_no_integrity/dpo_no_gate x seeds
  --battery se     self-earned policies se0,se1,se2,se4,se5 (se3/se6 ==
                   production policy, already covered by the main 'full' arm)
  --battery f      feedback channels F0-F7 (channel sets below)

Completed campaigns are skipped; interrupted ones resume at their last
completed generation. Identity = (arm, seed, se_arm, channels) read from the
campaign's own log.

Run:  python run_ablation.py                       # main matrix
      python run_ablation.py --battery se --seeds 11
      python run_ablation.py --battery f  --seeds 11
      python run_ablation.py --seeds 71,101        # extra headline seeds
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SI = Path("artifacts/self_improvement").resolve()
ALL_CH = "replay,surrogate,ranker,l4,value,selfearn"
#: AlphaZero decision recorder. Deliberately NOT part of ALL_CH: the F
#: battery ablates FEEDBACK channels, and folding a decision channel into F7
#: would make F6-vs-F7 measure two changes at once. The main and SE batteries
#: carry it so the headline campaigns produce decision evidence -- for every
#: design context the search states which structure it would deliver, and
#: since the campaign sizes every candidate anyway, that call is scored
#: against the measured outcomes at no extra SPICE cost.
SEARCH_CH = "search"
MAIN_CH = f"{ALL_CH},{SEARCH_CH}"
F_SETS = {"F0": "none", "F1": "replay",
          "F2": "replay,surrogate,ranker",
          "F3": "replay,l4", "F4": "replay,value", "F5": "replay,selfearn",
          "F6": "replay,surrogate,ranker,l4,value", "F7": ALL_CH}


def plan_for(battery, arms, seeds, main_ch=MAIN_CH):
    if battery == "main":
        return [{"arm": a, "seed": s, "se_arm": "se6", "channels": main_ch,
                 "label": f"{a}/s{s}"} for a in arms for s in seeds]
    if battery == "se":
        return [{"arm": "sft_only", "seed": s, "se_arm": se,
                 "channels": main_ch, "label": f"{se}/s{s}"}
                for se in ("se0", "se1", "se2", "se4", "se5") for s in seeds]
    if battery == "f":
        return [{"arm": "sft_only", "seed": s, "se_arm": "se6",
                 "channels": ch, "label": f"{f}/s{s}"}
                for f, ch in F_SETS.items() for s in seeds]
    raise SystemExit(f"unknown battery {battery}")


ENGINE = "sac_v2_pretrained"


def _identity(gens):
    g = gens[0]
    ch = g.get("channels")
    # channels==[] means NONE (F0); only channels==None (pre-battery
    # campaigns) defaults to ALL -- `or` would wrongly conflate the two
    ch_str = ",".join(sorted(ch)) if ch is not None else         ",".join(sorted(ALL_CH.split(",")))
    return (g.get("arm", "full"), g.get("campaign_seed", 0),
            g.get("se_arm", "se6"), ch_str,
            g.get("engine", "legacy"), g.get("profile", "full"))


def find(item, n_gen):
    want = (item["arm"], item["seed"], item["se_arm"],
            ",".join(sorted(set(item["channels"].split(","))))
            if item["channels"] != "none" else "", ENGINE,
            item.get("profile", "full"))
    complete = partial = None
    part_gens = 0
    for camp in sorted(SI.glob("camp_*")):
        f = camp / "logs/generations.jsonl"
        if not f.is_file():
            continue
        gens = [json.loads(x) for x in
                f.read_text(encoding="utf-8").splitlines()]
        if not gens:
            continue
        have = _identity(gens)
        if have != want:
            continue
        if len(gens) >= n_gen:
            complete = camp.name
        elif len(gens) > part_gens:
            partial, part_gens = camp.name, len(gens)
    return complete, partial, part_gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="main", choices=["main", "se", "f"])
    ap.add_argument("--arms",
                    default="full,sft_only,dpo_no_integrity,dpo_no_gate")
    ap.add_argument("--seeds", default="11,23,47")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--fast", action="store_true",
                    help="uniform fast profile for every arm in the battery")
    ap.add_argument("--no-search", action="store_true",
                    help="drop the AlphaZero decision recorder from the "
                         "main/SE batteries (F battery never carries it)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    plan = plan_for(args.battery, args.arms.split(","), seeds,
                    main_ch=ALL_CH if args.no_search else MAIN_CH)
    print(f"battery '{args.battery}': {len(plan)} campaigns x "
          f"{args.generations} generations")
    for item in plan:
        item["profile"] = "fast" if args.fast else "full"
    for i, item in enumerate(plan):
        done, partial, n = find(item, args.generations)
        if done:
            print(f"[{i+1}/{len(plan)}] {item['label']}: complete ({done}) "
                  f"— skipping")
            continue
        cmd = [sys.executable, "-u", "run_self_improvement.py",
               str(args.generations), "--arm", item["arm"],
               "--seed", str(item["seed"]), "--se-arm", item["se_arm"],
               "--channels", item["channels"]]             + (["--fast"] if args.fast else [])
        if partial:
            print(f"[{i+1}/{len(plan)}] {item['label']}: RESUMING {partial} "
                  f"at gen {n} ({time.strftime('%H:%M:%S')})")
            cmd += ["--resume-campaign", partial]
        else:
            print(f"[{i+1}/{len(plan)}] {item['label']}: starting "
                  f"({time.strftime('%H:%M:%S')})")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"FAILED (exit {r.returncode}) — fix and re-run; completed "
                  f"work is preserved")
            sys.exit(r.returncode)
    print("battery complete — aggregate with: "
          "python -m agentic_raptor.publication.aggregate")


if __name__ == "__main__":
    main()
