"""Publication figures — generated ONLY from saved experiment artifacts.
Each figure records the exact source file it was drawn from."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic_raptor.publication import FREEZE, PUB, ROOT

FIG = PUB / "figures"


def _save(fig, name, source):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.suptitle(fig._suptitle.get_text() if fig._suptitle else name,
                 fontsize=11)
    fig.text(0.99, 0.01, f"source: {source}", ha="right", fontsize=6,
             color="#888")
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(FIG / f"{name}.png")


def fig_learning_curves() -> list:
    agg = json.loads((PUB / "AGGREGATE.json").read_text())
    rows = agg["campaign_matrix"]["rows"]
    out = []
    for metric, label in (("match", "structure match (frozen validation)"),
                          ("exact_pass", "exact electrical pass rate"),
                          ("verified_earned", "verified self-earned/gen"),
                          ("unique", "unique structures")):
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        arms = sorted({r["arm"] for r in rows})
        for arm in arms:
            pts = {}
            for r in rows:
                if r["arm"] == arm and r[metric] is not None:
                    pts.setdefault(r["generation"], []).append(r[metric])
            if not pts:
                continue
            xs = sorted(pts)
            ys = [sum(pts[x]) / len(pts[x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=f"{arm} (n={max(len(v) for v in pts.values())})")
        ax.set_xlabel("generation")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.suptitle(label)
        out.append(_save(fig, f"curve_{metric}", "AGGREGATE.json"))
    return out


def fig_sizing_baselines() -> list:
    p = PUB / "sizing_baselines.json"
    if not p.is_file():
        return []
    d = json.loads(p.read_text())
    agg = {}
    for r in d["rows"]:
        agg.setdefault(r["method"], []).append(r)
    methods = sorted(agg)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(methods, [sum(x["exact_pass"] for x in agg[m]) for m in methods])
    axes[0].set_ylabel(f"exact passes / {len(next(iter(agg.values())))} tasks")
    axes[0].set_title("exact-spec passes (equal budget)")
    axes[1].bar(methods, [sum((1.0 if x["distance"] is None else x["distance"]) for x in agg[m]) / len(agg[m])
                          for m in methods], color="#c66")
    axes[1].set_title("mean distance to feasibility (lower=better)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"sizing baselines, budget {d['budget']} SPICE calls")
    return [_save(fig, "sizing_baselines", "sizing_baselines.json")]


def fig_reward_ablation() -> list:
    p = PUB / "reward_ablation.json"
    if not p.is_file():
        return []
    d = json.loads(p.read_text())
    agg = {}
    for r in d["rows"]:
        agg.setdefault(r["reward"], []).append(r)
    ks = sorted(agg)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(ks, [sum((1.0 if x["distance"] is None else x["distance"]) for x in agg[k]) / len(agg[k])
                for k in ks], color="#557")
    ax2 = ax.twinx()
    ax2.plot(ks, [sum(x["pm_excess_deg"] or 0 for x in agg[k]) / len(agg[k])
                  for k in ks], "r-o", label="mean PM excess (deg)")
    ax.set_ylabel("mean distance to feasibility")
    ax2.set_ylabel("mean PM excess (deg)", color="r")
    fig.suptitle("reward ablation: distance vs PM-excess chasing")
    return [_save(fig, "reward_ablation", "reward_ablation.json")]


def fig_failure_modes() -> list:
    p = FREEZE / "feasibility_audit.json"
    if not p.is_file():
        return []
    d = json.loads(p.read_text())
    counts = d["classification_counts"]
    ks = [k for k, v in counts.items() if v]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh(ks, [counts[k] for k in ks], color="#575")
    ax.set_xlabel("task profiles")
    fig.suptitle("feasibility audit / failure-mode classification")
    return [_save(fig, "failure_modes", "feasibility_audit.json")]


def fig_qwen_ablation() -> list:
    p = PUB / "qwen_ablation" / "SUMMARY.json"
    if not p.is_file():
        return []
    d = json.loads(p.read_text())
    arms = [a for a in sorted(d) if "unavailable" not in d[a]]
    if not arms:
        return []
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = range(len(arms))
    ax.bar([i - 0.2 for i in x], [d[a]["structure_match"] for a in arms],
           width=0.4, label="structure match")
    ax.bar([i + 0.2 for i in x], [d[a]["first_attempt_validity"]
                                  for a in arms],
           width=0.4, label="first-attempt validity")
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Qwen proposal quality: L-arms (frozen validation)")
    return [_save(fig, "qwen_ablation", "qwen_ablation/SUMMARY.json")]


def all_figures() -> list:
    figs = []
    for fn in (fig_learning_curves, fig_sizing_baselines,
               fig_reward_ablation, fig_failure_modes, fig_qwen_ablation):
        try:
            figs += fn()
        except Exception as exc:
            figs.append(f"SKIPPED {fn.__name__}: {exc}")
    return figs


if __name__ == "__main__":
    for f in all_figures():
        print(f)
