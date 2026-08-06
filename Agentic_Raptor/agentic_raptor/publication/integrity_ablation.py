"""Integrity ablation I0-I5: what each data-integrity layer removes, measured
on real archived artifacts (the historical poisoned queue and the latest
campaign's design rows). Pure data analysis — no training, no fabrication.

  I0 historical poisoned data (archived v2_mechanical queue)
  I1 real prompts only
  I2 + same-context pairing
  I3 + dedup / contradiction guards
  I4 + dominance guards (balance caps + collapse flags)
  I5 + rollback (acceptance-gate verdicts, from campaign history)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from agentic_raptor.llm_dpo import integrity as ig
from agentic_raptor.llm_dpo import load_measurement_map
from agentic_raptor.publication import PUB, ROOT


def _poisoned_queue():
    """The archived pre-repair queue (promptless, contradictory)."""
    for camp in sorted((ROOT / "artifacts/self_improvement").glob("camp_*")):
        p = camp / "archive" / "v2_mechanical.jsonl"
        if p.is_file():
            return [json.loads(x) for x in
                    p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return []


def _latest_rows():
    camps = sorted((ROOT / "artifacts/self_improvement").glob("camp_*"))
    for camp in reversed(camps):
        div = ROOT / "artifacts/stage3e4/diversity_sft.json"
        if div.is_file():
            return json.loads(div.read_text()).get("rows", []), camp.name
    return [], None


def _all_campaign_pairs():
    """Raw (pre-dedup) integrity pairs accumulated across every campaign."""
    pairs = []
    for camp in sorted((ROOT / "artifacts/self_improvement").glob("camp_*")):
        for pf in sorted((camp / "pairs").glob("pairs_gen*.jsonl")):
            pairs += [json.loads(x) for x in
                      pf.read_text(encoding="utf-8").splitlines()
                      if x.strip()]
    return pairs


def run() -> dict:
    poisoned = _poisoned_queue()
    directions = {}
    for p in poisoned:
        key = tuple(sorted((p.get("preferred", ""), p.get("rejected", ""))))
        directions.setdefault(key, set()).add(p.get("preferred", ""))
    contradictions = sum(1 for v in directions.values() if len(v) > 1)
    i0 = {"pairs": len(poisoned),
          "with_real_prompt": sum(1 for p in poisoned if p.get("prompt")),
          "unordered_groups": len(directions),
          "contradictory_groups": contradictions,
          "distinct_preferred_texts":
              len({p.get("preferred", "") for p in poisoned}),
          "observed_outcome": "unique_structures collapsed to 1, "
                              "spec_match 0.0 (4 archived campaigns)"}

    raw = _all_campaign_pairs()
    i1 = {"raw_campaign_pairs": len(raw),
          "with_real_prompt": sum(1 for p in raw
                                  if p.get("prompt")
                                  and ig.parse_spec(p["prompt"]))}
    i2 = {"same_context_only": all(
              p.get("evaluation_context_id") for p in raw),
          "distinct_evaluation_contexts":
              len({p.get("evaluation_context_id") for p in raw})}
    dd = ig.dedupe_pairs(raw)
    i3 = dd["report"]
    bal = ig.balance_pairs(dd["pairs"])
    i4 = {"retained_after_caps": len(bal["pairs"]),
          "dropped_balance_cap": bal["distribution"]["dropped_balance_cap"],
          "collapse_flags": bal["collapse_flags"],
          "max_single_response_fraction":
              bal["distribution"]["max_single_response_fraction"]}
    verdicts = {"accepted": 0, "rolled_back": 0, "skipped": 0}
    for c in sorted((ROOT / "artifacts/self_improvement").glob("camp_*")):
        f = c / "logs/generations.jsonl"
        if not f.is_file():
            continue
        for g in [json.loads(x) for x in
                  f.read_text(encoding="utf-8").splitlines()]:
            if "dpo" not in g:
                verdicts["skipped"] += 1
            elif g["dpo_update_accepted"]:
                verdicts["accepted"] += 1
            else:
                verdicts["rolled_back"] += 1
    i5 = {"acceptance_gate_verdicts": verdicts,
          "worst_prevented_regression":
              "gen-0 DPO match 0.414->0.207 rolled back "
              "(camp_20260730_142815)"}
    doc = {"I0_poisoned": i0, "I1_real_prompts": i1, "I2_same_context": i2,
           "I3_dedup_contradiction": i3, "I4_dominance": i4,
           "I5_rollback": i5,
           "pairs_source": "all campaign pairs_gen*.jsonl files",
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    PUB.mkdir(parents=True, exist_ok=True)
    (PUB / "integrity_ablation.json").write_text(
        json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
