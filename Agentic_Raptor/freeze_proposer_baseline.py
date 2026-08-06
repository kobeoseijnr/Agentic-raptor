"""Step 1: freeze the pre-repair proposer failure as an immutable baseline.

Written before any retraining touches the corpus or the checkpoint. After
Case C the old numbers become unreproducible -- the corpus changes, the
adapter changes -- so the failure evidence has to be captured now or it is
lost, and "the proposer used to collapse" becomes an assertion rather than a
measurement.

Run:  python freeze_proposer_baseline.py
"""
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "artifacts/publication_v2"
OUT = V2 / "proposer_repair"
DIVERSITY = V2 / "diversity/SUMMARY.json"
GATE = V2 / "pre_ablation_gate.json"
REPORT = V2 / "architecture_repair_report.md"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from agentic_raptor.ranking import directory_sha256
    from run_qwen_ablation import resolve_arms

    div = json.loads(DIVERSITY.read_text(encoding="utf-8"))
    corpus = json.loads(
        (ROOT / "artifacts/stage3e4/corpus.json").read_text(encoding="utf-8"))
    arms = resolve_arms()
    arm = div.get("arm", "L8")
    ckpt = arms.get(arm, {}).get("adapter")

    rows = div["rows"]
    baseline = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": "PROPOSER DIVERSITY PASS: False",
        "proposer": {
            "arm_label": arm,
            "checkpoint_path": ckpt,
            "checkpoint_sha256": directory_sha256(ckpt) if ckpt else None,
            "training_method": "SFT",
            "note": "arm labels are ambiguous; the SHA-256 is the identity"},
        "corpus": {
            "corpus_hash": corpus.get("split_manifest", {}).get("corpus_hash"),
            "split_hash": corpus.get("split_manifest", {}).get("split_hash"),
            "records": len(corpus["records"]),
            "targets_per_prompt": 1,
            "target_rule": "stages = 1 if gain<30 else 2 if gain<70 else 3; "
                           "comp = none/rc/miller by load and PM",
            "measured_rule_conformance": "85/85 (100%)"},
        "evaluated_spec_ids": [r["context_id"] for r in rows],
        "target_k": div["target_k"],
        "results": {
            "baseline_mean_distinct": div["baseline_mean_distinct"],
            "ladder_mean_distinct": div["ladder_mean_distinct"],
            "specs_reaching_target": div["specs_reaching_target"],
            "share_reaching_target": div["share_reaching_target"],
            "mean_attempts": div["mean_attempts"],
            "max_attempts_budget": 20,
            "max_temperature_used": max(r["max_temperature"] for r in rows),
            "classes_ever_produced": sorted(
                {c for r in rows for c in r["classes"]}),
            "classes_never_produced": sorted(
                {"2s_none", "2s_miller", "3s_none", "3s_miller", "3s_rc"}
                - {c for r in rows for c in r["classes"]})},
        "per_spec": rows,
        "root_cause": (
            "Training distribution is a delta per specification: every one of "
            "the 85 train prompts has exactly ONE target, and 85/85 match the "
            "deterministic stage-tier rule. Diversity absent from the data "
            "cannot be recovered by decoding temperature -- measured: 20 "
            "attempts at T=1.5 yielded at most 2 distinct graphs."),
        "preserved_files": []}

    for src in (DIVERSITY, GATE, REPORT):
        if src.is_file():
            dst = OUT / f"baseline_{src.name}"
            shutil.copy2(src, dst)
            baseline["preserved_files"].append(str(dst.relative_to(ROOT)))

    (OUT / "baseline_diversity.json").write_text(
        json.dumps(baseline, indent=1), encoding="utf-8")
    print(json.dumps({k: baseline[k] for k in
                      ("verdict", "proposer", "corpus", "results")},
                     indent=1))
    print(f"\nfrozen -> {(OUT / 'baseline_diversity.json').relative_to(ROOT)}")
    for f in baseline["preserved_files"]:
        print(f"preserved -> {f}")


if __name__ == "__main__":
    main()
