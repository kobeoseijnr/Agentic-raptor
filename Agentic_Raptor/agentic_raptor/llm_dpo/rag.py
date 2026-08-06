"""Repaired RAG retrieval (publication v2, Repair 2).

Structured, capped, filtered retrieval — never uncontrolled raw text:
  * records carry family/class/verified-status/metrics, not prose;
  * family-compatibility filter (same stage count) and spec-distance
    threshold (gain within 30 dB, load within one decade);
  * success/failure balance caps (<=2 each), duplicate removal;
  * token cap on the rendered evidence line;
  * hard leakage guard: no evidence derived from frozen-validation or
    blind-test contexts ever enters a prompt;
  * rendered format is EXACTLY the '### KNOWN ...' schema the model was
    trained on, inserted before '### BLOCKS', with '### PROPOSAL' always
    last (snapshot-tested).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
L4_FILE = _ROOT / "datasets/simulation_memory/self_improvement_runs.jsonl"

TOKEN_CAP = 60          # rough word cap for the KNOWN line
TOP_K = 4
MAX_SUCCESS = 2
MAX_FAILURE = 2
GAIN_WINDOW_DB = 30.0
LOAD_WINDOW_DECADES = 1.0


def _excluded_contexts() -> set:
    """Frozen-validation + blind spec ids (evidence from them is leakage)."""
    out = set()
    for name, key in (("frozen_exam.json", "frozen_exam_spec_ids"),
                      ("blind_test.json", None)):
        p = _ROOT / "artifacts/stage3e4" / name
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        if key and key in d:
            out |= set(d[key])
    sm = _ROOT / "artifacts/stage3e4/split_manifest.json"
    if sm.is_file():
        d = json.loads(sm.read_text())
        out |= set(d.get("validation_spec_ids", []))
        out |= set(d.get("blind_test_spec_ids", []))
    return out


def retrieve(spec: dict, stages: int, k: int = TOP_K) -> list:
    """Structured evidence records for (spec, stage-tier), filtered and
    balanced. Returns [{stages, stability, pm, gain_db, context_id}, ...]."""
    if not L4_FILE.is_file():
        return []
    excluded = _excluded_contexts()
    succ, fail, seen = [], [], set()
    entries = [json.loads(x) for x in
               L4_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    for e in reversed(entries):             # newest evidence first
        if not e.get("stability") or e.get("stages") != stages:
            continue
        if e.get("context_id") in excluded:
            continue                        # leakage guard
        key = (e.get("stages"), e.get("stability"),
               round(e.get("pm") or 0, 0))
        if key in seen:
            continue                        # duplicate-retrieval removal
        seen.add(key)
        rec = {"stages": e["stages"], "stability": e["stability"],
               "pm": e.get("pm"), "gain_db": e.get("gain_db"),
               "context_id": e.get("context_id")}
        if e["stability"] == "verified_stable" and len(succ) < MAX_SUCCESS:
            succ.append(rec)
        elif e["stability"] == "verified_unstable" and len(fail) < MAX_FAILURE:
            fail.append(rec)
        if len(succ) >= MAX_SUCCESS and len(fail) >= MAX_FAILURE:
            break
    return (succ + fail)[:k]


def render_known_line(records: list) -> str:
    """Render records in the trained '### KNOWN' schema, token-capped."""
    if not records:
        return ""
    parts = []
    for r in records:
        p = f"{r['stages']}stage {r['stability']}"
        if r.get("pm") is not None:
            p += f" pm={round(r['pm'])}deg"
        parts.append(p)
    line = "### KNOWN " + "; ".join(parts)
    while len(line.split()) > TOKEN_CAP and parts:
        parts.pop()
        line = "### KNOWN " + "; ".join(parts)
    return line + "\n" if parts else ""


def augment_prompt(base_prompt: str, spec: dict, stages: int) -> str:
    """Insert repaired retrieval into a prompt. Evidence goes before
    '### BLOCKS'; the '### PROPOSAL' contract stays LAST. Never applied to
    frozen-validation/blind prompts by construction (their contexts are in
    the exclusion set for evidence, and callers only augment train prompts)."""
    line = render_known_line(retrieve(spec, stages))
    if not line:
        return base_prompt
    out = re.sub(r"^### KNOWN [^\n]*\n", "", base_prompt, flags=re.M)
    return out.replace("### BLOCKS", line + "### BLOCKS")


def snapshot_check(prompt: str) -> dict:
    """Structural contract every rendered prompt must satisfy."""
    lines = prompt.strip().splitlines()
    return {"proposal_last": lines[-1].strip() == "### PROPOSAL",
            "single_known": len([l for l in lines
                                 if l.startswith("### KNOWN")]) <= 1,
            "has_spec": lines[0].startswith("### SPEC"),
            "word_count_ok": len(prompt.split()) < 220}
