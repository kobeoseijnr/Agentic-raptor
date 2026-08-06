"""Stage 3A: simulator-grounded dataset generation.

Field provenance convention (correction 7), used in every schema below via the
`field_provenance` map: OBSERVED = measured/recorded directly (SPICE output,
validator verdict, LLM text); DERIVED = deterministically computed from observed
data (hashes, margins, rewards, splits); PREDICTED = model estimates (dynamics/
ranker outputs) — never usable as ground-truth labels.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

SCHEMA_VERSION = "3a.0.1"
EXECUTION_MODES = ("REAL", "MOCK", "SYNTHETIC_TEST", "REPLAYED_REAL")


# -- identifiers (all DERIVED, deterministic) --------------------------------
def canonical_spec_hash(spec: dict[str, Any]) -> str:
    keep = {k: spec.get(k) for k in sorted(spec) if k != "source_metadata"}
    return "spec-" + hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def content_id(prefix: str, payload: Any) -> str:
    return f"{prefix}-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def fresh_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# -- content-addressed raw store (correction 2: copy, never move) ------------
class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put_text(self, kind: str, text: str, ext: str = ".txt") -> tuple[str, str]:
        digest = hashlib.sha256(text.encode()).hexdigest()[:20]
        p = self.root / kind / f"{digest}{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(text, encoding="utf-8")
        return digest, str(p)

    def copy_file(self, kind: str, source: str | Path) -> tuple[str, str] | None:
        src = Path(source)
        if not src.is_file():
            return None
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:20]
        dest = self.root / kind / f"{digest}{src.suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)  # original preserved in place
        return digest, str(dest)


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return len(rows)


# -- group-safe splitter (correction 3: fixture-tested; empty debug allowed) --
SPLITS = ("train", "validation", "test")


def assign_splits(groups: list[str], seed: int, ratios=(0.7, 0.15, 0.15)) -> dict[str, str]:
    ordered = sorted(set(groups))
    Random(seed).shuffle(ordered)
    n = len(ordered)
    n_test = round(n * ratios[2])
    n_val = round(n * ratios[1])
    out: dict[str, str] = {}
    for i, g in enumerate(ordered):
        out[g] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    return out


# -- validators ---------------------------------------------------------------
def validate_records(rows: list[dict], required: list[str], real_only: bool) -> tuple[list[dict], list[dict]]:
    ok, quarantine = [], []
    for r in rows:
        problems = [k for k in required if r.get(k) in (None, "", [])]
        if real_only and r.get("execution_mode") != "REAL":
            problems.append(f"execution_mode={r.get('execution_mode')} in real-only dataset")
        if r.get("real_or_imagined") == "MODEL":
            problems.append("imagined transition in real dataset")
        if r.get("reached_spice") is True and not r.get("spice_evaluation_id"):
            problems.append("reached_spice without spice_evaluation_id")
        if problems:
            quarantine.append({"record": r, "reasons": problems})
        else:
            ok.append(r)
    return ok, quarantine


def check_leakage(datasets: dict[str, list[dict]]) -> dict[str, Any]:
    report: dict[str, Any] = {"prohibited_leakage": 0, "checks": []}
    for name, rows in datasets.items():
        by_split: dict[str, set] = {}
        for r in rows:
            by_split.setdefault(r.get("split", "?"), set()).add(r.get("group_key"))
        splits = list(by_split)
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap = by_split[splits[i]] & by_split[splits[j]]
                if overlap:
                    report["prohibited_leakage"] += len(overlap)
                    report["checks"].append({"dataset": name, "splits": [splits[i], splits[j]], "groups": sorted(overlap)})
    report["checks"].append("group overlap across splits: " + ("NONE" if report["prohibited_leakage"] == 0 else "FOUND"))
    return report


def duplicate_report(rows: list[dict], key: str) -> dict[str, Any]:
    seen: dict[str, int] = {}
    for r in rows:
        seen[r.get(key, "?")] = seen.get(r.get(key, "?"), 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    return {"total": len(rows), "unique": len(seen), "duplicate_keys": dups,
            "duplicate_fraction": (len(rows) - len(seen)) / len(rows) if rows else 0.0}
