"""Preference-pair construction from verified historical outcomes.

The primary preference relation is LEXICOGRAPHIC — feasibility and
multi-objective priority are preserved, never collapsed into one scalar:

  a. SPICE-valid > simulation failure
  b. spec-passing > failing
  c. PVT-robust > nominal-only
  d. larger worst-case margin
  e. Pareto dominance on (worst margin, FoM, −spice_calls)
  f. higher FoM when feasibility comparable
  g. fewer SPICE calls to success
  h. lower runtime
  i. fewer topology repairs / MCTS edits

Ambiguous / nearly-tied pairs are excluded; each kept pair carries a
confidence weight derived from which rule decided it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random

from agentic_raptor.dpo.schemas import OutcomeRecord

#: rule → confidence weight (earlier rules are more decisive)
_RULE_CONFIDENCE = {"a": 1.0, "b": 1.0, "c": 0.9, "d": 0.8, "e": 0.8, "f": 0.7, "g": 0.6, "h": 0.5, "i": 0.5}


@dataclass
class PreferencePair:
    chosen: OutcomeRecord
    rejected: OutcomeRecord
    rule: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "chosen": self.chosen.to_dict(),
            "rejected": self.rejected.to_dict(),
            "rule": self.rule,
            "confidence": self.confidence,
        }


def _spec_key(record: OutcomeRecord) -> str:
    s = record.features.specifications
    return f"{s.get('circuit_class')}|{s.get('technology')}|{round(float(s.get('supply_voltage', 0)), 2)}"


def compare(a: OutcomeRecord, b: OutcomeRecord, tie_margin: float = 0.02) -> tuple[int, str]:
    """(-1: a preferred, +1: b preferred, 0: ambiguous/tied), deciding rule."""
    def decide(va: float, vb: float, rule: str, higher_better: bool = True) -> tuple[int, str] | None:
        diff = (va - vb) if higher_better else (vb - va)
        scale = max(abs(va), abs(vb), 1e-9)
        if abs(diff) / scale > tie_margin:
            return (-1 if diff > 0 else 1), rule
        return None

    if a.spice_success != b.spice_success:
        return (-1 if a.spice_success else 1), "a"
    if not a.spice_success:
        return 0, ""  # both failed simulation: ambiguous
    if a.passed_spec != b.passed_spec:
        return (-1 if a.passed_spec else 1), "b"
    pa, pb = a.pvt_pass_rate, b.pvt_pass_rate
    if (pa is not None and pa >= 0.999) != (pb is not None and pb >= 0.999):
        return (-1 if (pa is not None and pa >= 0.999) else 1), "c"
    if (r := decide(a.worst_margin, b.worst_margin, "d")) is not None:
        return r
    # e: Pareto dominance on (worst margin, FoM, −spice calls), tie-tolerant:
    # dominance requires no-worse in every dimension AND strictly better
    # (beyond tie_margin) in at least one — near-equal values never dominate.
    def better(x: float, y: float) -> bool:
        return (x - y) / max(abs(x), abs(y), 1e-9) > tie_margin

    fa, fb = a.fom or 0.0, b.fom or 0.0
    dims = [
        (a.worst_margin, b.worst_margin),
        (fa, fb),
        (-float(a.spice_calls_total), -float(b.spice_calls_total)),
    ]
    a_dom = all(not better(y, x) for x, y in dims) and any(better(x, y) for x, y in dims)
    b_dom = all(not better(x, y) for x, y in dims) and any(better(y, x) for x, y in dims)
    if a_dom != b_dom:
        return (-1 if a_dom else 1), "e"
    if (r := decide(fa, fb, "f")) is not None:
        return r
    ca = a.calls_to_first_pass if a.calls_to_first_pass is not None else a.spice_calls_total
    cb = b.calls_to_first_pass if b.calls_to_first_pass is not None else b.spice_calls_total
    if (r := decide(float(ca), float(cb), "g", higher_better=False)) is not None:
        return r
    if (r := decide(a.runtime_s, b.runtime_s, "h", higher_better=False)) is not None:
        return r
    if (r := decide(float(a.features.edit_count), float(b.features.edit_count), "i", higher_better=False)) is not None:
        return r
    return 0, ""


def build_pairs(
    records: list[OutcomeRecord],
    tie_margin: float = 0.02,
    max_pairs_per_specification: int = 50,
    seed: int = 0,
) -> list[PreferencePair]:
    """All non-ambiguous pairs among outcomes sharing a specification key."""
    by_spec: dict[str, list[OutcomeRecord]] = {}
    for record in records:
        by_spec.setdefault(_spec_key(record), []).append(record)
    rng = Random(seed)
    pairs: list[PreferencePair] = []
    for _key, group in sorted(by_spec.items()):
        group_pairs: list[PreferencePair] = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                verdict, rule = compare(group[i], group[j], tie_margin)
                if verdict == 0:
                    continue  # ambiguous / nearly tied: excluded
                chosen, rejected = (group[i], group[j]) if verdict < 0 else (group[j], group[i])
                group_pairs.append(
                    PreferencePair(chosen, rejected, rule, _RULE_CONFIDENCE[rule])
                )
        if len(group_pairs) > max_pairs_per_specification:
            group_pairs = rng.sample(group_pairs, max_pairs_per_specification)
        pairs.extend(group_pairs)
    return pairs


def split_pairs(
    pairs: list[PreferencePair], seed: int = 0, val_fraction: float = 0.15, test_fraction: float = 0.15
) -> tuple[list[PreferencePair], list[PreferencePair], list[PreferencePair]]:
    """Group-safe train/val/test split keyed by specification (no spec crosses splits)."""
    keys = sorted({_spec_key(p.chosen) for p in pairs})
    rng = Random(seed)
    rng.shuffle(keys)
    n_test = max(1, math.ceil(len(keys) * test_fraction)) if len(keys) > 2 else 0
    n_val = max(1, math.ceil(len(keys) * val_fraction)) if len(keys) > 2 else 0
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test:n_test + n_val])
    train, val, test = [], [], []
    for pair in pairs:
        key = _spec_key(pair.chosen)
        (test if key in test_keys else val if key in val_keys else train).append(pair)
    return train, val, test


class OutcomeStore:
    """Append-only JSONL store of verified outcomes (the DPO training source)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[OutcomeRecord] = []
        if self.path and self.path.is_file():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(OutcomeRecord.from_dict(json.loads(line)))

    def add(self, record: OutcomeRecord) -> None:
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.records)
