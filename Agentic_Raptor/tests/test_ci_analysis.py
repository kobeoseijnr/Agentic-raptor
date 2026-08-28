"""Paired bootstrap CI analysis (2026-08-17)."""
from __future__ import annotations

import json
import math

from agentic_raptor.publication import ci_analysis as ci


def _row(arm, spec, ok=True, p=True, fom=100.0, spice=10):
    return {"ablation_id": arm, "spec_index": spec, "pipeline_seed": 0,
            "trace_result": "OK" if ok else "ERROR",
            "nominal": {"complete_pass": p}, "fom": {"fom_value": fom},
            "spice": {"total_calls": spice}}


def test_bootstrap_ci_and_sign_test_basics():
    m, lo, hi = ci.bootstrap_ci([1.0] * 10)
    assert m == 1.0 and lo == 1.0 and hi == 1.0
    assert ci.sign_test_p(0, 0) == 1.0
    assert ci.sign_test_p(9, 0) < 0.01
    assert abs(ci.sign_test_p(5, 5) - 1.0) < 1e-9


def test_effective_n_counts_only_differing_pairs(tmp_path):
    rows = []
    for s in range(6):
        rows.append(_row("A0", s))
        rows.append(_row("AX", s))                          # identical -> eff n 0
        rows.append(_row("AY", s, fom=200.0, spice=5))      # differs on fom/spice
        rows.append(_row("AZ", s, ok=False))                # hard failure
    p = tmp_path / "r.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rep = ci.analyze(p)
    assert rep["arms"]["AX"]["effective_n_differing"] == 0
    assert rep["arms"]["AY"]["effective_n_differing"] == 6
    assert rep["arms"]["AY"]["spice_delta"][0] == -5.0
    assert abs(rep["arms"]["AY"]["log_fom_delta"][0] - math.log(2)) < 1e-9
    assert rep["arms"]["AZ"]["pass_delta"][0] == -1.0     # counted as loss
    md = ci.format_markdown(rep)
    assert "AX" in md and "|" in md
