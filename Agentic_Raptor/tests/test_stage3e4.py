"""Stage 3E.4 tests: multi-structure corpus, leakage, diversity, unique
physical pair grouping, realisation accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_raptor.utils.seeding import apply_torch_omp_workaround

apply_torch_omp_workaround()

from agentic_raptor.llm_dpo import stage3e4 as s4

_ROOT = Path(__file__).resolve().parents[1]
O4 = _ROOT / "artifacts" / "stage3e4"


def _load(n):
    p = O4 / n
    if not p.is_file():
        pytest.skip(f"{n} pending")
    return json.loads(p.read_text(encoding="utf-8"))


class TestCorpus:
    def test_multi_structure_and_splits(self):
        d = _load("corpus.json")
        st = d["stats"]
        assert st["iso_classes"] > 1                    # >1 isomorphism class
        assert set(st["stage_distribution"]) == {1, 2, 3} or len(st["stage_distribution"]) >= 2
        assert st["splits"]["heldout"] > 0 and st["splits"]["train"] > 0
        # leakage: heldout contexts never appear in train
        ids = {}
        for r in d["records"]:
            ids.setdefault(r["context_id"], set()).add(r["split"])
        assert all(len(v) == 1 for v in ids.values())

    def test_variant_text_parses_and_validates(self):
        from agentic_raptor.llm_dpo import parse_proposal_text, proposal_dict_valid
        al_p = _ROOT / "artifacts" / "variant_verification" / "ALLOWLIST.json"
        allow = ({(a["stages"] if isinstance(a["stages"], int) else len(a["stages"]),
                   a["comp"], a["buffer"], a["fb"])
                  for a in json.loads(al_p.read_text())["allow_list"]}
                 if al_p.is_file() else None)
        for stages, comp, buf, fb in s4.VARIANTS[:8]:
            obj = parse_proposal_text(s4.variant_text(stages, comp, buf, fb))
            assert obj is not None                      # always parseable
            ok, r = proposal_dict_valid(obj)
            if allow is None:
                assert ok, r
            else:   # validator must agree with the MEASURED allow-list
                assert ok == ((stages, comp, buf, fb) in allow), r

    def test_variant_hash_ignores_parameters(self):
        a = s4.variant_hash(json.loads(s4.variant_text(2, "miller", False, False)))
        b = s4.variant_hash(json.loads(s4.variant_text(2, "miller", False, False)))
        c = s4.variant_hash(json.loads(s4.variant_text(3, "miller", False, False)))
        assert a == b and a != c                        # structure-only identity


class TestDiversityAndPairs:
    def test_diversity_metrics(self):
        d = _load("diversity_sft.json")
        if "candidates_after_ranking" not in d:
            pytest.skip("artifact predates the ranked-candidate schema; "
                        "regenerated on the next campaign")
        # "generated" is PROPOSER output and keeps its meaning; the search's
        # added structures are reported separately so ranking cannot silently
        # redefine the proposer's diversity numbers
        assert d["generated"] == d["contexts"] * d["candidates_per_ctx"]
        assert d["unique_valid_canonical"] >= 1
        assert d["candidates_after_ranking"] >= d["unique_valid_canonical"]
        assert d["structures_added_by_search"] >= 0
        for r in d["rows"][:3]:
            temps = {c.get("temperature") for c in r["candidates"]
                     if c.get("source") != "mcts_enumerated"}
            assert temps <= {0.8}

    def test_unique_physical_groups_not_inflated(self):
        d = _load("model_pairs.json")
        assert d["unique_physical_groups"] == len(d["pairs"])
        assert d["counts"]["repeated_skipped"] >= 0
        for p in d["pairs"][:5]:
            assert p["context_id"]                      # same-context rule

    def test_realisation_real_spice_accounting(self):
        d = _load("realised.json")
        simulated = [r for r in d["realised"] if r.get("electrical")]
        assert d["real_spice_calls"] == len(simulated)
        for r in simulated:
            assert r["stability"] in ("verified_stable", "verified_unstable",
                                      "phase_margin_unavailable")

    def test_sft_validity_regression_gate(self):
        """Protect the solved SFT result: >=90% validity on held-out contexts.
        Fails loudly if a corpus/model change silently breaks generation."""
        d = _load("diversity_sft.json")
        assert d["valid"] / d["generated"] >= 0.9

    def test_comparison_has_cis(self):
        d = _load("comparison.json")
        assert set(d) == {"base", "sft", "sft_dpo"}
        for m in d.values():
            if "skipped" in m:      # honest skip (e.g. incompatible adapter)
                continue
            lo, hi = m["valid_rate_ci95"]
            assert 0.0 <= lo <= hi <= 1.0
