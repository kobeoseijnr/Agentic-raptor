"""MACE adapter (STAGE 5) -- Track B (multi-objective BO sizing) ONLY.

Status: the native C++ build is NOT REPRODUCIBLE in this environment (no
CMake/gcc toolchain; deps: Eigen, Boost, OpenMP, NLopt, GSL + author
submodules). APPROVED SUBSTITUTION (user decision 2026-08-28, documented,
never silent): the author's own Python reimplementation MACE_MCMC
(https://github.com/Alaya-in-Matrix/MACE_MCMC), to be cloned into
external_baselines/MACE_MCMC and pinned in repository_manifest.json at the
full phase. Every table row produced through it is labeled
"MACE (author's Python reimplementation)".
"""
from __future__ import annotations


def run_spec(*_a, **_k):
    raise NotImplementedError(
        "MACE is a Track-B sizing baseline; the MACE_MCMC substitution is "
        "wired at the full phase (see module docstring + "
        "repository_manifest.json).")
