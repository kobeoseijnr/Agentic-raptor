"""AutoCkt adapter (STAGE 5) -- Track B (fixed-topology sizing) ONLY.

Status: the native py3.6/TF1 environment is impractical on this host. The
validated reproduction path (documented in repository_manifest.json) is the
same original code under Ray 2.55.1 (oracle-ray311 env), already used for a
published-quality 1000-spec evaluation on the ORACLE benchmark.

Track-B plan (activated at the full phase, after pilot approval):
  1. train PPO on specs_train ranges ONLY (its native protocol); record
     TrainingCalls / TrainingRuntime;
  2. roll out on the frozen Track-B topology/spec set;
  3. rescore final designs through the shared ngspice judge;
  4. emit ExternalTopologyResult-compatible sizing rows.
Training runs are launched by the user (long GPU/CPU jobs), commands
prepared by this adapter.
"""
from __future__ import annotations


def run_spec(*_a, **_k):
    raise NotImplementedError(
        "AutoCkt is a Track-B sizing baseline; its training/rollout wiring is "
        "activated at the full phase (see module docstring + "
        "repository_manifest.json).")
