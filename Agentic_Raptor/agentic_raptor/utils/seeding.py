"""Deterministic seeding for all random processes.

Replicates the legacy repo's KMP_DUPLICATE_LIB_OK workaround (see
``mb_sac/sac_agent.py``) before any torch import on this machine.
"""

from __future__ import annotations

import os
import random


def apply_torch_omp_workaround() -> None:
    """Avoid the OpenMP runtime collision seen on this machine when importing torch."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and (if installed) ``torch``."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is expected to exist
        pass
    try:
        apply_torch_omp_workaround()
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch optional for pure-python paths
        pass


def make_rng(seed: int) -> random.Random:
    """Return an isolated, seeded RNG (no global mutable state)."""
    return random.Random(seed)
