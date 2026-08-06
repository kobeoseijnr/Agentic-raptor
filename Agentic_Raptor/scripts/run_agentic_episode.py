"""Run one agentic design episode with the default config (or a supplied one)."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from agentic_raptor.cli import main  # noqa: E402

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else str(_PKG_ROOT / "configs" / "default.yaml")
    sys.exit(main(["run-episode", "--config", config]))
