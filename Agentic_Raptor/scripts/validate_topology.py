"""Validate a circuit-graph JSON file: python scripts/validate_topology.py graph.json"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from agentic_raptor.cli import main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/validate_topology.py <graph.json>")
        sys.exit(2)
    sys.exit(main(["validate", "--graph", sys.argv[1]]))
