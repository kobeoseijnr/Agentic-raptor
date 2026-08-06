"""CLI: validate [--phase-only] | rebuild-memory | report | rerun-failed."""
import json
import sys

from agentic_raptor.electrical import run_stage3b


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if cmd in ("validate", "rebuild-memory", "rerun-failed"):
        out = run_stage3b()
        print(json.dumps(out["summary"], indent=1, default=str))
        return 0
    if cmd == "report":
        from pathlib import Path
        p = Path("datasets/simulation_memory/index_metadata.json")
        print(p.read_text(encoding="utf-8") if p.is_file() else "no memory built yet")
        return 0
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
