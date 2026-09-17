"""Child-process runner for PANDA's NATIVE offline topology components.

Executed via subprocess with cwd = the PANDA checkout; imports PANDA's own
modules from that checkout only (this file lives under src/evaluation, the
allow-listed adapter zone). Runs the baseline's local template generator and
its native validator UNMODIFIED, prints one JSON result to stdout.

Usage: python _panda_runner.py <design_spec_json_path>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")   # the PANDA checkout (cwd)

from analogxpert.topology_templates import generate_topology_template  # noqa: E402
from analogxpert.codex_engine import validate_topology_text            # noqa: E402


def main() -> int:
    req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    intent = req.get("design_intent", "")
    spec = req["design_spec"]
    t0 = time.time()
    out = generate_topology_template(intent, spec)
    wall = time.time() - t0
    if out is None:
        print(json.dumps({"generated": False, "wall_s": wall,
                          "note": "template generator returned None "
                                  "(no local template for this intent; the "
                                  "LLM path would be required)"}))
        return 0
    rep = validate_topology_text(out["netlist"])
    print(json.dumps({"generated": True, "wall_s": wall,
                      "generator": out.get("generator"),
                      "topology_summary": out.get("topology_summary"),
                      "netlist": out["netlist"],
                      "validator_ok": bool(rep.ok)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
