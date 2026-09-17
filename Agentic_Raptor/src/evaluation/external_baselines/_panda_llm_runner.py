"""Child-process runner for PANDA/AnalogXpert LLM topology generation.

Runs the baseline's OWN driver function (Analog_designer.single_run)
verbatim: their prompt suite, their self-refine loop (their own repair),
their model-candidate logic. Executed with cwd = PANDA/topology_gen so
their relative imports (Self_detect, post_result_process, ...) resolve.

SAFETY: single_run's default base_url is a third-party proxy
(bitexingai.com). This runner ALWAYS passes base_url explicitly from
OPENAI_BASE_URL (or the official api.openai.com) so the user's key is
never sent elsewhere.

Usage: python _panda_llm_runner.py <query_file> <log_file> [model]
Prints one JSON line: {netlist, their_check_pass, rounds, wall_s}
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, ".")           # the topology_gen dir (cwd)

from Analog_designer import single_run  # noqa: E402


def main() -> int:
    query = open(sys.argv[1], encoding="utf-8").read()
    testlog = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-5-mini"
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    t0 = time.time()
    try:
        response, memory, their_pass = single_run(
            query, testlog, base_url=base_url, model_name=model)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {str(e)[:300]}",
                          "wall_s": round(time.time() - t0, 1)}))
        return 0
    wall = time.time() - t0
    m = re.search(r"\*\*\*Netlist Start\*\*\*(.*?)\*\*\*Netlist End\*\*\*",
                  response or "", re.DOTALL)
    netlist = m.group(1).strip() if m else ""
    print(json.dumps({"netlist": netlist,
                      "their_check_pass": bool(their_pass),
                      "rounds": len(memory),
                      "raw_response_tail": (response or "")[-400:],
                      "wall_s": round(wall, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
