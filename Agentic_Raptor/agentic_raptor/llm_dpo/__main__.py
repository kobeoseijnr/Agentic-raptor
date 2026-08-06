import json
from agentic_raptor.llm_dpo import run_all
print(json.dumps(run_all(), indent=1, default=str))
