"""Fix 3: prove the topology PUCT selected is the topology that was built.

The failure this guards against is silent and total: the search computes
advice, something else is executed, and the feedback is attributed to the
search anyway. Every measured number would then describe a circuit the
search never chose, while appearing to validate it. That is unrecoverable
after the fact -- the only defence is a hash chain checked at run time.

Chain enforced end to end:

    PUCT selected topology hash
      == executed topology hash
      == C9 (sizing) input topology hash
      == feedback-attributed topology hash

Any mismatch raises TopologyExecutionMismatch and fails the run.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = _ROOT / "artifacts/publication_v2/execution_traces"


class TopologyExecutionMismatch(AssertionError):
    """PUCT chose one topology and a different one reached SPICE."""


@dataclass
class ExecutionTrace:
    """One design context, from specification to feedback record."""
    spec_id: str
    evaluation_context_id: str | None = None
    generation: int | None = None
    campaign_id: str | None = None

    proposal_id: str | None = None            # Qwen proposal
    proposal_hash: str | None = None
    proposal_class: str | None = None

    puct_root_id: str | None = None
    puct_selected_action: str | None = None
    puct_selected_topology_id: str | None = None
    puct_selected_topology_hash: str | None = None
    puct_visits: dict = field(default_factory=dict)
    puct_engine: str = "P8_repaired_multiply"

    executed_topology_id: str | None = None
    executed_topology_hash: str | None = None

    c9_input_topology_hash: str | None = None
    sizing_manifest_hash: str | None = None
    netlist_hash: str | None = None
    spice_result_id: str | None = None
    feedback_record_id: str | None = None
    feedback_topology_hash: str | None = None

    exact_pass: bool | None = None
    exact_failure_reason: str | None = None
    timestamp: float = field(default_factory=time.time)

    def verify(self, strict: bool = True) -> dict:
        """Check the hash chain. strict=False reports without raising."""
        links = [("puct_selected", self.puct_selected_topology_hash),
                 ("executed", self.executed_topology_hash),
                 ("c9_input", self.c9_input_topology_hash),
                 ("feedback", self.feedback_topology_hash)]
        present = [(n, h) for n, h in links if h is not None]
        distinct = {h for _n, h in present}
        ok = len(distinct) <= 1
        report = {"spec_id": self.spec_id, "aligned": ok,
                  "links_present": [n for n, _ in present],
                  "links_missing": [n for n, h in links if h is None],
                  "hashes": {n: h for n, h in present}}
        if not ok and strict:
            raise TopologyExecutionMismatch(
                f"topology hash chain broken for {self.spec_id}: "
                + ", ".join(f"{n}={h}" for n, h in present)
                + " -- PUCT advice was computed but a different topology "
                  "was executed; every downstream number would be "
                  "misattributed to the search")
        if strict and not self.puct_selected_topology_hash:
            raise TopologyExecutionMismatch(
                f"{self.spec_id}: no PUCT selection recorded -- cannot show "
                f"the search was consulted at all")
        return report

    def write(self, subdir: str = "campaign") -> Path:
        d = TRACE_DIR / subdir
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.spec_id}_gen{self.generation}.json"
        p.write_text(json.dumps(asdict(self), indent=1, default=str),
                     encoding="utf-8")
        return p


def verify_dir(subdir: str = "campaign", strict: bool = False) -> dict:
    """Verify every recorded trace; used by the pre-ablation gate."""
    d = TRACE_DIR / subdir
    if not d.is_dir():
        return {"traces": 0, "aligned": 0, "broken": [], "missing_puct": 0}
    aligned, broken, missing = 0, [], 0
    files = sorted(d.glob("*.json"))
    for f in files:
        t = ExecutionTrace(**json.loads(f.read_text(encoding="utf-8")))
        rep = t.verify(strict=False)
        if not t.puct_selected_topology_hash:
            missing += 1
        if rep["aligned"]:
            aligned += 1
        else:
            broken.append(rep)
    if strict and broken:
        raise TopologyExecutionMismatch(
            f"{len(broken)} of {len(files)} traces have a broken hash chain")
    return {"traces": len(files), "aligned": aligned, "broken": broken,
            "missing_puct": missing}
