"""Agentic RAPTOR v2 self-improvement.

Built entirely around `run_raptor_v2.run_pipeline`. It shares no code, no
checkpoints and no datasets with the older self-improvement loop, which drove
a different architecture (1 proposal, 5 enumerated candidates, 1 sized) and
whose artifacts are archived. Mixing them would pool measurements from two
architectures -- and, because the old data predates the input-common-mode
fix, measurements of two different circuits.

One verified A/B comparison teaches six things, and this package routes it to
each of them without letting any stream see data it must not:

    measured pair        -> post-SAC DPO ranker
    search visits + value-> PUCT policy/value
    every outcome        -> v2 RAG memory
    SAC trajectories     -> optional persistent replay
    VERIFIED SUCCESSES   -> measurement-grounded SFT queue
    topology comparisons -> proposer DPO preference queue

Failures never become positive SFT targets; they go to RAG and preference
data, where "this did not work" is exactly the useful signal.
"""

from agentic_raptor.selfimprove_v2.gates import (GateResult, proposer_gates,
                                                 puct_gate, ranker_gate)
from agentic_raptor.selfimprove_v2.streams import (STREAMS, GenerationPaths,
                                                   harvest_run,
                                                   sft_admission_reasons)

__all__ = ["GenerationPaths", "STREAMS", "harvest_run",
           "sft_admission_reasons", "GateResult", "ranker_gate", "puct_gate",
           "proposer_gates"]
