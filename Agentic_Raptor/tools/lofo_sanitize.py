"""Specification-only prompt sanitization for the LOFO family-generalization study.

WHAT IS REMOVED AND WHY (measured, not assumed):

  ### BLOCKS     REMOVED. Measured CONSTANT across all 142 records and all five
                 families: "five_transistor_first_stage,cs_gain_stage,miller_cap,
                 bias_mirror". It therefore carries ZERO family information -- it is
                 a fixed menu of available blocks, not a per-record hint. Removed
                 anyway because it names `miller_cap`, and a reviewer should not have
                 to verify constancy to trust the protocol.

  ### RAG        REMOVED. Names a reference topology (e.g. rag_l2_topology_0002).
                 Measured to span 3-5 response families each (16 tokens, NONE
                 family-determining), so it does not identify the held-out family --
                 but it is a reference-topology name, which the protocol excludes.
                 Retrieval is instead performed by the family-filtered RAG index at
                 run time, not by a token baked into the prompt.

WHAT IS RETAINED:

  ### SPEC       KEPT. Pure electrical/process request: gain, phase margin, load
                 capacitance, UGBW, technology. Topology-agnostic and legitimately
                 available from a design request.

  ### FORBIDDEN  KEPT. Measured constant ("raw_netlist,feedback_to_input"); a
                 legitimate output-format constraint, not a structural hint.

  ### PROPOSAL   KEPT. Empty generation marker.

The SAME function is used for LOFO training and LOFO test prompts.
"""
from __future__ import annotations

import re

#: Tokens that would identify a response family if they ever appeared in a prompt.
FAMILY_TOKENS = [
    "miller", "miller_cap", "rc_comp", "rc_compensation", "nulling",
    "cascode", "cas", "class_ab", "class-ab", "classab", "ab_output",
    "2s_", "3s_", "4s_", "1s_",
    "two_stage", "three_stage", "single_stage", "four_stage",
    "two-stage", "three-stage", "stages=", "stage_count",
    "comp=", "compensation", "topology_0", "topology_v2_", "rag_l",
    "five_transistor_first_stage", "cs_gain_stage", "bias_mirror",
    "buffer", "feedback",
]

KEEP_HEADERS = ("SPEC", "FORBIDDEN", "PROPOSAL")
DROP_HEADERS = ("BLOCKS", "RAG")


def sanitize(prompt: str) -> str:
    """Return the specification-only prompt. Deterministic, order-preserving."""
    out = []
    for line in prompt.splitlines():
        m = re.match(r"^###\s+(\w+)", line)
        if m and m.group(1) in DROP_HEADERS:
            continue
        out.append(line)
    return "\n".join(out)


def audit(prompt: str) -> list[str]:
    """Return every family token found in a (sanitized) prompt. Empty == clean."""
    low = prompt.lower()
    return sorted({t for t in FAMILY_TOKENS if t in low})


def discriminating_tokens(pairs) -> dict:
    """Tokens whose PRESENCE varies with family -- the only ones that can leak.

    `pairs` is an iterable of (sanitized_prompt, family). A token appearing in
    every record (or in none) carries no family information no matter how
    structural it sounds: `feedback_to_input` sits in the constant FORBIDDEN
    line of all 142 records and is a forbidden-construct name, not a hint.
    A token is reported ONLY if the set of families containing it is a proper
    non-empty subset of all families.
    """
    from collections import defaultdict
    fams, seen = set(), defaultdict(set)
    for prompt, fam in pairs:
        fams.add(fam)
        low = prompt.lower()
        for t in FAMILY_TOKENS:
            if t in low:
                seen[t].add(fam)
    return {t: sorted(f) for t, f in seen.items() if 0 < len(f) < len(fams)}
