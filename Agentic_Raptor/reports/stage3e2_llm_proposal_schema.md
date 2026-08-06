# Stage 3E.2 LLM-DPO — proposal schema

Date: 2026-07-26

Structured proposal = compact JSON (stages/ports/bias_roles/compensation/feedback_paths/polarity) serialised for the LM; executable fields only — free-form rationale is a separate field never parsed as connectivity; raw generated netlists are never executed (tested: module contains no SPICE invocation). Schema 3e2L.1; full TopologyProposal dataclass in stage3e2_edits.py carries provenance/decoding/seed fields.
