# Stage 3E.2 — edit library

Date: 2026-07-26

8 versioned templates (3e2.1) in stage3e2_edits.py; 7 executable + REMOVE (reversal). Executed on real ngspice: ['ADD_VERIFIED_STAGE', 'REPLACE_STAGE_WITH_COMPATIBLE_BLOCK', 'REPLACE_LOAD_WITH_COMPATIBLE_BLOCK', 'ADD_EXISTING_SUPPORTED_COMPENSATION_STRUCTURE', 'ADD_SUPPORTED_OUTPUT_STAGE', 'CONNECT_VERIFIED_FEEDBACK_PATH']. Rejected with reasons: {'REPLACE_SUPPORTED_COMPENSATION_STRUCTURE': 'no_compensation_to_replace'}. Parent immutability and add/remove round-trip hash recovery both verified. Base lineage: root+ADD_VERIFIED_STAGE.
