# Stage 3E.1 — topology actions

Date: 2026-07-26

Stage3E1Action (schema 3e1.1): 11 categories (KEEP, SELECT_EXISTING, ADD_VERIFIED_STAGE, REPLACE_STAGE/LOAD, ADD/REPLACE_COMPENSATION, ADD_OUTPUT_STAGE, REMOVE_OPTIONAL_STAGE, CONNECT_FEEDBACK, TERMINATE). Fields: action_id, type, source_ref, target_location, port_mapping, preconditions, compatibility, provenance, schema_version. No raw-netlist action exists; LLM proposals must pass parser->canonicaliser->validator (Part T interface). Structural-edit categories are generated but validator-gated until mapping support lands.
