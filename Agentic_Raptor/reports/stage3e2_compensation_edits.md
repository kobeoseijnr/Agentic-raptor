# Stage 3E.2 — compensation edits

Date: 2026-07-26

ADD Miller: {"label": "add_existing_supported_c", "static": "mapped_static_valid", "electrical": "validation_failed", "metrics": {"dc_gain_db": -202.39062704838295, "gain_margin_db": 125.83431444985953, "f3db_hz": 0.1498293859266252, "quiescent_power_w": 3.70791e-05, "output_dc_v": 2.07628e-05}, "stability": "phase_margin_unavailable", "spice_calls": 1, "run_ref": "C:\\Users\\kobeo\\OneDrive\\Desktop\\raptor1\\Agentic_Raptor\\artifacts\\stage3e2\\edits\\add_existing_supported_c\\run"}

REPLACE with RC nulling (RZ=5k, versioned template param): "no_compensation_to_replace"

Negative/unstable PM outcomes recorded as valid measurements, never as simulator failures.
