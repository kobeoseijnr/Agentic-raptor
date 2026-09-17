"""External-baseline adapter layer (STAGE 5). Baselines are invoked via
subprocess from their isolated checkouts; the RAPTOR runtime never
imports them (enforced by tests/test_external_baseline_leakage.py)."""
