"""Publication experiment package: freeze manifests, fixed benchmark,
feasibility audit, baselines, ablations, statistics, figures, report.

All experiment outputs are saved under artifacts/publication/ as JSON and are
the ONLY source for figures/tables/report (rule: generate directly from saved
experiment files)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "artifacts" / "publication"
FREEZE = ROOT / "artifacts" / "publication_freeze"
