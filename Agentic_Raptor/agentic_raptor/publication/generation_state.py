"""A9: GenerationState, atomic publication, idempotent resume.

Two independent lineages share this exact same state machine:
  "adaptive" -- may harvest/retrain between generations
  "static"   -- runs the identical adaptation task stream and simulator
                budget, logs outcomes, but NEVER updates RAG/SFT/PUCT/SAC/
                surrogate/DPO (learning_mode="static" in run_pipeline)

At G0 both lineages start from the exact same accepted-component hashes --
callers should build both G0 states from `GenerationState.new(0, None, ...)`
against the SAME frozen checkpoints and assert the hashes match (see
test_a9_generation.py) before diverging into G1.

Publication contract (Part: ATOMIC GENERATION PUBLICATION):
  - `checkpoint_progress()` persists in-progress state (write-temp, rename)
    so a crash mid-generation leaves a RESUMABLE record, not silent loss.
  - `publish_atomic()` refuses (ValueError) unless status=="COMPLETE" and
    every required hash field is present; only then does it atomically
    replace the ACTIVE state file and remove the in-progress checkpoint.
  - `resume_or_start()` is the single entry point: an in-progress file means
    resume THAT generation (ledger intact, no re-harvest of already-
    processed run_ids); otherwise the last COMPLETE generation's successor
    starts fresh; otherwise generation 0.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATUSES = ("BUILDING", "VALIDATING", "COMPLETE", "FAILED")

#: a generation cannot publish without these being set -- Part: ATOMIC
#: GENERATION PUBLICATION items 4 ("checkpoints validate") and 6 ("hashes
#: are created"). Static-lineage generations set these to the SAME values
#: as G0 forever (nothing changes), which is itself the point of A9.
REQUIRED_FOR_COMPLETE = ("rag_snapshot_hash", "sft_checkpoint_hash",
                        "puct_policy_hash", "puct_value_hash",
                        "dpo_checkpoint_hash", "evaluation_set_hash")


@dataclass
class GenerationState:
    generation_id: int
    parent_generation_id: int | None
    lineage: str                         # "adaptive" | "static"
    status: str = "BUILDING"
    git_commit: str | None = None
    rag_snapshot_hash: str | None = None
    sft_checkpoint_hash: str | None = None
    puct_policy_hash: str | None = None
    puct_value_hash: str | None = None
    sac_state_version: str | None = None
    sac_replay_hash: str | None = None
    surrogate_hash: str | None = None
    dpo_checkpoint_hash: str | None = None
    adaptation_batch_hash: str | None = None
    evaluation_set_hash: str | None = None
    seed_metadata: dict = field(default_factory=dict)
    simulator_version: str | None = None
    pdk_version: str | None = None
    true_spice_calls: int = 0
    #: idempotency ledger: run_ids already harvested THIS generation. A
    #: resumed generation re-checks this before processing each run_id, so
    #: a partial re-run cannot double-count a stream row.
    processed_run_ids: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    completed_at: str | None = None

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"invalid status {self.status!r}, must be one of {STATUSES}")
        if self.lineage not in ("adaptive", "static"):
            raise ValueError(f"invalid lineage {self.lineage!r}")

    @classmethod
    def new(cls, generation_id: int, parent_generation_id: int | None,
           lineage: str = "adaptive") -> "GenerationState":
        return cls(generation_id=generation_id,
                  parent_generation_id=parent_generation_id, lineage=lineage)

    def mark_run_processed(self, run_id: str) -> bool:
        """True if newly recorded (caller should harvest it); False if
        already in the ledger (caller must SKIP -- this is what makes
        harvest idempotent across a resume)."""
        if run_id in self.processed_run_ids:
            return False
        self.processed_run_ids.append(run_id)
        return True

    def is_complete_manifest(self) -> bool:
        return (self.status == "COMPLETE"
               and all(getattr(self, f) is not None for f in REQUIRED_FOR_COMPLETE))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationState":
        return cls(**d)


def _active_path(root: Path, lineage: str) -> Path:
    return root / f"STATE_{lineage}.json"


def _inprogress_path(root: Path, lineage: str) -> Path:
    return root / f"STATE_{lineage}_inprogress.json"


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)          # atomic rename on the same filesystem


def checkpoint_progress(state: GenerationState, root: Path) -> Path:
    """Persist in-progress state so a crash is resumable, not lost."""
    p = _inprogress_path(root, state.lineage)
    _atomic_write(p, state.to_dict())
    return p


def publish_atomic(state: GenerationState, root: Path) -> Path:
    """Refuses to publish an incomplete generation. On success, the ACTIVE
    state file is replaced atomically and the in-progress checkpoint for
    this lineage is removed (this generation is no longer "in progress")."""
    if not state.is_complete_manifest():
        missing = [f for f in REQUIRED_FOR_COMPLETE
                  if getattr(state, f) is None]
        raise ValueError(
            f"refusing to publish generation {state.generation_id} "
            f"({state.lineage}): status={state.status!r}, "
            f"missing={missing or 'none (status not COMPLETE)'}")
    if state.completed_at is None:
        state.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
    p = _active_path(root, state.lineage)
    _atomic_write(p, state.to_dict())
    ip = _inprogress_path(root, state.lineage)
    if ip.is_file():
        ip.unlink()
    return p


def resume_or_start(root: Path, lineage: str) -> GenerationState:
    """The single entry point for starting/continuing a lineage.

    1. An in-progress checkpoint exists -> resume that EXACT generation
       (ledger intact; already-processed run_ids are skipped by the
       caller's own `mark_run_processed` checks).
    2. Else the last COMPLETE (published) generation exists -> start its
       successor, fresh.
    3. Else -> generation 0.
    """
    ip = _inprogress_path(root, lineage)
    if ip.is_file():
        return GenerationState.from_dict(json.loads(ip.read_text(encoding="utf-8")))
    active = _active_path(root, lineage)
    if active.is_file():
        st = GenerationState.from_dict(json.loads(active.read_text(encoding="utf-8")))
        return GenerationState.new(st.generation_id + 1, st.generation_id, lineage)
    return GenerationState.new(0, None, lineage)


def accepted_component_hashes(*, rag_memory_path: Path, sft_adapter_path: Path,
                              puct_ckpt_path: Path, dpo_ckpt_path: Path,
                              evaluation_set_hash: str) -> dict:
    """The G0 snapshot both lineages must start from IDENTICALLY -- computed
    from files on disk, never invented, so adaptive/static G0 hashes can be
    asserted equal (Part: EVALUATION, "At G0: Adaptive == Static")."""
    from agentic_raptor.ranking.types import checkpoint_sha256, directory_sha256

    def _file_hash(p: Path) -> str | None:
        if not p.is_file():
            return None
        import hashlib
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    return {
        "rag_snapshot_hash": _file_hash(rag_memory_path),
        "sft_checkpoint_hash": (directory_sha256(str(sft_adapter_path))
                                if sft_adapter_path and sft_adapter_path.is_dir()
                                else None),
        "puct_policy_hash": checkpoint_sha256(puct_ckpt_path),
        "puct_value_hash": checkpoint_sha256(puct_ckpt_path),
        "dpo_checkpoint_hash": checkpoint_sha256(dpo_ckpt_path),
        "evaluation_set_hash": evaluation_set_hash,
    }
