"""Coordinator state machine with validated transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agentic_raptor.utils.exceptions import CoordinatorError


class CoordinatorState(str, Enum):
    INITIALIZE = "INITIALIZE"
    PARSE_SPECIFICATIONS = "PARSE_SPECIFICATIONS"
    RETRIEVE = "RETRIEVE"
    GENERATE = "GENERATE"
    VALIDATE = "VALIDATE"
    EDIT_TOPOLOGY = "EDIT_TOPOLOGY"
    SIZE = "SIZE"
    SIMULATE = "SIMULATE"
    DIAGNOSE = "DIAGNOSE"
    UPDATE_MEMORY = "UPDATE_MEMORY"
    TERMINATE = "TERMINATE"
    FAILED = "FAILED"


class Decision(str, Enum):
    RETRIEVE_MORE = "RETRIEVE_MORE"
    GENERATE_NEW_TOPOLOGY = "GENERATE_NEW_TOPOLOGY"
    REPAIR_OR_EDIT_TOPOLOGY = "REPAIR_OR_EDIT_TOPOLOGY"
    CONTINUE_SIZING = "CONTINUE_SIZING"
    RUN_SPICE = "RUN_SPICE"
    RUN_PVT = "RUN_PVT"
    ACCEPT_CANDIDATE = "ACCEPT_CANDIDATE"
    STOP_BUDGET_EXHAUSTED = "STOP_BUDGET_EXHAUSTED"
    STOP_SUCCESS = "STOP_SUCCESS"


ALLOWED_TRANSITIONS: dict[CoordinatorState, frozenset[CoordinatorState]] = {
    CoordinatorState.INITIALIZE: frozenset({CoordinatorState.PARSE_SPECIFICATIONS, CoordinatorState.FAILED}),
    CoordinatorState.PARSE_SPECIFICATIONS: frozenset({CoordinatorState.RETRIEVE, CoordinatorState.FAILED}),
    CoordinatorState.RETRIEVE: frozenset({CoordinatorState.GENERATE, CoordinatorState.RETRIEVE, CoordinatorState.FAILED}),
    CoordinatorState.GENERATE: frozenset({CoordinatorState.VALIDATE, CoordinatorState.RETRIEVE, CoordinatorState.FAILED}),
    CoordinatorState.VALIDATE: frozenset(
        {CoordinatorState.EDIT_TOPOLOGY, CoordinatorState.GENERATE, CoordinatorState.SIZE, CoordinatorState.DIAGNOSE, CoordinatorState.FAILED}
    ),
    CoordinatorState.EDIT_TOPOLOGY: frozenset(
        {CoordinatorState.VALIDATE, CoordinatorState.EDIT_TOPOLOGY, CoordinatorState.SIZE, CoordinatorState.DIAGNOSE, CoordinatorState.FAILED}
    ),
    CoordinatorState.SIZE: frozenset({CoordinatorState.SIMULATE, CoordinatorState.SIZE, CoordinatorState.DIAGNOSE, CoordinatorState.FAILED}),
    CoordinatorState.SIMULATE: frozenset(
        {CoordinatorState.DIAGNOSE, CoordinatorState.SIZE, CoordinatorState.SIMULATE, CoordinatorState.UPDATE_MEMORY, CoordinatorState.FAILED}
    ),
    CoordinatorState.DIAGNOSE: frozenset(
        {
            CoordinatorState.RETRIEVE,
            CoordinatorState.GENERATE,
            CoordinatorState.EDIT_TOPOLOGY,
            CoordinatorState.SIZE,
            CoordinatorState.SIMULATE,
            CoordinatorState.UPDATE_MEMORY,
            CoordinatorState.FAILED,
        }
    ),
    CoordinatorState.UPDATE_MEMORY: frozenset({CoordinatorState.TERMINATE, CoordinatorState.DIAGNOSE, CoordinatorState.FAILED}),
    CoordinatorState.TERMINATE: frozenset(),
    CoordinatorState.FAILED: frozenset(),
}


@dataclass
class StateMachine:
    state: CoordinatorState = CoordinatorState.INITIALIZE
    history: list[str] = field(default_factory=list)

    def transition(self, to: CoordinatorState) -> None:
        allowed = ALLOWED_TRANSITIONS[self.state]
        if to not in allowed:
            raise CoordinatorError(
                f"illegal transition {self.state.value} → {to.value}; allowed: "
                f"{sorted(s.value for s in allowed)}"
            )
        self.history.append(f"{self.state.value}->{to.value}")
        self.state = to

    @property
    def finished(self) -> bool:
        return self.state in (CoordinatorState.TERMINATE, CoordinatorState.FAILED)
