"""Exception hierarchy for Agentic RAPTOR."""

from __future__ import annotations


class AgenticRaptorError(Exception):
    """Base class for all Agentic RAPTOR errors."""


class ConfigurationError(AgenticRaptorError):
    """Raised when a configuration file is missing, malformed, or inconsistent."""


class SpecificationError(AgenticRaptorError):
    """Raised when design specifications are physically meaningless."""


class GraphError(AgenticRaptorError):
    """Raised on structurally invalid circuit-graph construction or mutation."""


class ActionError(AgenticRaptorError):
    """Raised when a topology action fails its precondition checks."""


class SimulationError(AgenticRaptorError):
    """Raised when a simulator adapter fails outside of a normal failed result."""


class BudgetExhaustedError(AgenticRaptorError):
    """Raised when an operation is attempted after its budget is exhausted."""


class CoordinatorError(AgenticRaptorError):
    """Raised on illegal coordinator state transitions."""


class GenerationError(AgenticRaptorError):
    """Raised when topology generation output cannot be parsed or repaired."""
