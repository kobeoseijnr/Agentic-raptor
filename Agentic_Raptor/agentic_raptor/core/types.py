"""Shared enums and type aliases for RAPTOR core models."""

from __future__ import annotations

from enum import Enum


class DeviceType(str, Enum):
    NMOS = "NMOS"
    PMOS = "PMOS"
    RESISTOR = "RESISTOR"
    CAPACITOR = "CAPACITOR"
    CURRENT_SOURCE = "CURRENT_SOURCE"
    VOLTAGE_SOURCE = "VOLTAGE_SOURCE"
    INPUT_PORT = "INPUT_PORT"
    OUTPUT_PORT = "OUTPUT_PORT"
    SUPPLY_PORT = "SUPPLY_PORT"
    GROUND_PORT = "GROUND_PORT"
    SUBCIRCUIT_BLOCK = "SUBCIRCUIT_BLOCK"


class TerminalType(str, Enum):
    # MOS terminals
    DRAIN = "D"
    GATE = "G"
    SOURCE = "S"
    BULK = "B"
    # Generic two-terminal devices
    PLUS = "P"
    MINUS = "N"
    # Ports and blocks
    PORT = "PORT"
    BLOCK_PIN = "PIN"


#: Terminals each device type must expose.
EXPECTED_TERMINALS: dict[DeviceType, tuple[TerminalType, ...]] = {
    DeviceType.NMOS: (TerminalType.DRAIN, TerminalType.GATE, TerminalType.SOURCE, TerminalType.BULK),
    DeviceType.PMOS: (TerminalType.DRAIN, TerminalType.GATE, TerminalType.SOURCE, TerminalType.BULK),
    DeviceType.RESISTOR: (TerminalType.PLUS, TerminalType.MINUS),
    DeviceType.CAPACITOR: (TerminalType.PLUS, TerminalType.MINUS),
    DeviceType.CURRENT_SOURCE: (TerminalType.PLUS, TerminalType.MINUS),
    DeviceType.VOLTAGE_SOURCE: (TerminalType.PLUS, TerminalType.MINUS),
    DeviceType.INPUT_PORT: (TerminalType.PORT,),
    DeviceType.OUTPUT_PORT: (TerminalType.PORT,),
    DeviceType.SUPPLY_PORT: (TerminalType.PORT,),
    DeviceType.GROUND_PORT: (TerminalType.PORT,),
    DeviceType.SUBCIRCUIT_BLOCK: (TerminalType.BLOCK_PIN,),
}

#: Device types considered "active" for signal-path checks.
ACTIVE_DEVICE_TYPES: frozenset[DeviceType] = frozenset(
    {DeviceType.NMOS, DeviceType.PMOS, DeviceType.CURRENT_SOURCE, DeviceType.SUBCIRCUIT_BLOCK}
)

#: Device types that are ports (single-terminal boundary nodes).
PORT_DEVICE_TYPES: frozenset[DeviceType] = frozenset(
    {DeviceType.INPUT_PORT, DeviceType.OUTPUT_PORT, DeviceType.SUPPLY_PORT, DeviceType.GROUND_PORT}
)


class GenerationSource(str, Enum):
    LLM = "llm"
    MOCK = "mock"
    RETRIEVED = "retrieved"
    EDITED = "edited"
    MANUAL = "manual"


class CandidateStatus(str, Enum):
    GENERATED = "generated"
    VALIDATED = "validated"
    INVALID = "invalid"
    EDITING = "editing"
    SIZING = "sizing"
    SIMULATED = "simulated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
