"""Send light phase codes to Arduino over serial."""

from __future__ import annotations

from typing import Protocol

from controller import Phase


# Single ASCII digit per line so the Arduino parser stays tiny.
_PHASE_CODE: dict[Phase, str] = {
    Phase.NS_LEFT_SOLID: "1",
    Phase.NS_LEFT_BLINK: "2",
    Phase.NS_LEFT_OFF_BUFFER: "3",
    Phase.NS_STRAIGHT_GREEN: "4",
    Phase.NS_STRAIGHT_YELLOW: "5",
    Phase.ALL_RED_AFTER_NS: "6",
    Phase.EW_LEFT_SOLID: "7",
    Phase.EW_LEFT_BLINK: "8",
    Phase.EW_LEFT_OFF_BUFFER: "9",
    Phase.EW_STRAIGHT_GREEN: "A",
    Phase.EW_STRAIGHT_YELLOW: "B",
    Phase.ALL_RED_AFTER_EW: "6",
}


class SerialLike(Protocol):
    def write(self, data: bytes) -> int: ...
    def flush(self) -> None: ...


def encode_phase(phase: Phase) -> bytes:
    code = _PHASE_CODE.get(phase, "6")
    return f"{code}\n".encode("ascii")


def open_serial(port: str, baud: int):
    import serial  # type: ignore

    return serial.Serial(port, baud, timeout=0.1)
