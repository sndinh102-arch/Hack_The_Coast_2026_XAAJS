"""
Adaptive traffic light controller with protected left-turn (blue) + straight (green).

Opposing approaches share the same signal group:
  - North and South run together
  - West and East run together

Sequence per group:
  SOLID_BLUE (protected left) → BLINK_BLUE → BLUE_OFF_BUFFER → GREEN → YELLOW → ALL_RED_CLEARANCE

Safety:
  - Never allows NS straight green/yellow while EW straight green/yellow (or vice versa)
  - Red is on by default unless that group's straight is green/yellow
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from config import TimingConfig


class Phase(Enum):
    NS_LEFT_SOLID = auto()
    NS_LEFT_BLINK = auto()
    NS_LEFT_OFF_BUFFER = auto()
    NS_STRAIGHT_GREEN = auto()
    NS_STRAIGHT_YELLOW = auto()
    ALL_RED_AFTER_NS = auto()

    EW_LEFT_SOLID = auto()
    EW_LEFT_BLINK = auto()
    EW_LEFT_OFF_BUFFER = auto()
    EW_STRAIGHT_GREEN = auto()
    EW_STRAIGHT_YELLOW = auto()
    ALL_RED_AFTER_EW = auto()


@dataclass
class LightOutputs:
    """Which bulb is lit per signal group."""

    ns_red: bool
    ns_yellow: bool
    ns_green: bool
    ns_blue: bool
    ns_blue_blink: bool

    ew_red: bool
    ew_yellow: bool
    ew_green: bool
    ew_blue: bool
    ew_blue_blink: bool


def lights_for_phase(phase: Phase) -> LightOutputs:
    # Default: all red
    out = LightOutputs(
        ns_red=True,
        ns_yellow=False,
        ns_green=False,
        ns_blue=False,
        ns_blue_blink=False,
        ew_red=True,
        ew_yellow=False,
        ew_green=False,
        ew_blue=False,
        ew_blue_blink=False,
    )

    if phase == Phase.NS_LEFT_SOLID:
        out.ns_blue = True
        return out
    if phase == Phase.NS_LEFT_BLINK:
        out.ns_blue_blink = True
        return out
    if phase == Phase.NS_LEFT_OFF_BUFFER:
        return out
    if phase == Phase.NS_STRAIGHT_GREEN:
        out.ns_red = False
        out.ns_green = True
        return out
    if phase == Phase.NS_STRAIGHT_YELLOW:
        out.ns_red = False
        out.ns_yellow = True
        return out
    if phase == Phase.ALL_RED_AFTER_NS:
        return out

    if phase == Phase.EW_LEFT_SOLID:
        out.ew_blue = True
        return out
    if phase == Phase.EW_LEFT_BLINK:
        out.ew_blue_blink = True
        return out
    if phase == Phase.EW_LEFT_OFF_BUFFER:
        return out
    if phase == Phase.EW_STRAIGHT_GREEN:
        out.ew_red = False
        out.ew_green = True
        return out
    if phase == Phase.EW_STRAIGHT_YELLOW:
        out.ew_red = False
        out.ew_yellow = True
        return out
    if phase == Phase.ALL_RED_AFTER_EW:
        return out

    return out


class AdaptiveTrafficController:
    def __init__(self, timing: TimingConfig, adaptive: bool = True) -> None:
        self.t = timing
        self.adaptive = adaptive
        self.phase = Phase.NS_LEFT_SOLID
        self.time_in_phase = 0.0
        self._ns_left_sec = timing.left_low_sec
        self._ns_straight_sec = timing.straight_low_sec
        self._ew_left_sec = timing.left_low_sec
        self._ew_straight_sec = timing.straight_low_sec

    def reset(self) -> None:
        self.phase = Phase.NS_LEFT_SOLID
        self.time_in_phase = 0.0

    def _corridor_counts(self, counts: dict[str, int], is_ns: bool) -> tuple[int, int]:
        if is_ns:
            left = int(counts.get("northL", 0)) + int(counts.get("southL", 0))
            straight = int(counts.get("north", 0)) + int(counts.get("south", 0))
        else:
            left = int(counts.get("westL", 0)) + int(counts.get("eastL", 0))
            straight = int(counts.get("west", 0)) + int(counts.get("east", 0))
        return left, straight

    def _plan_durations(self, counts: dict[str, int], is_ns: bool) -> tuple[float, float, bool, bool]:
        left_n, straight_n = self._corridor_counts(counts, is_ns)
        left_high = left_n >= self.t.high_threshold
        straight_high = (straight_n >= self.t.high_threshold) or left_high

        left_sec = self.t.left_high_sec if left_high else self.t.left_low_sec
        straight_sec = self.t.straight_high_sec if straight_high else self.t.straight_low_sec
        return left_sec, straight_sec, left_high, straight_high

    def tick(self, dt: float, counts: dict[str, int]) -> tuple[Phase, LightOutputs]:
        self.time_in_phase += dt

        if self.phase == Phase.NS_LEFT_SOLID:
            if self.adaptive:
                self._ns_left_sec, self._ns_straight_sec, *_ = self._plan_durations(counts, True)
            if self.time_in_phase >= self._ns_left_sec:
                self.phase = Phase.NS_LEFT_BLINK
                self.time_in_phase = 0.0

        elif self.phase == Phase.NS_LEFT_BLINK:
            if self.time_in_phase >= self.t.left_blink_sec:
                self.phase = Phase.NS_LEFT_OFF_BUFFER
                self.time_in_phase = 0.0

        elif self.phase == Phase.NS_LEFT_OFF_BUFFER:
            if self.time_in_phase >= self.t.left_off_buffer_sec:
                self.phase = Phase.NS_STRAIGHT_GREEN
                self.time_in_phase = 0.0

        elif self.phase == Phase.NS_STRAIGHT_GREEN:
            if self.time_in_phase >= self._ns_straight_sec:
                self.phase = Phase.NS_STRAIGHT_YELLOW
                self.time_in_phase = 0.0

        elif self.phase == Phase.NS_STRAIGHT_YELLOW:
            if self.time_in_phase >= self.t.yellow_sec:
                self.phase = Phase.ALL_RED_AFTER_NS
                self.time_in_phase = 0.0

        elif self.phase == Phase.ALL_RED_AFTER_NS:
            if self.time_in_phase >= self.t.all_red_clearance_sec:
                self.phase = Phase.EW_LEFT_SOLID
                self.time_in_phase = 0.0

        elif self.phase == Phase.EW_LEFT_SOLID:
            if self.adaptive:
                self._ew_left_sec, self._ew_straight_sec, *_ = self._plan_durations(counts, False)
            if self.time_in_phase >= self._ew_left_sec:
                self.phase = Phase.EW_LEFT_BLINK
                self.time_in_phase = 0.0

        elif self.phase == Phase.EW_LEFT_BLINK:
            if self.time_in_phase >= self.t.left_blink_sec:
                self.phase = Phase.EW_LEFT_OFF_BUFFER
                self.time_in_phase = 0.0

        elif self.phase == Phase.EW_LEFT_OFF_BUFFER:
            if self.time_in_phase >= self.t.left_off_buffer_sec:
                self.phase = Phase.EW_STRAIGHT_GREEN
                self.time_in_phase = 0.0

        elif self.phase == Phase.EW_STRAIGHT_GREEN:
            if self.time_in_phase >= self._ew_straight_sec:
                self.phase = Phase.EW_STRAIGHT_YELLOW
                self.time_in_phase = 0.0

        elif self.phase == Phase.EW_STRAIGHT_YELLOW:
            if self.time_in_phase >= self.t.yellow_sec:
                self.phase = Phase.ALL_RED_AFTER_EW
                self.time_in_phase = 0.0

        elif self.phase == Phase.ALL_RED_AFTER_EW:
            if self.time_in_phase >= self.t.all_red_clearance_sec:
                self.phase = Phase.NS_LEFT_SOLID
                self.time_in_phase = 0.0

        return self.phase, lights_for_phase(self.phase)


def summarize_demand(counts: dict[str, int]) -> tuple[int, int, str]:
    ns = counts.get("north", 0) + counts.get("south", 0) + counts.get("northL", 0) + counts.get("southL", 0)
    ew = counts.get("east", 0) + counts.get("west", 0) + counts.get("eastL", 0) + counts.get("westL", 0)
    level = "high" if (ns >= 4 or ew >= 4) else "low"
    return ns, ew, level
