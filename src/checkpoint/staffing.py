"""Staffing plans expressed as lanes open per staffing slot."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaffingPlan:
    """Lanes open in each staffing slot, anchored at minute zero of the simulated day."""

    lanes: np.ndarray
    slot_minutes: int
    start_minute: int
    name: str = "plan"

    def __post_init__(self):
        object.__setattr__(self, "lanes", np.asarray(self.lanes, dtype=int))

    @property
    def n_slots(self) -> int:
        return len(self.lanes)

    @property
    def end_minute(self) -> int:
        return self.start_minute + self.n_slots * self.slot_minutes

    def lanes_at(self, minute: float) -> int:
        idx = int((minute - self.start_minute) // self.slot_minutes)
        if idx < 0:
            return int(self.lanes[0])
        if idx >= self.n_slots:
            return int(self.lanes[-1])
        return int(self.lanes[idx])

    def lane_hours(self) -> float:
        return float(self.lanes.sum()) * self.slot_minutes / 60.0

    def cost(self, cost_per_lane_hour: float) -> float:
        return self.lane_hours() * cost_per_lane_hour

    def with_outage(self, start_minute: int, duration_minutes: int, lanes_down: int,
                    name: str | None = None) -> "StaffingPlan":
        lanes = self.lanes.copy()
        first = max(0, int((start_minute - self.start_minute) // self.slot_minutes))
        last = min(self.n_slots,
                   int(math.ceil((start_minute + duration_minutes - self.start_minute)
                                 / self.slot_minutes)))
        lanes[first:last] = np.maximum(1, lanes[first:last] - lanes_down)
        return StaffingPlan(lanes, self.slot_minutes, self.start_minute,
                            name or f"{self.name}+outage")

    def rename(self, name: str) -> "StaffingPlan":
        return StaffingPlan(self.lanes.copy(), self.slot_minutes, self.start_minute, name)


def flat_plan(n_slots: int, lanes: int, slot_minutes: int, start_minute: int,
              name: str = "flat") -> StaffingPlan:
    return StaffingPlan(np.full(n_slots, lanes, dtype=int), slot_minutes, start_minute, name)


def rule_of_thumb_plan(arrivals_per_slot: np.ndarray, lane_rate_per_hour: float,
                       slot_minutes: int, start_minute: int, min_lanes: int = 1,
                       max_lanes: int = 99, name: str = "rule_of_thumb") -> StaffingPlan:
    """Lanes equal to expected arrivals divided by lane throughput, rounded up.

    This is the plan an operations manager builds on a whiteboard: no queueing theory,
    no variability allowance, just demand over capacity.
    """
    per_hour = arrivals_per_slot * (60.0 / slot_minutes)
    lanes = np.ceil(per_hour / lane_rate_per_hour).astype(int)
    lanes = np.clip(lanes, min_lanes, max_lanes)
    return StaffingPlan(lanes, slot_minutes, start_minute, name)
