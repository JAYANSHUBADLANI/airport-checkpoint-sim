"""Bundle one airport-day into everything the simulation needs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .demand import DemandModel


@dataclass
class DayScenario:
    airport: str
    day: pd.Timestamp
    arrivals_per_slot: np.ndarray
    dep_lag_weights: np.ndarray
    demand_slot_minutes: int
    staffing_slot_minutes: int
    start_minute: int
    slot_starts: pd.DatetimeIndex

    @property
    def n_demand_slots(self) -> int:
        return len(self.arrivals_per_slot)

    @property
    def n_staffing_slots(self) -> int:
        per = self.staffing_slot_minutes // self.demand_slot_minutes
        return self.n_demand_slots // per

    def arrivals_per_staffing_slot(self) -> np.ndarray:
        per = self.staffing_slot_minutes // self.demand_slot_minutes
        return self.arrivals_per_slot.reshape(self.n_staffing_slots, per).sum(axis=1)

    def rate_per_hour_by_staffing_slot(self) -> np.ndarray:
        return self.arrivals_per_staffing_slot() * (60.0 / self.staffing_slot_minutes)

    def scaled(self, factor: float) -> "DayScenario":
        return DayScenario(self.airport, self.day, self.arrivals_per_slot * factor,
                           self.dep_lag_weights, self.demand_slot_minutes,
                           self.staffing_slot_minutes, self.start_minute, self.slot_starts)


def build_scenario(cfg: Config, model: DemandModel, dep: pd.DataFrame, airport: str,
                   day, connecting_share: float | None = None,
                   originating=None) -> DayScenario:
    day = pd.Timestamp(day).normalize()
    arr = model.arrivals(dep, airport, connecting_share, originating)
    sub = model.day_slice(arr, day)
    slot = model.slot_minutes
    staffing_slot = int(cfg["staffing"]["slot_minutes"])
    if staffing_slot % slot:
        raise ValueError("staffing slot must be a whole number of demand slots")
    start_minute = int(cfg["run"]["sim_start_hour"]) * 60
    weights = _lag_weights(model, dep, airport, sub, connecting_share, originating)
    return DayScenario(airport, day, sub["arrivals"].to_numpy(), weights, slot,
                       staffing_slot, start_minute,
                       pd.DatetimeIndex(sub["slot_start"]))


def _lag_weights(model: DemandModel, dep: pd.DataFrame, airport: str,
                 day_arrivals: pd.DataFrame, connecting_share, originating=None) -> np.ndarray:
    """For each arrival slot, the mix of departure lags the arrivals belong to."""
    sub = model.originating_passengers(dep, airport, connecting_share, originating)
    freq = pd.Timedelta(minutes=model.slot_minutes)
    origin = day_arrivals["slot_start"].iloc[0]
    n = len(day_arrivals)
    n_lags = len(model.pmf)
    horizon = n + model.base_offset + n_lags + 2
    seats = np.zeros(horizon, dtype=float)
    rel = ((sub["sched_dep"] - origin) // freq).to_numpy(dtype=np.int64)
    pax = sub["pax_originating"].to_numpy(dtype=float)
    ok = (rel >= 0) & (rel < horizon)
    np.add.at(seats, rel[ok], pax[ok])
    w = np.zeros((n, n_lags), dtype=float)
    for k in range(n_lags):
        lag = model.base_offset + k
        w[:, k] = model.pmf[k] * seats[lag:lag + n]
    return w
