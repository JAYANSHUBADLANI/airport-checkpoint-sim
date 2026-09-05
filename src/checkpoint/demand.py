"""Turn a flight schedule into checkpoint arrivals per slot.

Passengers per departure are seats times load factor times the share that is not
connecting, because connecting passengers stay airside and do not clear the checkpoint
again. Each flight's originating passengers are then pushed backwards in time through
the show-up profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .seats import attach_seats
from .showup import ShowupProfile


class DemandModel:
    def __init__(self, cfg: Config, showup: ShowupProfile | None = None,
                 load_factor: float | None = None):
        self.cfg = cfg
        self.slot_minutes = int(cfg["demand"]["slot_minutes"])
        self.showup = showup or ShowupProfile.from_config(cfg)
        self.load_factor = float(load_factor if load_factor is not None
                                 else cfg["demand"]["load_factor"])
        if self.showup.earliest % self.slot_minutes or self.showup.span % self.slot_minutes:
            raise ValueError("show-up window must be a whole number of slots")
        self.pmf = self.showup.slot_pmf(self.slot_minutes)
        self.base_offset = int(self.showup.earliest // self.slot_minutes)

    def originating_passengers(self, dep: pd.DataFrame, airport: str,
                               connecting_share: float | None = None,
                               originating=None) -> pd.DataFrame:
        """Passengers per departure who will clear the checkpoint.

        The originating share may be a single number or a function of the scheduled
        departure hour. The second form exists because a hub's evening banks carry a
        much larger connecting share than its morning banks, which a single number
        cannot represent.
        """
        sub = dep[dep["airport"] == airport].copy()
        sub = attach_seats(sub)
        sub["pax_onboard"] = sub["seats"] * self.load_factor
        if originating is None:
            share = 1.0 - (self.cfg.connecting_share(airport) if connecting_share is None
                           else connecting_share)
            sub["originating_share"] = share
        elif callable(originating):
            sub["originating_share"] = originating(sub["sched_dep"].dt.hour.to_numpy())
        else:
            sub["originating_share"] = float(originating)
        sub["pax_originating"] = sub["pax_onboard"] * sub["originating_share"]
        return sub

    def arrivals(self, dep: pd.DataFrame, airport: str,
                 connecting_share: float | None = None, originating=None) -> pd.DataFrame:
        """Checkpoint arrivals per slot across the whole loaded period."""
        sub = self.originating_passengers(dep, airport, connecting_share, originating)
        if sub.empty:
            raise ValueError(f"no departures for {airport}")
        start = sub["sched_dep"].min().normalize() - pd.Timedelta(days=1)
        end = sub["sched_dep"].max().normalize() + pd.Timedelta(days=1)
        freq = pd.Timedelta(minutes=self.slot_minutes)
        index = pd.date_range(start, end, freq=freq)
        n = len(index)

        dep_slot = ((sub["sched_dep"] - start) // freq).to_numpy(dtype=np.int64)
        pax = sub["pax_originating"].to_numpy(dtype=float)
        seats = np.zeros(n + len(self.pmf) + self.base_offset + 1, dtype=float)
        np.add.at(seats, dep_slot, pax)

        arrivals = np.zeros(n, dtype=float)
        for k, w in enumerate(self.pmf):
            lag = self.base_offset + k
            arrivals += w * seats[lag:lag + n]

        out = pd.DataFrame({"slot_start": index, "arrivals": arrivals})
        out["airport"] = airport
        out["date"] = out["slot_start"].dt.normalize()
        out["hour"] = out["slot_start"].dt.hour
        return out

    def day_slice(self, arrivals: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        day = pd.Timestamp(day).normalize()
        lo = day + pd.Timedelta(hours=int(self.cfg["run"]["sim_start_hour"]))
        hi = day + pd.Timedelta(hours=int(self.cfg["run"]["sim_end_hour"]))
        m = (arrivals["slot_start"] >= lo) & (arrivals["slot_start"] < hi)
        return arrivals.loc[m].reset_index(drop=True)


def profile_stats(day_arrivals: pd.DataFrame, slot_minutes: int) -> dict:
    a = day_arrivals["arrivals"].to_numpy()
    total = a.sum()
    per_hour = day_arrivals.groupby(day_arrivals["slot_start"].dt.floor("h"))["arrivals"].sum()
    slots_per_hour = 60 // slot_minutes
    peak_slot_idx = int(np.argmax(a))
    return {
        "total_pax": float(total),
        "peak_slot_start": day_arrivals["slot_start"].iloc[peak_slot_idx],
        "peak_slot_pax": float(a[peak_slot_idx]),
        "peak_slot_rate_per_hour": float(a[peak_slot_idx] * slots_per_hour),
        "mean_slot_pax": float(a.mean()),
        "peak_to_mean_slot": float(a[peak_slot_idx] / a.mean()) if a.mean() > 0 else float("nan"),
        "peak_hour": per_hour.idxmax(),
        "peak_hour_pax": float(per_hour.max()),
        "mean_hour_pax": float(per_hour.mean()),
        "peak_to_mean_hour": float(per_hour.max() / per_hour.mean()),
    }
