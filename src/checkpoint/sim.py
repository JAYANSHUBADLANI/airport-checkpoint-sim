"""Discrete event simulation of an airport security checkpoint.

Passengers arrive as a non-homogeneous Poisson process whose piecewise constant rate
comes from the demand model. Each passenger queues for a travel document check, then
queues for a screening lane. A share of passengers is selected for secondary screening,
which occupies the lane for an additional draw. Lanes open and close on the staffing
slot boundary and always finish the passenger in front of them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy

from .config import Config
from .staffing import StaffingPlan


def lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    sigma2 = np.log1p(cv ** 2)
    return float(np.log(mean) - 0.5 * sigma2), float(np.sqrt(sigma2))


@dataclass
class ServiceSpec:
    """A service time in minutes, declared in seconds.

    The default shape is lognormal, which is what screening time actually looks like.
    The exponential and deterministic shapes exist so the engine can be checked against
    results that only hold for those shapes.
    """

    mean_seconds: float
    cv: float
    dist: str = "lognormal"

    @property
    def mean_minutes(self) -> float:
        return self.mean_seconds / 60.0

    def sampler(self, rng: np.random.Generator):
        m = self.mean_minutes
        if m <= 0:
            return lambda: 0.0
        if self.dist == "deterministic" or self.cv == 0:
            return lambda: m
        if self.dist == "exponential":
            return lambda: float(rng.exponential(m))
        mu, sigma = lognormal_params(m, self.cv)
        return lambda: float(rng.lognormal(mu, sigma))


@dataclass
class Passenger:
    idx: int
    arrive: float
    flight_dep: float
    doc_start: float = np.nan
    doc_end: float = np.nan
    lane_start: float = np.nan
    lane_end: float = np.nan
    secondary: bool = False


@dataclass
class SimResult:
    passengers: pd.DataFrame
    plan: StaffingPlan
    seed: int
    meta: dict = field(default_factory=dict)


class CheckpointSim:
    def __init__(self, cfg: Config, plan: StaffingPlan, arrival_rate: np.ndarray,
                 slot_minutes: int, start_minute: int, rng: np.random.Generator,
                 dep_lag_weights: np.ndarray | None = None,
                 service_scale: float = 1.0,
                 fixed_arrivals: tuple[np.ndarray, np.ndarray] | None = None,
                 hard_stop_minutes: float = 360.0,
                 service_override: dict | None = None):
        self.cfg = cfg
        self.plan = plan
        self.rate = np.asarray(arrival_rate, dtype=float)
        self.slot_minutes = int(slot_minutes)
        self.start_minute = int(start_minute)
        self.rng = rng
        self.dep_lag_weights = dep_lag_weights
        self.service_scale = float(service_scale)
        self.fixed_arrivals = fixed_arrivals

        svc = cfg["service"]
        self.doc_spec = ServiceSpec(svc["doc_check"]["mean_seconds"] * service_scale,
                                    svc["doc_check"]["cv"])
        self.lane_spec = ServiceSpec(svc["screening"]["mean_seconds"] * service_scale,
                                     svc["screening"]["cv"])
        self.sec_spec = ServiceSpec(svc["secondary"]["mean_seconds"] * service_scale,
                                    svc["secondary"]["cv"])
        self.secondary_share = float(svc["secondary"]["share"])
        if service_override:
            for key, spec in service_override.items():
                if key == "doc":
                    self.doc_spec = spec
                elif key == "lane":
                    self.lane_spec = spec
                elif key == "secondary_share":
                    self.secondary_share = float(spec)
                else:
                    raise KeyError(f"unknown service override {key}")
        self.doc_per_lane = float(svc["doc_check"]["positions_per_lane"])
        self.doc_min = int(svc["doc_check"]["min_positions"])

        self.horizon_end = self.start_minute + len(self.rate) * self.slot_minutes
        self.max_lanes = int(self.plan.lanes.max())
        self.max_doc = self._doc_positions(self.max_lanes)
        self.hard_stop = self.horizon_end + float(hard_stop_minutes)
        self.records: list[Passenger] = []
        self.created: list[Passenger] = []

    def _doc_positions(self, lanes: int) -> int:
        return max(self.doc_min, int(np.ceil(lanes * self.doc_per_lane)))

    def _to_next_boundary(self, now: float) -> float:
        rel = now - self.start_minute
        nxt = (np.floor(rel / self.slot_minutes) + 1) * self.slot_minutes
        return float(nxt - rel)

    def _generate_arrivals(self) -> tuple[np.ndarray, np.ndarray]:
        if self.fixed_arrivals is not None:
            t, d = self.fixed_arrivals
            return np.asarray(t, dtype=float), np.asarray(d, dtype=float)
        counts = self.rng.poisson(self.rate)
        times, deps = [], []
        for i, c in enumerate(counts):
            if c == 0:
                continue
            lo = self.start_minute + i * self.slot_minutes
            t = lo + self.rng.uniform(0.0, self.slot_minutes, size=c)
            times.append(t)
            if self.dep_lag_weights is not None:
                w = self.dep_lag_weights[i]
                s = w.sum()
                if s <= 0:
                    lags = np.zeros(c, dtype=int)
                else:
                    lags = self.rng.choice(len(w), size=c, p=w / s)
                deps.append(t + (lags + 0.5) * self.slot_minutes)
            else:
                deps.append(np.full(c, np.inf))
        if not times:
            return np.array([]), np.array([])
        times = np.concatenate(times)
        deps = np.concatenate(deps)
        order = np.argsort(times)
        return times[order], deps[order]

    def run(self) -> SimResult:
        env = simpy.Environment(initial_time=float(self.start_minute))
        draw_doc = self.doc_spec.sampler(self.rng)
        draw_lane = self.lane_spec.sampler(self.rng)
        draw_sec = self.sec_spec.sampler(self.rng)
        times, deps = self._generate_arrivals()

        doc_queue: deque[Passenger] = deque()
        lane_queue: deque[Passenger] = deque()
        busy = {"doc": 0, "lane": 0}

        def doc_capacity() -> int:
            return self._doc_positions(self.plan.lanes_at(env.now))

        def lane_capacity() -> int:
            return self.plan.lanes_at(env.now)

        def serve_doc(p: Passenger):
            p.doc_start = env.now
            yield env.timeout(draw_doc())
            p.doc_end = env.now
            busy["doc"] -= 1
            lane_queue.append(p)
            dispatch_lane()
            dispatch_doc()

        def serve_lane(p: Passenger):
            p.lane_start = env.now
            duration = draw_lane()
            if self.rng.random() < self.secondary_share:
                p.secondary = True
                duration += draw_sec()
            yield env.timeout(duration)
            p.lane_end = env.now
            busy["lane"] -= 1
            self.records.append(p)
            dispatch_lane()

        def dispatch_doc():
            cap = doc_capacity()
            while doc_queue and busy["doc"] < cap:
                busy["doc"] += 1
                env.process(serve_doc(doc_queue.popleft()))

        def dispatch_lane():
            cap = lane_capacity()
            while lane_queue and busy["lane"] < cap:
                busy["lane"] += 1
                env.process(serve_lane(lane_queue.popleft()))

        def arrivals_proc():
            last = env.now
            for k in range(len(times)):
                yield env.timeout(max(0.0, times[k] - last))
                last = times[k]
                p = Passenger(idx=k, arrive=env.now, flight_dep=float(deps[k]))
                self.created.append(p)
                doc_queue.append(p)
                dispatch_doc()

        def boundary_proc():
            """Wake on every staffing boundary so newly opened lanes pick work up."""
            step = float(self.plan.slot_minutes)
            while env.now < self.hard_stop:
                rel = env.now - self.start_minute
                nxt = (np.floor(rel / step) + 1) * step
                yield env.timeout(float(nxt - rel))
                dispatch_doc()
                dispatch_lane()
                if not doc_queue and not lane_queue and env.now >= self.horizon_end:
                    return

        env.process(arrivals_proc())
        env.process(boundary_proc())
        env.run()

        unserved = len(self.created) - len(self.records)
        df = pd.DataFrame([{
            "idx": p.idx, "arrive": p.arrive, "flight_dep": p.flight_dep,
            "doc_start": p.doc_start, "doc_end": p.doc_end,
            "lane_start": p.lane_start, "lane_end": p.lane_end,
            "secondary": p.secondary,
        } for p in self.created])
        if df.empty:
            df = pd.DataFrame(columns=["idx", "arrive", "flight_dep", "doc_start", "doc_end",
                                       "lane_start", "lane_end", "secondary"])
        df["doc_wait"] = df["doc_start"] - df["arrive"]
        df["lane_wait"] = df["lane_start"] - df["doc_end"]
        df["total_wait"] = df["doc_wait"] + df["lane_wait"]
        df["time_in_system"] = df["lane_end"] - df["arrive"]
        cutoff = float(self.cfg["target"]["clearance_cutoff_minutes"])
        deadline = df["flight_dep"] - cutoff
        df["cleared_in_time"] = df["lane_end"] <= deadline
        df["would_miss_without_queueing"] = (df["lane_end"] - df["total_wait"]) > deadline
        df["arrive_hour"] = np.floor(df["arrive"] / 60.0).astype(int)
        df["clock_hour"] = df["arrive_hour"] % 24
        return SimResult(df, self.plan, seed=-1,
                         meta={"n_passengers": int(len(df)),
                               "n_unserved": int(unserved),
                               "service_scale": self.service_scale})
