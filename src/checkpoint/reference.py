"""An independent queue simulator used only to check the SimPy model.

This is a Lindley style recursion over a heap of server free times. It shares no code
with the SimPy engine, so agreement between the two is evidence that the event handling
in the main model is right rather than evidence that one bug is present twice.
"""

from __future__ import annotations

import heapq

import numpy as np


def lindley_waits(arrivals: np.ndarray, services: np.ndarray, servers: int) -> np.ndarray:
    """Waiting time of each arrival in a first come first served c server queue."""
    free = [0.0] * servers
    heapq.heapify(free)
    waits = np.empty(len(arrivals), dtype=float)
    for k, t in enumerate(arrivals):
        earliest = heapq.heappop(free)
        start = t if t > earliest else earliest
        waits[k] = start - t
        heapq.heappush(free, start + services[k])
    return waits


def poisson_arrivals(rng: np.random.Generator, minutes: int, rate_per_hour: float,
                     slot_minutes: int = 5) -> np.ndarray:
    """Poisson counts per slot with uniform placement inside the slot."""
    counts = rng.poisson(rate_per_hour * slot_minutes / 60.0, size=minutes // slot_minutes)
    parts = [i * slot_minutes + rng.uniform(0.0, slot_minutes, size=c)
             for i, c in enumerate(counts) if c]
    if not parts:
        return np.array([])
    return np.sort(np.concatenate(parts))


class SequenceSpec:
    """Feeds a fixed list of service times to the simulation, for exact comparisons."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        self.mean_seconds = float(self.values.mean() * 60.0) if len(self.values) else 0.0

    def sampler(self, rng):
        it = iter(self.values)
        return lambda: float(next(it))


class ZeroSpec:
    mean_seconds = 0.0

    def sampler(self, rng):
        return lambda: 0.0
