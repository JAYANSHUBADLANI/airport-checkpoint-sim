"""Closed form M/M/c results, used for verification and inside the optimiser.

Erlang B is computed by the standard recursion, which stays numerically stable for the
server counts a checkpoint reaches. Erlang C follows from it. The waiting time of a
delayed customer in M/M/c is exponential, so the whole waiting time distribution is a
mixture of a point mass at zero and that exponential tail.
"""

from __future__ import annotations

import math

import numpy as np


def erlang_b(c: int, a: float) -> float:
    inv = 1.0
    for k in range(1, c + 1):
        inv = 1.0 + inv * k / a
    return 1.0 / inv


def erlang_c(c: int, a: float) -> float:
    """Probability that an arriving customer has to wait, offered load a in erlangs."""
    if c <= 0:
        return 1.0
    rho = a / c
    if rho >= 1.0:
        return 1.0
    b = erlang_b(c, a)
    return b / (1.0 - rho * (1.0 - b))


def mmc_mean_wait(lam: float, mu: float, c: int) -> float:
    """Mean time in queue. lam and mu must share the same time unit."""
    if lam <= 0:
        return 0.0
    a = lam / mu
    if a >= c:
        return math.inf
    return erlang_c(c, a) / (c * mu - lam)


def mmc_wait_tail(t: float, lam: float, mu: float, c: int) -> float:
    a = lam / mu
    if a >= c:
        return 1.0
    return erlang_c(c, a) * math.exp(-(c * mu - lam) * t)


def mmc_wait_quantile(q: float, lam: float, mu: float, c: int) -> float:
    """Quantile q of the waiting time, zero when a share above q is served on arrival."""
    if lam <= 0:
        return 0.0
    a = lam / mu
    if a >= c:
        return math.inf
    cc = erlang_c(c, a)
    tail = 1.0 - q
    if cc <= tail:
        return 0.0
    return math.log(cc / tail) / (c * mu - lam)


def mmc_mean_system_time(lam: float, mu: float, c: int) -> float:
    return mmc_mean_wait(lam, mu, c) + 1.0 / mu


def required_servers(lam: float, mu: float, q: float, target: float,
                     c_max: int = 200, c_min: int = 1) -> int:
    for c in range(max(1, c_min), c_max + 1):
        if mmc_wait_quantile(q, lam, mu, c) <= target:
            return c
    return c_max


def effective_lane_rate_per_hour(cfg) -> float:
    """Passengers per hour a single open lane clears, including secondary screening."""
    svc = cfg["service"]
    mean_seconds = (svc["screening"]["mean_seconds"]
                    + svc["secondary"]["share"] * svc["secondary"]["mean_seconds"])
    return 3600.0 / mean_seconds


def doc_rate_per_hour(cfg) -> float:
    return 3600.0 / cfg["service"]["doc_check"]["mean_seconds"]


def required_lanes(lam_per_hour: float, cfg, target_minutes: float | None = None,
                   quantile: float = 0.95, c_max: int = 200) -> int:
    """Smallest lane count whose two stage M/M/c approximation meets the wait target.

    The checkpoint is two queues in series. I approximate the checkpoint wait as the sum
    of the two stage quantiles, which is conservative because the true quantile of a sum
    is below the sum of the quantiles. Document check positions scale with lanes exactly
    as they do in the simulation.
    """
    target = float(cfg["target"]["p95_wait_minutes"] if target_minutes is None
                   else target_minutes)
    mu_lane = effective_lane_rate_per_hour(cfg) / 60.0
    mu_doc = doc_rate_per_hour(cfg) / 60.0
    lam = lam_per_hour / 60.0
    ppl = float(cfg["service"]["doc_check"]["positions_per_lane"])
    dmin = int(cfg["service"]["doc_check"]["min_positions"])
    for c in range(1, c_max + 1):
        d = max(dmin, math.ceil(c * ppl))
        w = (mmc_wait_quantile(quantile, lam, mu_doc, d)
             + mmc_wait_quantile(quantile, lam, mu_lane, c))
        if w <= target:
            return c
    return c_max


def required_lanes_curve(cfg, max_rate: float = 8000.0, step: float = 25.0,
                         target_minutes: float | None = None) -> np.ndarray:
    rates = np.arange(0.0, max_rate + step, step)
    return np.array([[r, required_lanes(r, cfg, target_minutes)] for r in rates])
