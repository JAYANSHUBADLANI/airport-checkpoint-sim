"""Passenger show-up profile at the security checkpoint.

The functional form is the one used throughout airport terminal planning: a unimodal
density over minutes before scheduled departure, bounded below by a practical cutoff
and above by the point at which almost nobody has arrived yet. I use a Beta density
rescaled onto that window. The shape follows the domestic short-haul profiles
described in ACRP Report 25 (Airport Passenger Terminal Planning and Design,
Transportation Research Board, 2010) and the IATA Airport Development Reference
Manual. The specific parameter values are assumed, not measured, and are varied in
the sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class ShowupProfile:
    def __init__(self, earliest: float, latest: float, alpha: float, beta: float,
                 shift_minutes: float = 0.0):
        if latest <= earliest:
            raise ValueError("latest must exceed earliest")
        if earliest + shift_minutes < 0:
            raise ValueError("a shift that pushes show-up past departure is meaningless")
        self.earliest = float(earliest) + float(shift_minutes)
        self.latest = float(latest) + float(shift_minutes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._dist = stats.beta(self.alpha, self.beta)

    @classmethod
    def from_config(cls, cfg, shift_minutes: float = 0.0) -> "ShowupProfile":
        s = cfg["demand"]["showup"]
        if s["family"] != "beta":
            raise ValueError(f"unsupported show-up family {s['family']}")
        return cls(s["earliest_minutes_before_departure"],
                   s["latest_minutes_before_departure"],
                   s["alpha"], s["beta"], shift_minutes)

    @property
    def span(self) -> float:
        return self.latest - self.earliest

    def cdf_minutes_before(self, minutes_before) -> np.ndarray:
        """Probability that a passenger has already shown up, given minutes before departure.

        Larger minutes_before means earlier, so this is the upper tail of the Beta.
        """
        x = (np.asarray(minutes_before, dtype=float) - self.earliest) / self.span
        return 1.0 - self._dist.cdf(np.clip(x, 0.0, 1.0))

    def mean_minutes_before(self) -> float:
        return self.earliest + self.span * self.alpha / (self.alpha + self.beta)

    def mode_minutes_before(self) -> float:
        if self.alpha <= 1 or self.beta <= 1:
            return float("nan")
        return self.earliest + self.span * (self.alpha - 1) / (self.alpha + self.beta - 2)

    def slot_pmf(self, slot_minutes: int) -> np.ndarray:
        """Share of a flight's passengers arriving in each slot before departure.

        Element k covers the window [earliest + k*slot, earliest + (k+1)*slot) minutes
        before departure, so k = 0 is the latest arriving group.
        """
        n = int(np.ceil(self.span / slot_minutes))
        edges = self.earliest + slot_minutes * np.arange(n + 1)
        x = np.clip((edges - self.earliest) / self.span, 0.0, 1.0)
        pmf = np.maximum(np.diff(self._dist.cdf(x)), 0.0)
        total = pmf.sum()
        if total <= 0:
            raise ValueError("degenerate show-up profile")
        return pmf / total
