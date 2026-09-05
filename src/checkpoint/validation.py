"""Validation of the demand model against observed TSA checkpoint throughput.

This is separate from verification. Verification asks whether the simulation computes a
queue correctly. Validation asks whether the number of people arriving at the checkpoint
in the model looks like the number of people who actually turned up.

Two caveats shape how the comparison is read, and neither can be removed with the data
available here. The BTS on-time file covers domestic flights operated by reporting
carriers only, so international departures are missing from the model. The TSA count is
total passengers plus known crewmembers, so it includes staff the model never generates.
These push in opposite directions and cannot be separated with the data available, so
what I fit is a screening yield: checkpoint passengers per scheduled domestic seat. See
the ScreeningYield docstring for exactly what that absorbs.

Two parameters of the yield and one timing parameter of the show-up curve are fitted on
1 to 6 June 2026. Everything reported as validation is scored on 7 to 13 June 2026,
which the fit never saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .demand import DemandModel


def model_hourly(model: DemandModel, dep: pd.DataFrame, airport: str,
                 connecting_share: float = 0.0) -> pd.DataFrame:
    arr = model.arrivals(dep, airport, connecting_share=connecting_share)
    out = arr.groupby(["date", "hour"], as_index=False)["arrivals"].sum()
    out["airport"] = airport
    return out


def observed_hourly(tsa: pd.DataFrame, airport: str,
                    exclude: list[str] | None = None) -> pd.DataFrame:
    """Observed throughput per hour, after dropping checkpoints the model cannot generate.

    At ATL the F Arrival checkpoint screens international arrivals re-entering the
    sterile area to connect onwards, and the Private Terminal serves general aviation.
    Neither is a departing passenger produced by the scheduled departure file, so both
    are excluded. The excluded checkpoints are named in the config, not here.
    """
    sub = tsa[tsa["airport"] == airport]
    if exclude:
        sub = sub[~sub["checkpoint"].isin(exclude)]
    return sub.groupby(["date", "hour"], as_index=False)["throughput"].sum()


def excluded_share(tsa: pd.DataFrame, airport: str, exclude: list[str] | None) -> float:
    sub = tsa[tsa["airport"] == airport]
    if not exclude:
        return 0.0
    total = sub["throughput"].sum()
    return float(sub[sub["checkpoint"].isin(exclude)]["throughput"].sum() / total)


def fit_scalar_yield(model_h: pd.DataFrame, observed_h: pd.DataFrame,
                     days: list) -> dict:
    """The single multiplier on gross seats that reproduces observed volume.

    This is the flat version of ScreeningYield, kept as the baseline that the
    hour-varying fit has to beat.

    Because arrivals scale linearly in the originating share, matching total volume over
    the calibration days has a closed form: it is observed total over modelled total at a
    zero connecting share.
    """
    days = [pd.Timestamp(d).normalize() for d in days]
    m = model_h[model_h["date"].isin(days)]["arrivals"].sum()
    o = observed_h[observed_h["date"].isin(days)]["throughput"].sum()
    if m <= 0:
        raise ValueError("modelled volume is zero")
    factor = float(o / m)
    return {"scalar_yield": factor,
            "implied_non_originating_share": float(1.0 - factor),
            "modelled_gross_pax": float(m),
            "observed_pax": float(o),
            "calibration_days": days}


def score(model_h: pd.DataFrame, observed_h: pd.DataFrame, factor: float,
          days: list) -> dict:
    days = [pd.Timestamp(d).normalize() for d in days]
    m = model_h[model_h["date"].isin(days)].copy()
    m["modelled"] = m["arrivals"] * factor
    o = observed_h[observed_h["date"].isin(days)]
    j = m.merge(o, on=["date", "hour"], how="inner")
    if j.empty:
        raise ValueError("no overlapping hours between model and observation")
    err = j["modelled"] - j["throughput"]
    daily = j.groupby("date")[["modelled", "throughput"]].sum()
    busy = j[j["throughput"] >= 200]
    diurnal = j.groupby("hour")[["modelled", "throughput"]].mean()
    peak_model = int(diurnal["modelled"].idxmax())
    peak_obs = int(diurnal["throughput"].idxmax())
    morning = diurnal.loc[diurnal.index.isin(range(5, 10))]
    evening = diurnal.loc[diurnal.index.isin(range(17, 22))]
    return {
        "n_hours": int(len(j)),
        "hourly_mae": float(np.abs(err).mean()),
        "hourly_rmse": float(np.sqrt((err ** 2).mean())),
        "hourly_mape_busy": float((np.abs(busy["modelled"] - busy["throughput"])
                                   / busy["throughput"]).mean() * 100.0),
        "hourly_corr": float(np.corrcoef(j["modelled"], j["throughput"])[0, 1]),
        "level_ratio": float(j["modelled"].sum() / j["throughput"].sum()),
        "daily_mape": float((np.abs(daily["modelled"] - daily["throughput"])
                             / daily["throughput"]).mean() * 100.0),
        "peak_hour_model": peak_model,
        "peak_hour_observed": peak_obs,
        "morning_ratio": float(morning["modelled"].sum() / morning["throughput"].sum()),
        "evening_ratio": float(evening["modelled"].sum() / evening["throughput"].sum()),
    }


def showup_shift_diagnostic(cfg: Config, dep: pd.DataFrame, airport: str,
                            observed_h: pd.DataFrame, days: list,
                            shifts=range(-15, 61, 5)) -> pd.DataFrame:
    """How the hourly fit responds to moving the show-up curve earlier or later.

    The shift chosen for the model is the one that minimises calibration RMSE here, so
    this table is the evidence behind that choice rather than a separate exercise.
    """
    from .showup import ShowupProfile
    rows = []
    for shift in shifts:
        m = DemandModel(cfg, showup=ShowupProfile.from_config(cfg, shift_minutes=shift))
        yld = fit_screening_yield(m, dep, airport, observed_h, days)
        s = score_profile(m, dep, airport, observed_h, yld, days)
        rows.append({"shift_minutes": int(shift), "yield_intercept": yld.intercept,
                     "yield_slope": yld.slope,
                     **{k: s[k] for k in ["hourly_rmse", "hourly_corr", "hourly_mape_busy"]}})
    return pd.DataFrame(rows)


class ScreeningYield:
    """Checkpoint passengers per scheduled domestic seat, as a line in departure hour.

    This is deliberately not called a connecting share. What the data can identify is a
    yield: how many people turn up at the checkpoint for every seat the domestic
    schedule offers. That yield absorbs four things at once, and nothing available here
    separates them. Connecting passengers, who never clear the checkpoint again.
    International departures, which the BTS domestic file does not contain and which
    therefore add checkpoint passengers with no seats behind them. Crew, whom the TSA
    count includes and the model never generates. And any error in the assumed seats per
    departure and load factor.

    The yield is allowed to vary with departure hour because a hub's morning banks are
    mostly people starting a trip while its evening banks are mostly people changing
    planes, and one number cannot describe both. Two parameters fitted on six days and
    tested on the following seven is a defensible amount of fitting. The line is clipped
    to a sensible range and the clipping is reported rather than hidden.
    """

    LOW, HIGH = 0.02, 1.30

    def __init__(self, intercept: float, slope: float):
        self.intercept = float(intercept)
        self.slope = float(slope)

    def __call__(self, hour):
        h = np.asarray(hour, dtype=float)
        return np.clip(self.intercept + self.slope * (h - 12.0) / 12.0, self.LOW, self.HIGH)

    def clips(self) -> bool:
        h = np.arange(24)
        raw = self.intercept + self.slope * (h - 12.0) / 12.0
        return bool((raw < self.LOW).any() or (raw > self.HIGH).any())

    def as_dict(self) -> dict:
        return {"intercept": self.intercept, "slope": self.slope,
                "share_at_06": float(self(6)), "share_at_12": float(self(12)),
                "share_at_20": float(self(20)), "clipped": self.clips()}


def fit_screening_yield(model: DemandModel, dep: pd.DataFrame, airport: str,
                        observed_h: pd.DataFrame, days: list) -> ScreeningYield:
    """Least squares on the two design curves, with no intercept beyond them."""
    days = [pd.Timestamp(d).normalize() for d in days]
    base = model_hourly(model, dep, airport, connecting_share=0.0)
    lin = model.arrivals(dep, airport, originating=lambda h: (h - 12.0) / 12.0)
    lin = lin.groupby(["date", "hour"], as_index=False)["arrivals"].sum()
    j = (base.rename(columns={"arrivals": "a0"})
             .merge(lin.rename(columns={"arrivals": "a1"}), on=["date", "hour"])
             .merge(observed_h, on=["date", "hour"]))
    j = j[j["date"].isin(days)]
    if len(j) < 12:
        raise ValueError("not enough overlapping hours to fit the screening yield")
    X = j[["a0", "a1"]].to_numpy(dtype=float)
    y = j["throughput"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return ScreeningYield(coef[0], coef[1])


def model_hourly_with(model: DemandModel, dep: pd.DataFrame, airport: str,
                      originating) -> pd.DataFrame:
    arr = model.arrivals(dep, airport, originating=originating)
    out = arr.groupby(["date", "hour"], as_index=False)["arrivals"].sum()
    out["airport"] = airport
    return out


def score_profile(model: DemandModel, dep: pd.DataFrame, airport: str,
                  observed_h: pd.DataFrame, profile: ScreeningYield,
                  days: list) -> dict:
    mh = model_hourly_with(model, dep, airport, profile)
    return score(mh, observed_h, 1.0, days)


def select_showup_shift(cfg: Config, dep: pd.DataFrame, airport: str,
                        observed_h: pd.DataFrame, days: list,
                        shifts=range(-15, 61, 5)) -> tuple[int, ScreeningYield]:
    """Pick the one timing parameter that best matches the calibration week."""
    from .showup import ShowupProfile
    best = None
    for shift in shifts:
        model = DemandModel(cfg, showup=ShowupProfile.from_config(cfg, shift_minutes=shift))
        profile = fit_screening_yield(model, dep, airport, observed_h, days)
        s = score_profile(model, dep, airport, observed_h, profile, days)
        if best is None or s["hourly_rmse"] < best[0]:
            best = (s["hourly_rmse"], int(shift), profile)
    return best[1], best[2]
