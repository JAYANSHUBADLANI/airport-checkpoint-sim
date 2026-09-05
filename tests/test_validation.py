"""Validation plumbing, exercised on the committed sample so no network call is needed."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from checkpoint.demand import DemandModel
from checkpoint.schedule import read_sample
from checkpoint.showup import ShowupProfile
from checkpoint.validation import (ScreeningYield, excluded_share, fit_scalar_yield,
                                   fit_screening_yield, model_hourly, model_hourly_with,
                                   observed_hourly, score, score_profile)

ROOT = Path(__file__).resolve().parents[1]
DEP = ROOT / "data" / "sample" / "bna_2026-06-10_departures.csv"
TSA = ROOT / "data" / "sample" / "bna_2026-06-10_tsa_throughput.csv"
DAY = [pd.Timestamp("2026-06-10")]


@pytest.fixture(scope="module")
def sample():
    for p in (DEP, TSA):
        if not p.exists():
            pytest.fail(f"committed sample missing: {p}")
    return read_sample(DEP), pd.read_csv(TSA, parse_dates=["date"])


def test_observed_hourly_covers_the_day(sample, cfg):
    _, tsa = sample
    oh = observed_hourly(tsa, "BNA", cfg.excluded_checkpoints("BNA"))
    assert set(oh["hour"]) >= set(range(5, 22))
    assert oh["throughput"].sum() > 10_000


def test_exclusions_are_applied(sample):
    _, tsa = sample
    everything = observed_hourly(tsa, "BNA")["throughput"].sum()
    minus = observed_hourly(tsa, "BNA", ["BNA Central"])["throughput"].sum()
    assert everything > 0
    assert minus == 0
    assert excluded_share(tsa, "BNA", ["BNA Central"]) == pytest.approx(1.0)
    assert excluded_share(tsa, "BNA", None) == 0.0


def test_scalar_yield_reproduces_observed_volume(sample, cfg):
    dep, tsa = sample
    model = DemandModel(cfg)
    mh = model_hourly(model, dep, "BNA", connecting_share=0.0)
    oh = observed_hourly(tsa, "BNA", cfg.excluded_checkpoints("BNA"))
    fit = fit_scalar_yield(mh, oh, DAY)
    rescaled = mh[mh["date"].isin(DAY)]["arrivals"].sum() * fit["scalar_yield"]
    assert rescaled == pytest.approx(fit["observed_pax"], rel=1e-9)
    assert 0.0 < fit["scalar_yield"] < 1.5


def test_screening_yield_is_a_line_in_departure_hour():
    y = ScreeningYield(0.8, -0.36)
    assert y(12) == pytest.approx(0.8)
    assert y(0) == pytest.approx(0.8 + 0.36)
    assert y(24) == pytest.approx(0.8 - 0.36)
    assert not y.clips()
    assert ScreeningYield(0.2, -2.0).clips()
    assert float(ScreeningYield(0.2, -2.0)(23)) == pytest.approx(ScreeningYield.LOW)


def test_hour_varying_yield_beats_the_flat_one(sample, cfg):
    """The extra parameter has to earn its place on the day it was fitted."""
    dep, tsa = sample
    model = DemandModel(cfg)
    oh = observed_hourly(tsa, "BNA", cfg.excluded_checkpoints("BNA"))
    mh = model_hourly(model, dep, "BNA", connecting_share=0.0)
    flat = fit_scalar_yield(mh, oh, DAY)
    s_flat = score(mh, oh, flat["scalar_yield"], DAY)
    yld = fit_screening_yield(model, dep, "BNA", oh, DAY)
    s_prof = score_profile(model, dep, "BNA", oh, yld, DAY)
    assert s_prof["hourly_rmse"] < s_flat["hourly_rmse"]


def test_fitted_yield_stays_in_a_believable_range(sample, cfg):
    dep, tsa = sample
    model = DemandModel(cfg)
    oh = observed_hourly(tsa, "BNA", cfg.excluded_checkpoints("BNA"))
    yld = fit_screening_yield(model, dep, "BNA", oh, DAY)
    shares = yld(np.arange(4, 23))
    assert (shares > 0.2).all() and (shares < 1.3).all()


def test_flat_yield_fits_the_daily_total(sample, cfg):
    dep, tsa = sample
    model = DemandModel(cfg)
    mh = model_hourly(model, dep, "BNA", connecting_share=0.0)
    oh = observed_hourly(tsa, "BNA", cfg.excluded_checkpoints("BNA"))
    s = score(mh, oh, fit_scalar_yield(mh, oh, DAY)["scalar_yield"], DAY)
    assert s["level_ratio"] == pytest.approx(1.0, abs=0.02), (
        "the flat yield is fitted on total volume, so the level ratio should be one "
        "up to the hours the inner join drops")
    assert s["hourly_corr"] > 0.75


def test_the_hour_varying_yield_moves_the_peak_to_the_right_place(sample, cfg):
    """The flat yield puts the busiest hour in the wrong half of the day. The line does not.

    This is the single clearest reason the extra parameter is there: staffing is decided
    by where the peak is, and a constant yield puts it in the afternoon when the airport
    actually peaks in the morning.
    """
    dep, tsa = sample
    model = DemandModel(cfg)
    oh = observed_hourly(tsa, "BNA", cfg.excluded_checkpoints("BNA"))
    mh = model_hourly(model, dep, "BNA", connecting_share=0.0)
    flat = score(mh, oh, fit_scalar_yield(mh, oh, DAY)["scalar_yield"], DAY)
    yld = fit_screening_yield(model, dep, "BNA", oh, DAY)
    prof = score_profile(model, dep, "BNA", oh, yld, DAY)
    flat_gap = abs(flat["peak_hour_model"] - flat["peak_hour_observed"])
    prof_gap = abs(prof["peak_hour_model"] - prof["peak_hour_observed"])
    assert prof_gap < flat_gap
    assert prof_gap <= 2
