"""The demand model: does it conserve passengers and place them where it should."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from checkpoint.demand import DemandModel, profile_stats
from checkpoint.schedule import read_sample
from checkpoint.showup import ShowupProfile

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "bna_2026-06-10_departures.csv"


@pytest.fixture(scope="module")
def sample():
    if not SAMPLE.exists():
        pytest.fail(f"committed sample missing: {SAMPLE}")
    return read_sample(SAMPLE)


def test_showup_pmf_is_a_distribution(cfg):
    p = ShowupProfile.from_config(cfg)
    pmf = p.slot_pmf(cfg["demand"]["slot_minutes"])
    assert pmf.sum() == pytest.approx(1.0)
    assert (pmf >= 0).all()
    assert len(pmf) == int(p.span // cfg["demand"]["slot_minutes"])


def test_showup_mode_is_inside_the_window(cfg):
    p = ShowupProfile.from_config(cfg)
    assert p.earliest < p.mode_minutes_before() < p.latest
    assert p.earliest < p.mean_minutes_before() < p.latest


def test_showup_shift_moves_the_whole_window(cfg):
    base = ShowupProfile.from_config(cfg)
    later = ShowupProfile.from_config(cfg, shift_minutes=15)
    assert later.mode_minutes_before() == pytest.approx(base.mode_minutes_before() + 15)
    assert np.allclose(base.slot_pmf(5), later.slot_pmf(5))


def test_arrivals_conserve_passengers(cfg, sample):
    model = DemandModel(cfg)
    arr = model.arrivals(sample, "BNA", connecting_share=0.2)
    pax = model.originating_passengers(sample, "BNA", connecting_share=0.2)
    assert arr["arrivals"].sum() == pytest.approx(pax["pax_originating"].sum(), rel=1e-9)


def test_connecting_share_scales_arrivals_linearly(cfg, sample):
    model = DemandModel(cfg)
    a0 = model.arrivals(sample, "BNA", connecting_share=0.0)["arrivals"].sum()
    a5 = model.arrivals(sample, "BNA", connecting_share=0.5)["arrivals"].sum()
    assert a5 == pytest.approx(0.5 * a0, rel=1e-9)


def test_arrivals_lead_departures_by_the_show_up_mean(cfg, sample):
    """The passenger weighted centre of arrivals must sit one show-up mean earlier.

    Comparing centres rather than peaks, because the arrival curve is a smoothed
    version of the departure curve and their peaks need not line up.
    """
    model = DemandModel(cfg)
    arr = model.arrivals(sample, "BNA", connecting_share=0.2)
    pax = model.originating_passengers(sample, "BNA", connecting_share=0.2)
    origin = arr["slot_start"].iloc[0]
    dep_min = (pax["sched_dep"] - origin).dt.total_seconds().to_numpy() / 60.0
    arr_min = (arr["slot_start"] - origin).dt.total_seconds().to_numpy() / 60.0
    dep_centre = np.average(dep_min, weights=pax["pax_originating"].to_numpy())
    arr_centre = np.average(arr_min, weights=arr["arrivals"].to_numpy())
    lead = dep_centre - arr_centre
    edges = model.showup.earliest + model.slot_minutes * np.arange(len(model.pmf))
    within_slot = np.average(dep_min % model.slot_minutes,
                             weights=pax["pax_originating"].to_numpy())
    expected = float((model.pmf * edges).sum()) + within_slot
    assert lead == pytest.approx(expected, abs=1.0), (
        f"arrivals lead departures by {lead:.1f} min, the show-up profile says "
        f"{expected:.1f}")


def test_a_single_flight_lands_in_the_right_window(cfg):
    """One flight at noon puts every passenger inside the show-up window before noon."""
    dep = pd.DataFrame({
        "airport": ["BNA"], "flight_date": [pd.Timestamp("2026-06-10")],
        "Dest": ["ATL"], "Reporting_Airline": ["WN"],
        "Flight_Number_Reporting_Airline": [1],
        "sched_dep": [pd.Timestamp("2026-06-10 12:00")], "cancelled": [False],
    })
    model = DemandModel(cfg)
    arr = model.arrivals(dep, "BNA", connecting_share=0.0)
    nz = arr[arr["arrivals"] > 1e-12]
    lead = (pd.Timestamp("2026-06-10 12:00") - nz["slot_start"]).dt.total_seconds() / 60.0
    p = model.showup
    assert lead.min() >= p.earliest - 1e-9
    assert lead.max() <= p.latest + 1e-9
    assert nz["arrivals"].sum() == pytest.approx(168.0 * cfg["demand"]["load_factor"])


def test_profile_stats_are_self_consistent(cfg, sample):
    model = DemandModel(cfg)
    arr = model.arrivals(sample, "BNA", connecting_share=0.2)
    day = model.day_slice(arr, "2026-06-10")
    st = profile_stats(day, model.slot_minutes)
    assert st["peak_to_mean_hour"] > 1.0
    assert st["peak_slot_pax"] >= st["mean_slot_pax"]
    assert st["total_pax"] == pytest.approx(day["arrivals"].sum())
