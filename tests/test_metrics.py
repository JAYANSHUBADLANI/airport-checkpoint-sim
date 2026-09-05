"""Reproducibility and the Monte Carlo bookkeeping."""

import numpy as np
import pandas as pd
import pytest

from checkpoint.metrics import _stream_id, run_replications
from checkpoint.scenario import DayScenario
from checkpoint.staffing import flat_plan


def _scenario(total_pax=9000.0):
    n = (26 - 3) * 12
    shape = np.exp(-0.5 * ((np.arange(n) - 130) / 45.0) ** 2)
    rate = shape / shape.sum() * total_pax
    return DayScenario("BNA", pd.Timestamp("2026-06-10"), rate,
                       np.full((n, 38), 1.0 / 38), 5, 30, 180,
                       pd.date_range("2026-06-10 03:00", periods=n, freq="5min"))


def test_stream_ids_do_not_depend_on_the_interpreter(cfg):
    """Pinned on purpose.

    An earlier version derived the random stream from the built in hash of a tuple of
    strings. Python salts that hash differently in every process, so two runs of the same
    study disagreed while every test inside one process passed. Hard coding the expected
    value here means that class of bug cannot come back quietly.
    """
    assert _stream_id("ATL", "2026-06-26", "optimised") == 121659984
    assert _stream_id("BNA", "2026-06-21", "flat_feasible") == 4268312647
    assert _stream_id("a", "b") != _stream_id("b", "a")


def test_two_runs_with_the_same_seed_are_identical(cfg):
    sc = _scenario()
    plan = flat_plan(sc.n_staffing_slots, 8, 30, 180, "t")
    a = run_replications(cfg, sc, plan, n_reps=4, seed=123)["summary"]
    b = run_replications(cfg, sc, plan, n_reps=4, seed=123)["summary"]
    for k in ["n_pax", "mean_wait", "p95_wait", "max_wait", "miss_share"]:
        assert a[k] == b[k], k


def test_a_different_seed_gives_a_different_draw(cfg):
    sc = _scenario()
    plan = flat_plan(sc.n_staffing_slots, 8, 30, 180, "t")
    a = run_replications(cfg, sc, plan, n_reps=4, seed=123)["summary"]
    b = run_replications(cfg, sc, plan, n_reps=4, seed=987)["summary"]
    assert a["p95_wait"] != b["p95_wait"]


def test_monte_carlo_halfwidth_matches_the_replication_spread(cfg):
    """The reported half width must be the one implied by the replications themselves."""
    sc = _scenario()
    plan = flat_plan(sc.n_staffing_slots, 8, 30, 180, "t")
    out = run_replications(cfg, sc, plan, n_reps=10, seed=5)
    v = out["reps"]["p95_wait"].to_numpy()
    expected = 1.959963985 * v.std(ddof=1) / np.sqrt(len(v))
    assert out["summary"]["p95_wait"] == pytest.approx(v.mean())
    assert out["summary"]["p95_wait_mc_halfwidth"] == pytest.approx(expected)


def test_one_replication_reports_no_spurious_precision(cfg):
    sc = _scenario()
    plan = flat_plan(sc.n_staffing_slots, 8, 30, 180, "t")
    out = run_replications(cfg, sc, plan, n_reps=1, seed=5)
    assert out["summary"]["p95_wait_mc_halfwidth"] == 0.0


def test_more_lanes_never_increase_the_wait(cfg):
    sc = _scenario()
    waits = []
    for lanes in (6, 9, 14):
        plan = flat_plan(sc.n_staffing_slots, lanes, 30, 180, f"t{lanes}")
        waits.append(run_replications(cfg, sc, plan, n_reps=6, seed=11)["summary"]["p95_wait"])
    assert waits[0] > waits[1] > waits[2]


def test_every_passenger_is_accounted_for(cfg):
    sc = _scenario()
    plan = flat_plan(sc.n_staffing_slots, 12, 30, 180, "t")
    out = run_replications(cfg, sc, plan, n_reps=4, seed=3)
    assert out["summary"]["n_unserved"] == 0
    assert out["summary"]["n_pax"] == pytest.approx(sc.arrivals_per_slot.sum(), rel=0.05)


def test_by_slot_and_by_hour_agree_on_volume(cfg):
    sc = _scenario()
    plan = flat_plan(sc.n_staffing_slots, 12, 30, 180, "t")
    out = run_replications(cfg, sc, plan, n_reps=4, seed=3)
    assert out["by_slot"]["n_pax"].sum() == pytest.approx(out["by_hour"]["n_pax"].sum(), rel=1e-6)
