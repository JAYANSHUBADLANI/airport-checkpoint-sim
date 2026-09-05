"""Verification: does the simulation reproduce results I can derive independently.

These tests say nothing about whether the model resembles a real airport. They only
check that the engine computes what a queue is supposed to compute. Validation against
observed throughput is a separate exercise, in test_validation.py and the report.
"""

import numpy as np
import pytest

from checkpoint.analytic import erlang_c, mmc_mean_wait, mmc_wait_quantile
from checkpoint.reference import (SequenceSpec, ZeroSpec, lindley_waits,
                                  poisson_arrivals)
from checkpoint.sim import CheckpointSim, ServiceSpec
from checkpoint.staffing import flat_plan


def _single_stage_cfg(cfg):
    return cfg.with_overrides(
        service__doc_check__min_positions=40,
        service__doc_check__positions_per_lane=0.0,
        service__doc_check__mean_seconds=0.6,
        service__doc_check__cv=0.0,
        service__secondary__share=0.0,
    )


def test_erlang_c_matches_textbook_values():
    assert erlang_c(1, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert erlang_c(2, 1.0) == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert erlang_c(3, 2.0) == pytest.approx(0.4444444444, abs=1e-9)
    assert erlang_c(5, 6.0) == pytest.approx(1.0, abs=1e-12)


def test_mmc_quantile_is_consistent_with_its_own_tail():
    lam, mu, c = 5.0, 0.6666666667, 10
    t = mmc_wait_quantile(0.95, lam, mu, c)
    from checkpoint.analytic import mmc_wait_tail
    assert mmc_wait_tail(t, lam, mu, c) == pytest.approx(0.05, rel=1e-9)


def test_deterministic_queue_matches_hand_calculation(cfg):
    """D/D/1: arrivals every minute, service 90 seconds, so the nth wait is 0.5n minutes."""
    c = _single_stage_cfg(cfg)
    n = 40
    start = 180.0
    times = start + np.arange(n, dtype=float)
    deps = times + 1e6
    plan = flat_plan(4, 1, 30, int(start), "det")
    rng = np.random.default_rng(0)
    sim = CheckpointSim(c, plan, np.zeros(24), 5, int(start), rng,
                        fixed_arrivals=(times, deps))
    sim.doc_spec = ServiceSpec(0.6, 0.0, "deterministic")
    sim.lane_spec = ServiceSpec(90.0, 0.0, "deterministic")
    sim.secondary_share = 0.0
    df = sim.run().passengers.sort_values("idx")

    assert len(df) == n
    expected = 0.5 * np.arange(n)
    assert np.allclose(df["lane_wait"].to_numpy(), expected, atol=1e-9)
    assert df["doc_wait"].to_numpy() == pytest.approx(np.zeros(n), abs=1e-9)
    assert df["lane_end"].iloc[-1] == pytest.approx(start + 0.01 + 1.5 * n, abs=1e-9)


def test_deterministic_queue_stays_empty_when_capacity_exceeds_demand(cfg):
    c = _single_stage_cfg(cfg)
    n = 30
    start = 180.0
    times = start + 2.0 * np.arange(n, dtype=float)
    plan = flat_plan(4, 1, 30, int(start), "det2")
    sim = CheckpointSim(c, plan, np.zeros(24), 5, int(start),
                        np.random.default_rng(0),
                        fixed_arrivals=(times, times + 1e6))
    sim.doc_spec = ServiceSpec(0.6, 0.0, "deterministic")
    sim.lane_spec = ServiceSpec(90.0, 0.0, "deterministic")
    sim.secondary_share = 0.0
    df = sim.run().passengers
    assert df["total_wait"].max() == pytest.approx(0.0, abs=1e-9)


def _stationary_runs(cfg, lanes, rate_per_hour, days, reps, seed0, stat):
    """Replications of a stationary single stage checkpoint, one statistic per replication."""
    slot_minutes = 5
    minutes = days * 1440
    n_slots = minutes // slot_minutes
    rate = np.full(n_slots, rate_per_hour * slot_minutes / 60.0)
    plan = flat_plan(minutes // 30, lanes, 30, 0, f"mmc{lanes}")
    values = []
    for rep in range(reps):
        sim = CheckpointSim(cfg, plan, rate, slot_minutes, 0,
                            np.random.default_rng(seed0 + rep))
        sim.doc_spec = ServiceSpec(0.6, 0.0, "deterministic")
        sim.lane_spec = ServiceSpec(90.0, 1.0, "exponential")
        sim.secondary_share = 0.0
        df = sim.run().passengers
        df = df[df["arrive"] >= 60.0]
        values.append(stat(df["lane_wait"].to_numpy()))
    v = np.asarray(values, dtype=float)
    halfwidth = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(halfwidth)


def test_engine_matches_an_independent_queue_simulator(cfg):
    """Same arrivals, same service draws, two unrelated engines, identical waits.

    This is the sharpest check available. It removes sampling noise entirely, so any
    difference would be a defect in the event handling rather than randomness.
    """
    c = cfg.with_overrides(
        service__doc_check__min_positions=1,
        service__doc_check__positions_per_lane=0.0,
        service__doc_check__mean_seconds=0.0,
        service__secondary__share=0.0,
    )
    lanes, minutes = 6, 2 * 1440
    rng = np.random.default_rng(42)
    arrivals = poisson_arrivals(rng, minutes, 180.0)
    services = rng.exponential(1.5, size=len(arrivals))
    expected = lindley_waits(arrivals, services, lanes)

    plan = flat_plan(minutes // 30, lanes, 30, 0, "cross")
    sim = CheckpointSim(c, plan, np.zeros(minutes // 5), 5, 0, np.random.default_rng(0),
                        fixed_arrivals=(arrivals, arrivals + 1e7))
    sim.doc_spec = ZeroSpec()
    sim.lane_spec = SequenceSpec(services)
    sim.secondary_share = 0.0
    got = sim.run().passengers.sort_values("idx")["total_wait"].to_numpy()

    assert len(got) == len(arrivals) > 5000
    assert np.abs(got - expected).max() < 1e-9


def test_simulation_matches_mmc_mean_wait(cfg):
    """With Poisson arrivals and exponential service the mean wait must match Erlang C.

    This is a statistical comparison, so it needs enough simulated time to be worth
    making. A single simulated day of a checkpoint this size carries a standard
    deviation on the mean wait of around twenty percent, so the test runs long
    stationary replications and asks whether the analytic value sits inside the Monte
    Carlo interval. The first hour of each replication is dropped because the analytic
    result is stationary and the simulation starts empty.
    """
    c = _single_stage_cfg(cfg)
    lanes, rate_per_hour = 6, 180.0
    analytic = mmc_mean_wait(rate_per_hour / 60.0, 1.0 / 1.5, lanes)
    sim_mean, halfwidth = _stationary_runs(c, lanes, rate_per_hour, days=10, reps=12,
                                           seed0=1000, stat=np.mean)
    assert abs(sim_mean - analytic) <= max(0.05 * analytic, 2 * halfwidth), (
        f"simulated {sim_mean:.4f} vs analytic {analytic:.4f}, "
        f"Monte Carlo halfwidth {halfwidth:.4f}")


def test_simulation_matches_mmc_wait_quantile(cfg):
    c = _single_stage_cfg(cfg)
    lanes, rate_per_hour = 6, 180.0
    analytic = mmc_wait_quantile(0.95, rate_per_hour / 60.0, 1.0 / 1.5, lanes)
    sim_q, halfwidth = _stationary_runs(c, lanes, rate_per_hour, days=10, reps=12,
                                        seed0=5000,
                                        stat=lambda w: float(np.percentile(w, 95)))
    assert abs(sim_q - analytic) <= max(0.05 * analytic, 2 * halfwidth), (
        f"simulated p95 {sim_q:.4f} vs analytic {analytic:.4f}, "
        f"Monte Carlo halfwidth {halfwidth:.4f}")


def test_lanes_close_on_the_staffing_boundary(cfg):
    """A lane that closes finishes the passenger in front of it and takes no new one."""
    c = _single_stage_cfg(cfg)
    start = 0
    from checkpoint.staffing import StaffingPlan
    plan = StaffingPlan(np.array([1, 0, 1]), 30, start, "closing")
    times = np.array([1.0, 2.0, 31.0, 32.0])
    sim = CheckpointSim(c, plan, np.zeros(12), 5, start, np.random.default_rng(0),
                        fixed_arrivals=(times, times + 1e6))
    sim.doc_spec = ServiceSpec(0.6, 0.0, "deterministic")
    sim.lane_spec = ServiceSpec(60.0, 0.0, "deterministic")
    sim.secondary_share = 0.0
    df = sim.run().passengers.sort_values("idx")
    assert (df["lane_start"] < 30.0).sum() == 2
    assert (df["lane_start"] >= 60.0).sum() == 2
