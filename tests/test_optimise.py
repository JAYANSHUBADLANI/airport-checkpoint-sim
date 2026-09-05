"""The optimiser: shift structure, covering, and the loop that checks itself."""

import numpy as np
import pytest

from checkpoint.analytic import required_lanes
from checkpoint.optimise import requirement_from_rates, shift_matrix, solve_cover
from checkpoint.staffing import StaffingPlan, flat_plan, rule_of_thumb_plan


def test_shift_matrix_covers_every_slot():
    A, starts = shift_matrix(46, 8, 2)
    assert A.shape[0] == 46
    assert (A.sum(axis=1) > 0).all(), "some slot cannot be covered by any shift"
    assert all(A[:, j].sum() == 8 for j in range(A.shape[1]))


def test_shift_matrix_blocks_are_contiguous():
    A, starts = shift_matrix(20, 6, 2)
    for j in range(A.shape[1]):
        on = np.flatnonzero(A[:, j])
        assert (np.diff(on) == 1).all()


def test_cover_meets_the_requirement_everywhere(cfg):
    required = np.array([1] * 6 + [4] * 8 + [12] * 10 + [6] * 12 + [2] * 10)
    lanes, x, info = solve_cover(required, cfg)
    assert (lanes >= required).all()
    assert lanes.max() <= cfg["staffing"]["max_lanes"]


def test_cover_of_a_flat_requirement_is_flat(cfg):
    required = np.full(24, 5)
    lanes, x, info = solve_cover(required, cfg)
    assert (lanes >= 5).all()
    assert lanes.sum() <= 24 * 5 * 1.5


def test_cover_cost_rises_with_the_requirement(cfg):
    lo, _, _ = solve_cover(np.full(24, 4), cfg)
    hi, _, _ = solve_cover(np.full(24, 9), cfg)
    assert hi.sum() > lo.sum()


def test_requirement_rises_with_the_arrival_rate(cfg):
    rates = np.array([0.0, 200.0, 800.0, 2000.0, 4000.0])
    req = requirement_from_rates(rates, cfg)
    assert (np.diff(req) >= 0).all()
    assert req[0] == cfg["staffing"]["min_lanes"]


def test_safety_margin_never_reduces_the_requirement(cfg):
    rates = np.array([300.0, 1500.0, 3000.0])
    base = requirement_from_rates(rates, cfg, margin=0.0)
    tight = requirement_from_rates(rates, cfg, margin=0.15)
    assert (tight >= base).all()


@pytest.mark.parametrize("rate", [300.0, 1200.0, 2000.0, 4000.0, 6500.0])
def test_required_lanes_is_always_at_least_bare_capacity(cfg, rate):
    """Sizing for a wait target can never buy fewer lanes than throughput alone needs."""
    from checkpoint.analytic import effective_lane_rate_per_hour
    lane_rate = effective_lane_rate_per_hour(cfg)
    lanes = required_lanes(rate, cfg)
    assert lanes >= int(np.ceil(rate / lane_rate))
    assert rate / (lanes * lane_rate) < 1.0, "requirement leaves an unstable queue"


def test_required_lanes_exceeds_capacity_when_capacity_is_only_just_enough(cfg):
    """At a rate that fills every lane the wait target has to buy at least one more.

    This is the only regime where the queueing layer changes the answer. Below it the
    rounding up of bare capacity already leaves enough slack for a twenty minute
    ninety fifth percentile, which is itself a finding: at the scale of a large
    checkpoint the target is loose relative to capacity, and what actually breaks the
    target in the simulation is the shape of demand inside the slot, not the average.
    """
    from checkpoint.analytic import effective_lane_rate_per_hour
    lane_rate = effective_lane_rate_per_hour(cfg)
    rate = 10 * lane_rate
    assert required_lanes(rate, cfg) > 10


def test_rule_of_thumb_ignores_the_wait_target(cfg):
    from checkpoint.analytic import effective_lane_rate_per_hour
    arrivals = np.array([50.0, 200.0, 900.0])
    plan = rule_of_thumb_plan(arrivals, effective_lane_rate_per_hour(cfg), 30, 180)
    per_hour = arrivals * 2.0
    expected = np.ceil(per_hour / effective_lane_rate_per_hour(cfg)).astype(int)
    assert (plan.lanes == np.maximum(expected, 1)).all()


def test_plan_cost_and_lane_hours():
    plan = flat_plan(48, 10, 30, 180, "t")
    assert plan.lane_hours() == pytest.approx(48 * 10 * 0.5)
    assert plan.cost(160.0) == pytest.approx(48 * 10 * 0.5 * 160.0)


def test_outage_reduces_lanes_only_in_the_window():
    plan = flat_plan(10, 6, 30, 0, "t")
    hit = plan.with_outage(60, 120, 2)
    assert (hit.lanes[:2] == 6).all()
    assert (hit.lanes[2:6] == 4).all()
    assert (hit.lanes[6:] == 6).all()


def test_lanes_at_clamps_outside_the_plan():
    plan = StaffingPlan(np.array([3, 7]), 30, 180, "t")
    assert plan.lanes_at(0) == 3
    assert plan.lanes_at(180) == 3
    assert plan.lanes_at(215) == 7
    assert plan.lanes_at(9999) == 7
