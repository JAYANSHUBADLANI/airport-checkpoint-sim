"""Minimum cost staffing subject to a wait target.

The wait is a simulation output, so it cannot go into a linear program directly. I use
a two layer approach. The inner layer is a queueing approximation that turns an arrival
rate into a required lane count per slot. The outer layer is a set covering MILP that
buys shifts to cover those requirements at least cost. The plan that comes out is then
run through the simulation. Where the simulation says the target is missed, the
requirement for that slot is tightened by one lane and the MILP is re-solved. The gap
between what the MILP promised and what the simulation delivered is reported, not
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import LinearConstraint, milp

from .analytic import required_lanes
from .config import Config
from .metrics import run_replications
from .scenario import DayScenario
from .staffing import StaffingPlan


def shift_matrix(n_slots: int, shift_slots: int, stride: int) -> tuple[np.ndarray, list[int]]:
    """Coverage matrix A[s, j] = 1 when shift j is on duty in slot s."""
    starts = list(range(0, max(1, n_slots - shift_slots + 1), stride))
    last = n_slots - shift_slots
    if last >= 0 and last not in starts:
        starts.append(last)
    starts = sorted(set(s for s in starts if s >= 0))
    A = np.zeros((n_slots, len(starts)), dtype=float)
    for j, s0 in enumerate(starts):
        A[s0:min(n_slots, s0 + shift_slots), j] = 1.0
    return A, starts


def solve_cover(required: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, dict]:
    """Buy the cheapest set of shifts that covers the required lanes in every slot.

    Every shift is the same length, so cost is proportional to the number of shifts and
    the problem has many optima that cost exactly the same while stacking their spare
    capacity in different places. Some of those look ridiculous on a roster: a plan can
    park fifteen unneeded lanes in one afternoon half hour for free. A small secondary
    term on the busiest half hour breaks the tie towards the flattest of the optimal
    plans, and it is scaled so it can never outweigh buying one fewer shift, which keeps
    the primary objective intact. The peak also has a physical meaning: an airport cannot
    open more lanes than it has built.
    """
    st = cfg["staffing"]
    n_slots = len(required)
    shift_slots = int(st["min_shift_slots"])
    stride = int(st["shift_start_stride_slots"])
    max_lanes = int(st["max_lanes"])
    A, starts = shift_matrix(n_slots, shift_slots, stride)
    n_shifts = A.shape[1]

    shift_hours = shift_slots * int(st["slot_minutes"]) / 60.0
    shift_cost = shift_hours * float(st["cost_per_lane_hour"])
    epsilon = 0.4 * shift_cost / max(1, max_lanes)
    c = np.concatenate([np.full(n_shifts, shift_cost), [epsilon]])

    cover = np.hstack([A, np.zeros((n_slots, 1))])
    peak = np.hstack([A, -np.ones((n_slots, 1))])
    cons = [
        LinearConstraint(cover, lb=np.asarray(required, dtype=float),
                         ub=np.full(n_slots, float(max_lanes))),
        LinearConstraint(peak, lb=np.full(n_slots, -np.inf), ub=np.zeros(n_slots)),
    ]
    res = milp(c=c, constraints=cons, integrality=np.ones(n_shifts + 1),
               bounds=(0, max_lanes))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    x = np.round(res.x[:n_shifts]).astype(int)
    lanes = A @ x
    return lanes.astype(int), x, {"starts": starts, "status": res.message,
                                  "milp_cost": float(shift_cost * x.sum()),
                                  "shift_hours": shift_hours,
                                  "peak_lanes": int(lanes.max())}


def requirement_from_rates(rates_per_hour: np.ndarray, cfg: Config,
                           margin: float = 0.0, bump: np.ndarray | None = None,
                           target_minutes: float | None = None) -> np.ndarray:
    min_lanes = int(cfg["staffing"]["min_lanes"])
    req = np.array([required_lanes(r * (1.0 + margin), cfg, target_minutes)
                    for r in rates_per_hour], dtype=int)
    if bump is not None:
        req = req + bump.astype(int)
    return np.maximum(req, min_lanes)


@dataclass
class OptimiseResult:
    plan: StaffingPlan
    rounds: int
    history: list[dict] = field(default_factory=list)
    met_target: bool = False
    promised_p95: np.ndarray | None = None
    delivered_p95: np.ndarray | None = None
    required: np.ndarray | None = None


def optimise_staffing(cfg: Config, scenario: DayScenario, n_reps: int | None = None,
                      target_minutes: float | None = None, demand_factor: float = 1.0,
                      service_scale: float = 1.0, name: str = "optimised",
                      verbose: bool = False) -> OptimiseResult:
    st = cfg["staffing"]
    target = float(cfg["target"]["p95_wait_minutes"] if target_minutes is None
                   else target_minutes)
    rates = scenario.rate_per_hour_by_staffing_slot() * demand_factor
    n_slots = scenario.n_staffing_slots
    max_rounds = int(cfg["optimiser"]["max_rounds"])
    margin = float(cfg["optimiser"]["safety_margin"])

    bump = np.zeros(n_slots, dtype=int)
    history = []
    best = None

    for rnd in range(1, max_rounds + 1):
        required = requirement_from_rates(rates, cfg, margin=margin, bump=bump,
                                          target_minutes=target)
        lanes, shifts, info = solve_cover(required, cfg)
        plan = StaffingPlan(lanes, scenario.staffing_slot_minutes, scenario.start_minute,
                            f"{name}_r{rnd}")
        out = run_replications(cfg, scenario, plan, n_reps=n_reps,
                               demand_factor=demand_factor, service_scale=service_scale)
        by_slot = out["by_slot"]
        delivered = by_slot["p95_wait"].to_numpy()
        active = by_slot["n_pax"].to_numpy() > 0
        violations = np.where(active & (delivered > target))[0]
        record = {
            "round": rnd,
            "margin": margin,
            "required_lane_hours": float(required.sum() * scenario.staffing_slot_minutes / 60),
            "plan_lane_hours": plan.lane_hours(),
            "cost": plan.cost(float(st["cost_per_lane_hour"])),
            "n_violating_slots": int(len(violations)),
            "worst_slot_p95": float(delivered[active].max()) if active.any() else 0.0,
            "overall_p95": out["summary"]["p95_wait"],
            "overall_p95_mc": out["summary"]["p95_wait_mc_halfwidth"],
        }
        history.append(record)
        if verbose:
            print(f"  round {rnd}: cost {record['cost']:,.0f}, "
                  f"{len(violations)} slots over target, "
                  f"worst slot p95 {record['worst_slot_p95']:.1f} min")
        best = OptimiseResult(plan.rename(name), rnd, history, len(violations) == 0,
                              promised_p95=None, delivered_p95=delivered,
                              required=required)
        if len(violations) == 0:
            break
        bump[violations] += 1
        earlier = violations[violations > 0] - 1
        bump[earlier] = np.maximum(bump[earlier], bump[violations[violations > 0]] - 1)

    promised = np.array([
        _approx_p95(rates[s] * (1.0 + margin), best.required[s], cfg)
        for s in range(n_slots)
    ])
    best.promised_p95 = promised
    return best


def _approx_p95(rate_per_hour: float, lanes: int, cfg: Config) -> float:
    from .analytic import doc_rate_per_hour, effective_lane_rate_per_hour, mmc_wait_quantile
    import math
    mu_lane = effective_lane_rate_per_hour(cfg) / 60.0
    mu_doc = doc_rate_per_hour(cfg) / 60.0
    lam = rate_per_hour / 60.0
    ppl = float(cfg["service"]["doc_check"]["positions_per_lane"])
    dmin = int(cfg["service"]["doc_check"]["min_positions"])
    d = max(dmin, math.ceil(lanes * ppl))
    return (mmc_wait_quantile(0.95, lam, mu_doc, d)
            + mmc_wait_quantile(0.95, lam, mu_lane, lanes))


def smallest_feasible_flat(cfg: Config, scenario: DayScenario, n_reps: int | None = None,
                           target_minutes: float | None = None,
                           name: str = "flat_feasible") -> StaffingPlan:
    """Bisection on a constant lane count until the simulated p95 meets the target.

    The search starts from the queueing approximation at the busiest slot rather than
    from one lane, because a flat plan has to survive the peak and simulating obviously
    hopeless lane counts is wasted time.
    """
    target = float(cfg["target"]["p95_wait_minutes"] if target_minutes is None
                   else target_minutes)
    peak_rate = float(scenario.rate_per_hour_by_staffing_slot().max())
    lo = max(int(cfg["staffing"]["min_lanes"]),
             required_lanes(peak_rate, cfg, target) - 2)
    hi = int(cfg["staffing"]["max_lanes"])
    from .staffing import flat_plan
    feasible = None
    while lo <= hi:
        mid = (lo + hi) // 2
        plan = flat_plan(scenario.n_staffing_slots, mid, scenario.staffing_slot_minutes,
                         scenario.start_minute, f"{name}_{mid}")
        out = run_replications(cfg, scenario, plan, n_reps=n_reps)
        by_slot = out["by_slot"]
        active = by_slot["n_pax"].to_numpy() > 0
        worst = by_slot["p95_wait"].to_numpy()[active].max() if active.any() else 0.0
        if worst <= target:
            feasible = mid
            hi = mid - 1
        else:
            lo = mid + 1
    if feasible is None:
        feasible = int(cfg["staffing"]["max_lanes"])
    return flat_plan(scenario.n_staffing_slots, feasible, scenario.staffing_slot_minutes,
                     scenario.start_minute, name)
