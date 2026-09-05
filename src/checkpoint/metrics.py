"""Summarise simulation output and report Monte Carlo error.

Every reported percentile is the mean of that percentile across independent
replications, with the half width of a 95 percent confidence interval on that mean.
"""

from __future__ import annotations

import atexit
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import Config
from .scenario import DayScenario
from .sim import CheckpointSim
from .staffing import StaffingPlan

Z95 = 1.959963985
MAX_WORKERS = int(os.environ.get("CHECKPOINT_WORKERS", os.cpu_count() or 1))
_POOL = None


def _worker_count(n_reps: int) -> int:
    return max(1, min(MAX_WORKERS, n_reps))


def _stream_id(*parts) -> int:
    """A stable identifier for one airport, day and plan.

    Python salts the hash of a string differently in every interpreter, so using the
    built in hash here would make two runs of the same study disagree. This does not.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def _pool() -> ProcessPoolExecutor:
    """One pool for the life of the process.

    A study runs hundreds of replication batches. Building a pool per batch means paying
    to start and import into a handful of interpreters every time, which on this workload
    is a large share of the wall clock for no benefit.
    """
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(MAX_WORKERS)
        atexit.register(shutdown_pool)
    return _POOL


def shutdown_pool():
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None


def _stats_one(df: pd.DataFrame, cfg: Config) -> dict:
    if df.empty:
        return {"n_pax": 0, "mean_wait": 0.0, "p90_wait": 0.0, "p95_wait": 0.0,
                "max_wait": 0.0, "miss_share": 0.0, "miss_share_unavoidable": 0.0,
                "miss_share_from_queueing": 0.0, "mean_time_in_system": 0.0}
    w = df["total_wait"].to_numpy()
    return {
        "n_pax": int(len(df)),
        "mean_wait": float(w.mean()),
        "p90_wait": float(np.percentile(w, 90)),
        "p95_wait": float(np.percentile(w, 95)),
        "max_wait": float(w.max()),
        "miss_share": float((~df["cleared_in_time"]).mean()),
        "miss_share_unavoidable": float(df["would_miss_without_queueing"].mean()),
        "miss_share_from_queueing": float((~df["cleared_in_time"]).mean()
                                          - df["would_miss_without_queueing"].mean()),
        "mean_time_in_system": float(df["time_in_system"].mean()),
    }


def _by_hour(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["arrive_hour", "n_pax", "mean_wait", "p95_wait", "max_wait"])
    g = df.groupby("arrive_hour")["total_wait"]
    out = pd.DataFrame({
        "n_pax": g.size(),
        "mean_wait": g.mean(),
        "p95_wait": g.quantile(0.95),
        "max_wait": g.max(),
    }).reset_index()
    return out


def _by_staffing_slot(df: pd.DataFrame, start_minute: int, slot_minutes: int,
                      n_slots: int) -> pd.DataFrame:
    idx = np.arange(n_slots)
    if df.empty:
        return pd.DataFrame({"slot": idx, "n_pax": 0, "p95_wait": 0.0, "mean_wait": 0.0})
    s = ((df["arrive"] - start_minute) // slot_minutes).astype(int).clip(0, n_slots - 1)
    tmp = df.assign(slot=s)
    g = tmp.groupby("slot")["total_wait"]
    out = pd.DataFrame({"n_pax": g.size(), "mean_wait": g.mean(),
                        "p95_wait": g.quantile(0.95)}).reindex(idx).fillna(0.0)
    out.index.name = "slot"
    return out.reset_index()


def _run_one(args):
    """One replication. Module level so it can be shipped to a worker process."""
    (cfg, plan, rate, slot_minutes, start_minute, staffing_slot_minutes, n_staffing_slots,
     lag_weights, service_scale, seed_seq, rep, keep, service_override) = args
    rng = np.random.default_rng(seed_seq)
    sim = CheckpointSim(cfg, plan, rate, slot_minutes, start_minute, rng,
                        dep_lag_weights=lag_weights, service_scale=service_scale,
                        service_override=service_override)
    res = sim.run()
    df = res.passengers
    stats = _stats_one(df, cfg)
    stats["n_unserved"] = res.meta["n_unserved"]
    h = _by_hour(df)
    h["rep"] = rep
    s = _by_staffing_slot(df, start_minute, staffing_slot_minutes, n_staffing_slots)
    s["rep"] = rep
    return stats, h, s, (df if keep else None)


def run_replications(cfg: Config, scenario: DayScenario, plan: StaffingPlan,
                     n_reps: int | None = None, seed: int | None = None,
                     service_scale: float = 1.0, demand_factor: float = 1.0,
                     keep_passengers: bool = False, service_override: dict | None = None):
    """Run independent replications of one airport-day under one staffing plan."""
    n_reps = int(n_reps if n_reps is not None else cfg["run"]["replications"])
    seed = int(seed if seed is not None else cfg.seed)
    ss = np.random.SeedSequence([seed, _stream_id(scenario.airport, scenario.day,
                                                  plan.name)])
    children = ss.spawn(n_reps)
    rate = scenario.arrivals_per_slot * demand_factor

    jobs = [(cfg, plan, rate, scenario.demand_slot_minutes, scenario.start_minute,
             scenario.staffing_slot_minutes, scenario.n_staffing_slots,
             scenario.dep_lag_weights, service_scale,
             child, r, keep_passengers, service_override)
            for r, child in enumerate(children)]

    if _worker_count(n_reps) > 1:
        results = list(_pool().map(_run_one, jobs, chunksize=1))
    else:
        results = [_run_one(j) for j in jobs]

    per_rep = [r[0] for r in results]
    hour_frames = [r[1] for r in results]
    slot_frames = [r[2] for r in results]
    kept = [r[3] for r in results if r[3] is not None]

    reps = pd.DataFrame(per_rep)
    summary = {}
    for col in ["n_pax", "mean_wait", "p90_wait", "p95_wait", "max_wait", "miss_share",
                "miss_share_unavoidable", "miss_share_from_queueing",
                "mean_time_in_system", "n_unserved"]:
        v = reps[col].to_numpy(dtype=float)
        summary[col] = float(v.mean())
        summary[f"{col}_mc_halfwidth"] = float(Z95 * v.std(ddof=1) / np.sqrt(len(v))) \
            if len(v) > 1 else 0.0
    summary["n_reps"] = n_reps
    summary["seed"] = seed
    summary["plan"] = plan.name
    summary["airport"] = scenario.airport
    summary["date"] = scenario.day
    summary["lane_hours"] = plan.lane_hours()
    summary["cost"] = plan.cost(float(cfg["staffing"]["cost_per_lane_hour"]))

    hours = pd.concat(hour_frames, ignore_index=True)
    by_hour = hours.groupby("arrive_hour").agg(
        n_pax=("n_pax", "mean"),
        mean_wait=("mean_wait", "mean"),
        p95_wait=("p95_wait", "mean"),
        p95_wait_mc_halfwidth=("p95_wait", lambda x: Z95 * x.std(ddof=1) / np.sqrt(len(x))
                               if len(x) > 1 else 0.0),
        max_wait=("max_wait", "mean"),
    ).reset_index()

    slots = pd.concat(slot_frames, ignore_index=True)
    by_slot = slots.groupby("slot").agg(
        n_pax=("n_pax", "mean"),
        mean_wait=("mean_wait", "mean"),
        p95_wait=("p95_wait", "mean"),
        p95_wait_mc_halfwidth=("p95_wait", lambda x: Z95 * x.std(ddof=1) / np.sqrt(len(x))
                               if len(x) > 1 else 0.0),
    ).reset_index()

    return {"summary": summary, "by_hour": by_hour, "by_slot": by_slot,
            "reps": reps, "passengers": kept}
