"""End to end run: data, demand, validation, simulation, optimisation, robustness.

Every number that reaches the README or the memo is written to outputs/tables by this
module. Nothing is typed in by hand anywhere downstream.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .analytic import (effective_lane_rate_per_hour, mmc_mean_wait, mmc_wait_quantile,
                       required_lanes, required_lanes_curve)
from .config import Config, load_config
from .demand import DemandModel, profile_stats
from .metrics import Z95, run_replications
from .optimise import (optimise_staffing, requirement_from_rates, solve_cover,
                       smallest_feasible_flat)
from .paths import FIGURES, INTERIM, RAW, TABLES, ensure_dirs
from .reference import SequenceSpec, ZeroSpec, lindley_waits, poisson_arrivals
from .scenario import DayScenario, build_scenario
from .schedule import departures, load_ontime, write_sample
from .seats import SEATS_PER_DEPARTURE, coverage
from .showup import ShowupProfile
from .sim import CheckpointSim, ServiceSpec
from .staffing import StaffingPlan, flat_plan, rule_of_thumb_plan
from .tsa import load_weeks
from .validation import (ScreeningYield, excluded_share, fit_scalar_yield, model_hourly,
                         model_hourly_with, observed_hourly, score, score_profile,
                         select_showup_shift, showup_shift_diagnostic)


class Runner:
    def __init__(self, cfg: Config | None = None, quick: bool = False):
        self.cfg = cfg or load_config()
        self.quick = quick
        self.opt_reps = 4 if quick else 10
        self.eval_reps = 6 if quick else int(self.cfg["run"]["replications"])
        self.max_days = 2 if quick else None
        self.results: dict = {"quick": quick, "seed": self.cfg.seed}
        self.t0 = time.time()
        ensure_dirs()

    def log(self, msg: str):
        print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)

    def save(self, df: pd.DataFrame, name: str) -> Path:
        path = TABLES / f"{name}.csv"
        df.to_csv(path, index=False)
        self.log(f"wrote {path.relative_to(TABLES.parents[1])} ({len(df)} rows)")
        return path

    def run(self):
        self.load_data()
        self.build_demand()
        self.validate()
        self.verify()
        self.select_days()
        self.demand_profiles()
        self.fixed_plan_results()
        self.optimise_and_compare()
        self.replication_stability()
        self.shift_structure()
        self.target_sweep()
        self.checkpoint_split()
        self.robustness()
        self.sensitivity()
        self.write_results()
        return self.results

    def load_data(self):
        cfg = self.cfg
        year, month = int(cfg["run"]["schedule_year"]), int(cfg["run"]["schedule_month"])
        zip_path = RAW / f"ontime_{year}_{month}.zip"
        if not zip_path.exists():
            raise SystemExit(f"missing {zip_path}. Run scripts/fetch_data.py first.")
        self.codes = cfg.airport_codes
        self.log(f"loading {zip_path.name}")
        raw = load_ontime(zip_path, self.codes)
        self.dep = departures(raw)
        self.dep_all = departures(raw, include_cancelled=True)
        self.log(f"{len(raw):,} scheduled departures at {self.codes}, "
                 f"{len(self.dep):,} not cancelled")

        tsa_csv = INTERIM / "tsa_throughput.csv"
        pdfs = sorted(RAW.glob("tsa_throughput_*.pdf"))
        if not tsa_csv.exists() and not pdfs:
            raise SystemExit("no TSA throughput data. Run scripts/fetch_data.py first.")
        self.tsa = load_weeks(pdfs, cache=tsa_csv)
        self.log(f"TSA throughput: {len(self.tsa):,} airport-checkpoint-hours, "
                 f"{self.tsa.date.min().date()} to {self.tsa.date.max().date()}")

        write_sample(self.dep, "BNA", "2026-06-10",
                     Path(TABLES).parents[1] / "data" / "sample" / "bna_2026-06-10_departures.csv")
        sample_tsa = self.tsa[(self.tsa.airport == "BNA")
                              & (self.tsa.date == pd.Timestamp("2026-06-10"))]
        sample_tsa.to_csv(Path(TABLES).parents[1] / "data" / "sample"
                          / "bna_2026-06-10_tsa_throughput.csv", index=False)

        cov = coverage(self.dep)
        self.save(cov, "carrier_seat_coverage")
        self.results["carrier_default_share"] = float(
            cov.loc[cov.used_default, "flights"].sum() / cov["flights"].sum())

        counts = self.dep.groupby(["airport", "flight_date"]).size().rename("flights")
        cancelled = self.dep_all.groupby(["airport", "flight_date"])["cancelled"].mean()
        summary = pd.concat([counts, cancelled.rename("cancelled_share")], axis=1).reset_index()
        self.save(summary, "schedule_daily_counts")
        self.results["flights_loaded"] = int(len(self.dep))
        self.results["flights_including_cancelled"] = int(len(self.dep_all))

    def build_demand(self):
        """The prior show-up curve, before the calibration week moves it."""
        self.profile = ShowupProfile.from_config(self.cfg)
        self.model = DemandModel(self.cfg, showup=self.profile)
        plots.plot_showup_profile(self.profile, self.cfg["demand"]["slot_minutes"],
                                  FIGURES / "showup_profile.png")
        self.results["showup"] = {
            "family": self.cfg["demand"]["showup"]["family"],
            "earliest_minutes_before": self.profile.earliest,
            "latest_minutes_before": self.profile.latest,
            "alpha": self.profile.alpha, "beta": self.profile.beta,
            "mode_minutes_before": round(self.profile.mode_minutes_before(), 1),
            "mean_minutes_before": round(self.profile.mean_minutes_before(), 1),
        }
        self.results["effective_lane_rate_per_hour"] = round(
            effective_lane_rate_per_hour(self.cfg), 1)
        curve = required_lanes_curve(self.cfg, max_rate=8000, step=50)
        plots.plot_required_lanes_curve(curve, FIGURES / "required_lanes_curve.png",
                                        float(self.cfg["target"]["p95_wait_minutes"]))

    def validate(self):
        """Fit on one week, score on the next.

        Three parameters are fitted per airport and nowhere else: the intercept and
        slope of the screening yield, and one show-up timing shift. Everything reported
        as validation comes from the week that fitting never saw.
        """
        cfg = self.cfg
        cal_days = pd.date_range(cfg["days"]["calibration_start"],
                                 cfg["days"]["calibration_end"]).normalize().tolist()
        test_days = pd.date_range(cfg["days"]["anchor_week_start"],
                                  cfg["days"]["anchor_week_end"]).normalize().tolist()
        self.cal_days, self.test_days = cal_days, test_days
        self.models, self.yields = {}, {}
        rows, cal_rows = [], []

        for code in self.codes:
            excl = cfg.excluded_checkpoints(code)
            oh = observed_hourly(self.tsa, code, excl)
            shift, yld = select_showup_shift(cfg, self.dep, code, oh, cal_days)
            model = DemandModel(cfg, showup=ShowupProfile.from_config(cfg, shift_minutes=shift))
            self.models[code] = model
            self.yields[code] = yld

            flat_model = DemandModel(cfg)
            mh_flat = model_hourly(flat_model, self.dep, code, connecting_share=0.0)
            scalar = fit_scalar_yield(mh_flat, oh, cal_days)

            cal_rows.append({
                "airport": code, "showup_shift_minutes": shift,
                "yield_intercept": yld.intercept, "yield_slope": yld.slope,
                "yield_at_06": float(yld(6)), "yield_at_12": float(yld(12)),
                "yield_at_20": float(yld(20)), "yield_clipped": yld.clips(),
                "flat_yield": scalar["scalar_yield"],
                "prior_connecting_share": cfg.connecting_share(code),
                "modelled_gross_seat_pax": scalar["modelled_gross_pax"],
                "observed_pax": scalar["observed_pax"],
                "excluded_checkpoints": "; ".join(excl) or "none",
                "excluded_share_of_throughput": excluded_share(self.tsa, code, excl),
            })
            for label, days in [("calibration", cal_days), ("holdout", test_days)]:
                rows.append({"airport": code, "window": label, "model": "flat yield",
                             **score(mh_flat, oh, scalar["scalar_yield"], days)})
                rows.append({"airport": code, "window": label,
                             "model": "hour varying yield",
                             **score_profile(model, self.dep, code, oh, yld, days)})

            mh = model_hourly_with(model, self.dep, code, yld) \
                .rename(columns={"arrivals": "modelled"})
            j = mh.merge(oh, on=["date", "hour"])
            j = j[j["date"].isin(cal_days + test_days)] \
                .sort_values(["date", "hour"]).reset_index(drop=True)
            plots.plot_model_vs_observed(
                j, FIGURES / f"model_vs_observed_{code}.png",
                f"{code}: modelled checkpoint arrivals against TSA throughput. "
                f"Fitted on 1 to 6 June, held out from 7 June.")
            self.results.setdefault("fitted_shift", {})[code] = int(shift)
            self.log(f"{code}: shift {shift} min, yield {yld.intercept:.3f} "
                     f"{yld.slope:+.3f} per 12h")

        self.save(pd.DataFrame(cal_rows), "validation_calibration")
        self.save(pd.DataFrame(rows), "validation_scores")
        self.results["screening_yield"] = {c: self.yields[c].as_dict() for c in self.codes}

        diag = []
        for code in self.codes:
            oh = observed_hourly(self.tsa, code, cfg.excluded_checkpoints(code))
            d = showup_shift_diagnostic(cfg, self.dep, code, oh, cal_days)
            d["airport"] = code
            diag.append(d)
        self.save(pd.concat(diag, ignore_index=True), "showup_shift_diagnostic")

    def verify(self):
        """Two verification results, reported rather than only asserted in the tests.

        First, the SimPy engine is run against an independent Lindley recursion on the
        same arrivals and the same service draws, which removes randomness entirely.
        Second, the SimPy engine is run in the one regime where a closed form exists,
        Poisson arrivals and exponential service at a fixed number of servers, and
        compared with Erlang C. The second comparison is statistical, so the Monte Carlo
        half width is reported next to every gap.
        """
        reps, days = (4, 3) if self.quick else (10, 6)
        single_stage = self.cfg.with_overrides(
            service__doc_check__min_positions=1,
            service__doc_check__positions_per_lane=0.0,
            service__doc_check__mean_seconds=0.0,
        )

        rng = np.random.default_rng(4242)
        arrivals = poisson_arrivals(rng, 2 * 1440, 180.0)
        services = rng.exponential(1.5, size=len(arrivals))
        expected = lindley_waits(arrivals, services, 6)
        cross = CheckpointSim(single_stage, flat_plan(96, 6, 30, 0, "cross"),
                              np.zeros(576), 5, 0, np.random.default_rng(0),
                              fixed_arrivals=(arrivals, arrivals + 1e7),
                              service_override={"doc": ZeroSpec(),
                                                "lane": SequenceSpec(services),
                                                "secondary_share": 0.0})
        got = cross.run().passengers.sort_values("idx")["total_wait"].to_numpy()
        max_diff = float(np.abs(got - expected).max())
        self.results["cross_engine"] = {
            "passengers": int(len(expected)),
            "max_abs_difference_minutes": max_diff,
        }
        self.log(f"cross engine check on {len(expected):,} passengers: "
                 f"max difference {max_diff:.2e} minutes")

        rows = []
        mu = 1.0 / 1.5
        override = {"doc": ServiceSpec(0.0, 0.0, "deterministic"),
                    "lane": ServiceSpec(90.0, 1.0, "exponential"),
                    "secondary_share": 0.0}
        for lanes, rate in [(4, 120.0), (6, 180.0), (10, 300.0), (14, 460.0)]:
            n_slots = days * 1440 // 5
            sc = DayScenario("MMC", pd.Timestamp("2026-06-01"),
                             np.full(n_slots, rate * 5.0 / 60.0), None, 5, 30, 0,
                             pd.date_range("2026-06-01", periods=n_slots, freq="5min"))
            plan = flat_plan(sc.n_staffing_slots, lanes, 30, 0, f"mmc{lanes}")
            out = run_replications(single_stage, sc, plan, n_reps=reps, seed=777,
                                   service_override=override)
            r = out["reps"]
            rows.append({
                "lanes": lanes, "arrivals_per_hour": rate,
                "utilisation": rate / 60.0 / mu / lanes,
                "sim_mean_wait": r["mean_wait"].mean(),
                "sim_mean_wait_mc_halfwidth": 1.96 * r["mean_wait"].std(ddof=1)
                                              / np.sqrt(len(r)),
                "analytic_mean_wait": mmc_mean_wait(rate / 60.0, mu, lanes),
                "sim_p95_wait": r["p95_wait"].mean(),
                "sim_p95_wait_mc_halfwidth": 1.96 * r["p95_wait"].std(ddof=1)
                                             / np.sqrt(len(r)),
                "analytic_p95_wait": mmc_wait_quantile(0.95, rate / 60.0, mu, lanes),
            })
        df = pd.DataFrame(rows)
        df["mean_wait_error_pct"] = (df.sim_mean_wait / df.analytic_mean_wait - 1) * 100
        df["p95_wait_error_pct"] = (df.sim_p95_wait / df.analytic_p95_wait - 1) * 100
        df["mean_within_mc_interval"] = (df.sim_mean_wait - df.analytic_mean_wait).abs() \
            <= 2 * df.sim_mean_wait_mc_halfwidth
        self.save(df, "verification_mmc")
        self.results["verification_all_within_interval"] = bool(df.mean_within_mc_interval.all())

    def select_days(self):
        """The anchor week plus the busiest and quietest day of the month, per airport."""
        anchor = pd.date_range(self.cfg["days"]["anchor_week_start"],
                               self.cfg["days"]["anchor_week_end"]).normalize().tolist()
        self.days = {}
        rows = []
        for code in self.codes:
            arr = self.models[code].arrivals(self.dep, code, originating=self.yields[code])
            month = int(self.cfg["run"]["schedule_month"])
            daily = arr[arr["slot_start"].dt.month == month] \
                .groupby(arr["slot_start"].dt.normalize())["arrivals"].sum()
            busiest, quietest = daily.idxmax(), daily.idxmin()
            days = sorted(set(anchor) | {busiest, quietest})
            if self.max_days:
                days = sorted({busiest, quietest})
            self.days[code] = days
            for d in days:
                rows.append({"airport": code, "date": d, "modelled_pax": daily.get(d, np.nan),
                             "role": ("busiest" if d == busiest else
                                      "quietest" if d == quietest else "anchor_week")})
            self.log(f"{code}: {len(days)} days, busiest {busiest.date()} "
                     f"({daily.max():,.0f} pax), quietest {quietest.date()} "
                     f"({daily.min():,.0f} pax)")
            self.results.setdefault("design_day", {})[code] = str(busiest.date())
        self.save(pd.DataFrame(rows), "selected_days")

    def scenario(self, code: str, day) -> "object":
        key = (code, str(pd.Timestamp(day).date()))
        if not hasattr(self, "_scen"):
            self._scen = {}
        if key not in self._scen:
            self._scen[key] = build_scenario(
                self.cfg, self.models[code], self.dep, code, day,
                originating=self.yields[code])
        return self._scen[key]

    def demand_profiles(self):
        rows, curves = [], {}
        for code in self.codes:
            model = self.models[code]
            arr = model.arrivals(self.dep, code, originating=self.yields[code])
            per_airport = {}
            for day in self.days[code]:
                sub = model.day_slice(arr, day)
                st = profile_stats(sub, model.slot_minutes)
                rows.append({"airport": code, "date": day, **st})
                per_airport[str(pd.Timestamp(day).date())] = sub
            busiest = self.results["design_day"][code]
            quietest = min(rows[-len(self.days[code]):], key=lambda r: r["total_pax"])["date"]
            pick = {k: v for k, v in per_airport.items()
                    if k in {busiest, str(pd.Timestamp(quietest).date()),
                             "2026-06-10"}}
            curves[code] = pick
            plots.plot_arrival_curves(
                pick, FIGURES / f"arrival_curves_{code}.png",
                f"{code}: modelled checkpoint arrivals, selected days")
        self.save(pd.DataFrame(rows), "demand_profile")

    def fixed_plan_results(self):
        """Question two: what happens under a plan that ignores the shape of the day."""
        rows, hours = [], []
        rate_per_lane = effective_lane_rate_per_hour(self.cfg)
        for code in self.codes:
            day = pd.Timestamp(self.results["design_day"][code])
            sc = self.scenario(code, day)
            total = sc.arrivals_per_slot.sum()
            open_hours = sc.n_staffing_slots * sc.staffing_slot_minutes / 60.0
            lanes = int(np.ceil(total / (rate_per_lane * open_hours)))
            plan = flat_plan(sc.n_staffing_slots, lanes, sc.staffing_slot_minutes,
                             sc.start_minute, f"flat_average_{lanes}")
            out = run_replications(self.cfg, sc, plan, n_reps=self.eval_reps)
            rows.append({"airport": code, "date": day, "lanes": lanes,
                         **{k: v for k, v in out["summary"].items()
                            if k not in ("airport", "date")}})
            h = out["by_hour"].copy()
            h["clock_hour"] = h["arrive_hour"] % 24
            h["airport"] = code
            h["date"] = day
            h["lanes"] = lanes
            hours.append(h)
            self.log(f"{code} fixed plan of {lanes} lanes: p95 "
                     f"{out['summary']['p95_wait']:.1f} min, max "
                     f"{out['summary']['max_wait']:.1f} min")
        self.save(pd.DataFrame(rows), "fixed_plan_summary")
        self.save(pd.concat(hours, ignore_index=True), "fixed_plan_by_hour")

    def _plans_for(self, code: str, sc) -> tuple[dict, dict]:
        cfg = self.cfg
        opt = optimise_staffing(cfg, sc, n_reps=self.opt_reps, name="optimised")
        rot = rule_of_thumb_plan(sc.arrivals_per_staffing_slot(),
                                 effective_lane_rate_per_hour(cfg),
                                 sc.staffing_slot_minutes, sc.start_minute,
                                 min_lanes=int(cfg["staffing"]["min_lanes"]),
                                 max_lanes=int(cfg["staffing"]["max_lanes"]))
        peak = flat_plan(sc.n_staffing_slots, int(opt.plan.lanes.max()),
                         sc.staffing_slot_minutes, sc.start_minute, "peak_everywhere")
        flat = smallest_feasible_flat(cfg, sc, n_reps=self.opt_reps)
        return {"optimised": opt.plan, "rule_of_thumb": rot,
                "flat_feasible": flat, "peak_everywhere": peak}, {"optimised": opt}

    def optimise_and_compare(self):
        rows, rounds, slot_frames, promise_frames = [], [], [], []
        self.stability_reps = {}
        target = float(self.cfg["target"]["p95_wait_minutes"])
        self.opt_plans = {}
        for code in self.codes:
            for day in self.days[code]:
                sc = self.scenario(code, day)
                plans, extra = self._plans_for(code, sc)
                self.opt_plans[(code, str(pd.Timestamp(day).date()))] = plans
                opt = extra["optimised"]
                for r in opt.history:
                    rounds.append({"airport": code, "date": day, **r})
                start_h = int(self.cfg["run"]["sim_start_hour"])
                promise_frames.append(pd.DataFrame({
                    "airport": code, "date": day,
                    "slot": np.arange(sc.n_staffing_slots),
                    "clock_hour": (start_h
                                   + np.arange(sc.n_staffing_slots)
                                   * sc.staffing_slot_minutes / 60.0) % 24,
                    "arrivals_per_hour": sc.rate_per_hour_by_staffing_slot(),
                    "lanes": opt.plan.lanes,
                    "required_by_approximation": opt.required,
                    "promised_p95": opt.promised_p95,
                    "delivered_p95": opt.delivered_p95,
                }))
                by_slot = {}
                for label, plan in plans.items():
                    out = run_replications(self.cfg, sc, plan, n_reps=self.eval_reps)
                    s = out["summary"]
                    rows.append({"airport": code, "date": day, "plan": label,
                                 "lanes_peak": int(plan.lanes.max()),
                                 "lane_hours": plan.lane_hours(), "cost": s["cost"],
                                 "p95_wait": s["p95_wait"],
                                 "p95_wait_mc_halfwidth": s["p95_wait_mc_halfwidth"],
                                 "mean_wait": s["mean_wait"],
                                 "max_wait": s["max_wait"],
                                 "max_wait_mc_halfwidth": s["max_wait_mc_halfwidth"],
                                 "miss_share": s["miss_share"],
                                 "miss_share_from_queueing": s["miss_share_from_queueing"],
                                 "miss_share_mc_halfwidth": s["miss_share_mc_halfwidth"],
                                 "n_pax": s["n_pax"], "meets_target": s["p95_wait"] <= target})
                    by_slot[label] = out["by_slot"]
                    if (label == "optimised"
                            and str(pd.Timestamp(day).date())
                            == self.results["design_day"][code]):
                        self.stability_reps[code] = out["reps"]
                    sf = out["by_slot"].copy()
                    sf["airport"], sf["date"], sf["plan"] = code, day, label
                    slot_frames.append(sf)
                if str(pd.Timestamp(day).date()) == self.results["design_day"][code]:
                    plots.plot_plan_comparison(
                        plans, by_slot, target,
                        FIGURES / f"plan_comparison_{code}.png",
                        f"{code} on {pd.Timestamp(day).date()}: staffing plans and the "
                        f"waits they produce", int(self.cfg["run"]["sim_start_hour"]),
                        sc.staffing_slot_minutes)
                self.log(f"{code} {pd.Timestamp(day).date()} done "
                         f"({opt.rounds} optimiser rounds, "
                         f"target met: {opt.met_target})")
        self.save(pd.DataFrame(rows), "plan_comparison")
        self.save(pd.DataFrame(rounds), "optimiser_rounds")
        self.save(pd.concat(slot_frames, ignore_index=True), "plan_by_slot")
        self.save(pd.concat(promise_frames, ignore_index=True),
                  "optimiser_promise_vs_delivery")

    def replication_stability(self):
        """Evidence that the replication count is enough for the percentiles quoted.

        The same replications are reused, so this costs nothing beyond arithmetic: it
        reads off what the estimate and its interval would have been had the run stopped
        earlier.
        """
        rows = []
        for code, reps in self.stability_reps.items():
            v = reps["p95_wait"].to_numpy(dtype=float)
            for n in [5, 10, 15, 20, 25, 30]:
                if n > len(v):
                    continue
                sub = v[:n]
                rows.append({
                    "airport": code, "replications": n,
                    "p95_wait": float(sub.mean()),
                    "mc_halfwidth": float(Z95 * sub.std(ddof=1) / np.sqrt(n)),
                    "halfwidth_as_pct_of_estimate":
                        float(100 * Z95 * sub.std(ddof=1) / np.sqrt(n) / sub.mean()),
                })
        self.save(pd.DataFrame(rows), "replication_stability")

    def shift_structure(self):
        """What the shift rules cost, separately from what the queue costs.

        The queueing approximation says how many lanes each half hour needs. A plan that
        could set lanes freely per half hour would buy exactly that. Real rosters cannot,
        so the MILP has to cover the requirement with whole shifts, and the difference
        between the two is the price of the roster rather than the price of the service
        target. Every structure here is solved exactly; the coarsest and finest are then
        run through the simulation to confirm the target still holds.
        """
        cfg = self.cfg
        slot_h = int(cfg["staffing"]["slot_minutes"]) / 60.0
        rate = float(cfg["staffing"]["cost_per_lane_hour"])
        rows = []
        for code in self.codes:
            day = pd.Timestamp(self.results["design_day"][code])
            sc = self.scenario(code, day)
            required = requirement_from_rates(sc.rate_per_hour_by_staffing_slot(), cfg)
            free_hours = float(required.sum()) * slot_h
            for shift_slots in [4, 8, 12, 16]:
                for stride in [1, 2, 4]:
                    if stride > shift_slots:
                        continue
                    vcfg = cfg.with_overrides(
                        staffing__min_shift_slots=int(shift_slots),
                        staffing__shift_start_stride_slots=int(stride))
                    try:
                        lanes, x, info = solve_cover(required, vcfg)
                    except RuntimeError:
                        continue
                    plan = StaffingPlan(lanes, sc.staffing_slot_minutes,
                                        sc.start_minute,
                                        f"shift{shift_slots}_stride{stride}")
                    row = {"airport": code, "date": day,
                           "shift_hours": shift_slots * slot_h,
                           "start_every_hours": stride * slot_h,
                           "shifts_bought": int(x.sum()),
                           "free_lane_hours": free_hours,
                           "plan_lane_hours": plan.lane_hours(),
                           "cost": plan.cost(rate),
                           "roster_overhead_pct":
                               100.0 * (plan.lane_hours() / free_hours - 1.0),
                           "simulated_p95": np.nan, "simulated_p95_mc_halfwidth": np.nan}
                    if (shift_slots, stride) in {(4, 1), (16, 4), (8, 2)}:
                        out = run_replications(cfg, sc, plan, n_reps=self.eval_reps)
                        row["simulated_p95"] = out["summary"]["p95_wait"]
                        row["simulated_p95_mc_halfwidth"] = \
                            out["summary"]["p95_wait_mc_halfwidth"]
                    rows.append(row)
            self.log(f"{code} shift structure done")
        self.save(pd.DataFrame(rows), "shift_structure")

    def target_sweep(self):
        """Where the wait target starts to matter and where capacity alone decides.

        At the volumes these two checkpoints run, rounding bare capacity up to a whole
        lane already leaves enough slack for a generous percentile target, so most of the
        cost is throughput, not queueing. This table finds the point at which tightening
        the target actually buys lanes.
        """
        cfg = self.cfg
        rate = float(cfg["staffing"]["cost_per_lane_hour"])
        rows = []
        for code in self.codes:
            day = pd.Timestamp(self.results["design_day"][code])
            sc = self.scenario(code, day)
            rates = sc.rate_per_hour_by_staffing_slot()
            for target in [3.0, 5.0, 10.0, 20.0, 30.0]:
                required = requirement_from_rates(rates, cfg, target_minutes=target)
                lanes, x, info = solve_cover(required, cfg)
                plan = StaffingPlan(lanes, sc.staffing_slot_minutes, sc.start_minute,
                                    f"target{target:g}")
                out = run_replications(cfg, sc, plan, n_reps=self.eval_reps)
                rows.append({
                    "airport": code, "date": day, "target_p95_minutes": target,
                    "peak_lanes": int(plan.lanes.max()),
                    "lane_hours": plan.lane_hours(), "cost": plan.cost(rate),
                    "simulated_p95": out["summary"]["p95_wait"],
                    "simulated_p95_mc_halfwidth": out["summary"]["p95_wait_mc_halfwidth"],
                    "meets_target": out["summary"]["p95_wait"] <= target,
                })
            self.log(f"{code} target sweep done")
        self.save(pd.DataFrame(rows), "target_sweep")

    def checkpoint_split(self):
        """What pooling every checkpoint into one queue is worth, measured rather than waved at.

        The model treats an airport as a single queue. BNA really is one checkpoint, so
        nothing is lost there. ATL has eight, and passengers in the wrong hall cannot use
        a free lane in another one. This splits the demand across the real checkpoints in
        the proportions TSA recorded, gives each its own queue, and reports both what the
        same total lanes deliver once split and what it costs to buy the target back.
        """
        cfg = self.cfg
        rate_per_hour = float(cfg["staffing"]["cost_per_lane_hour"])
        target = float(cfg["target"]["p95_wait_minutes"])
        rows = []
        for code in self.codes:
            excl = cfg.excluded_checkpoints(code)
            sub = self.tsa[(self.tsa["airport"] == code)
                           & (~self.tsa["checkpoint"].isin(excl))
                           & (self.tsa["date"].isin(self.cal_days))]
            shares = sub.groupby("checkpoint")["throughput"].sum()
            shares = (shares / shares.sum()).sort_values(ascending=False)
            day = pd.Timestamp(self.results["design_day"][code])
            sc = self.scenario(code, day)
            pooled_plan = self.opt_plans[(code, str(day.date()))]["optimised"]
            pooled = run_replications(cfg, sc, pooled_plan, n_reps=self.eval_reps)
            rows.append({"airport": code, "layout": "pooled", "checkpoint": "all",
                         "share": 1.0, "lane_hours": pooled_plan.lane_hours(),
                         "cost": pooled_plan.cost(rate_per_hour),
                         "p95_wait": pooled["summary"]["p95_wait"],
                         "p95_wait_mc_halfwidth": pooled["summary"]["p95_wait_mc_halfwidth"],
                         "mean_wait": pooled["summary"]["mean_wait"],
                         "n_pax": pooled["summary"]["n_pax"]})
            if len(shares) < 2:
                self.log(f"{code} has one checkpoint, nothing to split")
                continue

            for layout in ["split, same lanes", "split, re-optimised"]:
                for cp, share in shares.items():
                    part = sc.scaled(float(share))
                    if layout == "split, same lanes":
                        lanes = np.maximum(1, np.round(pooled_plan.lanes * share)).astype(int)
                        plan = StaffingPlan(lanes, sc.staffing_slot_minutes,
                                            sc.start_minute, f"{cp}_same")
                    else:
                        plan = optimise_staffing(cfg, part, n_reps=self.opt_reps,
                                                 name=f"{cp}_opt").plan
                    out = run_replications(cfg, part, plan, n_reps=self.eval_reps)
                    rows.append({"airport": code, "layout": layout, "checkpoint": cp,
                                 "share": float(share), "lane_hours": plan.lane_hours(),
                                 "cost": plan.cost(rate_per_hour),
                                 "p95_wait": out["summary"]["p95_wait"],
                                 "p95_wait_mc_halfwidth":
                                     out["summary"]["p95_wait_mc_halfwidth"],
                                 "mean_wait": out["summary"]["mean_wait"],
                                 "n_pax": out["summary"]["n_pax"]})
                self.log(f"{code} {layout} done over {len(shares)} checkpoints")
        df = pd.DataFrame(rows)
        self.save(df, "checkpoint_split")

        summary = []
        for (code, layout), g in df.groupby(["airport", "layout"]):
            summary.append({
                "airport": code, "layout": layout,
                "checkpoints": int(len(g)),
                "lane_hours": float(g["lane_hours"].sum()),
                "cost": float(g["cost"].sum()),
                "worst_checkpoint_p95": float(g["p95_wait"].max()),
                "pax_weighted_mean_wait":
                    float((g["mean_wait"] * g["n_pax"]).sum() / g["n_pax"].sum()),
                "meets_target": bool(g["p95_wait"].max() <= target),
            })
        self.save(pd.DataFrame(summary), "checkpoint_split_summary")

    def robustness(self):
        cfg = self.cfg
        rb = cfg["robustness"]
        target = float(cfg["target"]["p95_wait_minutes"])
        rows = []
        self.insurance = {}
        for code in self.codes:
            day = pd.Timestamp(self.results["design_day"][code])
            sc = self.scenario(code, day)
            plans = dict(self.opt_plans[(code, str(day.date()))])

            uplift = float(rb["demand_uplift"])
            hardened = optimise_staffing(cfg, sc, n_reps=self.opt_reps,
                                         demand_factor=1.0 + uplift,
                                         name="optimised_plus10")
            plans["optimised_plus10"] = hardened.plan
            base_cost = plans["optimised"].cost(float(cfg["staffing"]["cost_per_lane_hour"]))
            self.insurance[code] = {
                "base_cost": base_cost,
                "hardened_cost": hardened.plan.cost(float(cfg["staffing"]["cost_per_lane_hour"])),
                "extra_cost": hardened.plan.cost(float(cfg["staffing"]["cost_per_lane_hour"]))
                              - base_cost,
            }

            peak_slot = int(np.argmax(sc.arrivals_per_staffing_slot()))
            outage_start = sc.start_minute + peak_slot * sc.staffing_slot_minutes
            scenarios = {
                "as planned": dict(),
                f"demand +{uplift:.0%}": dict(demand_factor=1.0 + uplift),
                f"service {rb['service_slowdown']:.0%} slower":
                    dict(service_scale=1.0 + float(rb["service_slowdown"])),
                f"{rb['lane_outage_lanes']} lane down {rb['lane_outage_hours']}h at peak":
                    dict(outage=True),
            }
            for label, kwargs in scenarios.items():
                for pname, plan in plans.items():
                    use = plan
                    if kwargs.get("outage"):
                        use = plan.with_outage(outage_start,
                                               int(rb["lane_outage_hours"]) * 60,
                                               int(rb["lane_outage_lanes"]),
                                               plan.name)
                    out = run_replications(
                        cfg, sc, use, n_reps=self.eval_reps,
                        demand_factor=kwargs.get("demand_factor", 1.0),
                        service_scale=kwargs.get("service_scale", 1.0))
                    s = out["summary"]
                    rows.append({"airport": code, "date": day, "scenario": label,
                                 "plan": pname, "cost": plan.cost(
                                     float(cfg["staffing"]["cost_per_lane_hour"])),
                                 "p95_wait": s["p95_wait"],
                                 "p95_wait_mc_halfwidth": s["p95_wait_mc_halfwidth"],
                                 "max_wait": s["max_wait"],
                                 "miss_share": s["miss_share"],
                                 "miss_share_from_queueing": s["miss_share_from_queueing"],
                                 "meets_target": s["p95_wait"] <= target})
                self.log(f"{code} robustness scenario '{label}' done")
            sub = pd.DataFrame([r for r in rows if r["airport"] == code])
            plots.plot_robustness(sub, FIGURES / f"robustness_{code}.png",
                                  f"{code} on {day.date()}: how each plan degrades", target)
        self.save(pd.DataFrame(rows), "robustness")
        self.results["insurance"] = self.insurance

    def sensitivity(self):
        """How the answer moves when each assumption moves, on the design day."""
        cfg = self.cfg
        sens = cfg["sensitivity"]
        rows = []
        for code in self.codes:
            day = pd.Timestamp(self.results["design_day"][code])
            base_yield = self.yields[code]
            base_shift = int(self.results["fitted_shift"][code])
            variants = []
            for v in sens["showup_shift_minutes"]:
                variants.append(("show-up shift vs fitted (min)", v, cfg,
                                 base_shift + int(v), base_yield))
            for v in sens["screening_mean_seconds"]:
                variants.append(("screening mean (s)", v,
                                 cfg.with_overrides(service__screening__mean_seconds=float(v)),
                                 base_shift, base_yield))
            for v in sens["load_factor"]:
                variants.append(("load factor", v,
                                 cfg.with_overrides(demand__load_factor=float(v)),
                                 base_shift, base_yield))
            for v in sens["screening_yield_delta"]:
                variants.append(("screening yield delta", v, cfg, base_shift,
                                 ScreeningYield(base_yield.intercept + float(v),
                                                base_yield.slope)))

            for factor, value, vcfg, shift, yld in variants:
                model = DemandModel(vcfg,
                                    showup=ShowupProfile.from_config(vcfg, shift_minutes=shift),
                                    load_factor=vcfg["demand"]["load_factor"])
                sc = build_scenario(vcfg, model, self.dep, code, day, originating=yld)
                opt = optimise_staffing(vcfg, sc, n_reps=self.opt_reps,
                                        name=f"sens_{factor}_{value}")
                out = run_replications(vcfg, sc, opt.plan, n_reps=self.eval_reps)
                rows.append({
                    "airport": code, "date": day, "factor": factor, "value": value,
                    "daily_pax": float(sc.arrivals_per_slot.sum()),
                    "peak_lanes": int(opt.plan.lanes.max()),
                    "lane_hours": opt.plan.lane_hours(),
                    "cost": opt.plan.cost(float(cfg["staffing"]["cost_per_lane_hour"])),
                    "p95_wait": out["summary"]["p95_wait"],
                    "p95_wait_mc_halfwidth": out["summary"]["p95_wait_mc_halfwidth"],
                    "optimiser_rounds": opt.rounds,
                })
            self.log(f"{code} sensitivity done ({len(variants)} variants)")
        self.save(pd.DataFrame(rows), "sensitivity")

    def write_results(self):
        self.results["runtime_seconds"] = round(time.time() - self.t0, 1)
        self.results["eval_replications"] = self.eval_reps
        self.results["optimiser_replications"] = self.opt_reps
        path = TABLES / "results.json"
        path.write_text(json.dumps(self.results, indent=2, default=str))
        self.log(f"wrote {path}")
