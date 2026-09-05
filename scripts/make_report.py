"""Turn the output tables into the markdown blocks the README and memo quote.

Nothing here computes a result. It only formats what scripts/run_all.py produced, so no
number can enter the write-up without having come out of a run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from checkpoint.config import load_config
from checkpoint.paths import OUTPUTS, TABLES
from checkpoint.seats import DEFAULT_SEATS, SEATS_PER_DEPARTURE


def md(df: pd.DataFrame, floatfmt="{:,.2f}") -> str:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].map("{:,}".format)
    header = "| " + " | ".join(str(c) for c in out.columns) + " |"
    rule = "| " + " | ".join("---" for _ in out.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in out.itertuples(index=False)]
    return "\n".join([header, rule] + rows)


def dotted(cfg_raw, path):
    node = cfg_raw
    for k in path.split("."):
        node = node[k]
    return node


def assumptions_table(cfg) -> pd.DataFrame:
    rows = []
    for path, source in cfg["sources"].items():
        if path == "seats_per_departure":
            value = (f"{len(SEATS_PER_DEPARTURE)} carriers, "
                     f"{min(SEATS_PER_DEPARTURE.values()):.0f} to "
                     f"{max(SEATS_PER_DEPARTURE.values()):.0f} seats, "
                     f"default {DEFAULT_SEATS:.0f}")
        else:
            value = dotted(cfg.raw, path)
        rows.append({"setting": path, "value": value,
                     "source": " ".join(str(source).split())})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    results = json.loads((TABLES / "results.json").read_text())
    parts = []

    def add(title, body):
        parts.append(f"### {title}\n\n{body}\n")

    add("Assumptions", md(assumptions_table(cfg), "{:,.4g}"))

    v = pd.read_csv(TABLES / "verification_mmc.csv")
    add("Verification against M/M/c",
        md(v[["lanes", "arrivals_per_hour", "utilisation", "sim_mean_wait",
              "analytic_mean_wait", "mean_wait_error_pct", "sim_p95_wait",
              "analytic_p95_wait", "p95_wait_error_pct"]], "{:,.3f}"))

    c = pd.read_csv(TABLES / "validation_calibration.csv")
    add("Calibration of the screening yield", md(c, "{:,.4f}"))
    s = pd.read_csv(TABLES / "validation_scores.csv")
    add("Validation against TSA throughput", md(s, "{:,.3f}"))
    d = pd.read_csv(TABLES / "showup_shift_diagnostic.csv")
    add("Show-up curve shift diagnostic", md(d, "{:,.3f}"))

    p = pd.read_csv(TABLES / "demand_profile.csv", parse_dates=["date"])
    p["date"] = p["date"].dt.date
    add("Demand profile by airport-day",
        md(p[["airport", "date", "total_pax", "peak_hour_pax", "mean_hour_pax",
              "peak_to_mean_hour", "peak_slot_rate_per_hour"]], "{:,.1f}"))

    f = pd.read_csv(TABLES / "fixed_plan_summary.csv", parse_dates=["date"])
    f["date"] = f["date"].dt.date
    add("A fixed plan sized to the daily average",
        md(f[["airport", "date", "lanes", "n_pax", "mean_wait", "p95_wait",
              "p95_wait_mc_halfwidth", "max_wait", "miss_share"]], "{:,.2f}"))

    pc = pd.read_csv(TABLES / "plan_comparison.csv", parse_dates=["date"])
    pc["date"] = pc["date"].dt.date
    design = {k: pd.Timestamp(v).date() for k, v in results["design_day"].items()}
    day = pc[[r.airport in design and r.date == design[r.airport]
              for r in pc.itertuples()]]
    add("Plan comparison on the design day",
        md(day[["airport", "date", "plan", "lanes_peak", "lane_hours", "cost",
                "p95_wait", "p95_wait_mc_halfwidth", "max_wait", "miss_share",
                "miss_share_from_queueing", "meets_target"]], "{:,.3f}"))

    agg = pc.groupby(["airport", "plan"]).agg(
        days=("date", "nunique"), cost=("cost", "mean"), p95_wait=("p95_wait", "mean"),
        max_wait=("max_wait", "mean"), miss_share=("miss_share", "mean"),
        days_meeting_target=("meets_target", "sum")).reset_index()
    add("Plan comparison averaged over every selected day", md(agg, "{:,.2f}"))

    r = pd.read_csv(TABLES / "optimiser_rounds.csv", parse_dates=["date"])
    r["date"] = r["date"].dt.date
    add("What the optimiser promised and what the simulation delivered",
        md(r[["airport", "date", "round", "margin", "cost", "n_violating_slots",
              "worst_slot_p95", "overall_p95", "overall_p95_mc"]], "{:,.2f}"))

    rb = pd.read_csv(TABLES / "robustness.csv", parse_dates=["date"])
    rb["date"] = rb["date"].dt.date
    add("Robustness", md(rb[["airport", "scenario", "plan", "cost", "p95_wait",
                             "p95_wait_mc_halfwidth", "max_wait",
                             "miss_share_from_queueing", "meets_target"]], "{:,.3f}"))

    st = pd.read_csv(TABLES / "shift_structure.csv")
    add("What the roster rules cost, separately from the service target",
        md(st[["airport", "shift_hours", "start_every_hours", "shifts_bought",
               "free_lane_hours", "plan_lane_hours", "roster_overhead_pct", "cost",
               "simulated_p95"]], "{:,.2f}"))

    ts = pd.read_csv(TABLES / "target_sweep.csv", parse_dates=["date"])
    add("How much the wait target itself costs",
        md(ts[["airport", "target_p95_minutes", "peak_lanes", "lane_hours", "cost",
               "simulated_p95", "simulated_p95_mc_halfwidth", "meets_target"]], "{:,.2f}"))

    cs = pd.read_csv(TABLES / "checkpoint_split_summary.csv")
    add("What pooling the halls into one queue is worth", md(cs, "{:,.2f}"))
    csd = pd.read_csv(TABLES / "checkpoint_split.csv")
    add("Per checkpoint, once ATL is split",
        md(csd[csd.airport == "ATL"][["layout", "checkpoint", "share", "lane_hours",
                                      "p95_wait", "mean_wait", "n_pax"]], "{:,.3f}"))

    rs = pd.read_csv(TABLES / "replication_stability.csv")
    add("Is the replication count enough", md(rs, "{:,.3f}"))

    pv = pd.read_csv(TABLES / "optimiser_promise_vs_delivery.csv", parse_dates=["date"])
    pv = pv[pv["arrivals_per_hour"] > 0]
    add("Promised against delivered, by half hour, on the ATL design day",
        md(pv[pv.airport == "ATL"][["clock_hour", "arrivals_per_hour", "lanes",
                                    "required_by_approximation", "promised_p95",
                                    "delivered_p95"]], "{:,.2f}"))

    sn = pd.read_csv(TABLES / "sensitivity.csv", parse_dates=["date"])
    add("Sensitivity", md(sn[["airport", "factor", "value", "daily_pax", "peak_lanes",
                              "lane_hours", "cost", "p95_wait", "optimiser_rounds"]],
                          "{:,.2f}"))

    sd = pd.read_csv(TABLES / "schedule_daily_counts.csv")
    add("Schedule loaded",
        md(sd.groupby("airport").agg(days=("flight_date", "nunique"),
                                     flights=("flights", "sum"),
                                     mean_daily_flights=("flights", "mean"),
                                     cancelled_share=("cancelled_share", "mean")
                                     ).reset_index(), "{:,.3f}"))

    path = OUTPUTS / "report_fragments.md"
    path.write_text("\n".join(parts))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
