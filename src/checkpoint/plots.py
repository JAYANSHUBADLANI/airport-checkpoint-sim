"""Figures. Plain matplotlib, no styling libraries, readable in black and white."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGSIZE = (10.0, 5.5)


def _finish(fig, ax, path: Path, title: str, xlabel: str, ylabel: str, legend=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.6)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_showup_profile(profile, slot_minutes: int, path: Path):
    pmf = profile.slot_pmf(slot_minutes)
    centres = profile.earliest + slot_minutes * (np.arange(len(pmf)) + 0.5)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(centres, pmf * 100.0, width=slot_minutes * 0.9, color="0.4")
    ax.axvline(profile.mode_minutes_before(), color="black", linestyle="--", linewidth=1.2,
               label=f"mode {profile.mode_minutes_before():.0f} min")
    ax.axvline(profile.mean_minutes_before(), color="black", linestyle=":", linewidth=1.2,
               label=f"mean {profile.mean_minutes_before():.0f} min")
    ax.invert_xaxis()
    return _finish(fig, ax, path, "Assumed checkpoint show-up profile",
                   "minutes before scheduled departure",
                   f"share of a flight's passengers per {slot_minutes} min")


def plot_arrival_curves(curves: dict, path: Path, title: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    styles = ["-", "--", ":", "-."]
    greys = np.linspace(0.0, 0.6, max(1, len(curves)))
    for i, (label, df) in enumerate(sorted(curves.items())):
        hours = df["slot_start"].dt.hour + df["slot_start"].dt.minute / 60.0
        hours = np.where(hours < 3, hours + 24, hours)
        ax.plot(hours, df["arrivals"] * 12.0, linewidth=1.5, label=label,
                color=str(greys[i]), linestyle=styles[i % len(styles)])
    ax.set_xticks(range(3, 27, 2))
    ax.set_xticklabels([f"{h % 24:02d}" for h in range(3, 27, 2)])
    return _finish(fig, ax, path, title, "hour of day (local)",
                   "checkpoint arrivals per hour")


def plot_model_vs_observed(joined: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(joined))
    ax.plot(x, joined["throughput"], linewidth=1.4, color="black", label="TSA observed")
    ax.plot(x, joined["modelled"], linewidth=1.4, color="0.55", linestyle="--",
            label="model")
    first = joined.reset_index().groupby("date")["index"].min()
    ax.set_xticks(first.to_numpy())
    ax.set_xticklabels([str(d.date())[5:] for d in first.index], rotation=45,
                       ha="right", fontsize=8)
    return _finish(fig, ax, path, title, "date", "passengers screened per hour")


def plot_plan_comparison(plans: dict, by_slot: dict, target: float, path: Path,
                         title: str, start_hour: int, slot_minutes: int):
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.5), sharex=True)
    styles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    greys = np.linspace(0.0, 0.62, max(1, len(plans)))
    for i, (label, plan) in enumerate(plans.items()):
        hours = start_hour + np.arange(plan.n_slots) * slot_minutes / 60.0
        axes[0].step(hours, plan.lanes, where="post", linewidth=1.5, label=label,
                     color=str(greys[i]), linestyle=styles[i % len(styles)])
    axes[0].set_ylabel("lanes open")
    axes[0].grid(alpha=0.3, linewidth=0.6)
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[0].set_title(title)

    for i, (label, df) in enumerate(by_slot.items()):
        hours = start_hour + df["slot"].to_numpy() * slot_minutes / 60.0
        axes[1].plot(hours, df["p95_wait"], linewidth=1.5, label=label,
                     color=str(greys[i]), linestyle=styles[i % len(styles)])
    axes[1].axhline(target, color="black", linestyle="--", linewidth=1.2,
                    label=f"target {target:.0f} min")
    axes[1].set_ylabel("95th percentile wait (min)")
    axes[1].set_xlabel("hour of day (local)")
    axes[1].grid(alpha=0.3, linewidth=0.6)
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    axes[1].set_xticks(range(start_hour, 27, 2))
    axes[1].set_xticklabels([f"{h % 24:02d}" for h in range(start_hour, 27, 2)])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_robustness(df: pd.DataFrame, path: Path, title: str, target: float):
    """Bars are capped at three times the target.

    A plan that fails here does not fail by a little. Left on a linear axis the failures
    run to hours and flatten everything that still works into the baseline, so anything
    above the cap is drawn at the cap and labelled with the number it actually reached.
    """
    scenarios = list(dict.fromkeys(df["scenario"]))
    plans = list(dict.fromkeys(df["plan"]))
    width = 0.8 / len(plans)
    cap = 3.0 * target
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    x = np.arange(len(scenarios))
    greys = np.linspace(0.15, 0.8, len(plans))
    hatches = ["", "//", "..", "xx", "\\\\"]
    for i, plan in enumerate(plans):
        sub = df[df["plan"] == plan].set_index("scenario").reindex(scenarios)
        values = sub["p95_wait"].to_numpy(dtype=float)
        shown = np.minimum(values, cap)
        err = np.where(values <= cap, sub["p95_wait_mc_halfwidth"].to_numpy(dtype=float), 0.0)
        ax.bar(x + i * width, shown, width, yerr=err, capsize=2,
               color=str(greys[i]), edgecolor="black", linewidth=0.5,
               hatch=hatches[i % len(hatches)], label=plan)
        for xi, v, sv in zip(x + i * width, values, shown):
            if v > cap:
                ax.text(xi, cap * 1.01, f"{v:,.0f}", ha="center", va="bottom",
                        fontsize=7.5, rotation=90)
    ax.axhline(target, color="black", linestyle="--", linewidth=1.2,
               label=f"target {target:.0f} min")
    ax.set_ylim(0, cap * 1.35)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(scenarios, rotation=12, ha="right", fontsize=9)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left")
    ax.set_title(title)
    ax.set_ylabel("95th percentile wait (min), capped at " f"{cap:.0f}")
    ax.grid(alpha=0.3, linewidth=0.6, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_required_lanes_curve(curve: np.ndarray, path: Path, target: float):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(curve[:, 0], curve[:, 1], linewidth=1.6, color="0.3")
    return _finish(fig, ax, path,
                   f"Lanes needed to hold the 95th percentile wait under {target:.0f} minutes",
                   "checkpoint arrivals per hour", "lanes", legend=False)
