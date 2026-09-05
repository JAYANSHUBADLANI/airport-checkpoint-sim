"""Load the BTS On-Time Reporting Carrier On-Time Performance extract.

One zip per month holds every scheduled domestic flight operated by a reporting
carrier. I keep the scheduled departure side only: origin airport, scheduled local
departure time, operating carrier and the cancellation flag.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

USECOLS = [
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "Cancelled",
]


def _parse_crs_dep(date: pd.Series, crs: pd.Series) -> pd.Series:
    """CRSDepTime is local clock time as HHMM with 2400 meaning midnight ending the day."""
    hhmm = pd.to_numeric(crs, errors="coerce")
    hours = (hhmm // 100).astype("Int64")
    minutes = (hhmm % 100).astype("Int64")
    rollover = hours == 24
    hours = hours.where(~rollover, 0)
    stamp = date + pd.to_timedelta(hours.astype("float"), unit="h") \
                 + pd.to_timedelta(minutes.astype("float"), unit="m") \
                 + pd.to_timedelta(rollover.astype("float"), unit="D")
    return stamp


def load_ontime(zip_path: str | Path, airports: list[str] | None = None) -> pd.DataFrame:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one csv in {zip_path.name}, found {names}")
        with zf.open(names[0]) as fh:
            df = pd.read_csv(fh, usecols=USECOLS, encoding="latin-1", low_memory=False)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    if airports:
        df = df[df["Origin"].isin(airports)]
    df = df.copy()
    df["sched_dep"] = _parse_crs_dep(df["FlightDate"], df["CRSDepTime"])
    df = df.dropna(subset=["sched_dep"])
    df["cancelled"] = df["Cancelled"].fillna(0).astype(int).astype(bool)
    return df.reset_index(drop=True)


def departures(df: pd.DataFrame, include_cancelled: bool = False) -> pd.DataFrame:
    out = df if include_cancelled else df[~df["cancelled"]]
    keep = ["FlightDate", "Origin", "Dest", "Reporting_Airline",
            "Flight_Number_Reporting_Airline", "sched_dep", "cancelled"]
    return out[keep].rename(columns={"FlightDate": "flight_date", "Origin": "airport"}) \
                    .reset_index(drop=True)


def daily_counts(dep: pd.DataFrame) -> pd.DataFrame:
    return dep.groupby(["airport", "flight_date"], as_index=False) \
              .agg(flights=("sched_dep", "size"))


def write_sample(dep: pd.DataFrame, airport: str, day: str, path: str | Path) -> Path:
    """Commit one airport-day so the test suite runs with no network call."""
    sub = dep[(dep["airport"] == airport) & (dep["flight_date"] == pd.Timestamp(day))]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def read_sample(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["flight_date", "sched_dep"])
    return df
