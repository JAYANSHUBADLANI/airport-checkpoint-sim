"""Seats offered per scheduled departure.

The BTS on-time performance file identifies the operating carrier but carries no seat
count. The canonical seat source is BTS T-100 Domestic Segment, which has no stable
direct download URL and is served only behind an ASP.NET form that could not be driven
from this environment, so the seat figures below are an assumption, not a measurement.
They are average seats per domestic departure by reporting carrier, consistent with the
published mainline and regional fleet configurations for each carrier. Every number
here is labelled "assumed" in the README, and the load factor that multiplies them is
varied in the sensitivity analysis.
"""

from __future__ import annotations

import pandas as pd

SEATS_PER_DEPARTURE = {
    "AA": 137.0,
    "AS": 154.0,
    "B6": 162.0,
    "DL": 140.0,
    "F9": 195.0,
    "G4": 172.0,
    "HA": 156.0,
    "NK": 195.0,
    "UA": 148.0,
    "WN": 168.0,
    "9E": 73.0,
    "OO": 71.0,
    "YX": 73.0,
    "MQ": 66.0,
    "OH": 70.0,
    "YV": 73.0,
    "ZW": 50.0,
    "C5": 50.0,
    "PT": 50.0,
    "EM": 30.0,
    "KS": 30.0,
    "4B": 30.0,
    "VX": 149.0,
}

DEFAULT_SEATS = 110.0
SOURCE = "assumed"


def attach_seats(df: pd.DataFrame, carrier_col: str = "Reporting_Airline") -> pd.DataFrame:
    out = df.copy()
    out["seats"] = out[carrier_col].map(SEATS_PER_DEPARTURE).fillna(DEFAULT_SEATS)
    out["seats_source"] = SOURCE
    return out


def coverage(df: pd.DataFrame, carrier_col: str = "Reporting_Airline") -> pd.DataFrame:
    """Report which carriers fell through to the default, so the gap is visible."""
    counts = df[carrier_col].value_counts().rename_axis("carrier").reset_index(name="flights")
    counts["seats"] = counts["carrier"].map(SEATS_PER_DEPARTURE)
    counts["used_default"] = counts["seats"].isna()
    counts["seats"] = counts["seats"].fillna(DEFAULT_SEATS)
    return counts
