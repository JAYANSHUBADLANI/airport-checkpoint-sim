"""Parse the TSA FOIA reading room weekly checkpoint throughput PDFs.

Each PDF holds one row per airport, checkpoint and clock hour, with the total
passengers plus known crewmember passengers screened in that hour. Date, hour,
airport code, airport name, city and state are merged cells in the source table,
so they are forward filled down the page and across page boundaries.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pdfplumber

COLUMNS = ["date", "hour", "airport", "airport_name", "city", "state", "checkpoint", "throughput"]
_HOUR_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")


def _clean(value):
    if value is None:
        return None
    text = str(value).replace("\n", " ").strip()
    return text or None


def _parse_pages(args):
    path, first, last = args
    rows = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[first:last]:
            table = page.extract_table()
            if not table:
                continue
            for raw in table:
                cells = [_clean(c) for c in raw]
                if len(cells) < 8:
                    cells = cells + [None] * (8 - len(cells))
                date, hour, code, name, city, state, checkpoint, total = cells[:8]
                if date is not None and not _DATE_RE.match(date):
                    continue
                if hour is not None and not _HOUR_RE.match(hour):
                    hour = None
                if total is None or not re.fullmatch(r"-?[\d,]+", total):
                    continue
                rows.append((date, hour, code, name, city, state, checkpoint,
                             int(total.replace(",", ""))))
    return rows


def parse_throughput_pdf(path: str | Path, workers: int = 8) -> pd.DataFrame:
    path = str(path)
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
    stride = max(1, (n_pages + workers - 1) // workers)
    chunks = [(path, i, min(n_pages, i + stride)) for i in range(0, n_pages, stride)]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_parse_pages, chunks))
    rows = [r for chunk in results for r in chunk]
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in ["date", "hour", "airport", "airport_name", "city", "state"]:
        df[col] = df[col].ffill()
    df = df.dropna(subset=["date", "hour", "airport", "checkpoint"])
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["hour"] = df["hour"].str.slice(0, 2).astype(int)
    df = df.groupby(["date", "hour", "airport", "checkpoint"], as_index=False)["throughput"].sum()
    return df.sort_values(["airport", "date", "hour", "checkpoint"]).reset_index(drop=True)


def load_weeks(paths, cache: str | Path | None = None) -> pd.DataFrame:
    if cache is not None and Path(cache).exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        return df
    frames = [parse_throughput_pdf(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["date", "hour", "airport", "checkpoint"])
    df = df.sort_values(["airport", "date", "hour", "checkpoint"]).reset_index(drop=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def airport_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse checkpoints to one row per airport, date and hour."""
    out = df.groupby(["airport", "date", "hour"], as_index=False)["throughput"].sum()
    return out
