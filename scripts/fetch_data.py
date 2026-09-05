"""Fetch every raw input the pipeline needs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint.config import load_config
from checkpoint.fetch import fetch_bts_ontime, fetch_tsa_throughput
from checkpoint.paths import RAW, ensure_dirs


def main():
    ensure_dirs()
    cfg = load_config()
    year = int(cfg["run"]["schedule_year"])
    month = int(cfg["run"]["schedule_month"])
    fetch_bts_ontime(year, month, RAW / f"ontime_{year}_{month}.zip")
    fetch_tsa_throughput(RAW)


if __name__ == "__main__":
    main()
