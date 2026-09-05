"""Parse the downloaded TSA weekly throughput PDFs into one tidy CSV."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint.paths import INTERIM, RAW, ensure_dirs
from checkpoint.tsa import load_weeks


def main():
    ensure_dirs()
    paths = sorted(RAW.glob("tsa_throughput_*.pdf"))
    if not paths:
        raise SystemExit("no TSA throughput PDFs in data/raw")
    t0 = time.time()
    df = load_weeks(paths, cache=INTERIM / "tsa_throughput.csv")
    print(f"parsed {len(paths)} pdfs in {time.time() - t0:.1f}s, rows={len(df)}")
    print(f"dates {df.date.min().date()} to {df.date.max().date()}, airports={df.airport.nunique()}")


if __name__ == "__main__":
    main()
