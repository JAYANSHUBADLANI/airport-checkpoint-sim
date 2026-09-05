"""Download the raw inputs.

The BTS on-time zip is served from a static path but the origin throttles a single
connection hard, so the file is pulled with parallel range requests. The TSA reading
room refuses plain HTTP clients with a 403 regardless of headers; where that happens the
function says so and prints the exact URLs to fetch by hand, rather than failing
silently or pretending the data is unavailable.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from pathlib import Path

import requests

BTS_URL = ("https://transtats.bts.gov/PREZIP/"
           "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
TSA_URL = "https://www.tsa.gov/sites/default/files/foia-readingroom/{name}"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

TSA_FILES = [
    ("tsa-throughput-data-may-31-2026-to-june-6-2026.pdf",
     "tsa_throughput_2026-05-31_to_2026-06-06.pdf"),
    ("tsa-throughput-data-june-7-2026-to-june-13-2026.pdf",
     "tsa_throughput_2026-06-07_to_2026-06-13.pdf"),
]


def fetch_bts_ontime(year: int, month: int, dest: Path, workers: int = 16,
                     force: bool = False) -> Path:
    url = BTS_URL.format(year=year, month=month)
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"already present: {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    head = requests.head(url, headers={"User-Agent": UA}, timeout=120, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers["Content-Length"])
    print(f"downloading {url}\n  {total:,} bytes in {workers} parallel ranges")
    size = (total + workers - 1) // workers

    def part(i: int):
        lo, hi = i * size, min(total - 1, i * size + size - 1)
        for attempt in range(5):
            try:
                r = requests.get(url, headers={"Range": f"bytes={lo}-{hi}", "User-Agent": UA},
                                 timeout=1800)
                if r.status_code in (200, 206) and len(r.content) == hi - lo + 1:
                    return i, r.content
            except requests.RequestException:
                pass
            time.sleep(3)
        raise RuntimeError(f"range {i} failed after 5 attempts")

    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as pool:
        parts = dict(pool.map(part, range(workers)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"".join(parts[i] for i in range(workers)))
    print(f"  wrote {dest} in {time.time() - t0:.0f}s")
    return dest


def fetch_tsa_throughput(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    got, missing = [], []
    for remote, local in TSA_FILES:
        target = raw_dir / local
        if target.exists():
            print(f"already present: {local} ({target.stat().st_size:,} bytes)")
            got.append(target)
            continue
        url = TSA_URL.format(name=remote)
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=300)
        except requests.RequestException as exc:
            missing.append((url, str(exc)))
            continue
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            target.write_bytes(r.content)
            got.append(target)
            print(f"  wrote {target}")
        else:
            missing.append((url, f"HTTP {r.status_code}"))
    if missing:
        print("\nThe TSA reading room refused this client. Download these by hand into "
              f"{raw_dir} and keep the local names:")
        for url, why in missing:
            print(f"  {why}: {url}")
            print(f"    save as: {dict(TSA_FILES)[url.rsplit('/', 1)[-1]]}")
    return got
