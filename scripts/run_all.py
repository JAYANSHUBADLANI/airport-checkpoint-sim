"""Run the whole study and write every table and figure."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint.pipeline import Runner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="fewer replications, for a smoke test rather than a result")
    args = ap.parse_args()
    Runner(quick=args.quick).run()


if __name__ == "__main__":
    main()
