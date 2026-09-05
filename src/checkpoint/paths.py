"""Filesystem layout for the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "config.yaml"
DATA = ROOT / "data"
RAW = DATA / "raw"
SAMPLE = DATA / "sample"
INTERIM = DATA / "interim"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"


def ensure_dirs():
    for d in (RAW, SAMPLE, INTERIM, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
