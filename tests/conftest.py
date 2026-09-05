import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from checkpoint.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()
