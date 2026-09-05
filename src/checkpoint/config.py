"""Single source of truth for every model assumption.

Everything that is not read from a downloaded file is declared in config/config.yaml
and reaches the rest of the code through this module.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import yaml

from .paths import CONFIG


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default=None) -> Any:
        return self.raw.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.raw["run"]["seed"])

    @property
    def airport_codes(self) -> list[str]:
        return [v["code"] for v in self.raw["airports"].values()]

    def airport_entry(self, code: str) -> dict[str, Any]:
        for v in self.raw["airports"].values():
            if v["code"] == code:
                return v
        raise KeyError(f"airport {code} is not in the config")

    def connecting_share(self, code: str) -> float:
        return float(self.airport_entry(code)["connecting_share"])

    def excluded_checkpoints(self, code: str) -> list[str]:
        return list(self.airport_entry(code).get("excluded_checkpoints") or [])

    def with_overrides(self, **paths) -> "Config":
        """Return a copy with dotted-path values replaced, used by the sensitivity runs."""
        new = copy.deepcopy(self.raw)
        for dotted, value in paths.items():
            keys = dotted.split("__")
            node = new
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value
        return Config(new)


def load_config(path=None) -> Config:
    with open(path or CONFIG) as fh:
        return Config(yaml.safe_load(fh))
