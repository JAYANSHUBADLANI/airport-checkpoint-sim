"""The config file is the only place an assumption may live."""

import pytest

from checkpoint.analytic import effective_lane_rate_per_hour
from checkpoint.config import load_config


def test_every_section_is_present(cfg):
    for section in ["run", "airports", "days", "demand", "service", "staffing",
                    "target", "optimiser", "robustness", "sensitivity"]:
        assert section in cfg.raw, f"missing config section {section}"


def test_airport_codes_are_the_two_studied(cfg):
    assert cfg.airport_codes == ["ATL", "BNA"]


def test_staffing_slot_is_a_whole_number_of_demand_slots(cfg):
    assert cfg["staffing"]["slot_minutes"] % cfg["demand"]["slot_minutes"] == 0


def test_effective_lane_rate_sits_in_the_published_range(cfg):
    """A standard screening lane clears roughly 150 to 180 passengers an hour."""
    rate = effective_lane_rate_per_hour(cfg)
    assert 140.0 <= rate <= 190.0, rate


def test_overrides_do_not_mutate_the_original(cfg):
    before = cfg["demand"]["load_factor"]
    other = cfg.with_overrides(demand__load_factor=0.5)
    assert other["demand"]["load_factor"] == 0.5
    assert cfg["demand"]["load_factor"] == before


def test_unknown_airport_raises(cfg):
    with pytest.raises(KeyError):
        cfg.connecting_share("XXX")
