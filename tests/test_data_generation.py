"""Tests for the Phase 1 simulator. Run with:  pytest"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pricing_engine.config import SimulationConfig
from pricing_engine.data_generation import (
    EXPECTED_COLUMNS,
    circular_bump,
    generate_dataset,
    price_response,
    validate_dataset,
)

SMALL_CONFIG = SimulationConfig(n_days=14)


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    return generate_dataset(SMALL_CONFIG)


# ----------------------------- math tests ---------------------------------
def test_circular_bump_wraps_around_midnight():
    # 23:00 and 03:00 are both 2 hours away from 01:00 on a clock
    assert circular_bump(23, center=1, width=2) == pytest.approx(circular_bump(3, center=1, width=2))


def test_price_response_is_one_at_reference_price():
    assert price_response(12.0, 12.0, 12.0, beta=0.08, gamma=0.04) == pytest.approx(1.0)


def test_elasticity_equals_minus_beta_times_price():
    # Elasticity = d ln(D) / d ln(p). For D = A * exp(-beta * p) it should be -beta * p.
    beta, p, h = 0.08, 15.0, 1e-4

    def log_d(price):
        return np.log(price_response(price, 12.0, 12.0, beta, 0.04))

    numeric_elasticity = (log_d(p + h) - log_d(p - h)) / (np.log(p + h) - np.log(p - h))
    assert numeric_elasticity == pytest.approx(-beta * p, rel=1e-4)


def test_revenue_peaks_at_one_over_beta():
    # Revenue R(p) = p * D(p) is maximised at p* = 1 / beta (we derive this with calculus).
    beta = 0.08
    prices = np.linspace(1, 40, 39_001)
    revenue = prices * price_response(prices, 12.0, 12.0, beta, 0.04)
    assert prices[np.argmax(revenue)] == pytest.approx(1 / beta, abs=0.01)


# ----------------------------- data tests ---------------------------------
def test_shape_and_columns(small_df):
    assert list(small_df.columns) == EXPECTED_COLUMNS
    assert len(small_df) == SMALL_CONFIG.n_days * 24 * len(SMALL_CONFIG.zones)


def test_same_seed_gives_identical_data(small_df):
    pd.testing.assert_frame_equal(small_df, generate_dataset(SMALL_CONFIG))


def test_different_seed_gives_different_data(small_df):
    other = generate_dataset(dataclasses.replace(SMALL_CONFIG, seed=SMALL_CONFIG.seed + 1))
    assert not small_df["demand"].equals(other["demand"])


def test_rides_respect_demand_and_supply(small_df):
    capacity = small_df["drivers_available"] * SMALL_CONFIG.rides_per_driver_per_hour
    assert (small_df["rides_completed"] <= small_df["demand"]).all()
    assert (small_df["rides_completed"] <= capacity).all()


def test_surge_happens_at_rush_hour(small_df):
    downtown = small_df[(small_df["zone"] == "downtown") & ~small_df["is_weekend"]]
    rush = downtown[downtown["hour"] == 18]["our_price"].mean()
    night = downtown[downtown["hour"] == 3]["our_price"].mean()
    assert rush > night


def test_validation_catches_bad_data(small_df):
    broken = small_df.copy()
    broken.loc[0, "our_price"] = -5.0
    with pytest.raises(ValueError, match="negative"):
        validate_dataset(broken, SMALL_CONFIG)
