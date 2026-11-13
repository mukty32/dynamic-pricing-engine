"""Tests for Phase 3: the price optimizer. Run with:  pytest

Most tests use a FAKE demand model with a known formula, D(p) = C * exp(-beta * p),
so we can check the optimizer against answers derived by hand with calculus.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from fakes import BETA, C, FakeExponentialModel

from pricing_engine.config import SimulationConfig
from pricing_engine.optimizer import (
    PriceOptimizer,
    PricingConstraints,
    PricingContext,
    exponential_closed_form,
)

CONFIG = SimulationConfig()
FARE = 12.0  # downtown base fare -> bounds [9.60, 36.00] with default rules


def context(drivers: int, competitor: float = 40.0, current: float | None = None) -> PricingContext:
    return PricingContext("2025-11-18 18:00", "downtown", False, competitor, drivers, current)


@pytest.fixture
def optimizer() -> PriceOptimizer:
    return PriceOptimizer(FakeExponentialModel(), config=CONFIG)


def test_plenty_of_drivers_gives_textbook_optimum(optimizer):
    rec = optimizer.recommend(context(drivers=500))
    assert rec.price == pytest.approx(1 / BETA, abs=0.01)
    assert rec.binding_constraint.startswith("none")


def test_few_drivers_gives_market_clearing_price(optimizer):
    capacity = 30 * CONFIG.rides_per_driver_per_hour  # 60 rides
    rec = optimizer.recommend(context(drivers=30))
    assert rec.price == pytest.approx(np.log(C / capacity) / BETA, abs=0.02)  # ~ $23.71
    assert rec.binding_constraint == "supply"
    assert rec.expected_demand == pytest.approx(capacity, rel=0.01)


def test_extreme_shortage_hits_surge_cap(optimizer):
    rec = optimizer.recommend(context(drivers=2))  # market-clearing price would be ~$57
    assert rec.price == pytest.approx(3.0 * FARE)
    assert rec.binding_constraint == "surge_cap"


def test_competitor_premium_rule(optimizer):
    rec = optimizer.recommend(context(drivers=30, competitor=15.0))  # cap = 1.5 x 15 = 22.50
    assert rec.price == pytest.approx(22.50)
    assert rec.binding_constraint == "competitor_premium"


def test_price_stability_rule():
    rules = PricingConstraints(max_change=0.10)
    optimizer = PriceOptimizer(FakeExponentialModel(), constraints=rules, config=CONFIG)
    rec = optimizer.recommend(context(drivers=30, current=15.0))  # may move only to [13.50, 16.50]
    assert rec.price == pytest.approx(16.50)
    assert rec.binding_constraint == "price_stability"


def test_engine_matches_pen_and_paper_calculus(optimizer):
    rng = np.random.default_rng(0)
    for drivers in rng.integers(1, 120, size=15):
        rec = optimizer.recommend(context(drivers=int(drivers)))
        expected = exponential_closed_form(C, BETA, rec.capacity, rec.lower_bound, rec.upper_bound)
        assert rec.price == pytest.approx(expected, abs=0.02)


def test_no_grid_price_beats_the_optimum(optimizer):
    ctx = context(drivers=30)
    rec = optimizer.recommend(ctx)
    prices = np.linspace(rec.lower_bound, rec.upper_bound, 2000)
    revenue = prices * np.minimum(C * np.exp(-BETA * prices), rec.capacity)
    assert rec.expected_revenue >= revenue.max() - 0.5


def test_batch_and_single_agree(optimizer):
    contexts = pd.concat([context(d).to_frame() for d in (3, 30, 500)], ignore_index=True)
    batch = optimizer.recommend_many(contexts)
    singles = [optimizer.recommend(context(d)).price for d in (3, 30, 500)]
    np.testing.assert_allclose(batch["price"], singles, atol=0.10)


def test_doubling_demand_raises_surge_price_by_ln2_over_beta(optimizer):
    # p_K = ln(m * C / K) / beta, so multiplying demand by m adds ln(m) / beta to the price.
    normal = optimizer.recommend(context(drivers=30))
    concert = optimizer.recommend(dataclasses.replace(context(drivers=30), demand_multiplier=2.0))
    assert concert.price - normal.price == pytest.approx(np.log(2) / BETA, abs=0.03)  # ~ +$8.66


def test_demand_shock_does_not_move_the_natural_peak(optimizer):
    # With plenty of drivers, the optimum is 1/beta no matter how big demand is.
    rec = optimizer.recommend(dataclasses.replace(context(drivers=5000), demand_multiplier=3.0))
    assert rec.price == pytest.approx(1 / BETA, abs=0.01)


def test_unknown_zone_is_rejected(optimizer):
    bad = PricingContext("2025-11-18 18:00", "mars", False, 12.0, 10)
    with pytest.raises(ValueError, match="unknown zone"):
        optimizer.recommend(bad)
