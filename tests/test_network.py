"""Tests for Phase 10: the multi-zone convex optimiser. Run with:  pytest

Most tests use hand-built markets with known demand curves, so every answer can be
checked against calculus: the closed form, marginal revenue and the KKT conditions.
"""

import math

import numpy as np
import pandas as pd
import pytest
from fakes import BETA, C, FakeExponentialModel

from pricing_engine.network import RepositioningCosts, ZoneMarket, markets_from_model, solve_network
from pricing_engine.optimizer import exponential_closed_form

R = 2.0  # rides per driver-hour


def market(name="downtown", C=400.0, beta=0.08, drivers=30.0, fare=12.0, lower=9.6, upper=36.0):
    return ZoneMarket(name=name, C=C, beta=beta, drivers=drivers, base_fare=fare, lower=lower, upper=upper)


def city(downtown_drivers=25.0, suburb_drivers=45.0):
    return [
        market("downtown", drivers=downtown_drivers),
        market("airport", C=500.0, beta=0.035, drivers=15.0, fare=25.0, lower=20.0, upper=56.25),
        market("suburbs", C=120.0, beta=0.10, drivers=suburb_drivers, fare=10.0, lower=8.0, upper=22.5),
    ]


# ----------------------------- the math behind the formulation -------------
def test_revenue_is_concave_in_quantity():
    # R(x) = x ln(C/x) / beta  ->  R''(x) = -1 / (beta x) < 0
    x, h = np.linspace(5, 300, 50), 1e-3

    def revenue(q):
        return q * np.log(C / q) / BETA

    second_derivative = (revenue(x + h) - 2 * revenue(x) + revenue(x - h)) / h**2
    np.testing.assert_allclose(second_derivative, -1 / (BETA * x), rtol=1e-3)


@pytest.mark.parametrize("drivers", [2, 10, 30, 60, 500])
def test_single_zone_matches_the_phase3_closed_form(drivers):
    m = market(drivers=drivers)
    plan = solve_network([m], rides_per_driver=R, allow_moves=False)
    expected = exponential_closed_form(m.C, m.beta, R * drivers, m.lower, m.upper)
    assert plan.zones.loc["downtown", "price"] == pytest.approx(expected, abs=0.02)


def test_shadow_price_is_marginal_revenue_when_supply_binds():
    plan = solve_network([market(drivers=30)], rides_per_driver=R, allow_moves=False)  # supply-limited
    row = plan.zones.loc["downtown"]
    assert row.value_per_ride == pytest.approx(row.price - 1 / BETA, abs=0.05)  # lambda = p - 1/beta


def test_spare_drivers_are_worth_nothing():
    plan = solve_network([market(drivers=500)], rides_per_driver=R, allow_moves=False)
    assert plan.zones.loc["downtown", "value_per_ride"] == pytest.approx(0.0, abs=1e-3)
    assert plan.zones.loc["downtown", "price"] == pytest.approx(1 / BETA, abs=0.02)


# ----------------------------- repositioning -------------------------------
def test_moving_drivers_never_lowers_profit():
    markets = city()
    separate = solve_network(markets, rides_per_driver=R, allow_moves=False)
    network = solve_network(markets, rides_per_driver=R)
    assert network.profit >= separate.profit - 1e-6
    assert network.profit > separate.profit * 1.05  # this city clearly benefits
    # Drivers flow TOWARD the most valuable zone (the airport, ~$56 per extra ride here) and nobody
    # leaves it. They come from both other zones: the optimiser weighs travel time against value.
    assert (network.moves["from"] != "airport").all()
    assert network.zones.loc["airport", "drivers_after"] > network.zones.loc["airport", "drivers_before"]


def test_kkt_every_move_exactly_pays_for_itself():
    """At the optimum, a used route breaks even and no unused route would make money."""
    costs = RepositioningCosts()
    plan = solve_network(city(), costs, rides_per_driver=R)
    value = plan.zones["value_per_ride"]
    for a in value.index:
        for b in value.index:
            if a == b:
                continue
            keep = 1 - costs.travel_minutes(a, b) / 60
            net_gain = R * keep * value[b] - R * value[a] - costs.move_cost(a, b)
            used = ((plan.moves["from"] == a) & (plan.moves["to"] == b)).any()
            if used:
                assert net_gain == pytest.approx(0.0, abs=0.1)  # complementary slackness
            else:
                assert net_gain <= 0.1  # no profitable move left


def test_expensive_moves_are_not_made():
    plan = solve_network(city(), RepositioningCosts(bonus_per_move=1_000), rides_per_driver=R)
    assert plan.moves.empty
    assert plan.move_cost == pytest.approx(0.0)


def test_cannot_move_more_drivers_than_exist():
    plan = solve_network(city(suburb_drivers=5.0), rides_per_driver=R)
    moved_out = plan.moves.groupby("from")["drivers"].sum()
    for zone, drivers in moved_out.items():
        assert drivers <= plan.zones.loc[zone, "drivers_before"] + 1e-6


# ----------------------------- reading markets off a model -----------------
def test_markets_from_model_recovers_the_demand_curve():
    contexts = pd.DataFrame({"timestamp": pd.Timestamp("2025-11-21 18:00"), "zone": ["downtown"],
                             "is_raining": False, "competitor_price": [18.0], "drivers_available": [30]})  # fmt: skip
    (m,) = markets_from_model(FakeExponentialModel(), contexts)
    assert m.beta == pytest.approx(BETA, rel=1e-6)
    assert pytest.approx(C, rel=1e-6) == m.C
    assert (m.lower, m.upper) == (pytest.approx(9.6), pytest.approx(27.0))  # floor, 1.5 x competitor
    assert m.price_for(m.C * math.exp(-BETA * 20)) == pytest.approx(20.0)  # inverse demand
