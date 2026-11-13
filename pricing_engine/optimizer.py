"""Phase 3: the price optimization engine.

The question it answers:
    "Given this situation (zone, time, weather, competitor price, drivers online),
     which price earns the most revenue without breaking any business rule?"

The math problem:

    maximise   R(p) = p * min( D(p), K )          revenue = price x rides we can actually serve
    subject to lower_bound <= p <= upper_bound    business rules (price floor, surge cap, ...)

where D(p) is the demand forecast from our Phase 2 model and K = drivers x rides_per_driver
is the supply capacity.

How we solve it (works for ANY demand model, even a black box):
    1. Global search: evaluate R(p) on a grid of 301 prices, all in ONE model call.
    2. Local polish:  zoom into the best grid cell with SciPy's bounded scalar
                      optimiser (Brent's method) to get an exact answer.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pricing_engine.config import SimulationConfig
from pricing_engine.model import predict_demand

CONTEXT_COLUMNS = ["timestamp", "zone", "is_raining", "competitor_price", "drivers_available"]
PRICE_TOLERANCE = 0.011  # $; "at a bound" means within about a cent of it


@dataclass(frozen=True)
class PricingConstraints:
    """Business rules. Each one becomes a lower or upper bound on the price."""

    min_multiplier: float = 0.8  # never charge less than 80% of the base fare
    max_multiplier: float = 3.0  # surge cap: at most 3x the base fare (regulation / brand trust)
    max_competitor_premium: float = 0.5  # never more than 50% above the competitor
    max_change: float | None = None  # optional stability rule: at most +/- this fraction vs current price


@dataclass(frozen=True)
class PricingContext:
    """Everything we know about one zone at one moment."""

    timestamp: str | pd.Timestamp
    zone: str
    is_raining: bool
    competitor_price: float
    drivers_available: int
    current_price: float | None = None  # only needed for the max_change rule
    demand_multiplier: float = 1.0  # manual demand shock, e.g. 2.0 when a concert ends nearby

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dataclasses.asdict(self)])


@dataclass(frozen=True)
class PriceRecommendation:
    price: float
    surge_multiplier: float  # price / base fare
    expected_demand: float  # riders who would request at this price
    expected_rides: float  # rides we can actually serve = min(demand, capacity)
    expected_revenue: float
    fulfillment_rate: float  # expected_rides / expected_demand
    capacity: float
    lower_bound: float
    upper_bound: float
    binding_constraint: str  # which rule decided the price (see _binding_constraint)
    base_fare_revenue: float  # what charging the plain base fare would earn
    revenue_uplift: float  # (expected_revenue / base_fare_revenue) - 1


def exponential_closed_form(C: float, beta: float, capacity: float, lower: float, upper: float) -> float:
    """Exact optimum for D(p) = C * exp(-beta * p), found with pen-and-paper calculus.

    * Without a supply limit, R(p) = C p e^(-beta p) peaks where R'(p) = 0  ->  p = 1 / beta.
    * If demand at that price exceeds capacity K, raise the price until demand = K:
          C e^(-beta p) = K   ->   p = ln(C / K) / beta      (the "market-clearing" surge price)
    * R is single-peaked (log-concave), so the best price inside [lower, upper] is the
      peak, clipped to the bounds.
    """
    peak = 1.0 / beta
    if capacity > 0 and C * np.exp(-beta * peak) > capacity:
        peak = np.log(C / capacity) / beta
    return float(np.clip(peak, lower, upper))


class PriceOptimizer:
    """Turns a demand model + business rules into revenue-maximising prices."""

    def __init__(
        self,
        model,
        constraints: PricingConstraints | None = None,
        config: SimulationConfig | None = None,
        grid_size: int = 301,
    ):
        self.model = model
        self.constraints = constraints or PricingConstraints()
        self.config = config or SimulationConfig()
        self.grid_size = grid_size
        self._fares = {zone.name: zone.base_fare for zone in self.config.zones}

    # ------------------------------------------------------------------ public API
    def recommend(self, context: PricingContext, polish: bool = True) -> PriceRecommendation:
        """Best price for ONE situation (used by the API and dashboard)."""
        row = self.recommend_many(context.to_frame(), polish=polish).iloc[0]
        return PriceRecommendation(**row.to_dict())

    def recommend_many(self, contexts: pd.DataFrame, polish: bool = False, chunk_size: int = 200) -> pd.DataFrame:
        """Best price for MANY situations at once (used for backtests). One row out per row in."""
        contexts = self._prepare(contexts)
        parts = [
            self._solve(contexts.iloc[start : start + chunk_size], polish)
            for start in range(0, len(contexts), chunk_size)
        ]
        return pd.concat(parts, ignore_index=True)

    def demand(self, contexts: pd.DataFrame, prices) -> np.ndarray:
        """Model forecast of demand for each context at the given price(s).

        An optional `demand_multiplier` column scales the forecast for events the
        model can't know about (concerts, stadium games, a train strike, ...).
        """
        X = contexts[["timestamp", "zone", "is_raining", "competitor_price"]].copy()
        X["our_price"] = np.asarray(prices, dtype=float)
        multiplier = contexts["demand_multiplier"].to_numpy(dtype=float) if "demand_multiplier" in contexts else 1.0
        return predict_demand(self.model, X) * multiplier

    def price_bounds(self, contexts: pd.DataFrame) -> pd.DataFrame:
        """Turn every business rule into a bound; the tightest ones win.

        If rules conflict (lowest allowed > highest allowed), the UPPER bounds win:
        a regulator's cap beats our own price floor.
        """
        c = self.constraints
        fare = contexts["zone"].map(self._fares).to_numpy(dtype=float)
        lowers = {"min_price": fare * c.min_multiplier}
        competitor = contexts["competitor_price"].to_numpy(dtype=float)
        uppers = {
            "surge_cap": fare * c.max_multiplier,
            "competitor_premium": competitor * (1 + c.max_competitor_premium),
        }
        if c.max_change is not None:
            current = contexts["current_price"].to_numpy(dtype=float)  # NaN when unknown -> rule skipped
            lowers["price_stability"] = current * (1 - c.max_change)
            uppers["price_stability"] = current * (1 + c.max_change)
        lowers_df, uppers_df = pd.DataFrame(lowers), pd.DataFrame(uppers)
        upper = uppers_df.min(axis=1)
        return pd.DataFrame(
            {
                "lower_bound": np.minimum(lowers_df.max(axis=1), upper),
                "upper_bound": upper,
                "lower_reason": lowers_df.idxmax(axis=1),
                "upper_reason": uppers_df.idxmin(axis=1),
            }
        )

    # ------------------------------------------------------------------ internals
    def _prepare(self, contexts: pd.DataFrame) -> pd.DataFrame:
        missing = set(CONTEXT_COLUMNS) - set(contexts.columns)
        if missing:
            raise ValueError(f"contexts is missing columns: {sorted(missing)}")
        unknown = set(contexts["zone"]) - set(self._fares)
        if unknown:
            raise ValueError(f"unknown zone(s): {sorted(unknown)}")
        prepared = contexts.reset_index(drop=True).copy()
        if "current_price" not in prepared:
            prepared["current_price"] = np.nan
        prepared["current_price"] = pd.to_numeric(prepared["current_price"])
        if "demand_multiplier" not in prepared:
            prepared["demand_multiplier"] = 1.0
        return prepared

    def _solve(self, ctx: pd.DataFrame, polish: bool) -> pd.DataFrame:
        n, g = len(ctx), self.grid_size
        bounds = self.price_bounds(ctx)
        lo, hi = bounds["lower_bound"].to_numpy(), bounds["upper_bound"].to_numpy()
        capacity = ctx["drivers_available"].to_numpy(dtype=float) * self.config.rides_per_driver_per_hour

        # 1) Global search: an (n contexts x g prices) table of revenue, from a single model call.
        prices = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, g)[None, :]
        expanded = ctx.loc[ctx.index.repeat(g)].reset_index(drop=True)
        demand = self.demand(expanded, prices.ravel()).reshape(n, g)
        revenue = prices * np.minimum(demand, capacity[:, None])
        best = revenue.argmax(axis=1)
        price = prices[np.arange(n), best]

        # 2) Local polish with SciPy inside the winning grid cell.
        if polish:
            for i in range(n):
                price[i] = self._polish(ctx.iloc[[i]], prices[i], best[i], capacity[i])

        # Real prices are in cents, and must stay inside the bounds.
        price = np.clip(np.round(price, 2), np.ceil(lo * 100) / 100, np.floor(hi * 100) / 100)

        fare = ctx["zone"].map(self._fares).to_numpy(dtype=float)
        demand_star = self.demand(ctx, price)
        rides = np.minimum(demand_star, capacity)
        revenue_star = price * rides
        base_revenue = fare * np.minimum(self.demand(ctx, fare), capacity)
        return pd.DataFrame(
            {
                "price": price,
                "surge_multiplier": np.round(price / fare, 3),
                "expected_demand": demand_star,
                "expected_rides": rides,
                "expected_revenue": revenue_star,
                "fulfillment_rate": np.divide(rides, demand_star, out=np.ones(n), where=demand_star > 0),
                "capacity": capacity,
                "lower_bound": lo,
                "upper_bound": hi,
                "binding_constraint": self._binding_constraint(price, lo, hi, bounds, demand_star, capacity),
                "base_fare_revenue": base_revenue,
                "revenue_uplift": np.divide(revenue_star, base_revenue, out=np.zeros(n), where=base_revenue > 0) - 1,
            }
        )

    def _polish(self, ctx_row: pd.DataFrame, grid: np.ndarray, best: int, capacity: float) -> float:
        left, right = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        if right - left < 1e-9:
            return float(grid[best])

        def negative_revenue(p: float) -> float:  # SciPy MINIMISES, so we flip the sign
            return -p * min(self.demand(ctx_row, [p])[0], capacity)

        result = minimize_scalar(negative_revenue, bounds=(left, right), method="bounded", options={"xatol": 1e-3})
        # Keep the polished price only if it really is better than the grid winner.
        return float(result.x) if result.fun < negative_revenue(grid[best]) else float(grid[best])

    @staticmethod
    def _binding_constraint(price, lo, hi, bounds, demand, capacity) -> np.ndarray:
        """Explain WHY the price is what it is (this is the KKT idea in plain words):

        either some rule stopped us (a bound is "binding"), supply stopped us
        (demand = capacity), or we sit at the natural peak where marginal revenue = 0.
        """
        at_upper = price >= hi - PRICE_TOLERANCE
        at_lower = price <= lo + PRICE_TOLERANCE
        supply_limited = np.abs(demand / np.maximum(capacity, 1e-9) - 1) < 0.02
        return np.select(
            [capacity <= 0, at_upper, at_lower, supply_limited],
            ["no_drivers", bounds["upper_reason"].to_numpy(), bounds["lower_reason"].to_numpy(), "supply"],
            default="none (natural peak: marginal revenue = 0)",
        )
