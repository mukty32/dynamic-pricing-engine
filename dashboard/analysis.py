"""Pure math and text helpers for the dashboard (no Streamlit, so they're easy to test)."""

from __future__ import annotations

import math

import numpy as np


def fit_exponential(prices, demand) -> tuple[float, float]:
    """Recover beta and C from a demand curve, assuming D(p) = C * exp(-beta * p).

    Taking logs turns the curve into a straight line:  ln D = ln C - beta * p.
    A least-squares line fit then gives slope = -beta and intercept = ln C.
    """
    prices, demand = np.asarray(prices, dtype=float), np.asarray(demand, dtype=float)
    mask = demand > 0
    slope, intercept = np.polyfit(prices[mask], np.log(demand[mask]), deg=1)
    return float(-slope), float(math.exp(intercept))


def pen_and_paper_optimum(beta: float, C: float, capacity: float, lower: float, upper: float) -> dict:
    """The Phase 3 calculus, step by step, so the dashboard can show its working."""
    natural_peak = 1.0 / beta
    market_clearing = math.log(C / capacity) / beta if capacity > 0 else None
    candidate = natural_peak
    if market_clearing is not None and market_clearing > natural_peak:
        candidate = market_clearing
    return {
        "natural_peak": natural_peak,
        "market_clearing": market_clearing,
        "candidate": candidate,
        "optimum": min(max(candidate, lower), upper),
    }


def explain_binding(
    binding: str, *, price: float, capacity: float, competitor_price: float, base_fare: float, rules: dict
) -> tuple[str, str]:
    """Plain-language answer to 'why this price?'. Returns (headline, explanation)."""
    if binding.startswith("none"):
        return (
            "Natural revenue peak",
            "There are enough drivers. Raising the price further would lose more riders than it earns, "
            "so this is where marginal revenue hits zero (p = 1/β).",
        )
    explanations = {
        "supply": (
            "Supply-limited: market-clearing surge",
            f"Only {capacity:.0f} rides/hour can be served. The price rises until expected demand "
            f"equals that capacity. Any lower and riders would be turned away anyway.",
        ),
        "surge_cap": (
            "Hit the surge cap",
            f"The optimizer wanted to go higher, but prices may not exceed "
            f"{rules['max_multiplier']:.1f}× the ${base_fare:.2f} base fare.",
        ),
        "competitor_premium": (
            "Limited by the competitor rule",
            f"Prices may be at most {rules['max_competitor_premium']:.0%} above the "
            f"competitor's ${competitor_price:.2f}, so ${price:.2f} is the ceiling.",
        ),
        "min_price": (
            "Held at the price floor",
            f"Prices may not go below {rules['min_multiplier']:.0%} of the base fare.",
        ),
        "price_stability": (
            "Limited by the price-stability rule",
            "The price may only move a limited percentage from the current price.",
        ),
        "no_drivers": ("No drivers online", "Nothing can be served, so no price earns revenue."),
    }
    return explanations.get(binding, (binding, ""))
