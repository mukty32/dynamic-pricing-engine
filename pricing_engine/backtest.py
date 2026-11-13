"""Phase 3: backtest. How much more money would the optimizer have made?

Run from the project root (after Phases 1 and 2):

    python -m pricing_engine.backtest

We replay every hour of the test period (Nov-Dec) under four pricing policies
and score each one with the TRUE market formula from the simulator:

    static base fare        -> no dynamic pricing at all
    historical surge rule   -> the old rule-of-thumb that generated our data
    optimizer + GLM model   -> OUR engine, using the Phase 2 demand forecast
    optimizer + true demand -> the same engine with perfect knowledge (the "oracle")

In real life you can't do this (nobody knows the true formula); companies run
A/B tests instead. A simulator lets us measure the counterfactual exactly.
Limitation: the competitor's prices are replayed as-is (they don't react to us).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pricing_engine.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, REPORTS_DIR, SimulationConfig
from pricing_engine.data_generation import true_expected_demand
from pricing_engine.model import load_model
from pricing_engine.optimizer import CONTEXT_COLUMNS, PriceOptimizer, PricingContext
from pricing_engine.train import DEFAULT_TEST_START, load_dataset, time_split

logger = logging.getLogger(__name__)


class TrueDemandModel:
    """Wraps the simulator's ground-truth formula so it looks like a trained model."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return true_expected_demand(X, self.config)


def score_policy(contexts: pd.DataFrame, prices, config: SimulationConfig) -> dict[str, float]:
    """Evaluate a set of prices with the TRUE market (expected values, no noise)."""
    prices = np.asarray(prices, dtype=float)
    X = contexts[["timestamp", "zone", "is_raining", "competitor_price"]].copy()
    X["our_price"] = prices
    demand = true_expected_demand(X, config)
    capacity = contexts["drivers_available"].to_numpy(dtype=float) * config.rides_per_driver_per_hour
    rides = np.minimum(demand, capacity)
    return {
        "avg_price": prices.mean(),
        "total_revenue": (prices * rides).sum(),
        "rides_served": rides.sum(),
        "fulfillment_rate": rides.sum() / demand.sum(),
    }


def run_backtest(test: pd.DataFrame, model, config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    contexts = test[CONTEXT_COLUMNS].reset_index(drop=True)
    fares = contexts["zone"].map({z.name: z.base_fare for z in config.zones})

    logger.info("Optimizing %s hourly prices with the GLM model ...", f"{len(contexts):,}")
    ours = PriceOptimizer(model, config=config).recommend_many(contexts)
    logger.info("Optimizing the same hours with the true demand formula (oracle) ...")
    oracle = PriceOptimizer(TrueDemandModel(config), config=config).recommend_many(contexts)

    policies = {
        "static base fare": fares,
        "historical surge rule": test["our_price"],
        "optimizer + GLM model": ours["price"],
        "optimizer + true demand (oracle)": oracle["price"],
    }
    table = pd.DataFrame({name: score_policy(contexts, prices, config) for name, prices in policies.items()}).T
    table["uplift_vs_historical"] = table["total_revenue"] / table.loc["historical surge rule", "total_revenue"] - 1
    return table, ours


def demo_recommendations(optimizer: PriceOptimizer) -> pd.DataFrame:
    """A few hand-picked situations to show WHY the engine picks each price."""
    scenarios = {
        "downtown 3am, quiet": PricingContext("2025-11-18 03:00", "downtown", False, 12.0, 20),
        "downtown 6pm, 200 drivers": PricingContext("2025-11-18 18:00", "downtown", False, 18.0, 200),
        "downtown 6pm, 40 drivers": PricingContext("2025-11-18 18:00", "downtown", False, 18.0, 40),
        "downtown 6pm, 5 drivers": PricingContext("2025-11-18 18:00", "downtown", False, 18.0, 5),
        "downtown 6pm, 5 drivers, rival surging": PricingContext("2025-11-18 18:00", "downtown", False, 30.0, 5),
        "airport 6pm, rain, cheap rival": PricingContext("2025-11-18 18:00", "airport", True, 22.0, 15),
    }
    rows = {}
    for name, context in scenarios.items():
        rec = optimizer.recommend(context)
        rows[name] = {
            "drivers": context.drivers_available,
            "price": rec.price,
            "surge": rec.surge_multiplier,
            "demand": round(rec.expected_demand, 1),
            "capacity": rec.capacity,
            "revenue": round(rec.expected_revenue, 2),
            "binding": rec.binding_constraint,
        }
    return pd.DataFrame(rows).T


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Backtest pricing policies on the test period.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    args = parser.parse_args(argv)

    config = SimulationConfig()
    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} not found. Run `python -m pricing_engine.train` first.")
    model = load_model(args.model)
    _, test = time_split(load_dataset(args.data), args.test_start)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n1) Example recommendations (Tuesday, champion model):")
        print(demo_recommendations(PriceOptimizer(model, config=config)).to_string())

        table, ours = run_backtest(test, model, config)
        print("\n2) Backtest on the test period, scored with the TRUE market:")
        print(
            table.round(
                {
                    "avg_price": 2,
                    "total_revenue": 0,
                    "rides_served": 0,
                    "fulfillment_rate": 3,
                    "uplift_vs_historical": 4,
                }
            ).to_string()
        )
        print("\n3) How often each rule decided the optimizer's price:")
        print(ours["binding_constraint"].value_counts(normalize=True).round(3).to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "backtest.json"
    out.write_text(
        json.dumps({"test_start": args.test_start, "policies": table.round(4).to_dict(orient="index")}, indent=2)
    )
    logger.info("Saved backtest -> %s", out)


if __name__ == "__main__":
    main()
