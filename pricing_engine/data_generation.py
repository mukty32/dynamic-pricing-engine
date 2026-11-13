"""Phase 1: synthetic ride-hailing market simulator.

Run it from the project root:

    python -m pricing_engine.data_generation
    python -m pricing_engine.data_generation --days 90 --seed 7 --output data/raw/small.csv

It creates one row per (hour, zone) containing:
    supply   -> how many drivers were online
    prices   -> our price and a competitor's price
    demand   -> how many riders requested a ride  (what Phase 2 will forecast)
    outcome  -> rides actually completed and the revenue they earned

The core "law of physics" of this market (the ground truth) is:

    expected_demand = base_demand
                      x time_pattern(hour, weekend)
                      x seasonality x growth x weather
                      x exp(-beta  * (our_price        - base_fare))
                      x exp(+gamma * (competitor_price - base_fare))

    demand ~ Poisson(expected_demand)
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pricing_engine.config import DEFAULT_DATASET_PATH, SimulationConfig

logger = logging.getLogger(__name__)

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365

EXPECTED_COLUMNS = [
    "timestamp",
    "zone",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "is_raining",
    "drivers_available",
    "competitor_price",
    "our_price",
    "demand",
    "rides_completed",
    "revenue",
]


# ---------------------------------------------------------------------------
# 1. Pure math functions: no randomness, so they're easy to test and we can
#    reuse them in Phase 3 to check the optimizer against the true curve.
# ---------------------------------------------------------------------------
def circular_bump(hour, center: float, width: float) -> np.ndarray:
    """A bell curve (Gaussian) on a 24-hour clock.

    Equals 1.0 at `center` and fades smoothly to 0 further away. "Circular"
    means 23:00 counts as 1 hour away from 00:00, not 23 hours.
    """
    hour = np.asarray(hour, dtype=float)
    distance = np.abs(hour - center)
    distance = np.minimum(distance, HOURS_PER_DAY - distance)
    return np.exp(-0.5 * (distance / width) ** 2)


def demand_time_pattern(hour, is_weekend) -> np.ndarray:
    """How busy each hour is, as a multiplier (about 0.15 at night to about 1.2 at rush hour)."""
    weekday = (
        0.15
        + 1.00 * circular_bump(hour, center=8, width=1.5)  # morning commute
        + 0.35 * circular_bump(hour, center=13, width=3)  # midday errands
        + 1.10 * circular_bump(hour, center=18, width=2)  # evening commute
    )
    weekend = (
        0.20
        + 0.50 * circular_bump(hour, center=11, width=3)  # late brunch
        + 0.70 * circular_bump(hour, center=19, width=3)  # evening out
        + 0.90 * circular_bump(hour, center=23.5, width=2)  # nightlife
    )
    return np.where(is_weekend, weekend, weekday)


def supply_time_pattern(hour) -> np.ndarray:
    """Share of drivers online: most drivers work around midday, few at night.

    Supply peaks at 2 pm while demand peaks at 8 am and 6 pm. That mismatch is
    exactly what creates shortages, and therefore surge pricing.
    """
    return 0.35 + 0.65 * circular_bump(hour, center=14, width=5)


def seasonality(day_of_year) -> np.ndarray:
    """A gentle yearly wave: busiest around late December, about +/-8%."""
    day_of_year = np.asarray(day_of_year, dtype=float)
    return 1.0 + 0.08 * np.cos(2 * np.pi * (day_of_year - 355) / DAYS_PER_YEAR)


def price_response(price, competitor_price, reference_price, beta, gamma) -> np.ndarray:
    """Multiplier on demand caused by prices. Equals 1.0 when both prices equal the reference.

    exp(-beta * (p - p_ref)): each extra $1 on OUR price keeps a share e^-beta of riders.
    exp(+gamma * (c - p_ref)): each extra $1 on the COMPETITOR's price sends us e^gamma more.
    """
    price = np.asarray(price, dtype=float)
    competitor_price = np.asarray(competitor_price, dtype=float)
    return np.exp(-beta * (price - reference_price) + gamma * (competitor_price - reference_price))


def demand_before_price(hour, is_weekend, day_of_year, day_index, is_raining, base_demand, config) -> np.ndarray:
    """What the market 'wants' this hour, before any price effect is applied."""
    growth = 1.0 + config.annual_growth * np.asarray(day_index, dtype=float) / DAYS_PER_YEAR
    weather = np.where(is_raining, config.rain_demand_boost, 1.0)
    return base_demand * demand_time_pattern(hour, is_weekend) * seasonality(day_of_year) * growth * weather


def true_expected_demand(context: pd.DataFrame, config: SimulationConfig | None = None) -> np.ndarray:
    """The GROUND TRUTH average demand for any situation.

    `context` needs the columns: timestamp, zone, is_raining, our_price, competitor_price.
    In the real world nobody knows this function; we only have it because we
    built the simulation. We use it to grade how well our ML model learned.
    """
    config = config or SimulationConfig()
    zones = {zone.name: zone for zone in config.zones}
    ts = pd.to_datetime(context["timestamp"])
    zone_of_row = context["zone"].map(zones)

    def param(attribute: str) -> np.ndarray:
        return zone_of_row.map(lambda zone: getattr(zone, attribute)).to_numpy(dtype=float)

    base_fare = param("base_fare")
    day_index = (ts - pd.Timestamp(config.start_date)).dt.days.to_numpy()
    before_price = demand_before_price(
        hour=ts.dt.hour.to_numpy(),
        is_weekend=ts.dt.day_of_week.to_numpy() >= 5,
        day_of_year=ts.dt.day_of_year.to_numpy(),
        day_index=day_index,
        is_raining=context["is_raining"].to_numpy(dtype=bool),
        base_demand=param("base_demand"),
        config=config,
    )
    return before_price * price_response(
        context["our_price"],
        context["competitor_price"],
        base_fare,
        param("price_sensitivity"),
        config.cross_price_sensitivity,
    )


# ---------------------------------------------------------------------------
# 2. The simulator
# ---------------------------------------------------------------------------
def generate_dataset(config: SimulationConfig | None = None) -> pd.DataFrame:
    """Simulate `config.n_days` days of hourly market data for every zone."""
    config = config or SimulationConfig()
    # One random generator, seeded -> the exact same dataset every run.
    rng = np.random.default_rng(config.seed)

    # --- Build the grid: every hour x every zone ---
    timestamps = pd.date_range(config.start_date, periods=config.n_days * HOURS_PER_DAY, freq="h")
    zone_names = [zone.name for zone in config.zones]
    grid = pd.MultiIndex.from_product([timestamps, zone_names], names=["timestamp", "zone"]).to_frame(index=False)
    n_rows = len(grid)

    def zone_param(attribute: str) -> np.ndarray:
        """Look up one zone parameter for every row, e.g. base_fare for each row's zone."""
        lookup = {zone.name: getattr(zone, attribute) for zone in config.zones}
        return grid["zone"].map(lookup).to_numpy(dtype=float)

    base_demand = zone_param("base_demand")
    base_supply = zone_param("base_supply")
    base_fare = zone_param("base_fare")
    beta = zone_param("price_sensitivity")

    # --- Calendar features ---
    ts = grid["timestamp"]
    hour = ts.dt.hour.to_numpy()
    day_of_week = ts.dt.day_of_week.to_numpy()  # Monday=0 ... Sunday=6
    is_weekend = day_of_week >= 5
    day_index = (ts - ts.min()).dt.days.to_numpy()  # 0, 1, 2, ... days since start

    # --- Weather: each day is rainy or dry (city-wide) ---
    rainy_days = rng.random(config.n_days) < config.rain_probability
    is_raining = rainy_days[day_index]

    # --- Supply: drivers online (Poisson = random count around an average) ---
    expected_drivers = base_supply * supply_time_pattern(hour) * np.where(is_raining, config.rain_supply_factor, 1.0)
    drivers_available = rng.poisson(expected_drivers)
    capacity = drivers_available * config.rides_per_driver_per_hour

    # --- Demand BEFORE prices are applied (what the market "wants" this hour) ---
    market_demand = demand_before_price(
        hour, is_weekend, ts.dt.day_of_year.to_numpy(), day_index, is_raining, base_demand, config
    )

    # --- Historical pricing: a simple surge rule based on demand / supply ---
    pressure = market_demand / np.maximum(capacity, 1)  # >1 means a shortage
    our_surge = np.clip(1 + config.surge_sensitivity * (pressure - 1), 1.0, config.max_surge)
    experiment = rng.uniform(1 - config.price_experiment_noise, 1 + config.price_experiment_noise, n_rows)
    our_price = np.round(base_fare * our_surge * experiment, 2)

    competitor_surge = np.clip(
        1 + config.competitor_surge_sensitivity * (pressure - 1), 1.0, config.competitor_max_surge
    )
    competitor_noise = rng.lognormal(mean=0.0, sigma=config.competitor_price_noise, size=n_rows)
    competitor_price = np.round(base_fare * competitor_surge * competitor_noise, 2)

    # --- Final demand: riders react to our price and the competitor's price ---
    expected_demand = market_demand * price_response(
        our_price, competitor_price, base_fare, beta, config.cross_price_sensitivity
    )
    demand = rng.poisson(expected_demand)

    # --- Outcome: we can't complete more rides than drivers can handle ---
    rides_completed = np.minimum(demand, capacity)
    revenue = np.round(rides_completed * our_price, 2)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "zone": grid["zone"],
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "month": ts.dt.month.to_numpy(),
            "is_raining": is_raining,
            "drivers_available": drivers_available,
            "competitor_price": competitor_price,
            "our_price": our_price,
            "demand": demand,
            "rides_completed": rides_completed,
            "revenue": revenue,
        }
    )
    validate_dataset(df, config)
    return df


# ---------------------------------------------------------------------------
# 3. Data quality checks: fail loudly instead of training on broken data
# ---------------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame, config: SimulationConfig | None = None) -> None:
    """Raise ValueError if the dataset breaks any rule that must always hold."""
    config = config or SimulationConfig()
    problems: list[str] = []

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[EXPECTED_COLUMNS].isna().any().any():
        problems.append("dataset contains missing values")

    numeric = ["drivers_available", "competitor_price", "our_price", "demand", "rides_completed", "revenue"]
    if (df[numeric] < 0).any().any():
        problems.append("negative values found")
    if (df["our_price"] <= 0).any() or (df["competitor_price"] <= 0).any():
        problems.append("prices must be strictly positive")
    if (df["rides_completed"] > df["demand"]).any():
        problems.append("completed more rides than were requested")
    if (df["rides_completed"] > df["drivers_available"] * config.rides_per_driver_per_hour).any():
        problems.append("completed more rides than drivers could handle")
    if df.duplicated(subset=["timestamp", "zone"]).any():
        problems.append("duplicate (timestamp, zone) rows")

    if problems:
        raise ValueError("Dataset failed validation: " + "; ".join(problems))


def save_dataset(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-zone averages: a quick check that the numbers look sensible."""
    summary = df.groupby("zone").agg(
        avg_price=("our_price", "mean"),
        avg_competitor=("competitor_price", "mean"),
        avg_demand=("demand", "mean"),
        avg_drivers=("drivers_available", "mean"),
        total_revenue=("revenue", "sum"),
    )
    summary["fulfillment_rate"] = df.groupby("zone")["rides_completed"].sum() / df.groupby("zone")["demand"].sum()
    return summary.round(2)


# ---------------------------------------------------------------------------
# 4. Command-line entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(description="Generate the synthetic ride-hailing dataset.")
    parser.add_argument("--days", type=int, default=defaults.n_days, help="number of days to simulate")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="random seed (same seed = same data)")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH, help="where to save the CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    config = dataclasses.replace(SimulationConfig(), n_days=args.days, seed=args.seed)

    logger.info("Simulating %d days for %d zones (seed=%d)...", config.n_days, len(config.zones), config.seed)
    df = generate_dataset(config)
    path = save_dataset(df, args.output)

    logger.info("Saved %s rows x %s columns to %s", f"{len(df):,}", df.shape[1], path)
    logger.info("Date range: %s -> %s", df["timestamp"].min(), df["timestamp"].max())
    print("\nPer-zone summary:\n", summarize(df).to_string(), sep="")
    print("\nFirst rows:\n", df.head(6).to_string(index=False), sep="")


if __name__ == "__main__":
    main()
