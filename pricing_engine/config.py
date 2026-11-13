"""Central configuration for the synthetic ride-hailing market.

Every "magic number" in the simulation lives here. That gives us two benefits:
  1. The simulation is easy to tweak without touching the logic.
  2. Later phases (model, optimizer, API) can import the SAME ground-truth
     parameters, so we can check how close our ML model gets to the truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Paths are built from this file's location, so the code works no matter
# which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_DATASET_PATH = RAW_DATA_DIR / "rides.csv"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "demand_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLOPS_DIR = PROJECT_ROOT / "mlops"  # MLflow tracking database + artifacts (local, not committed)


@dataclass(frozen=True)
class ZoneConfig:
    """Parameters for one area of the city.

    frozen=True makes the object read-only: nobody can accidentally change
    the "laws of physics" of our simulated market halfway through a run.
    """

    name: str
    base_demand: float  # ride requests per hour at a busy hour, when everyone charges base_fare
    base_supply: float  # drivers online at the busiest supply hour (~2 pm) on a dry day
    base_fare: float  # our normal, non-surge price in $ (also the reference price)
    price_sensitivity: float  # beta: how strongly riders react to OUR price


@dataclass(frozen=True)
class SimulationConfig:
    """All knobs of the simulation in one place."""

    # --- Time & reproducibility ---
    start_date: str = "2025-01-01"
    n_days: int = 365
    seed: int = 42

    # --- Demand ---
    cross_price_sensitivity: float = 0.04  # gamma: how much a competitor's price pushes riders to us
    annual_growth: float = 0.10  # the business grows 10% over a year
    rain_probability: float = 0.20  # chance that any given day is rainy
    rain_demand_boost: float = 1.30  # rain -> 30% more ride requests

    # --- Supply ---
    rides_per_driver_per_hour: int = 2
    rain_supply_factor: float = 0.85  # rain -> 15% fewer drivers online

    # --- Our historical pricing policy (the "old system" that produced the data) ---
    surge_sensitivity: float = 0.5  # surge = 1 + 0.5 * (demand/supply - 1)
    max_surge: float = 2.5
    price_experiment_noise: float = 0.10  # random +/-10% price tests (A/B experiments)

    # --- Competitor ---
    competitor_surge_sensitivity: float = 0.4
    competitor_max_surge: float = 2.2
    competitor_price_noise: float = 0.08  # sigma of the log-normal noise
    competitor_price_level: float = 1.0  # < 1 means the competitor got cheaper (used for drift scenarios)

    zones: tuple[ZoneConfig, ...] = (
        ZoneConfig("downtown", base_demand=120, base_supply=45, base_fare=12.0, price_sensitivity=0.080),
        ZoneConfig("airport", base_demand=50, base_supply=20, base_fare=25.0, price_sensitivity=0.035),
        ZoneConfig("suburbs", base_demand=35, base_supply=15, base_fare=10.0, price_sensitivity=0.100),
    )
