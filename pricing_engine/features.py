"""Phase 2: feature engineering, turning raw rows into numbers a model can learn from.

Every model receives the SAME raw input columns (MODEL_INPUT_COLUMNS). Each model
then builds its own "view" of the data:

  * glm_design_matrix   -> for the Poisson regression (a formula-like, interpretable model)
  * tree_design_matrix  -> for XGBoost (a flexible, tree-based model)

Keeping the raw input identical matters for Phase 4: the API only has to send
these five fields, whichever model is the champion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing_engine.config import SimulationConfig

_DEFAULT_CONFIG = SimulationConfig()
REFERENCE_DATE = pd.Timestamp(_DEFAULT_CONFIG.start_date)  # "day 0" of our business
ZONES: tuple[str, ...] = tuple(zone.name for zone in _DEFAULT_CONFIG.zones)

# What the model needs to know about a situation. Notice what is NOT here:
# drivers_available, rides_completed and revenue. They are outcomes/supply, not
# drivers of rider demand, and including outcomes would be "data leakage".
MODEL_INPUT_COLUMNS = ["timestamp", "zone", "is_raining", "our_price", "competitor_price"]
TARGET_COLUMN = "demand"


def cyclical_encode(values, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Map a repeating quantity onto a circle: angle = 2*pi * value / period.

    As raw numbers, hour 23 and hour 0 look far apart (23 vs 0). On the circle
    they are neighbours, which is the truth.
    """
    angle = 2 * np.pi * np.asarray(values, dtype=float) / period
    return np.sin(angle), np.cos(angle)


def calendar_features(X: pd.DataFrame) -> pd.DataFrame:
    """Derive time features from the timestamp."""
    ts = pd.to_datetime(X["timestamp"])
    doy_sin, doy_cos = cyclical_encode(ts.dt.day_of_year.to_numpy(), 365)
    return pd.DataFrame(
        {
            "hour": ts.dt.hour.to_numpy(),
            "is_weekend": (ts.dt.day_of_week >= 5).to_numpy(),
            # time in years since launch -> lets the model learn business growth
            "years_since_start": ((ts - REFERENCE_DATE).dt.days / 365).to_numpy(),
            # the day of the year as a point on a circle -> Dec 31 sits next to Jan 1
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
        },
        index=X.index,
    )


def glm_design_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Columns for the Poisson GLM, where   log(demand) = w . features.

    Each column is chosen so that its weight has a clear meaning:
      zone_*          -> how much bigger/smaller each zone is than downtown
      slot_*          -> one switch per (weekday/weekend, hour): the daily rhythm
      price_x_<zone>  -> the price sensitivity (-beta) of each zone separately
      competitor_price-> the cross-price effect (+gamma)
    """
    cal = calendar_features(X)
    zone = X["zone"].to_numpy()
    columns: dict[str, np.ndarray] = {}

    # One-hot zones. Downtown is the "baseline" (absorbed by the intercept), so we drop it.
    for name in ZONES[1:]:
        columns[f"zone_{name}"] = (zone == name).astype(float)

    # One-hot time slots: 2 day types x 24 hours = 48 slots, minus 1 baseline (weekday midnight).
    is_weekend = cal["is_weekend"].to_numpy()
    hour = cal["hour"].to_numpy()
    for weekend_flag, label in ((False, "wd"), (True, "we")):
        for h in range(24):
            if label == "wd" and h == 0:
                continue
            columns[f"slot_{label}_{h:02d}"] = ((is_weekend == weekend_flag) & (hour == h)).astype(float)

    columns["is_raining"] = X["is_raining"].to_numpy(dtype=float)
    price = X["our_price"].to_numpy(dtype=float)
    for name in ZONES:
        columns[f"price_x_{name}"] = price * (zone == name)
    columns["competitor_price"] = X["competitor_price"].to_numpy(dtype=float)
    columns["years_since_start"] = cal["years_since_start"].to_numpy()
    columns["doy_sin"] = cal["doy_sin"].to_numpy()
    columns["doy_cos"] = cal["doy_cos"].to_numpy()
    return pd.DataFrame(columns, index=X.index)


def tree_design_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Columns for XGBoost. Trees discover interactions and curves on their own,
    so we give them plainer features."""
    cal = calendar_features(X)
    features = pd.DataFrame(
        {
            "hour": cal["hour"],
            "is_weekend": cal["is_weekend"].astype(int),
            "is_raining": X["is_raining"].astype(int),
            "our_price": X["our_price"].astype(float),
            "competitor_price": X["competitor_price"].astype(float),
            "years_since_start": cal["years_since_start"],
            "doy_sin": cal["doy_sin"],
            "doy_cos": cal["doy_cos"],
        },
        index=X.index,
    )
    for name in ZONES:
        features[f"zone_{name}"] = (X["zone"] == name).astype(int)
    return features
