"""Phase 9: model monitoring. Is the world still the one our model learned?

Three kinds of drift, from "something changed" to "the model is now wrong":

1. DATA DRIFT: the inputs look different (e.g. the competitor got cheaper).
   Measured with PSI and the Kolmogorov-Smirnov test. A WARNING, not an alarm:
   the model may handle the new inputs perfectly well.

2. PERFORMANCE DRIFT: the forecasts got worse (WAPE up) or biased (too high or low).

3. CONCEPT DRIFT: the RELATIONSHIP changed, e.g. riders became more price-sensitive
   (beta moved). Detected with the Phase 7 Bayesian learner: we fit a small
   correction on top of the model and check if it's significantly different from zero.

Performance or concept drift -> ALERT -> retrain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pricing_engine.config import SimulationConfig
from pricing_engine.model import predict_demand
from pricing_engine.online_learning import BayesianPriceLearner
from pricing_engine.train import estimate_price_sensitivity

PSI_WARNING, PSI_ALERT = 0.10, 0.25  # industry rule of thumb
NUMERIC_FEATURES = ["competitor_price", "our_price", "drivers_available", "demand"]
CATEGORICAL_FEATURES = ["is_raining"]
WAPE_TOLERANCE = 1.20  # alert if forecast error grows by more than 20%
BIAS_TOLERANCE = 0.05  # alert if forecasts are systematically off by more than 5%
MIN_BETA_CHANGE = 0.005  # ignore statistically significant but tiny beta changes


# ----------------------------------------------------------------------------- 1. data drift
def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index:  PSI = sum_i (c_i - r_i) * ln(c_i / r_i)

    r_i, c_i = share of reference / current values falling in bin i. Bins are the
    reference's deciles, so each holds ~10% of the reference. PSI is a symmetric
    version of KL divergence: 0 when the distributions match, growing as they differ.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def categorical_psi(reference, current, eps: float = 1e-4) -> float:
    reference, current = pd.Series(reference), pd.Series(current)  # also accepts plain lists
    categories = sorted(set(reference.unique()) | set(current.unique()), key=str)
    r = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    c = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_status(value: float) -> str:
    return "alert" if value >= PSI_ALERT else "warning" if value >= PSI_WARNING else "ok"


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_FEATURES:
        ks = stats.ks_2samp(reference[feature], current[feature])
        value = psi(reference[feature], current[feature])
        rows.append({"feature": feature, "psi": value, "ks_statistic": float(ks.statistic),
                     "ks_pvalue": float(ks.pvalue), "status": psi_status(value)})  # fmt: skip
    for feature in CATEGORICAL_FEATURES:
        value = categorical_psi(reference[feature], current[feature])
        rows.append({"feature": feature, "psi": value, "ks_statistic": np.nan, "ks_pvalue": np.nan,
                     "status": psi_status(value)})  # fmt: skip
    return pd.DataFrame(rows).set_index("feature")


# ----------------------------------------------------------------------------- 2. performance drift
def performance_check(model, current: pd.DataFrame, reference_wape: float) -> dict:
    predicted = predict_demand(model, current)
    actual = current["demand"].to_numpy(dtype=float)
    wape = float(np.abs(actual - predicted).sum() / actual.sum())
    bias = float(actual.sum() / predicted.sum())  # 1.0 = unbiased; 1.1 = model under-forecasts by 10%
    problems = []
    if wape > WAPE_TOLERANCE * reference_wape:
        problems.append(f"WAPE {wape:.1%} is above {WAPE_TOLERANCE:.0%} of the reference {reference_wape:.1%}")
    if abs(bias - 1) > BIAS_TOLERANCE:
        problems.append(f"forecasts are biased: actual/predicted = {bias:.3f}")
    return {"wape": wape, "reference_wape": reference_wape, "bias": bias,
            "status": "alert" if problems else "ok", "problems": problems}  # fmt: skip


# ----------------------------------------------------------------------------- 3. concept drift
def price_sensitivity_check(model, current: pd.DataFrame, config: SimulationConfig | None = None,
                            z: float = 1.96) -> pd.DataFrame:  # fmt: skip
    """Has beta changed? Fit   demand ~ Poisson( model_forecast * exp(a - b * (price - fare)) ).

    If the model is still right, the correction b is ~0. A 95% interval for b that
    excludes 0 (and isn't trivially small) means riders' price sensitivity has changed:
    the true beta is now about  model_beta + b.
    """
    config = config or SimulationConfig()
    rows = []
    for zone in config.zones:
        window = current[current["zone"] == zone.name]
        if window.empty:
            continue
        forecast = predict_demand(model, window)
        learner = BayesianPriceLearner(prior_mean=(0.0, 0.0), prior_sd=(1.0, 0.5))  # weak prior: let data speak
        for m, price, y in zip(forecast, window["our_price"], window["demand"], strict=True):
            learner.observe(max(m, 1e-6), price - zone.base_fare, int(y))
        mean, cov = learner.posterior()
        change, sd = float(mean[1]), float(np.sqrt(cov[1, 1]))
        model_beta = float(estimate_price_sensitivity(lambda X: predict_demand(model, X), window)[zone.name])
        significant = abs(change) > z * sd and abs(change) > MIN_BETA_CHANGE
        rows.append({"zone": zone.name, "model_beta": model_beta, "estimated_beta": model_beta + change,
                     "ci_low": model_beta + change - z * sd, "ci_high": model_beta + change + z * sd,
                     "status": "alert" if significant else "ok"})  # fmt: skip
    return pd.DataFrame(rows).set_index("zone")


# ----------------------------------------------------------------------------- the report
def drift_report(model, reference: pd.DataFrame, current: pd.DataFrame, reference_wape: float,
                 config: SimulationConfig | None = None) -> dict:  # fmt: skip
    data = feature_drift(reference, current)
    performance = performance_check(model, current, reference_wape)
    sensitivity = price_sensitivity_check(model, current, config)

    reasons = list(performance["problems"])
    for zone, row in sensitivity[sensitivity["status"] == "alert"].iterrows():
        reasons.append(f"{zone}: beta changed {row.model_beta:.3f} -> {row.estimated_beta:.3f}")
    warnings = [f"{feature}: PSI {row.psi:.2f}" for feature, row in data.iterrows() if row.status != "ok"]
    status = "alert" if reasons else "warning" if warnings else "ok"
    return {"status": status, "reasons": reasons, "warnings": warnings,
            "data_drift": data, "performance": performance, "price_sensitivity": sensitivity}  # fmt: skip
