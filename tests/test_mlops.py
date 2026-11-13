"""Tests for Phase 9: drift monitoring, MLflow tracking/registry and the retrain loop. Run with:  pytest"""

import json

import numpy as np
import pandas as pd
import pytest

from pricing_engine import tracking
from pricing_engine.backtest import TrueDemandModel
from pricing_engine.config import SimulationConfig
from pricing_engine.mlops_pipeline import (
    drifted_config,
    ensure_champion,
    recency_weights,
    run_cycle,
    should_promote,
    simulate_window,
    train_challenger,
)
from pricing_engine.model import build_models, predict_demand, save_model
from pricing_engine.monitoring import categorical_psi, feature_drift, price_sensitivity_check, psi

CONFIG = SimulationConfig()


# ----------------------------- PSI & data drift ---------------------------
def test_psi_is_zero_for_identical_and_large_for_shifted_distributions():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 20_000)
    assert psi(reference, rng.normal(0, 1, 20_000)) < 0.01
    assert psi(reference, rng.normal(1.0, 1, 20_000)) > 0.25


def test_categorical_psi():
    assert categorical_psi([True] * 20 + [False] * 80, [True] * 20 + [False] * 80) == pytest.approx(0)
    assert categorical_psi([True] * 20 + [False] * 80, [True] * 60 + [False] * 40) > 0.25


def test_feature_drift_flags_only_the_column_that_moved():
    rng = np.random.default_rng(1)
    n = 5000
    ref = pd.DataFrame({"competitor_price": rng.normal(15, 3, n), "our_price": rng.normal(16, 3, n),
                        "drivers_available": rng.poisson(30, n), "demand": rng.poisson(60, n),
                        "is_raining": rng.random(n) < 0.2})  # fmt: skip
    cur = ref.sample(frac=1, random_state=2).reset_index(drop=True)
    cur["competitor_price"] = cur["competitor_price"] * 0.7  # the competitor got 30% cheaper
    table = feature_drift(ref, cur)
    assert table.loc["competitor_price", "status"] != "ok"
    assert (table.drop(index="competitor_price")["status"] == "ok").all()


# ----------------------------- concept drift (beta) ------------------------
@pytest.mark.parametrize("scenario, expect_alert", [("stable", False), ("drift", True)])
def test_price_sensitivity_check_detects_a_beta_change(scenario, expect_alert):
    window, _ = simulate_window(scenario, weeks=3)
    model = TrueDemandModel(CONFIG)  # a "perfect" model of the OLD market
    table = price_sensitivity_check(model, window, CONFIG)
    assert (table["status"] == "alert").all() == expect_alert
    if expect_alert:  # the estimate should land near the new truth (1.5 x the old beta)
        truth = {z.name: z.price_sensitivity for z in drifted_config(CONFIG).zones}
        for zone, row in table.iterrows():
            assert row.estimated_beta == pytest.approx(truth[zone], rel=0.08)


# ----------------------------- retraining helpers --------------------------
def test_recency_weights_halve_every_half_life():
    ts = pd.Series(pd.to_datetime(["2026-01-29", "2026-01-15", "2026-01-01"]))
    np.testing.assert_allclose(recency_weights(ts, half_life_days=14), [1.0, 0.5, 0.25])


def test_promotion_requires_a_clear_improvement():
    assert should_promote(champion_deviance=1.00, challenger_deviance=0.90)
    assert not should_promote(champion_deviance=1.00, challenger_deviance=0.99)  # within the 2% margin
    assert not should_promote(champion_deviance=1.00, challenger_deviance=1.10)


# ----------------------------- MLflow tracking & registry ------------------
@pytest.fixture
def small_history():
    return simulate_window("stable", start="2025-09-01", weeks=8)[0]


@pytest.fixture(scope="module")
def year_history():
    """Jan-Oct 2025, like the real training set. A champion needs a long history to learn the
    yearly trend and seasonality; one trained on 2 months is biased by ~75% in January and
    monitoring (correctly) flags it."""
    return simulate_window("stable", start="2025-01-01", weeks=43)[0]


@pytest.fixture
def registry(tmp_path):
    tracking.configure(tmp_path / "mlops")
    return tmp_path


def test_each_configure_uses_its_own_database(tmp_path):
    first = tracking.configure(tmp_path / "a")
    second = tracking.configure(tmp_path / "b")  # must NOT be hijacked by the env var the first call set
    assert first != second
    assert second.endswith("b/mlflow.db")


def test_register_promote_and_load_the_champion(registry, small_history):
    assert tracking.get_champion() is None
    model = train_challenger(small_history)
    out = tracking.log_run("unit", params={"a": 1}, metrics={"score (x)": 0.5}, model=model, register=True,
                           model_tags={"reference_wape": "0.12"})  # fmt: skip
    tracking.promote(out["version"])
    champion = tracking.get_champion()
    assert champion["version"] == out["version"]
    assert champion["tags"]["reference_wape"] == "0.12"
    np.testing.assert_allclose(predict_demand(champion["model"], small_history),
                               predict_demand(model, small_history))  # fmt: skip
    assert tracking.list_versions()[0]["aliases"] == ["champion"]


def test_full_cycle_stable_keeps_and_drift_promotes(registry, year_history):
    model_path = registry / "models" / "demand_model.joblib"
    save_model(build_models()["poisson_glm"].fit(year_history, year_history["demand"]), model_path)
    model_path.with_name("metrics.json").write_text(
        json.dumps({"champion": "poisson_glm", "test_metrics": {"poisson_glm": {"WAPE": 0.11}}}))  # fmt: skip
    ensure_champion(model_path)

    stable_window, _ = simulate_window("stable", weeks=4)
    stable = run_cycle(year_history, stable_window, "stable", report_dir=registry / "reports")
    assert stable["status"] != "alert", stable["reasons"]
    assert not stable["retrained"]

    drift_window, drift_config = simulate_window("drift", weeks=4)
    drift = run_cycle(year_history, drift_window, "drift", report_dir=registry / "reports", config=drift_config)
    assert drift["status"] == "alert"
    assert drift["promoted"]
    assert drift["champion_after"] == 2
    assert tracking.get_champion()["version"] == 2
    assert (registry / "reports" / "drift_report_drift.json").exists()
