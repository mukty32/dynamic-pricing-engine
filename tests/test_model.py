"""Tests for Phase 2: features, models and the ground-truth function. Run with:  pytest"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_poisson_deviance

from pricing_engine.config import SimulationConfig
from pricing_engine.data_generation import generate_dataset, true_expected_demand
from pricing_engine.features import MODEL_INPUT_COLUMNS, cyclical_encode, glm_design_matrix, tree_design_matrix
from pricing_engine.model import build_models, load_model, predict_demand, save_model
from pricing_engine.train import estimate_price_sensitivity, time_split

CONFIG = SimulationConfig(n_days=120)


@pytest.fixture(scope="module")
def data():
    df = generate_dataset(CONFIG)
    train, test = time_split(df, "2025-04-11")  # last 20 days are the test set
    return df, train, test


@pytest.fixture(scope="module")
def fitted(data):
    _, train, _ = data
    models = build_models()
    for model in models.values():
        model.fit(train[MODEL_INPUT_COLUMNS], train["demand"])
    return models


# ----------------------------- features -----------------------------------
def test_cyclical_encoding_puts_midnight_next_to_11pm():
    sin, cos = cyclical_encode([23, 0, 12], period=24)
    points = np.column_stack([sin, cos])
    assert np.linalg.norm(points[0] - points[1]) < np.linalg.norm(points[0] - points[2])


def test_design_matrices_are_complete(data):
    df, _, _ = data
    glm_X = glm_design_matrix(df[MODEL_INPUT_COLUMNS])
    tree_X = tree_design_matrix(df[MODEL_INPUT_COLUMNS])
    # 2 zones + 47 time slots + rain + 3 price columns + competitor + growth + 2 season
    assert glm_X.shape == (len(df), 57)
    assert not glm_X.isna().any().any()
    assert not tree_X.isna().any().any()


# ----------------------------- ground truth -------------------------------
def test_true_formula_matches_average_simulated_demand(data):
    df, _, _ = data
    expected = true_expected_demand(df, CONFIG)
    assert df["demand"].mean() == pytest.approx(expected.mean(), rel=0.01)


# ----------------------------- models -------------------------------------
def test_glm_recovers_true_price_sensitivity(fitted, data):
    _, _, test = data
    betas = estimate_price_sensitivity(lambda X: predict_demand(fitted["poisson_glm"], X), test)
    for zone in CONFIG.zones:
        assert betas[zone.name] == pytest.approx(zone.price_sensitivity, rel=0.10)


def test_xgboost_demand_never_rises_with_our_price(fitted, data):
    _, _, test = data
    row = test[MODEL_INPUT_COLUMNS].iloc[[0]]
    sweep = pd.concat([row] * 50, ignore_index=True)
    sweep["our_price"] = np.linspace(5, 60, 50)
    predictions = predict_demand(fitted["xgboost"], sweep)
    assert np.all(np.diff(predictions) <= 1e-9)


def test_price_aware_models_beat_baseline(fitted, data):
    _, _, test = data
    score = {name: mean_poisson_deviance(test["demand"], predict_demand(m, test)) for name, m in fitted.items()}
    assert score["poisson_glm"] < score["baseline"]
    assert score["xgboost"] < score["baseline"]


def test_save_and_load_roundtrip(fitted, data, tmp_path):
    _, _, test = data
    path = save_model(fitted["poisson_glm"], tmp_path / "model.joblib")
    reloaded = load_model(path)
    np.testing.assert_allclose(predict_demand(reloaded, test), predict_demand(fitted["poisson_glm"], test))
