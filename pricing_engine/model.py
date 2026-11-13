"""Phase 2: the three candidate demand models.

  1. SeasonalBaseline: "average demand at this zone & hour". Ignores price.
                        Every real ML project starts with a dumb baseline:
                        if a fancy model can't beat it, the fancy model is useless.
  2. Poisson GLM:      log(demand) = w . features. Same SHAPE as our true market,
                        so its weights are readable (we can read beta straight off it).
  3. XGBoost:          hundreds of small decision trees, flexible but a black box.

All three follow scikit-learn's fit / predict interface, so they're interchangeable.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from pricing_engine.features import MODEL_INPUT_COLUMNS, calendar_features, glm_design_matrix, tree_design_matrix


class SeasonalBaseline(RegressorMixin, BaseEstimator):
    """Predict the historical average demand for each (zone, weekend?, hour)."""

    def fit(self, X: pd.DataFrame, y) -> SeasonalBaseline:
        keys = self._keys(X)
        self.table_ = pd.Series(np.asarray(y, dtype=float), index=X.index).groupby(keys).mean()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._keys(X).map(self.table_).fillna(self.global_mean_).to_numpy()

    @staticmethod
    def _keys(X: pd.DataFrame) -> pd.Series:
        cal = calendar_features(X)
        return X["zone"].astype(str) + "|" + cal["is_weekend"].astype(str) + "|" + cal["hour"].astype(str)


def build_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Fresh, untrained copies of every candidate model."""
    # Imported here, not at the top of the file: the API only LOADS a saved model, and when
    # the champion is the GLM, it never needs XGBoost's large native library in memory.
    from xgboost import XGBRegressor

    glm = Pipeline(
        [
            ("features", FunctionTransformer(glm_design_matrix)),
            # alpha=0 -> no penalty: we want the pure maximum-likelihood weights.
            ("model", PoissonRegressor(alpha=0.0, solver="newton-cholesky", max_iter=300)),
        ]
    )
    xgb = Pipeline(
        [
            ("features", FunctionTransformer(tree_design_matrix)),
            (
                "model",
                XGBRegressor(
                    objective="count:poisson",  # target is a count -> Poisson loss
                    n_estimators=600,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    # Domain knowledge as a hard rule: demand may only FALL when our price rises
                    # and only RISE when the competitor's price rises.
                    monotone_constraints={"our_price": -1, "competitor_price": 1},
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {"baseline": SeasonalBaseline(), "poisson_glm": glm, "xgboost": xgb}


def predict_demand(model: BaseEstimator, context: pd.DataFrame) -> np.ndarray:
    """Predicted average demand for each row of `context` (needs MODEL_INPUT_COLUMNS)."""
    return np.asarray(model.predict(context[MODEL_INPUT_COLUMNS]), dtype=float)


def save_model(model: BaseEstimator, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> BaseEstimator:
    # Security note: joblib/pickle files can run code when loaded.
    # Only ever load model files that YOU created.
    return joblib.load(path)
