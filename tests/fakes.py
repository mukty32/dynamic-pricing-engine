"""Test doubles shared by several test files."""

import numpy as np
import pandas as pd

BETA, C = 0.08, 400.0  # downtown-like: natural revenue peak at 1 / 0.08 = $12.50


class FakeExponentialModel:
    """A stand-in demand model with a known formula: D(p) = C * exp(-beta * p).

    Because we know the formula, we know the right answers in advance (via calculus),
    and the tests don't depend on training a real model.
    """

    def __init__(self, C: float = C, beta: float = BETA):
        self.C, self.beta = C, beta

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.C * np.exp(-self.beta * X["our_price"].to_numpy(dtype=float))
