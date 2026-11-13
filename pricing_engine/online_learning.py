"""Phase 7: online learning. The engine keeps learning price sensitivity while it runs.

The problem
-----------
The Phase 2 model learned beta from LAST year's data. Markets change: a new competitor
arrives, the economy shifts, riders become more price-sensitive. A model that never
updates keeps charging yesterday's optimal price.

The solution: Bayesian learning + Thompson Sampling
----------------------------------------------------
* Keep a BELIEF about the unknown parameters theta = (alpha, beta): a probability
  distribution, not a single number.
* Every hour:  1. SAMPLE one plausible theta from the belief,
               2. charge the optimal price FOR THAT SAMPLE,
               3. observe how many riders requested a ride,
               4. update the belief with Bayes' rule.
* Uncertain belief -> samples differ -> prices vary   -> we EXPLORE (and learn).
  Confident belief -> samples agree -> prices settle -> we EXPLOIT what we know.

Demand model learned online, for the situation at hour t:

    lambda_t(p) = m_t * exp( alpha - beta * (p - p0) )

    m_t   : the Phase 2 model's forecast at the base fare p0. It still knows the
            time-of-day, weather and competitor patterns well.
    alpha : correction to the overall demand level   (prior belief: 0, the model is right)
    beta  : price sensitivity                         (prior belief: the model's estimate)
"""

from __future__ import annotations

import math

import numpy as np

from pricing_engine.optimizer import exponential_closed_form

MIN_BETA = 0.005  # a sampled beta <= 0 would mean "higher prices attract riders"; never price on that


class BayesianPriceLearner:
    """Bayesian Poisson regression for theta = (alpha, beta), updated one observation at a time.

    The exact posterior has no neat formula, so we use the LAPLACE APPROXIMATION:
    find the most likely theta (the MAP) with Newton's method, then approximate the
    posterior by a Gaussian centred there, whose spread comes from the curvature
    (the Hessian) of the log-posterior.

    `forgetting` < 1 shrinks the weight of old observations by that factor every hour,
    so the learner can track a market that changes over time.
    """

    def __init__(self, prior_mean, prior_sd, forgetting: float = 1.0):
        self.prior_mean = np.asarray(prior_mean, dtype=float)
        self.prior_precision = np.diag(1.0 / np.asarray(prior_sd, dtype=float) ** 2)
        if not 0 < forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        self.forgetting = forgetting
        self._log_m: list[float] = []
        self._offset: list[float] = []
        self._requests: list[float] = []
        self._theta = self.prior_mean.copy()  # warm start for Newton
        self._cov = np.linalg.inv(self.prior_precision)
        self._stale = False

    @property
    def n_observations(self) -> int:
        return len(self._requests)

    def observe(self, market_size: float, price_offset: float, requests: int) -> None:
        """Record one hour: forecast m_t, how far our price was from p0, and the ride requests seen."""
        self._log_m.append(math.log(market_size))
        self._offset.append(price_offset)
        self._requests.append(float(requests))
        self._stale = True

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """(mean, covariance) of the Gaussian belief about theta = (alpha, beta)."""
        if self._stale:
            self._fit()
            self._stale = False
        return self._theta.copy(), self._cov.copy()

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One plausible theta drawn from the current belief: the heart of Thompson Sampling."""
        mean, cov = self.posterior()
        return rng.multivariate_normal(mean, cov)

    def _fit(self, max_iter: int = 50, tol: float = 1e-9) -> None:
        log_m = np.asarray(self._log_m)
        y = np.asarray(self._requests)
        X = np.column_stack([np.ones_like(y), -np.asarray(self._offset)])  # eta = log m + alpha - beta*offset
        ages = np.arange(len(y))[::-1]  # the newest observation has age 0
        w = self.forgetting**ages

        theta = self._theta.copy()
        for _ in range(max_iter):
            lam = np.exp(log_m + X @ theta)
            # Gradient of the log-posterior = data pull (y - lambda) + prior pull back to its mean
            gradient = X.T @ (w * (y - lam)) - self.prior_precision @ (theta - self.prior_mean)
            # Minus the Hessian: always positive-definite, because the log-posterior is CONCAVE
            curvature = (X * (w * lam)[:, None]).T @ X + self.prior_precision
            step = np.linalg.solve(curvature, gradient)
            step_size = np.linalg.norm(step)
            if step_size > 1.0:  # damping: never jump too far in one Newton step
                step /= step_size
            theta += step
            if step_size < tol:
                break

        lam = np.exp(log_m + X @ theta)
        curvature = (X * (w * lam)[:, None]).T @ X + self.prior_precision
        self._theta, self._cov = theta, np.linalg.inv(curvature)


def price_from_theta(theta, market_size: float, base_fare: float, capacity: float, lower: float, upper: float) -> float:
    """Optimal price IF theta were the truth, using the Phase 3 closed form.

    lambda(p) = m * exp(alpha - beta*(p - p0)) = [m * e^(alpha + beta*p0)] * e^(-beta*p) = C * e^(-beta*p)
    """
    alpha, beta = float(theta[0]), max(float(theta[1]), MIN_BETA)
    C = market_size * math.exp(alpha + beta * base_fare)
    return exponential_closed_form(C, beta, capacity, lower, upper)
