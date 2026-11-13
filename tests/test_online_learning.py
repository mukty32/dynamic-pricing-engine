"""Tests for Phase 7: Bayesian online learning and Thompson Sampling. Run with:  pytest"""

import numpy as np
import pandas as pd
import pytest

from pricing_engine.online_experiment import DriftScenario, fixed_context_experiment, run_policy
from pricing_engine.online_learning import BayesianPriceLearner, price_from_theta
from pricing_engine.optimizer import exponential_closed_form

FARE = 12.0


def feed(learner, rng, n, alpha=0.0, beta=0.12, market_size=100.0):
    """Show the learner n hours of data generated from a KNOWN theta, at random prices."""
    for price in rng.uniform(9, 30, size=n):
        lam = market_size * np.exp(alpha - beta * (price - FARE))
        learner.observe(market_size, price - FARE, rng.poisson(lam))


# ----------------------------- the Bayesian learner ------------------------
def test_with_no_data_the_belief_is_the_prior():
    learner = BayesianPriceLearner((0.0, 0.08), (0.1, 0.02))
    mean, cov = learner.posterior()
    np.testing.assert_allclose(mean, [0.0, 0.08])
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), [0.1, 0.02])


def test_learner_finds_the_true_parameters():
    learner = BayesianPriceLearner((0.0, 0.08), (0.1, 0.02))
    feed(learner, np.random.default_rng(0), n=2000, alpha=0.05, beta=0.12)
    mean, _ = learner.posterior()
    assert mean[0] == pytest.approx(0.05, abs=0.02)
    assert mean[1] == pytest.approx(0.12, abs=0.003)


def test_uncertainty_shrinks_like_one_over_sqrt_n():
    rng = np.random.default_rng(1)
    flat_prior = (10.0, 10.0)  # nearly no prior information, so the data decides
    small, big = BayesianPriceLearner((0, 0.1), flat_prior), BayesianPriceLearner((0, 0.1), flat_prior)
    feed(small, rng, n=400)
    feed(big, rng, n=1600)  # 4x the data
    ratio = np.sqrt(small.posterior()[1][1, 1]) / np.sqrt(big.posterior()[1][1, 1])
    assert ratio == pytest.approx(2.0, rel=0.15)  # -> about half the uncertainty


def test_forgetting_lets_the_learner_track_a_change():
    rng = np.random.default_rng(2)
    remembers, forgets = (BayesianPriceLearner((0, 0.08), (0.1, 0.02), forgetting=f) for f in (1.0, 0.99))
    for learner in (remembers, forgets):
        feed(learner, rng, n=1500, beta=0.08)  # old market
        feed(learner, rng, n=400, beta=0.12)  # new market
    assert abs(forgets.posterior()[0][1] - 0.12) < abs(remembers.posterior()[0][1] - 0.12)
    assert forgets.posterior()[0][1] == pytest.approx(0.12, abs=0.01)


def test_thompson_samples_follow_the_belief():
    learner = BayesianPriceLearner((0.0, 0.08), (0.1, 0.02))
    feed(learner, np.random.default_rng(3), n=300)
    mean, cov = learner.posterior()
    rng = np.random.default_rng(4)
    samples = np.array([learner.sample(rng) for _ in range(4000)])
    standard_error = np.sqrt(np.diag(cov) / len(samples))
    assert np.all(np.abs(samples.mean(axis=0) - mean) < 4 * standard_error)  # centred on the belief
    np.testing.assert_allclose(samples.std(axis=0), np.sqrt(np.diag(cov)), rtol=0.1)  # same spread


def test_bad_forgetting_value_is_rejected():
    with pytest.raises(ValueError):
        BayesianPriceLearner((0, 0.1), (1, 1), forgetting=0.0)


# ----------------------------- pricing from a belief ----------------------
def test_price_from_theta_uses_the_closed_form():
    theta = (0.1, 0.09)
    C = 100 * np.exp(0.1 + 0.09 * FARE)
    assert price_from_theta(theta, 100, FARE, 60, 9.6, 36) == pytest.approx(
        exponential_closed_form(C, 0.09, 60, 9.6, 36)
    )


def test_impossible_negative_beta_never_breaks_pricing():
    price = price_from_theta((0.0, -0.5), 100, FARE, 1e9, 9.6, 36)
    assert 9.6 <= price <= 36  # guarded: stays inside the business rules


# ----------------------------- the experiments ----------------------------
@pytest.fixture(scope="module")
def tiny_env():
    """A 10-day market where beta jumps from 0.08 to 0.12 on day 3."""
    n = 240
    hour = np.arange(n) % 24
    env = pd.DataFrame(
        {
            "market_size": 60 + 40 * np.sin(hour / 24 * 2 * np.pi) ** 2,
            "capacity": np.where(hour % 3 == 0, 40.0, 1e6),  # sometimes supply-limited
            "lower": 9.6,
            "upper": 36.0,
        }
    )
    env["true_base"] = env["market_size"]
    env["beta_true"] = np.where(np.arange(n) < 72, 0.08, 0.12)
    env["oracle_price"] = [
        exponential_closed_form(r.true_base * np.exp(r.beta_true * FARE), r.beta_true, r.capacity, r.lower, r.upper)
        for r in env.itertuples()
    ]
    env["oracle_revenue"] = env["oracle_price"] * np.minimum(
        env["true_base"] * np.exp(-env["beta_true"] * (env["oracle_price"] - FARE)), env["capacity"]
    )
    return env


def test_no_policy_ever_beats_the_oracle(tiny_env):
    scenario = DriftScenario(shift_day=3)
    for policy in ("static model", "greedy Bayesian", "Thompson sampling"):
        run = run_policy(tiny_env, FARE, policy, (0.0, 0.08), scenario, seed=0)
        assert (run["regret"] >= -1e-6).all()


def test_learning_beats_the_static_model_after_a_shift(tiny_env):
    scenario = DriftScenario(shift_day=3)
    regret = {
        policy: run_policy(tiny_env, FARE, policy, (0.0, 0.08), scenario, seed=0)["regret"].iloc[120:].sum()
        for policy in ("static model", "Thompson sampling")
    }
    assert regret["Thompson sampling"] < 0.5 * regret["static model"]


def test_greedy_gets_stuck_without_natural_price_variation():
    table = fixed_context_experiment(n_hours=600, seeds=range(4))
    greedy, thompson = table.loc["greedy Bayesian"], table.loc["Thompson sampling"]
    assert thompson["total_regret"] < 0.25 * greedy["total_regret"]
    assert abs(thompson["final_beta_belief"] - 0.12) < abs(greedy["final_beta_belief"] - 0.12)
