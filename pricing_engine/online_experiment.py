"""Phase 7: the market-shift experiment. Can the engine notice that riders changed?

Run from the project root (after Phases 1 and 2):

    python -m pricing_engine.online_experiment

Story: we replay the 61 test days (Nov-Dec) for one zone, hour by hour. On day 15 a
cheap new competitor launches and riders become MORE price-sensitive: beta jumps from
0.08 to 0.12. Nobody tells our engine; it only sees ride-request counts.

Three strategies, all using the same pricing formula and business rules:
    static model      : trusts the Phase 2 model forever (never updates)
    greedy Bayesian   : updates its belief, prices on the single most likely theta
    Thompson sampling : updates its belief, prices on a RANDOM SAMPLE from it

We score them by REGRET: the revenue lost compared with an oracle that always knows
the true market. Lower regret is better; regret that flattens out means "learned it".
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (used as matplotlib.ticker)
import numpy as np
import pandas as pd

from pricing_engine.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, FIGURES_DIR, REPORTS_DIR, SimulationConfig
from pricing_engine.data_generation import true_expected_demand
from pricing_engine.model import load_model
from pricing_engine.online_learning import BayesianPriceLearner, price_from_theta
from pricing_engine.optimizer import CONTEXT_COLUMNS, PriceOptimizer, exponential_closed_form
from pricing_engine.train import DEFAULT_TEST_START, load_dataset, time_split

logger = logging.getLogger(__name__)

POLICIES = ("static model", "greedy Bayesian", "Thompson sampling")


@dataclass(frozen=True)
class DriftScenario:
    zone: str = "downtown"
    new_beta: float = 0.12  # riders' price sensitivity after the shift (was 0.08)
    shift_day: int = 15
    forgetting: float = 0.995  # old data loses 0.5% weight per hour -> memory of roughly 200 hours
    prior_sd: tuple[float, float] = (0.10, 0.02)  # how unsure we start about (alpha, beta)


# ----------------------------------------------------------------------------- environment
def build_environment(test: pd.DataFrame, model, config: SimulationConfig, scenario: DriftScenario):
    """Everything known about each hour, plus the hidden truth used only for scoring."""
    zone = next(z for z in config.zones if z.name == scenario.zone)
    ctx = test[test["zone"] == zone.name][CONTEXT_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    n = len(ctx)
    optimizer = PriceOptimizer(model, config=config)
    fare = np.full(n, zone.base_fare)
    bounds = optimizer.price_bounds(ctx)

    at_fare = ctx[["timestamp", "zone", "is_raining", "competitor_price"]].copy()
    at_fare["our_price"] = fare
    hour = np.arange(n)
    env = pd.DataFrame(
        {
            "timestamp": ctx["timestamp"],
            "market_size": optimizer.demand(ctx, fare),  # m_t: what the model expects at the base fare
            "true_base": true_expected_demand(at_fare, config),  # truth at the base fare (both regimes agree here)
            "beta_true": np.where(hour < scenario.shift_day * 24, zone.price_sensitivity, scenario.new_beta),
            "capacity": ctx["drivers_available"].to_numpy(float) * config.rides_per_driver_per_hour,
            "lower": bounds["lower_bound"].to_numpy(),
            "upper": bounds["upper_bound"].to_numpy(),
        }
    )
    env["oracle_price"] = [
        exponential_closed_form(
            r.true_base * np.exp(r.beta_true * zone.base_fare), r.beta_true, r.capacity, r.lower, r.upper
        )
        for r in env.itertuples()
    ]
    env["oracle_revenue"] = expected_revenue(env, env["oracle_price"].to_numpy(), zone.base_fare)

    # The model's own beta, read off its demand curve: this is our prior belief.
    up, down = optimizer.demand(ctx, fare + 0.5), optimizer.demand(ctx, fare - 0.5)
    model_beta = float(np.mean(np.log(down / up)))
    return env, zone.base_fare, model_beta


def true_demand(env: pd.DataFrame, prices, base_fare: float) -> np.ndarray:
    return env["true_base"].to_numpy() * np.exp(-env["beta_true"].to_numpy() * (np.asarray(prices) - base_fare))


def expected_revenue(env: pd.DataFrame, prices, base_fare: float) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    return prices * np.minimum(true_demand(env, prices, base_fare), env["capacity"].to_numpy())


# ----------------------------------------------------------------------------- simulation
def run_policy(
    env: pd.DataFrame, base_fare: float, policy: str, prior_mean, scenario: DriftScenario, seed: int
) -> pd.DataFrame:
    """Replay every hour with one strategy. Returns prices, revenue and the belief over time."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    rng = np.random.default_rng(seed)
    prior_mean = np.asarray(prior_mean, dtype=float)
    learner = (
        None if policy == "static model" else BayesianPriceLearner(prior_mean, scenario.prior_sd, scenario.forgetting)
    )
    rows = []
    for r in env.itertuples():
        if learner is None:
            theta, sd = prior_mean, np.array([np.nan, np.nan])
        else:
            mean, cov = learner.posterior()
            sd = np.sqrt(np.diag(cov))
            theta = mean if policy == "greedy Bayesian" else learner.sample(rng)
        price = price_from_theta(theta, r.market_size, base_fare, r.capacity, r.lower, r.upper)
        lam = r.true_base * np.exp(-r.beta_true * (price - base_fare))
        requests = rng.poisson(lam)  # the ONLY feedback the engine gets
        if learner is not None:
            learner.observe(r.market_size, price - base_fare, requests)
        belief = learner.posterior()[0] if learner is not None else prior_mean
        rows.append({"price": price, "beta_belief": belief[1], "beta_sd": sd[1]})

    out = pd.DataFrame(rows)
    out["revenue"] = expected_revenue(env, out["price"].to_numpy(), base_fare)
    out["regret"] = env["oracle_revenue"].to_numpy() - out["revenue"]
    return out


def run_experiment(env, base_fare, model_beta, scenario, seeds) -> dict[str, list[pd.DataFrame]]:
    prior_mean = (0.0, model_beta)
    return {
        policy: [run_policy(env, base_fare, policy, prior_mean, scenario, seed) for seed in seeds]
        for policy in POLICIES
    }


def summarize(env: pd.DataFrame, results: dict, scenario: DriftScenario) -> pd.DataFrame:
    shift = scenario.shift_day * 24
    oracle_total = env["oracle_revenue"].sum()
    rows = {}
    for policy, runs in results.items():
        revenue = np.array([run["revenue"].sum() for run in runs])
        before = np.array([run["regret"].iloc[:shift].sum() for run in runs])
        after = np.array([run["regret"].iloc[shift:].sum() for run in runs])
        last_week = np.array([run["regret"].iloc[-7 * 24 :].sum() for run in runs])
        rows[policy] = {
            "revenue": revenue.mean(),
            "pct_of_oracle": revenue.mean() / oracle_total,
            "regret_before_shift": before.mean(),
            "regret_after_shift": after.mean(),
            "regret_last_week": last_week.mean(),
            "final_beta_belief": np.mean([run["beta_belief"].iloc[-1] for run in runs]),
        }
    rows["oracle (knows the truth)"] = {
        "revenue": oracle_total,
        "pct_of_oracle": 1.0,
        "regret_before_shift": 0.0,
        "regret_after_shift": 0.0,
        "regret_last_week": 0.0,
        "final_beta_belief": scenario.new_beta,
    }
    return pd.DataFrame(rows).T


def fixed_context_experiment(
    n_hours: int = 1500,
    seeds=range(20),
    market_size: float = 100.0,
    base_fare: float = 12.0,
    beta_true: float = 0.12,
    prior_beta: float = 0.08,
    lower: float = 9.6,
    upper: float = 36.0,
) -> pd.DataFrame:
    """The SAME situation every hour (plenty of drivers, nothing changes).

    Now nothing forces prices to vary on their own. Greedy keeps charging one price,
    and one price can't separate "fewer riders overall" (alpha) from "riders hate high
    prices" (beta), so it can get stuck believing the wrong beta forever. Thompson
    sampling keeps varying the price a little, and that variation is what lets it learn.
    """
    capacity = 1e9
    true_optimum = exponential_closed_form(
        market_size * np.exp(beta_true * base_fare), beta_true, capacity, lower, upper
    )

    def revenue(price: float) -> float:
        return price * market_size * np.exp(-beta_true * (price - base_fare))

    rows = {}
    for policy in ("greedy Bayesian", "Thompson sampling"):
        betas, prices, regrets = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            learner = BayesianPriceLearner((0.0, prior_beta), DriftScenario().prior_sd)
            regret = 0.0
            for _ in range(n_hours):
                mean, _ = learner.posterior()
                theta = mean if policy == "greedy Bayesian" else learner.sample(rng)
                price = price_from_theta(theta, market_size, base_fare, capacity, lower, upper)
                regret += revenue(true_optimum) - revenue(price)
                learner.observe(
                    market_size, price - base_fare, rng.poisson(market_size * np.exp(-beta_true * (price - base_fare)))
                )
            final = learner.posterior()[0]
            betas.append(final[1])
            prices.append(price_from_theta(final, market_size, base_fare, capacity, lower, upper))
            regrets.append(regret)
        rows[policy] = {
            "final_beta_belief": np.mean(betas),
            "final_price": np.mean(prices),
            "true_optimal_price": true_optimum,
            "total_regret": np.mean(regrets),
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- plot
INK = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
COLORS = {"Thompson sampling": "#2a78d6", "static model": "#eb6834", "greedy Bayesian": "#1baf7a"}


def _style(ax) -> None:
    ax.set_facecolor(INK["surface"])
    ax.grid(color=INK["grid"], linewidth=0.8)
    ax.tick_params(colors=INK["muted"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])


def plot_results(env, results, scenario, model_beta, path: Path) -> Path:
    days = np.arange(len(env)) / 24
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=INK["surface"])

    ends = {}
    for policy in POLICIES:
        curves = np.array([run["regret"].cumsum().to_numpy() for run in results[policy]])
        mean = curves.mean(axis=0)
        left.plot(days, mean, color=COLORS[policy], linewidth=2, label=policy)
        left.fill_between(
            days,
            np.percentile(curves, 10, axis=0),
            np.percentile(curves, 90, axis=0),
            color=COLORS[policy],
            alpha=0.15,
            linewidth=0,
        )
        ends[policy] = mean[-1]
    # End-of-line labels, nudged apart so close lines don't print on top of each other.
    min_gap, previous = 0.06 * max(ends.values()), -np.inf
    for policy, end in sorted(ends.items(), key=lambda item: item[1]):
        y = max(end, previous + min_gap)
        left.annotate(
            f"{policy}  ${end:,.0f}",
            xy=(days[-1], end),
            xytext=(days[-1] + 1, y),
            va="center",
            color=INK["ink2"],
            fontsize=9,
        )
        previous = y
    left.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    left.set_title("Cumulative regret: revenue lost vs. a perfect oracle", loc="left", color=INK["ink"], fontsize=11)
    left.set_xlabel("Day", color=INK["ink2"])
    left.set_ylabel("Cumulative lost revenue ($)", color=INK["ink2"])

    ts = results["Thompson sampling"][0]
    band = 2 * ts["beta_sd"].to_numpy()
    right.fill_between(
        days,
        ts["beta_belief"] - band,
        ts["beta_belief"] + band,
        color=COLORS["Thompson sampling"],
        alpha=0.18,
        linewidth=0,
        label="belief ± 2 sd",
    )
    right.plot(
        days, ts["beta_belief"], color=COLORS["Thompson sampling"], linewidth=2, label="Thompson sampling belief"
    )
    right.plot(days, env["beta_true"], color=INK["ink"], linestyle="--", linewidth=2, label="true β")
    right.axhline(model_beta, color=COLORS["static model"], linewidth=2, label="static model β")
    right.set_title("What the engine believes about price sensitivity β", loc="left", color=INK["ink"], fontsize=11)
    right.set_xlabel("Day", color=INK["ink2"])
    right.set_ylabel("β", color=INK["ink2"])
    right.legend(frameon=False, labelcolor=INK["ink"], loc="lower right")

    for ax in (left, right):
        _style(ax)
        ax.axvline(scenario.shift_day, color=INK["muted"], linestyle=":", linewidth=1)
        ax.annotate(
            "riders become more\nprice-sensitive",
            xy=(scenario.shift_day, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            color=INK["ink2"],
            fontsize=9,
        )
    left.set_xlim(0, days[-1] * 1.38)  # room for the end-of-line labels
    left.set_xticks(np.arange(0, days[-1] + 1, 10))
    fig.suptitle(
        f"Online learning after a market shift ({scenario.zone}, β {model_beta:.2f} → {scenario.new_beta})",
        x=0.01,
        ha="left",
        color=INK["ink"],
        fontsize=13,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=INK["surface"])
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Market-shift experiment for online price learning.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--zone", default="downtown")
    parser.add_argument("--new-beta", type=float, default=0.12)
    parser.add_argument("--seeds", type=int, default=10, help="independent repeats (randomness check)")
    parser.add_argument("--test-start", default=DEFAULT_TEST_START, help="first timestamp of the replayed period")
    args = parser.parse_args(argv)

    config = SimulationConfig()
    scenario = DriftScenario(zone=args.zone, new_beta=args.new_beta)
    _, test = time_split(load_dataset(args.data), args.test_start)
    env, base_fare, model_beta = build_environment(test, load_model(args.model), config, scenario)
    logger.info(
        "Replaying %d hours of %s, %d seeds x %d policies ...", len(env), scenario.zone, args.seeds, len(POLICIES)
    )
    results = run_experiment(env, base_fare, model_beta, scenario, seeds=range(args.seeds))

    table = summarize(env, results, scenario)
    with pd.option_context("display.width", 200):
        print(f"\nModel's beta = {model_beta:.4f}; true beta jumps to {scenario.new_beta} on day {scenario.shift_day}")
        print(
            table.round(
                {
                    "revenue": 0,
                    "pct_of_oracle": 4,
                    "regret_before_shift": 0,
                    "regret_after_shift": 0,
                    "regret_last_week": 0,
                    "final_beta_belief": 4,
                }
            ).to_string()
        )

    logger.info("Experiment 2: the same situation every hour (no natural price variation) ...")
    fixed = fixed_context_experiment(seeds=range(2 * args.seeds))
    print("\nExperiment 2: no natural price variation. Does greedy get stuck?")
    print(fixed.round(4).to_string())

    figure = plot_results(env, results, scenario, model_beta, FIGURES_DIR / "online_learning.png")
    out = REPORTS_DIR / "online_learning.json"
    out.write_text(
        json.dumps(
            {
                "scenario": {
                    "zone": scenario.zone,
                    "new_beta": scenario.new_beta,
                    "shift_day": scenario.shift_day,
                    "seeds": args.seeds,
                },
                "model_beta": round(model_beta, 4),
                "results": table.round(4).to_dict(orient="index"),
                "fixed_context": fixed.round(4).to_dict(orient="index"),
            },
            indent=2,
        )
    )
    logger.info("Saved figure -> %s", figure)
    logger.info("Saved results -> %s", out)


if __name__ == "__main__":
    main()
