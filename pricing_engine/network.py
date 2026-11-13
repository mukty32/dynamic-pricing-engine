"""Phase 10: multi-zone pricing + driver repositioning as ONE convex optimisation problem.

Until now each zone was priced on its own. But drivers can MOVE: when downtown is
short of drivers and the suburbs have spare ones, paying a few suburban drivers to
drive downtown can earn more than it costs. That couples all zones together.

The trick that makes it solvable: think in QUANTITY (rides sold), not price.
--------------------------------------------------------------------------------
With exponential demand  D(p) = C e^(-beta p), selling x rides means charging the price

        p(x) = ln(C / x) / beta              (the "inverse demand" curve)

so revenue as a function of quantity is

        R(x) = x * ln(C / x) / beta = [x ln C + entr(x)] / beta,   entr(x) = -x ln x

and  R''(x) = -1 / (beta x) < 0:  revenue is CONCAVE in quantity. (In price it isn't
concave everywhere, only log-concave.) A surge cap p_max adds  R(x) <= p_max * x, and the
minimum of two concave functions is still concave. All constraints are linear, so the
whole city is a convex problem: one global optimum, solved exactly by cvxpy.

The problem, for zones i, j and drivers moved f_ij (from i to j):

    maximise    sum_i R_i(x_i)  -  sum_ij cost_ij * f_ij
    subject to  x_i <= r * (K_i - sum_j f_ij) + sum_j r * (1 - t_ji / 60) * f_ji   (capacity)
                sum_j f_ij <= K_i                                                (can't move more than you have)
                x_i <= D_i(p_min)                                                (price floor)
                x, f >= 0

r = rides per driver-hour, K_i = drivers in zone i, t_ji = travel minutes (a moving
driver loses that part of the hour).

Shadow prices (duality): the dual value lambda_i of zone i's capacity constraint is the
extra revenue from ONE more ride of capacity there. At the optimum,

    lambda_i = R_i'(x_i) = p_i - 1/beta_i     (marginal revenue), or p_max if the cap binds,

and a driver moves from i to j only if   r(1 - t/60) lambda_j - r lambda_i > cost_ij.
Moving drivers raises lambda where they leave and lowers it where they arrive, until
no move pays for itself: spatial arbitrage, exactly like prices equalising across markets.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing_engine.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, FIGURES_DIR, REPORTS_DIR, SimulationConfig
from pricing_engine.data_generation import true_expected_demand
from pricing_engine.model import load_model
from pricing_engine.optimizer import PriceOptimizer, PricingConstraints
from pricing_engine.train import DEFAULT_TEST_START, load_dataset, time_split

logger = logging.getLogger(__name__)

# Minutes to drive between zones (symmetric). A moving driver loses that share of the hour.
TRAVEL_MINUTES = {("downtown", "airport"): 30, ("downtown", "suburbs"): 20, ("airport", "suburbs"): 40}


@dataclass(frozen=True)
class ZoneMarket:
    """One zone's demand curve D(p) = C e^(-beta p), supply and allowed price range."""

    name: str
    C: float
    beta: float
    drivers: float
    base_fare: float
    lower: float  # lowest allowed price
    upper: float  # highest allowed price (surge cap / competitor rule)

    def demand(self, price):
        return self.C * np.exp(-self.beta * np.asarray(price, dtype=float))

    def price_for(self, rides: float) -> float:
        """Inverse demand, clipped to the allowed range (above the cap, riders are turned away)."""
        if rides <= 1e-9:
            return self.upper
        return float(np.clip(math.log(self.C / rides) / self.beta, self.lower, self.upper))


@dataclass(frozen=True)
class RepositioningCosts:
    bonus_per_move: float = 4.0  # $ incentive paid to a driver who relocates
    cost_per_minute: float = 0.25  # $ per minute of driving empty (fuel, wear)

    def travel_minutes(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return float(TRAVEL_MINUTES.get((a, b), TRAVEL_MINUTES.get((b, a), 0.0)))

    def move_cost(self, a: str, b: str) -> float:
        return 0.0 if a == b else self.bonus_per_move + self.cost_per_minute * self.travel_minutes(a, b)


@dataclass(frozen=True)
class NetworkPlan:
    zones: pd.DataFrame  # one row per zone: price, rides, capacity, shadow price, ...
    moves: pd.DataFrame  # drivers moved: from -> to
    revenue: float  # ride revenue
    move_cost: float  # repositioning spend
    status: str  # solver status

    @property
    def profit(self) -> float:
        return self.revenue - self.move_cost


def solve_network(markets: list[ZoneMarket], costs: RepositioningCosts | None = None,
                  rides_per_driver: float = 2.0, allow_moves: bool = True) -> NetworkPlan:  # fmt: skip
    """Jointly choose every zone's rides (hence price) and the driver moves."""
    costs = costs or RepositioningCosts()
    n = len(markets)
    names = [m.name for m in markets]
    K = np.array([m.drivers for m in markets], dtype=float)
    keep = np.array([[1.0 - costs.travel_minutes(a, b) / 60 for b in names] for a in names])  # usable share
    move_cost = np.array([[costs.move_cost(a, b) for b in names] for a in names])

    x = cp.Variable(n, nonneg=True, name="rides")
    f = cp.Variable((n, n), nonneg=True, name="moves")
    inflow = cp.sum(cp.multiply(keep, f), axis=0)  # effective drivers arriving in each zone
    outflow = cp.sum(f, axis=1)
    capacity = x <= rides_per_driver * (K - outflow + inflow)
    constraints = [
        capacity,
        outflow <= K,
        cp.diag(f) == 0,
        x <= np.array([m.demand(m.lower) for m in markets]),  # price floor
    ]
    if not allow_moves:
        constraints.append(f == 0)

    revenue_terms = [
        cp.minimum(m.upper * x[i], (x[i] * math.log(m.C) + cp.entr(x[i])) / m.beta) for i, m in enumerate(markets)
    ]
    objective = cp.Maximize(cp.sum(cp.hstack(revenue_terms)) - cp.sum(cp.multiply(move_cost, f)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"network optimisation failed: {problem.status}")

    rides = np.maximum(np.asarray(x.value, dtype=float), 0.0)
    flows = np.maximum(np.asarray(f.value, dtype=float), 0.0)
    flows[flows < 1e-6] = 0.0
    eff_drivers = K - flows.sum(axis=1) + (keep * flows).sum(axis=0)
    prices = np.array([m.price_for(r) for m, r in zip(markets, rides, strict=True)])
    demand = np.array([float(m.demand(p)) for m, p in zip(markets, prices, strict=True)])
    shadow = np.asarray(capacity.dual_value, dtype=float)
    zones = pd.DataFrame({
        "drivers_before": K,
        "drivers_after": eff_drivers,
        "capacity": rides_per_driver * eff_drivers,
        "price": prices,
        "surge": prices / np.array([m.base_fare for m in markets]),
        "demand": demand,
        "rides": rides,
        "fulfillment": np.divide(rides, demand, out=np.ones(n), where=demand > 0),
        "revenue": prices * rides,
        "value_per_ride": shadow,  # shadow price of capacity ($ per extra ride-slot)
        "value_per_driver": shadow * rides_per_driver,
    }, index=pd.Index(names, name="zone"))  # fmt: skip
    moves = pd.DataFrame(
        [
            {"from": names[i], "to": names[j], "drivers": flows[i, j], "cost": flows[i, j] * move_cost[i, j]}
            for i in range(n)
            for j in range(n)
            if flows[i, j] > 1e-3
        ],  # fmt: skip
        columns=["from", "to", "drivers", "cost"],
    )
    return NetworkPlan(zones=zones, moves=moves, revenue=float((prices * rides).sum()),
                       move_cost=float((move_cost * flows).sum()), status=str(problem.status))  # fmt: skip


# ----------------------------------------------------------------------------- from the demand model
def markets_from_model(model, contexts: pd.DataFrame, config: SimulationConfig | None = None,
                       constraints: PricingConstraints | None = None) -> list[ZoneMarket]:  # fmt: skip
    """Read each zone's demand curve off the trained model at this moment.

    `contexts` has one row per zone with: timestamp, zone, is_raining, competitor_price, drivers_available.
    beta comes from the slope of ln(demand) around the base fare (exact for the Poisson GLM).
    """
    config = config or SimulationConfig()
    optimizer = PriceOptimizer(model, constraints=constraints, config=config)
    fares = contexts["zone"].map({z.name: z.base_fare for z in config.zones}).to_numpy(dtype=float)
    at_fare = optimizer.demand(contexts, fares)
    beta = np.log(optimizer.demand(contexts, fares - 0.5) / optimizer.demand(contexts, fares + 0.5))
    bounds = optimizer.price_bounds(contexts)
    return [
        ZoneMarket(name=row.zone, C=float(at_fare[i] * np.exp(beta[i] * fares[i])), beta=float(beta[i]),
                   drivers=float(row.drivers_available), base_fare=float(fares[i]),
                   lower=float(bounds["lower_bound"].iloc[i]), upper=float(bounds["upper_bound"].iloc[i]))
        for i, row in enumerate(contexts.reset_index(drop=True).itertuples())
    ]  # fmt: skip


# ----------------------------------------------------------------------------- backtest
def evaluate_plan(plan: NetworkPlan, contexts: pd.DataFrame, config: SimulationConfig) -> tuple[float, float]:
    """Score a plan with the TRUE market: (revenue, rides served). Moves change each zone's capacity."""
    X = contexts[["timestamp", "zone", "is_raining", "competitor_price"]].copy()
    X["our_price"] = plan.zones.loc[contexts["zone"], "price"].to_numpy()
    demand = true_expected_demand(X, config)
    capacity = plan.zones.loc[contexts["zone"], "capacity"].to_numpy()
    rides = np.minimum(demand, capacity)
    return float((X["our_price"].to_numpy() * rides).sum()), float(rides.sum())


def run_backtest(test: pd.DataFrame, model, config: SimulationConfig,
                 costs: RepositioningCosts | None = None) -> pd.DataFrame:  # fmt: skip
    """Every hour of the test period: price zones separately vs. jointly with driver moves."""
    rows = []
    for timestamp, hour in test.groupby("timestamp", sort=True):
        contexts = hour.sort_values("zone").reset_index(drop=True)
        markets = markets_from_model(model, contexts, config)
        row = {"timestamp": timestamp}
        for label, allow in (("separate", False), ("network", True)):
            plan = solve_network(markets, costs, config.rides_per_driver_per_hour, allow_moves=allow)
            revenue, rides = evaluate_plan(plan, contexts, config)
            row |= {f"{label}_revenue": revenue, f"{label}_rides": rides, f"{label}_move_cost": plan.move_cost,
                    f"{label}_moved": plan.moves["drivers"].sum()}  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- demo + plot
def demo_contexts() -> pd.DataFrame:
    """Friday 6 pm: downtown is short of drivers, the suburbs have plenty."""
    return pd.DataFrame({
        "timestamp": pd.Timestamp("2025-11-21 18:00"),
        "zone": ["downtown", "airport", "suburbs"],
        "is_raining": False,
        "competitor_price": [18.0, 37.5, 15.0],
        "drivers_available": [25, 15, 45],
    })  # fmt: skip


COLORS = {"before": "#eb6834", "after": "#2a78d6"}
INK = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "axis": "#c3c2b7"}  # fmt: skip


def plot_demo(separate: NetworkPlan, network: NetworkPlan, path: Path) -> Path:
    zones = list(separate.zones.index)
    x = np.arange(len(zones))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=INK["surface"])
    panels = [("drivers_after", "Drivers available", "{:.0f}"),
              ("value_per_driver", "Value of one more driver ($/hour)", "${:.0f}")]  # fmt: skip
    for ax, (column, title, fmt) in zip(axes, panels, strict=True):
        for k, (label, plan) in enumerate((("without moves", separate), ("with moves", network))):
            values = plan.zones[column].to_numpy()
            color = COLORS["before" if k == 0 else "after"]
            bars = ax.bar(x + (k - 0.5) * width, values, width - 0.04, color=color, label=label)
            for bar, value in zip(bars, values, strict=True):
                ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9,
                            color=INK["ink2"])  # fmt: skip
        ax.set_xticks(x, [z.title() for z in zones])
        ax.set_title(title, loc="left", color=INK["ink"], fontsize=11)
        ax.set_facecolor(INK["surface"])
        ax.grid(axis="y", color=INK["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK["muted"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK["axis"])
    axes[0].legend(frameon=False, labelcolor=INK["ink"])
    fig.suptitle("Drivers move until no further move pays for its cost (Friday 6 pm)",
                 x=0.01, ha="left", color=INK["ink"], fontsize=13)  # fmt: skip
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=INK["surface"])
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Multi-zone pricing with driver repositioning.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--max-hours", type=int, help="only backtest the first N hours (quick runs)")
    args = parser.parse_args(argv)

    config = SimulationConfig()
    model = load_model(args.model)
    contexts = demo_contexts()
    markets = markets_from_model(model, contexts, config)
    separate = solve_network(markets, rides_per_driver=config.rides_per_driver_per_hour, allow_moves=False)
    network = solve_network(markets, rides_per_driver=config.rides_per_driver_per_hour)
    columns = ["drivers_before", "drivers_after", "price", "surge", "rides", "fulfillment", "value_per_driver"]
    with pd.option_context("display.width", 200):
        for label, plan in (("WITHOUT moves (each zone alone)", separate), ("WITH moves (whole city)", network)):
            print(f"\n{label}: revenue ${plan.revenue:,.0f} - moves ${plan.move_cost:,.0f} = ${plan.profit:,.0f}")
            print(plan.zones[columns].round(2).to_string())
        print("\nDriver moves:")
        print(network.moves.round(2).to_string(index=False) if len(network.moves) else "  (none)")

    logger.info("Backtesting every hour of the test period ...")
    _, test = time_split(load_dataset(args.data), args.test_start)
    if args.max_hours:
        test = test[test["timestamp"].isin(sorted(test["timestamp"].unique())[: args.max_hours])]
    hourly = run_backtest(test, model, config)
    summary = pd.DataFrame({
        label: {"revenue": hourly[f"{label}_revenue"].sum(), "move_cost": hourly[f"{label}_move_cost"].sum(),
                "profit": (hourly[f"{label}_revenue"] - hourly[f"{label}_move_cost"]).sum(),
                "rides_served": hourly[f"{label}_rides"].sum(),
                "hours_with_moves": int((hourly[f"{label}_moved"] > 0.01).sum()),
                "avg_drivers_moved": hourly[f"{label}_moved"].mean()}
        for label in ("separate", "network")
    }).T  # fmt: skip
    uplift = summary.loc["network", "profit"] / summary.loc["separate", "profit"] - 1
    print(f"\nBacktest, {len(hourly):,} hours scored with the TRUE market (profit = revenue - move costs):")
    print(summary.round(1).to_string())
    print(f"\nNetwork optimisation profit uplift: {uplift:+.2%}")

    figure = plot_demo(separate, network, FIGURES_DIR / "network_repositioning.png")
    out = REPORTS_DIR / "network.json"
    out.write_text(json.dumps({
        "demo": {"separate": separate.zones.round(3).to_dict(orient="index"),
                 "network": network.zones.round(3).to_dict(orient="index"),
                 "moves": network.moves.round(3).to_dict(orient="records")},
        "backtest": summary.round(2).to_dict(orient="index"), "profit_uplift": round(float(uplift), 4),
        "costs": dataclasses.asdict(RepositioningCosts()),
    }, indent=2, default=str))  # fmt: skip
    logger.info("Saved figure -> %s", figure)
    logger.info("Saved report -> %s", out)


if __name__ == "__main__":
    main()
