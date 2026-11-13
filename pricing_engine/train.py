"""Phase 2: train, evaluate and compare the demand models.

Run from the project root (after generating the data in Phase 1):

    python -m pricing_engine.train

Steps:
  1. Show the confounding trap in the raw data.
  2. Split by TIME: train on Jan-Oct, test on Nov-Dec (the "future").
  3. Fit baseline, Poisson GLM and XGBoost; score them on the test months.
  4. Check which model learned the TRUE price sensitivity (beta).
  5. Save the champion model + metrics, and draw demand/revenue curves.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error

from pricing_engine.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, FIGURES_DIR, SimulationConfig
from pricing_engine.data_generation import true_expected_demand
from pricing_engine.features import MODEL_INPUT_COLUMNS, TARGET_COLUMN, ZONES
from pricing_engine.model import build_models, predict_demand, save_model

logger = logging.getLogger(__name__)

DEFAULT_TEST_START = "2025-11-01"
PRICE_STEP = 0.50  # $ used for the finite-difference slope estimate


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------
def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m pricing_engine.data_generation` first.")
    return pd.read_csv(path, parse_dates=["timestamp"])


def time_split(df: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Past -> train, future -> test. NEVER shuffle time series randomly:
    a random split lets the model peek at the future and makes scores look too good."""
    cutoff = pd.Timestamp(test_start)
    return df[df["timestamp"] < cutoff].copy(), df[df["timestamp"] >= cutoff].copy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)  # deviance needs > 0
    return {
        "MAE": mean_absolute_error(y_true, y_pred),  # average miss, in riders
        "RMSE": root_mean_squared_error(y_true, y_pred),  # punishes big misses more
        "WAPE": np.abs(y_true - y_pred).sum() / y_true.sum(),  # total miss as % of total demand
        "poisson_deviance": mean_poisson_deviance(y_true, y_pred),  # the loss our models minimise
    }


def estimate_price_sensitivity(predict_fn, X: pd.DataFrame, step: float = PRICE_STEP) -> pd.Series:
    """Estimate beta per zone:  beta ~ -d ln(D) / dp,  using a central finite difference.

    For each row we nudge OUR price up and down by `step` and see how much the
    log of predicted demand moves. Averaging over many rows gives one number per zone.
    """
    up, down = X.copy(), X.copy()
    up["our_price"] += step
    down["our_price"] -= step
    slope = (np.log(predict_fn(up)) - np.log(predict_fn(down))) / (2 * step)
    return (-pd.Series(slope, index=X.index)).groupby(X["zone"]).mean()


def confounding_report(df: pd.DataFrame) -> pd.Series:
    """Raw correlation between our price and demand per zone. Spoiler: it's POSITIVE."""
    return df.groupby("zone")[["our_price", "demand"]].apply(lambda g: g["our_price"].corr(g["demand"]))


# ---------------------------------------------------------------------------
# Plot: learned vs true demand & revenue curves
# ---------------------------------------------------------------------------
LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "glm": "#2a78d6",
    "xgb": "#eb6834",
}


def plot_price_curves(models: dict, test: pd.DataFrame, path: Path, config: SimulationConfig) -> Path:
    """For one rush-hour situation per zone, sweep our price and compare curves."""
    when = pd.Timestamp("2025-11-18 18:00")  # a dry Tuesday at 6 pm in the test period
    fig, axes = plt.subplots(2, len(ZONES), figsize=(13, 7.5), facecolor=LIGHT["surface"])

    for col, zone in enumerate(config.zones):
        competitor = test.loc[(test["zone"] == zone.name) & (test["hour"] == 18), "competitor_price"].median()
        prices = np.linspace(0.5 * zone.base_fare, 3.0 * zone.base_fare, 200)
        context = pd.DataFrame(
            {
                "timestamp": when,
                "zone": zone.name,
                "is_raining": False,
                "our_price": prices,
                "competitor_price": competitor,
            }
        )
        curves = {
            "True market": (true_expected_demand(context, config), LIGHT["ink"], "--"),
            "Poisson GLM": (predict_demand(models["poisson_glm"], context), LIGHT["glm"], "-"),
            "XGBoost": (predict_demand(models["xgboost"], context), LIGHT["xgb"], "-"),
        }
        for row, ylabel in enumerate(["Expected ride requests / hour", "Expected revenue / hour ($)"]):
            ax = axes[row, col]
            ax.set_facecolor(LIGHT["surface"])
            for label, (demand, color, style) in curves.items():
                values = demand if row == 0 else prices * demand  # row 0: demand, row 1: revenue
                ax.plot(prices, values, color=color, linestyle=style, linewidth=2, label=label)
            if row == 1:
                p_star = 1 / zone.price_sensitivity
                ax.axvline(p_star, color=LIGHT["muted"], linewidth=1, linestyle=":")
                ax.annotate(
                    f"true p* = 1/β = ${p_star:.2f}",
                    xy=(p_star, ax.get_ylim()[0]),
                    xytext=(4, 6),
                    textcoords="offset points",
                    color=LIGHT["ink2"],
                    fontsize=9,
                )
            ax.set_title(
                f"{zone.name} (β = {zone.price_sensitivity})" if row == 0 else "",
                color=LIGHT["ink"],
                fontsize=11,
                loc="left",
            )
            ax.set_xlabel("Our price ($)" if row == 1 else "", color=LIGHT["ink2"])
            ax.set_ylabel(ylabel if col == 0 else "", color=LIGHT["ink2"])
            ax.grid(color=LIGHT["grid"], linewidth=0.8)
            ax.tick_params(colors=LIGHT["muted"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(LIGHT["axis"])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        ncol=3,
        frameon=False,
        labelcolor=LIGHT["ink"],
    )
    fig.suptitle(
        "Learned vs. true price response: weekday 6 pm, dry weather",
        y=0.995,
        color=LIGHT["ink"],
        fontsize=13,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=LIGHT["surface"])
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame, test_start: str = DEFAULT_TEST_START) -> dict:
    config = SimulationConfig()
    train, test = time_split(df, test_start)
    X_train, y_train = train[MODEL_INPUT_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[MODEL_INPUT_COLUMNS], test[TARGET_COLUMN]
    logger.info(
        "Train: %s rows (until %s) | Test: %s rows (from %s)",
        f"{len(train):,}",
        train["timestamp"].max().date(),
        f"{len(test):,}",
        test_start,
    )

    models = build_models()
    metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)
        metrics[name] = evaluate(y_test, predict_demand(model, X_test))
    # The "perfect" model: knows the true formula. The best any model could hope for.
    metrics["true_formula (ceiling)"] = evaluate(y_test, true_expected_demand(X_test, config))

    # Did the models learn beta? (The baseline ignores price, so it learned beta = 0.)
    betas = pd.DataFrame(
        {
            "true_beta": pd.Series({z.name: z.price_sensitivity for z in config.zones}),
            "poisson_glm": estimate_price_sensitivity(lambda X: predict_demand(models["poisson_glm"], X), X_test),
            "xgboost": estimate_price_sensitivity(lambda X: predict_demand(models["xgboost"], X), X_test),
        }
    )

    # Champion = best test deviance among models that can actually be used for pricing.
    price_aware = ["poisson_glm", "xgboost"]
    champion = min(price_aware, key=lambda name: metrics[name]["poisson_deviance"])
    return {
        "models": models,
        "metrics": metrics,
        "betas": betas,
        "champion": champion,
        "train": train,
        "test": test,
        "config": config,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare demand models.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START, help="first timestamp of the test period")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--track", action="store_true", help="log this run to MLflow and register the champion")
    return parser.parse_args(argv)


def track_training_run(result: dict, args: argparse.Namespace, artifacts: list[Path]) -> int | None:
    """Log every model's scores to MLflow and register the champion as a new version."""
    # Imported here so training (e.g. inside the Docker build) doesn't need MLflow installed.
    from pricing_engine import tracking

    tracking.configure()
    champion = result["champion"]
    metrics: dict[str, float] = {}
    for name, scores in result["metrics"].items():
        metrics.update(tracking.flatten_metrics(name, scores))
    for column in ("poisson_glm", "xgboost"):
        metrics.update(tracking.flatten_metrics(f"beta.{column}", result["betas"][column].to_dict()))
    run = tracking.log_run(
        f"train: {champion}",
        params={"test_start": args.test_start, "train_rows": len(result["train"]), "test_rows": len(result["test"]),
                "champion": champion},
        metrics=metrics,
        tags={"stage": "training"},
        artifacts=artifacts,
        model=result["models"][champion],
        register=True,
        model_tags={"reference_wape": str(result["metrics"][champion]["WAPE"]), "origin": "train.py"},
    )  # fmt: skip
    tracking.promote(run["version"])
    return run["version"]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    df = load_dataset(args.data)

    print("\n1) Confounding check: raw correlation(our_price, demand) per zone")
    print(confounding_report(df).round(3).to_string())

    result = train_and_evaluate(df, args.test_start)
    metrics_table = pd.DataFrame(result["metrics"]).T
    print("\n2) Test-set accuracy (Nov-Dec, never seen in training). Lower is better:")
    print(metrics_table.round(3).to_string())
    print("\n3) Price sensitivity beta: did the model learn the truth?")
    print(result["betas"].round(4).to_string())

    champion = result["champion"]
    model_path = save_model(result["models"][champion], args.model_out)
    metrics_path = model_path.with_name("metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "champion": champion,
                "test_start": args.test_start,
                "test_metrics": {
                    k: {m: round(float(v), 4) for m, v in d.items()} for k, d in result["metrics"].items()
                },
                "estimated_beta": result["betas"].round(4).to_dict(),
            },
            indent=2,
        )
    )
    figure = plot_price_curves(
        result["models"], result["test"], FIGURES_DIR / "price_response_curves.png", result["config"]
    )

    print(f"\n4) Champion: {champion}")
    if args.track:
        version = track_training_run(result, args, [figure, metrics_path])
        print(f"5) Logged to MLflow; registered and promoted as demand-model v{version}")
    logger.info("Saved model   -> %s", model_path)
    logger.info("Saved metrics -> %s", metrics_path)
    logger.info("Saved figure  -> %s", figure)


if __name__ == "__main__":
    main()
