"""Phase 9: the MLOps loop. Monitor -> (retrain -> compare -> promote).

Run from the project root (after Phases 1 and 2):

    python -m pricing_engine.mlops_pipeline              # both scenarios, fresh registry
    mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db   # then browse http://127.0.0.1:5000

Each "cycle" is what a scheduled job would do every few weeks in production:

  1. MONITOR  the champion model on the newest data (data, performance and concept drift)
  2. if there's an ALERT:
       RETRAIN  a challenger on all data, weighting recent weeks more (exponential decay)
       COMPARE  champion vs challenger on the most recent days, which neither has seen
       PROMOTE  the challenger only if it is clearly better (champion/challenger pattern)
  3. LOG      everything to MLflow, so every decision is explainable later

Two simulated futures (Jan-Feb 2026) show both outcomes:
    stable : the market behaves as before            -> expect no alert, no retraining
    drift  : riders 50% more price-sensitive, a 15%-cheaper competitor and more rain
                                                     -> expect an alert, retraining and promotion
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pricing_engine import tracking
from pricing_engine.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, MLOPS_DIR, SimulationConfig
from pricing_engine.data_generation import generate_dataset
from pricing_engine.features import MODEL_INPUT_COLUMNS, TARGET_COLUMN
from pricing_engine.model import build_models, load_model, predict_demand, save_model
from pricing_engine.monitoring import drift_report
from pricing_engine.train import estimate_price_sensitivity, evaluate, load_dataset

logger = logging.getLogger(__name__)

WINDOW_START = "2026-01-01"
WINDOW_WEEKS = 8
HOLDOUT_DAYS = 14  # newest days, kept back to compare champion vs challenger fairly
HALF_LIFE_DAYS = 14.0  # a 2-week-old row counts half as much as today's
MIN_IMPROVEMENT = 0.02  # challenger must beat the champion's deviance by at least 2%


# ----------------------------------------------------------------------------- simulated futures
def drifted_config(base: SimulationConfig) -> SimulationConfig:
    zones = tuple(dataclasses.replace(z, price_sensitivity=z.price_sensitivity * 1.5) for z in base.zones)
    return dataclasses.replace(base, zones=zones, competitor_price_level=0.85, rain_probability=0.35)


def simulate_window(scenario: str, seed: int = 2026, start: str = WINDOW_START, weeks: int = WINDOW_WEEKS):
    """New data after training. The simulation runs from the original start date so growth
    and seasonality continue smoothly, and we keep only the new window."""
    base = SimulationConfig()
    if scenario not in {"stable", "drift"}:
        raise ValueError(f"unknown scenario {scenario!r}")
    config = drifted_config(base) if scenario == "drift" else base
    n_days = (pd.Timestamp(start) - pd.Timestamp(base.start_date)).days + 7 * weeks
    df = generate_dataset(dataclasses.replace(config, seed=seed, n_days=n_days))
    return df[df["timestamp"] >= pd.Timestamp(start)].reset_index(drop=True), config


# ----------------------------------------------------------------------------- retraining
def recency_weights(timestamps: pd.Series, half_life_days: float = HALF_LIFE_DAYS) -> np.ndarray:
    """w = 0.5 ** (age / half_life): the newest row weighs 1, a half-life-old row 0.5, and so on.
    Same idea as the forgetting factor in Phase 7, applied to batch retraining."""
    ts = pd.to_datetime(timestamps)
    age_days = (ts.max() - ts).dt.total_seconds().to_numpy() / 86_400
    return 0.5 ** (age_days / half_life_days)


def train_challenger(history: pd.DataFrame, half_life_days: float = HALF_LIFE_DAYS):
    model = build_models()["poisson_glm"]
    model.fit(history[MODEL_INPUT_COLUMNS], history[TARGET_COLUMN],
              model__sample_weight=recency_weights(history["timestamp"], half_life_days))  # fmt: skip
    return model


def should_promote(champion_deviance: float, challenger_deviance: float,
                   min_improvement: float = MIN_IMPROVEMENT) -> bool:  # fmt: skip
    return challenger_deviance < champion_deviance * (1 - min_improvement)


# ----------------------------------------------------------------------------- registry bootstrap
def ensure_champion(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """If the registry is empty, register the currently deployed model as version 1 champion."""
    champion = tracking.get_champion()
    if champion is not None:
        return champion
    metrics = json.loads(model_path.with_name("metrics.json").read_text())
    name = metrics["champion"]
    reference_wape = metrics["test_metrics"][name]["WAPE"]
    out = tracking.log_run(
        "bootstrap: deployed model",
        params={"source": str(model_path.name), "model_type": name},
        metrics={"reference_wape": reference_wape},
        model=load_model(model_path),
        register=True,
        model_tags={"reference_wape": str(reference_wape), "origin": "bootstrap from deployed model"},
    )
    tracking.promote(out["version"])
    return tracking.get_champion()  # type: ignore[return-value]


# ----------------------------------------------------------------------------- one cycle
def run_cycle(history: pd.DataFrame, window: pd.DataFrame, label: str, *, report_dir: Path,
              config: SimulationConfig | None = None, force_retrain: bool = False) -> dict:  # fmt: skip
    config = config or SimulationConfig()
    champion = tracking.get_champion()
    if champion is None:
        raise RuntimeError("no champion in the registry; call ensure_champion() first")

    cutoff = window["timestamp"].max().normalize() - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    monitor, holdout = window[window["timestamp"] < cutoff], window[window["timestamp"] >= cutoff]

    # 1. MONITOR ------------------------------------------------------------------------------
    report = drift_report(champion["model"], history, monitor, float(champion["tags"]["reference_wape"]), config)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"drift_report_{label}.json"
    report_path.write_text(json.dumps({
        "status": report["status"], "reasons": report["reasons"], "warnings": report["warnings"],
        "data_drift": report["data_drift"].round(4).to_dict(orient="index"),
        "performance": report["performance"],
        "price_sensitivity": report["price_sensitivity"].round(4).to_dict(orient="index"),
    }, indent=2, default=str))  # fmt: skip
    monitor_metrics = {
        **tracking.flatten_metrics("psi", report["data_drift"]["psi"].to_dict()),
        **tracking.flatten_metrics("perf", {k: report["performance"][k] for k in ("wape", "bias")}),
        **tracking.flatten_metrics("beta", report["price_sensitivity"]["estimated_beta"].to_dict()),
    }
    tracking.log_run(
        f"monitor: {label}",
        params={"scenario": label, "champion_version": champion["version"],
                "window": f"{monitor['timestamp'].min():%Y-%m-%d} to {monitor['timestamp'].max():%Y-%m-%d}"},
        metrics=monitor_metrics,
        tags={"stage": "monitoring", "drift_status": report["status"]},
        artifacts=[report_path],
    )  # fmt: skip
    result = {"scenario": label, "status": report["status"], "reasons": report["reasons"],
              "warnings": report["warnings"], "report": report, "champion_before": champion["version"],
              "retrained": False, "promoted": False, "champion_after": champion["version"]}  # fmt: skip
    if report["status"] != "alert" and not force_retrain:
        return result

    # 2. RETRAIN -> COMPARE -> PROMOTE ------------------------------------------------------------
    challenger = train_challenger(pd.concat([history, monitor], ignore_index=True))
    scores = {
        "champion": evaluate(holdout[TARGET_COLUMN], predict_demand(champion["model"], holdout)),
        "challenger": evaluate(holdout[TARGET_COLUMN], predict_demand(challenger, holdout)),
    }
    betas = {name: estimate_price_sensitivity(lambda X, m=model: predict_demand(m, X), holdout)
             for name, model in (("champion", champion["model"]), ("challenger", challenger))}  # fmt: skip
    promote = should_promote(scores["champion"]["poisson_deviance"], scores["challenger"]["poisson_deviance"])
    run = tracking.log_run(
        f"retrain: {label}",
        params={"scenario": label, "half_life_days": HALF_LIFE_DAYS, "holdout_days": HOLDOUT_DAYS,
                "trained_through": f"{monitor['timestamp'].max():%Y-%m-%d}", "rows": len(history) + len(monitor)},
        metrics={**tracking.flatten_metrics("champion", scores["champion"]),
                 **tracking.flatten_metrics("challenger", scores["challenger"]),
                 **tracking.flatten_metrics("challenger.beta", betas["challenger"].to_dict())},
        tags={"stage": "retraining", "decision": "promoted" if promote else "kept champion",
              "trigger": "; ".join(report["reasons"]) or "forced"},
        model=challenger,
        register=True,
        model_tags={"reference_wape": str(scores["challenger"]["WAPE"]), "origin": f"retrained after {label}"},
    )  # fmt: skip
    if promote:
        tracking.promote(run["version"])
    result.update(retrained=True, promoted=promote, scores=scores, betas=betas, challenger=challenger,
                  champion_after=run["version"] if promote else champion["version"])  # fmt: skip
    return result


# ----------------------------------------------------------------------------- CLI
def print_cycle(result: dict) -> None:
    report = result["report"]
    print(f"\n{'=' * 78}\nScenario: {result['scenario'].upper()}   ->   status: {result['status'].upper()}")
    print(f"{'=' * 78}\nData drift (PSI: <0.10 ok, 0.10-0.25 warning, >0.25 big shift):")
    print(report["data_drift"].round(3).to_string())
    perf = report["performance"]
    print(f"\nPerformance: WAPE {perf['wape']:.1%} (reference {perf['reference_wape']:.1%}), "
          f"bias actual/predicted = {perf['bias']:.3f}")  # fmt: skip
    print("\nPrice sensitivity (concept drift):")
    print(report["price_sensitivity"].round(4).to_string())
    for reason in result["reasons"]:
        print(f"  ALERT: {reason}")
    for warning in result["warnings"]:
        print(f"  warning: {warning}")
    if not result["retrained"]:
        print(f"\nDecision: no alert -> keep champion v{result['champion_after']}, no retraining.")
        return
    scores = pd.DataFrame(result["scores"]).T[["WAPE", "poisson_deviance"]]
    scores["beta_downtown"] = [result["betas"][name]["downtown"] for name in scores.index]
    print(f"\nRetrained a challenger. Holdout = last {HOLDOUT_DAYS} days (unseen by both):")
    print(scores.round(4).to_string())
    verdict = "PROMOTED" if result["promoted"] else "kept the champion (not clearly better)"
    print(f"\nDecision: {verdict} -> champion is now v{result['champion_after']}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Monitor -> retrain -> promote, tracked in MLflow.")
    parser.add_argument("--scenario", choices=["stable", "drift", "both"], default="both")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tracking-dir", type=Path, default=MLOPS_DIR)
    parser.add_argument("--keep-history", action="store_true", help="don't wipe the registry first")
    parser.add_argument("--deploy-to", type=Path, help="export the final champion to this .joblib path")
    args = parser.parse_args(argv)

    if not args.keep_history and args.tracking_dir.exists():
        try:
            shutil.rmtree(args.tracking_dir)  # start the demo from a clean registry
        except PermissionError as exc:  # Windows can't delete a database another program has open
            raise SystemExit(
                f"Can't reset {args.tracking_dir}: a file is in use ({exc.filename}).\n"
                "Stop `mlflow ui` (or anything else using it) and run again, or pass --keep-history."
            ) from exc
    tracking.configure(args.tracking_dir)
    champion = ensure_champion(args.model)
    print(f"Registry ready: champion is v{champion['version']} ({champion['tags'].get('origin', '')})")

    history = load_dataset(args.data)
    scenarios = ["stable", "drift"] if args.scenario == "both" else [args.scenario]
    for scenario in scenarios:
        window, config = simulate_window(scenario)
        print_cycle(run_cycle(history, window, scenario, report_dir=args.tracking_dir / "reports", config=config))

    print("\nModel registry:")
    for version in tracking.list_versions():
        alias = f"  <- {', '.join(version['aliases'])}" if version["aliases"] else ""
        print(f"  v{version['version']}: {version['tags'].get('origin', '')}{alias}")
    if args.deploy_to:
        final = tracking.get_champion()
        save_model(final["model"], args.deploy_to)  # type: ignore[index]
        print(f"\nExported champion v{final['version']} -> {args.deploy_to}")  # type: ignore[index]
    print(f"\nBrowse every run:  mlflow ui --backend-store-uri sqlite:///{args.tracking_dir.name}/mlflow.db")


if __name__ == "__main__":
    main()
