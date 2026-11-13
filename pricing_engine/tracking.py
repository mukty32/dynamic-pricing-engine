"""Phase 9: experiment tracking and the model registry, with MLflow.

Why: after a few months nobody remembers which data, settings and code produced
the model running in production, or how good it was. MLflow records every run:

    params    -> the settings (test period, number of rows, half-life, ...)
    metrics   -> the scores (WAPE, Poisson deviance, estimated beta, PSI, ...)
    artifacts -> files (figures, drift reports)
    models    -> the trained model itself, as a numbered VERSION in the registry

The registry keeps every version of "demand-model". An ALIAS called "champion"
points at the version that should serve traffic. Promoting a new model just moves
the alias, and rolling back means moving it back.

Everything is stored locally in mlops/ (a SQLite database + an artifact folder).
Browse it with:   mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep the CLI output quiet

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from pricing_engine.config import MLOPS_DIR  # noqa: E402

EXPERIMENT_NAME = "dynamic-pricing"
REGISTERED_MODEL = "demand-model"
CHAMPION_ALIAS = "champion"


def configure(tracking_dir: Path | None = None) -> str:
    """Point MLflow at a local SQLite database and artifact folder. Returns the tracking URI.

    An explicit `tracking_dir` always wins. Only without one do we honour a MLFLOW_TRACKING_URI
    environment variable (e.g. a shared MLflow server). Careful: mlflow.set_tracking_uri() itself
    writes that variable, so trusting it first would silently reuse the PREVIOUS database.
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    artifact_location: str | None = None
    if tracking_dir is None and env_uri:
        uri = env_uri
    else:
        folder = Path(tracking_dir or MLOPS_DIR).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{(folder / 'mlflow.db').as_posix()}"
        artifact_location = (folder / "artifacts").as_uri()
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def clean_key(key: str) -> str:
    """MLflow keys may only contain letters, digits, _ - . space and /."""
    return re.sub(r"[^\w\-. /]", "", key).strip()


def flatten_metrics(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    return {clean_key(f"{prefix}.{name}" if prefix else name): float(v) for name, v in values.items()}


def log_run(
    run_name: str,
    *,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    artifacts: Iterable[Path] = (),
    model: Any = None,
    register: bool = False,
    model_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Record one run. If `register`, the model becomes a new numbered version in the registry."""
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params({clean_key(k): v for k, v in params.items()})
        if metrics:
            mlflow.log_metrics({clean_key(k): float(v) for k, v in metrics.items()})
        if tags:
            mlflow.set_tags(tags)
        for path in artifacts:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
        version = None
        if model is not None:
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=REGISTERED_MODEL if register else None,
                serialization_format="cloudpickle",  # our pipelines contain custom Python functions
                pip_requirements=[],  # the image's requirements files are the source of truth
            )
            version = info.registered_model_version
            if register and version is not None and model_tags:
                client = MlflowClient()
                for key, value in model_tags.items():
                    client.set_model_version_tag(REGISTERED_MODEL, str(version), key, value)
        return {"run_id": run.info.run_id, "version": version}


def promote(version: int | str) -> None:
    """Make `version` the champion. The previous champion stays in the registry for rollback."""
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, str(version))


def get_champion() -> dict[str, Any] | None:
    """The current champion's version, tags and loaded model, or None if there is none yet."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@{CHAMPION_ALIAS}")
    return {"version": int(mv.version), "tags": dict(mv.tags), "model": model}


def list_versions() -> list[dict[str, Any]]:
    client = MlflowClient()
    # search_model_versions doesn't fill in aliases, so read them from the registered model itself.
    aliases: dict[str, list[str]] = {}
    for alias, version in client.get_registered_model(REGISTERED_MODEL).aliases.items():
        aliases.setdefault(str(version), []).append(alias)
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    return sorted(
        (
            {"version": int(v.version), "aliases": aliases.get(str(v.version), []), "tags": dict(v.tags)}
            for v in versions
        ),
        key=lambda row: row["version"],
    )
