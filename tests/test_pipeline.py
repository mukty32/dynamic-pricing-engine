"""End-to-end smoke test: every command-line step runs, in order, on a small dataset.

This is exactly what the Docker build does (generate -> train), plus the two
evaluation scripts. Output folders are redirected to a temporary directory so the
test never overwrites the real reports and figures.
"""

import json

import pricing_engine.backtest as backtest
import pricing_engine.mlops_pipeline as mlops_pipeline
import pricing_engine.network as network
import pricing_engine.online_experiment as online_experiment
import pricing_engine.train as train
from pricing_engine import data_generation, tracking


def test_full_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    for module in (train, online_experiment, network):
        monkeypatch.setattr(module, "FIGURES_DIR", tmp_path / "figures")
    for module in (backtest, online_experiment, network):
        monkeypatch.setattr(module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(tracking, "MLOPS_DIR", tmp_path / "mlops")  # `train --track` default location
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)  # left behind by other tests' configure()

    data, model = tmp_path / "rides.csv", tmp_path / "models" / "demand_model.joblib"
    test_start = "2025-04-01"  # 120 simulated days -> roughly the last month is the test period

    data_generation.main(["--days", "120", "--output", str(data)])
    train.main(["--data", str(data), "--test-start", test_start, "--model-out", str(model), "--track"])
    backtest.main(["--data", str(data), "--model", str(model), "--test-start", test_start])
    online_experiment.main(["--data", str(data), "--model", str(model), "--test-start", test_start, "--seeds", "1"])
    network.main(["--data", str(data), "--model", str(model), "--test-start", test_start, "--max-hours", "24"])
    assert json.loads((tmp_path / "reports" / "network.json").read_text())["backtest"]["network"]["profit"] > 0
    deployed = tmp_path / "deployed" / "demand_model.joblib"
    # --keep-history: the champion registered by `train --track` above must be picked up from the registry.
    mlops_pipeline.main(["--data", str(data), "--model", str(model), "--tracking-dir", str(tmp_path / "mlops"),
                         "--scenario", "drift", "--keep-history", "--deploy-to", str(deployed)])  # fmt: skip
    assert deployed.exists()
    assert tracking.list_versions()[0]["tags"]["origin"] == "train.py"  # v1 came from training, not bootstrap

    metrics = json.loads((model.parent / "metrics.json").read_text())
    assert metrics["champion"] in {"poisson_glm", "xgboost"}
    policies = json.loads((tmp_path / "reports" / "backtest.json").read_text())["policies"]
    assert policies["optimizer + GLM model"]["total_revenue"] > policies["static base fare"]["total_revenue"]
    assert (tmp_path / "figures" / "price_response_curves.png").exists()
    assert (tmp_path / "figures" / "online_learning.png").exists()
