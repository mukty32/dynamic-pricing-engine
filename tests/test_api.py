"""Tests for Phase 4: the FastAPI service. Run with:  pytest

TestClient sends real HTTP requests to the app in memory, with no server needed.
We inject the fake exponential model, so we know the correct answers in advance.
"""

import numpy as np
import pytest
from fakes import BETA, C, FakeExponentialModel
from fastapi.testclient import TestClient

from pricing_engine.api.main import create_app

CONTEXT = {
    "timestamp": "2025-11-18T18:00:00",
    "zone": "downtown",
    "is_raining": False,
    "competitor_price": 40.0,
    "drivers_available": 500,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(model=FakeExponentialModel(), model_name="fake")) as test_client:
        yield test_client  # the `with` block runs the startup (lifespan) code


# ----------------------------- system -------------------------------------
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_name"] == "fake"


def test_every_response_has_timing_and_request_id(client):
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert float(response.headers["X-Process-Time-ms"]) >= 0


def test_zones(client):
    names = {zone["name"] for zone in client.get("/v1/zones").json()}
    assert names == {"downtown", "airport", "suburbs"}


def test_api_zone_list_matches_the_config():
    # The Zone enum is written out by hand; this guards against it drifting from the config.
    from pricing_engine.api.schemas import Zone
    from pricing_engine.config import SimulationConfig

    assert {zone.value for zone in Zone} == {zone.name for zone in SimulationConfig().zones}


# ----------------------------- pricing ------------------------------------
def test_price_with_plenty_of_drivers_is_one_over_beta(client):
    body = client.post("/v1/price", json=CONTEXT).json()
    assert body["price"] == pytest.approx(1 / BETA, abs=0.01)
    assert body["binding_constraint"].startswith("none")


def test_price_with_few_drivers_is_market_clearing(client):
    body = client.post("/v1/price", json={**CONTEXT, "drivers_available": 30}).json()
    assert body["price"] == pytest.approx(np.log(C / 60) / BETA, abs=0.02)
    assert body["binding_constraint"] == "supply"


def test_custom_rules_override_defaults(client):
    request = {**CONTEXT, "drivers_available": 30, "rules": {"max_multiplier": 1.5}}
    body = client.post("/v1/price", json=request).json()
    assert body["price"] == pytest.approx(18.0)  # 1.5 x $12 base fare
    assert body["binding_constraint"] == "surge_cap"


def test_demand_shock_raises_surge_price(client):
    normal = client.post("/v1/price", json={**CONTEXT, "drivers_available": 30}).json()["price"]
    concert = client.post("/v1/price", json={**CONTEXT, "drivers_available": 30, "demand_multiplier": 2.0}).json()
    assert concert["price"] - normal == pytest.approx(np.log(2) / BETA, abs=0.03)


def test_timezone_suffix_is_accepted(client):
    response = client.post("/v1/price", json={**CONTEXT, "timestamp": "2025-11-18T18:00:00+06:00"})
    assert response.status_code == 200
    assert response.json()["timestamp"] == "2025-11-18T18:00:00"


def test_forecast_matches_the_demand_formula(client):
    body = client.post("/v1/forecast", json={**CONTEXT, "our_price": 20.0}).json()
    assert body["expected_demand"] == pytest.approx(C * np.exp(-BETA * 20.0))


def test_batch_keeps_order(client):
    items = [{**CONTEXT, "drivers_available": d} for d in (500, 30, 2)]
    results = client.post("/v1/price/batch", json={"items": items}).json()["results"]
    assert [r["binding_constraint"][:4] for r in results] == ["none", "supp", "surg"]


def test_demand_curve_peaks_at_the_optimum(client):
    body = client.post("/v1/demand-curve", json={**CONTEXT, "drivers_available": 30, "n_points": 200}).json()
    assert len(body["points"]) == 200
    best_point = max(body["points"], key=lambda p: p["expected_revenue"])
    assert best_point["price"] == pytest.approx(body["optimum"]["price"], abs=0.2)


# ----------------------------- validation ---------------------------------
@pytest.mark.parametrize(
    "bad_fields",
    [
        {"competitor_price": -5},  # prices must be positive
        {"drivers_available": -1},  # can't have negative drivers
        {"zone": "mars"},  # unknown zone
        {"timestamp": "not-a-date"},  # unparseable time
        {"competitor_prcie": 10},  # typo -> extra field is forbidden
        {"rules": {"max_multiplier": 0.5}},  # surge cap below 1x makes no sense
    ],
)
def test_bad_input_is_rejected_with_422(client, bad_fields):
    response = client.post("/v1/price", json={**CONTEXT, **bad_fields})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in CONTEXT.items() if k != "drivers_available"}
    response = client.post("/v1/price", json=incomplete)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "drivers_available"


def test_batch_size_limit(client):
    response = client.post("/v1/price/batch", json={"items": [CONTEXT] * 1001})
    assert response.status_code == 422
