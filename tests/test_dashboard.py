"""Tests for Phase 5: the dashboard. Run with:  pytest

Streamlit's AppTest runs the real dashboard script headlessly: we can read its
metrics and move its sliders from Python. The API underneath is the real FastAPI
app, running in memory with the fake exponential model.
"""

from pathlib import Path

import numpy as np
import pytest
from fakes import BETA, C, FakeExponentialModel
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from dashboard.analysis import explain_binding, fit_exponential, pen_and_paper_optimum
from dashboard.client import ApiError, PricingClient
from pricing_engine.api.main import create_app

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture
def in_memory_api(monkeypatch):
    """Make every PricingClient talk to an in-memory API instead of the network."""
    with TestClient(create_app(model=FakeExponentialModel(), model_name="fake")) as api:
        monkeypatch.setattr(PricingClient, "http_factory", staticmethod(lambda base_url, timeout: api))
        yield api


# ----------------------------- math helpers -------------------------------
def test_fit_exponential_recovers_beta_and_C():
    prices = np.linspace(5, 40, 50)
    beta, constant = fit_exponential(prices, C * np.exp(-BETA * prices))
    assert beta == pytest.approx(BETA)
    assert constant == pytest.approx(C)


def test_pen_and_paper_optimum_cases():
    plenty = pen_and_paper_optimum(BETA, C, capacity=1000, lower=9.6, upper=36)
    scarce = pen_and_paper_optimum(BETA, C, capacity=60, lower=9.6, upper=36)
    assert plenty["optimum"] == pytest.approx(12.5)
    assert scarce["optimum"] == pytest.approx(np.log(C / 60) / BETA)


def test_every_binding_constraint_has_an_explanation():
    rules = {"min_multiplier": 0.8, "max_multiplier": 3.0, "max_competitor_premium": 0.5}
    for binding in [
        "none (natural peak...)",
        "supply",
        "surge_cap",
        "competitor_premium",
        "min_price",
        "price_stability",
        "no_drivers",
    ]:
        headline, text = explain_binding(binding, price=20, capacity=60, competitor_price=18, base_fare=12, rules=rules)
        assert headline and text


# ----------------------------- client -------------------------------------
def test_client_talks_to_the_api(in_memory_api):
    client = PricingClient("http://test")
    assert client.health()["model_name"] == "fake"
    assert {zone["name"] for zone in client.zones()} == {"downtown", "airport", "suburbs"}


def test_client_raises_friendly_error_when_api_is_down():
    client = PricingClient("http://127.0.0.1:9", timeout=0.5)  # nothing listens on port 9
    with pytest.raises(ApiError, match="Cannot reach"):
        client.health()


# ----------------------------- the app itself -----------------------------
def test_app_shows_help_when_api_is_down(monkeypatch):
    monkeypatch.setenv("PRICING_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert "Cannot reach" in at.error[0].value


def test_app_updates_price_when_sliders_move(in_memory_api, monkeypatch):
    monkeypatch.setenv("PRICING_API_URL", "http://in-memory")
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    # Defaults: downtown, 40 drivers -> capacity 80 -> market-clearing price ln(400/80)/0.08
    assert at.metric[0].value == f"${np.log(C / 80) / BETA:.2f}"

    at.slider(key="drivers").set_value(300).run()  # plenty of drivers -> natural peak 1/beta
    assert at.metric[0].value == "$12.50"

    at.slider(key="drivers").set_value(40).run()
    at.slider(key="shock").set_value(2.0).run()  # a concert doubles demand -> wants ln(2*400/80)/0.08 = $28.78 ...
    assert at.metric[0].value == "$27.00"  # ... but the competitor rule caps it at 1.5 x $18
    assert "competitor" in at.info[0].value

    at.slider(key="premium").set_value(100).run()  # loosen the rule -> the market-clearing price is allowed
    assert at.metric[0].value == f"${np.log(2 * C / 80) / BETA:.2f}"
