"""Pydantic schemas: the API's contract with the outside world.

Every request is checked BEFORE it reaches our math. A negative price, an unknown
zone or a typo in a field name is rejected with a clear 422 error that says
exactly which field is wrong, so bad input can never produce a silently wrong price.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pricing_engine.optimizer import PricingConstraints

_DEFAULT_RULES = PricingConstraints()


class Zone(StrEnum):
    """Zones the API accepts. Written out explicitly so type checkers and the API docs
    can see every value. A test makes sure it always matches SimulationConfig.zones."""

    DOWNTOWN = "downtown"
    AIRPORT = "airport"
    SUBURBS = "suburbs"


EXAMPLE_CONTEXT: dict[str, Any] = {
    "timestamp": "2025-11-18T18:00:00",
    "zone": "downtown",
    "is_raining": False,
    "competitor_price": 18.0,
    "drivers_available": 40,
}


class StrictModel(BaseModel):
    # extra="forbid": an unexpected field (e.g. a typo like "competitor_prcie") is an error,
    # not something we silently ignore.
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- requests
class MarketContext(StrictModel):
    """What's happening in one zone right now."""

    timestamp: datetime = Field(description="Local city time. Any timezone suffix is ignored.")
    zone: Zone
    is_raining: bool = False
    competitor_price: float = Field(gt=0, le=1_000, description="Competitor's current price in $")
    drivers_available: int = Field(ge=0, le=100_000, description="Drivers online in this zone")
    current_price: float | None = Field(
        default=None, gt=0, le=1_000, description="Our current price (only used by max_change)"
    )
    demand_multiplier: float = Field(
        default=1.0, gt=0, le=10, description="Manual demand shock, e.g. 2.0 when a big event ends nearby"
    )

    @field_validator("timestamp")
    @classmethod
    def use_local_wall_clock(cls, value: datetime) -> datetime:
        # Our model learned "6 pm = rush hour" in LOCAL time, so we keep the wall-clock
        # reading and drop the timezone label.
        return value.replace(tzinfo=None)


class PricingRules(StrictModel):
    """Optional per-request override of the business rules."""

    min_multiplier: float = Field(default=_DEFAULT_RULES.min_multiplier, gt=0, le=1)
    max_multiplier: float = Field(default=_DEFAULT_RULES.max_multiplier, ge=1, le=10)
    max_competitor_premium: float = Field(default=_DEFAULT_RULES.max_competitor_premium, ge=0, le=5)
    max_change: float | None = Field(default=_DEFAULT_RULES.max_change, gt=0, le=1)

    def to_constraints(self) -> PricingConstraints:
        return PricingConstraints(**self.model_dump())


class PriceRequest(MarketContext):
    rules: PricingRules | None = None

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_CONTEXT]})


class ForecastRequest(MarketContext):
    our_price: float = Field(gt=0, le=1_000, description="The price to forecast demand for")

    model_config = ConfigDict(json_schema_extra={"examples": [{**EXAMPLE_CONTEXT, "our_price": 20.0}]})


class BatchPriceRequest(StrictModel):
    items: list[MarketContext] = Field(min_length=1, max_length=1_000)
    rules: PricingRules | None = None


class DemandCurveRequest(PriceRequest):
    n_points: int = Field(60, ge=5, le=500, description="How many prices to sample between the bounds")


# ----------------------------------------------------------------------------- responses
class PriceResponse(BaseModel):
    zone: Zone
    timestamp: datetime
    price: float
    surge_multiplier: float
    expected_demand: float
    expected_rides: float
    expected_revenue: float
    fulfillment_rate: float
    capacity: float
    lower_bound: float
    upper_bound: float
    binding_constraint: str
    base_fare_revenue: float
    revenue_uplift: float
    model_name: str


class ForecastResponse(BaseModel):
    zone: Zone
    timestamp: datetime
    our_price: float
    expected_demand: float
    expected_rides: float
    expected_revenue: float
    capacity: float
    model_name: str


class BatchPriceResponse(BaseModel):
    results: list[PriceResponse]


class CurvePoint(BaseModel):
    price: float
    expected_demand: float
    expected_rides: float
    expected_revenue: float


class DemandCurveResponse(BaseModel):
    points: list[CurvePoint]
    optimum: PriceResponse


class ZoneInfo(BaseModel):
    name: str
    base_fare: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_trained_at: str | None
    version: str
