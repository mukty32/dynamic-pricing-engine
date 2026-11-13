"""Phase 4: the FastAPI service.

Run it locally from the project root:

    uvicorn pricing_engine.api.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive documentation where you
can send requests from your browser.

Endpoints
    GET  /health              is the service up, and which model is loaded?
    GET  /v1/zones            the zones we price, with their base fares
    POST /v1/forecast         "how many riders would request at THIS price?"
    POST /v1/price            "what's the BEST price right now?"   <- the main one
    POST /v1/price/batch      the best price for up to 1,000 situations at once
    POST /v1/demand-curve     demand & revenue across the allowed price range (for charts)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from pricing_engine import __version__
from pricing_engine.api.schemas import (
    BatchPriceRequest,
    BatchPriceResponse,
    CurvePoint,
    DemandCurveRequest,
    DemandCurveResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MarketContext,
    PriceRequest,
    PriceResponse,
    PricingRules,
    ZoneInfo,
)
from pricing_engine.config import DEFAULT_MODEL_PATH, SimulationConfig
from pricing_engine.model import load_model
from pricing_engine.optimizer import PriceOptimizer, PriceRecommendation, PricingContext

logger = logging.getLogger("pricing_engine.api")

MODEL_PATH_ENV = "PRICING_MODEL_PATH"  # lets Docker/servers point to a different model file


@dataclass(frozen=True)
class Engine:
    """Everything loaded ONCE at startup and shared by all requests."""

    model: object
    model_name: str
    trained_at: str | None
    config: SimulationConfig


def load_engine(model_path: Path) -> Engine:
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}. Run `python -m pricing_engine.train` first.")
    metrics_path = model_path.with_name("metrics.json")
    name = json.loads(metrics_path.read_text())["champion"] if metrics_path.exists() else model_path.stem
    trained_at = datetime.fromtimestamp(model_path.stat().st_mtime, tz=UTC).isoformat(timespec="seconds")
    return Engine(load_model(model_path), name, trained_at, SimulationConfig())


# ----------------------------------------------------------------------------- helpers
def to_context(ctx: MarketContext) -> PricingContext:
    """Convert the validated API object into the engine's own type."""
    return PricingContext(
        timestamp=ctx.timestamp,
        zone=ctx.zone.value,
        is_raining=ctx.is_raining,
        competitor_price=ctx.competitor_price,
        drivers_available=ctx.drivers_available,
        current_price=ctx.current_price,
        demand_multiplier=ctx.demand_multiplier,
    )


def build_optimizer(engine: Engine, rules: PricingRules | None) -> PriceOptimizer:
    constraints = (rules or PricingRules()).to_constraints()
    return PriceOptimizer(engine.model, constraints=constraints, config=engine.config)


def to_price_response(ctx: MarketContext, rec: PriceRecommendation | dict, engine: Engine) -> PriceResponse:
    fields = dataclasses.asdict(rec) if dataclasses.is_dataclass(rec) else rec
    return PriceResponse(zone=ctx.zone, timestamp=ctx.timestamp, model_name=engine.model_name, **fields)


def get_engine(request: Request) -> Engine:
    """FastAPI 'dependency': hands each endpoint the shared engine."""
    return request.app.state.engine


# ----------------------------------------------------------------------------- app factory
def create_app(model: object | None = None, model_name: str | None = None) -> FastAPI:
    """Build the app. Tests pass in a `model`; in production it's loaded from disk.

    This "factory" pattern means tests never need a real model file.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once when the server starts: load the model before accepting any request.
        if model is not None:
            app.state.engine = Engine(model, model_name or type(model).__name__, None, SimulationConfig())
        else:
            app.state.engine = load_engine(Path(os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH)))
        logger.info("Model '%s' loaded", app.state.engine.model_name)
        yield

    app = FastAPI(
        title="Dynamic Pricing Engine",
        version=__version__,
        description="Demand forecasting + revenue-maximising price optimisation for a ride-hailing market.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_timing(request: Request, call_next):
        """Tag every response with an ID and how long it took. Essential for debugging in production."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "%s %s -> %d (%.1f ms) [%s]", request.method, request.url.path, response.status_code, elapsed_ms, request_id
        )
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        # Anything our engine rejects becomes a clean 422 instead of a 500 crash.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # NOTE: the endpoints below are plain `def`, not `async def`. Our maths is CPU work,
    # so FastAPI runs each call in a worker thread and the server stays responsive.

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(engine: Engine = Depends(get_engine)) -> HealthResponse:
        return HealthResponse(
            status="ok", model_name=engine.model_name, model_trained_at=engine.trained_at, version=__version__
        )

    @app.get("/v1/zones", response_model=list[ZoneInfo], tags=["pricing"])
    def zones(engine: Engine = Depends(get_engine)) -> list[ZoneInfo]:
        return [ZoneInfo(name=z.name, base_fare=z.base_fare) for z in engine.config.zones]

    @app.post("/v1/forecast", response_model=ForecastResponse, tags=["pricing"])
    def forecast(req: ForecastRequest, engine: Engine = Depends(get_engine)) -> ForecastResponse:
        optimizer = build_optimizer(engine, None)
        demand = float(optimizer.demand(to_context(req).to_frame(), [req.our_price])[0])
        capacity = float(req.drivers_available * engine.config.rides_per_driver_per_hour)
        rides = min(demand, capacity)
        return ForecastResponse(
            zone=req.zone,
            timestamp=req.timestamp,
            our_price=req.our_price,
            expected_demand=demand,
            expected_rides=rides,
            expected_revenue=req.our_price * rides,
            capacity=capacity,
            model_name=engine.model_name,
        )

    @app.post("/v1/price", response_model=PriceResponse, tags=["pricing"])
    def price(req: PriceRequest, engine: Engine = Depends(get_engine)) -> PriceResponse:
        rec = build_optimizer(engine, req.rules).recommend(to_context(req))
        return to_price_response(req, rec, engine)

    @app.post("/v1/price/batch", response_model=BatchPriceResponse, tags=["pricing"])
    def price_batch(req: BatchPriceRequest, engine: Engine = Depends(get_engine)) -> BatchPriceResponse:
        # Batch mode skips the SciPy polish step for speed (prices may differ by a few cents).
        frame = pd.DataFrame([dataclasses.asdict(to_context(item)) for item in req.items])
        recs = build_optimizer(engine, req.rules).recommend_many(frame)
        return BatchPriceResponse(
            results=[
                to_price_response(item, row, engine)
                for item, row in zip(req.items, recs.to_dict(orient="records"), strict=True)
            ]
        )

    @app.post("/v1/demand-curve", response_model=DemandCurveResponse, tags=["pricing"])
    def demand_curve(req: DemandCurveRequest, engine: Engine = Depends(get_engine)) -> DemandCurveResponse:
        optimizer = build_optimizer(engine, req.rules)
        context = to_context(req)
        rec = optimizer.recommend(context)
        prices = np.linspace(rec.lower_bound, rec.upper_bound, req.n_points)
        frame = context.to_frame()
        demand = optimizer.demand(frame.loc[frame.index.repeat(req.n_points)], prices)
        rides = np.minimum(demand, rec.capacity)
        points = [
            CurvePoint(price=float(p), expected_demand=float(d), expected_rides=float(r), expected_revenue=float(p * r))
            for p, d, r in zip(prices, demand, rides, strict=True)
        ]
        return DemandCurveResponse(points=points, optimum=to_price_response(req, rec, engine))

    return app


app = create_app()
