# syntax=docker/dockerfile:1
#
# Single-container DEMO image: API + dashboard together, for free hosting platforms
# that run one container with one public port (Render, Hugging Face Spaces, ...).
# For a real deployment use compose.yaml, which keeps them as separate services.

# ---------------------------------------------------------------------------- base
FROM python:3.14-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# ---------------------------------------------------------------------------- deps (API + dashboard)
FROM base AS deps
RUN python -m venv /opt/venv
COPY requirements/api.txt requirements/dashboard.txt requirements/
RUN pip install -r requirements/api.txt -r requirements/dashboard.txt

# ---------------------------------------------------------------------------- trainer
FROM deps AS trainer
COPY requirements/train.txt requirements/
RUN pip install -r requirements/train.txt
COPY pricing_engine/ pricing_engine/
RUN python -m pricing_engine.data_generation \
    && python -m pricing_engine.train

# ---------------------------------------------------------------------------- runtime
FROM base AS runtime
ENV PRICING_MODEL_PATH=/app/models/demand_model.joblib \
    PORT=8501
RUN useradd --create-home --uid 10001 appuser
COPY --from=deps /opt/venv /opt/venv
COPY pricing_engine/ pricing_engine/
COPY dashboard/ dashboard/
COPY deploy/ deploy/
COPY --from=trainer /app/models/ models/
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "deploy/run_demo.py", "--check"]
CMD ["python", "deploy/run_demo.py"]
