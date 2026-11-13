# syntax=docker/dockerfile:1
#
# Pricing API image, built in stages so the final image only holds what it needs.
#
#   base     -> slim Python + the one system library XGBoost needs
#   deps     -> the API's Python packages, in a virtual environment
#   trainer  -> deps + matplotlib; generates the data and TRAINS the model
#   runtime  -> base + deps' packages + our code + the trained model (no data, no plotting libs)
#
# Training inside the build makes the image self-contained and reproducible:
# the same seed always produces the same data and the same model.

# ---------------------------------------------------------------------------- base
FROM python:3.14-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"
# XGBoost uses OpenMP for parallel trees; slim images don't ship that library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# ---------------------------------------------------------------------------- deps
FROM base AS deps
RUN python -m venv /opt/venv
# Copy ONLY the requirements first: Docker caches this layer, so editing code
# doesn't trigger a slow reinstall of every package.
COPY requirements/api.txt requirements/api.txt
RUN pip install -r requirements/api.txt

# ---------------------------------------------------------------------------- trainer
FROM deps AS trainer
COPY requirements/train.txt requirements/train.txt
RUN pip install -r requirements/train.txt
COPY pricing_engine/ pricing_engine/
RUN python -m pricing_engine.data_generation \
    && python -m pricing_engine.train

# ---------------------------------------------------------------------------- runtime
FROM base AS runtime
ENV PRICING_MODEL_PATH=/app/models/demand_model.joblib
# Never run a web server as root: if it's ever compromised, the attacker gets no admin rights.
RUN useradd --create-home --uid 10001 appuser
COPY --from=deps /opt/venv /opt/venv
COPY pricing_engine/ pricing_engine/
COPY --from=trainer /app/models/ models/
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "pricing_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
