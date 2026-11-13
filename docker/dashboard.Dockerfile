# syntax=docker/dockerfile:1
#
# Dashboard image: Streamlit + an HTTP client. It holds no ML code and no model;
# it asks the API for everything.

FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PRICING_API_URL=http://api:8000
WORKDIR /app

COPY requirements/dashboard.txt requirements/dashboard.txt
RUN pip install -r requirements/dashboard.txt

RUN useradd --create-home --uid 10001 appuser
COPY dashboard/ dashboard/
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
