"""A small HTTP client for the pricing API.

The dashboard never imports the pricing engine directly; it only talks to the API.
That's how real companies split systems: the UI team and the pricing team can
deploy independently, and the API is the contract between them.
"""

from __future__ import annotations

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"


class ApiError(RuntimeError):
    """Raised when the API can't be reached or rejects a request."""


def _default_http(base_url: str, timeout: float) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout)


class PricingClient:
    # Tests swap this factory for FastAPI's in-memory TestClient.
    http_factory = staticmethod(_default_http)

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 10.0):
        self.base_url = base_url
        self._http = self.http_factory(base_url, timeout)

    def health(self) -> dict:
        return self._request("GET", "/health")

    def zones(self) -> list[dict]:
        return self._request("GET", "/v1/zones")

    def price(self, payload: dict) -> dict:
        return self._request("POST", "/v1/price", payload)

    def demand_curve(self, payload: dict) -> dict:
        return self._request("POST", "/v1/demand-curve", payload)

    def _request(self, method: str, path: str, payload: dict | None = None):
        try:
            response = self._http.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise ApiError(f"Cannot reach the pricing API at {self.base_url} ({exc.__class__.__name__}).") from exc
        if response.status_code >= 400:
            raise ApiError(f"The API rejected the request ({response.status_code}): {response.text}")
        return response.json()
