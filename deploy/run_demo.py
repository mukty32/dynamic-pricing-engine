"""Run the API and the dashboard together in ONE container, for free hosting platforms.

Free hosts (Render, Hugging Face Spaces, ...) give you one container with one public
port. So this script:
  1. starts the API PRIVATELY on 127.0.0.1:8000 (not reachable from the internet),
  2. waits until its /health check passes,
  3. starts the dashboard on the PUBLIC port ($PORT, which the host sets),
  4. shuts everything down if either process dies, so the platform restarts the container.

In a real company you'd run them as separate services (see compose.yaml). One container
is a cost-saving shortcut that's fine for a portfolio demo.

    python deploy/run_demo.py           # start both
    python deploy/run_demo.py --check   # health check used by Docker
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_PORT = 8000
API_URL = f"http://127.0.0.1:{API_PORT}"


def public_port() -> str:
    return os.environ.get("PORT", "8501")


def is_healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


def wait_for(url: str, process: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited with code {process.returncode} before becoming healthy")
        if is_healthy(url):
            return
        time.sleep(0.5)
    raise TimeoutError(f"{url} not healthy after {timeout:.0f}s")


def main() -> int:
    if "--check" in sys.argv:  # Docker HEALTHCHECK: is the public dashboard up?
        return 0 if is_healthy(f"http://127.0.0.1:{public_port()}/_stcore/health") else 1

    # Docker stops containers with SIGTERM; turn it into a normal exit so `finally` cleans up.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    env = {**os.environ, "PRICING_API_URL": API_URL}
    processes: list[subprocess.Popen] = []
    try:
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "pricing_engine.api.main:app",
             "--host", "127.0.0.1", "--port", str(API_PORT), "--workers", "1"],
            cwd=ROOT, env=env,
        )  # fmt: skip
        processes.append(api)
        wait_for(f"{API_URL}/health", api)
        print(f"[demo] API healthy on {API_URL}; starting dashboard on port {public_port()}", flush=True)

        dashboard = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
             f"--server.port={public_port()}", "--server.address=0.0.0.0",
             "--server.headless=true", "--browser.gatherUsageStats=false"],
            cwd=ROOT, env=env,
        )  # fmt: skip
        processes.append(dashboard)

        while all(p.poll() is None for p in processes):
            time.sleep(1)
        dead = next(p for p in processes if p.poll() is not None)
        print(f"[demo] a process exited with code {dead.returncode}; shutting down", flush=True)
        return dead.returncode or 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
