"""Regenerate `docs/screenshot.png` for the README.

Boots the Streamlit app on a temporary port, drives a headless Chromium via
Playwright until the Plotly chart's data trace has actually rendered, takes a
full-page screenshot at retina resolution, then kills the app.

Run with:
    uv run python scripts/take_screenshot.py

Requires the optional `screenshot` extra:
    uv sync --extra screenshot
    uv run playwright install chromium

The first time you ever run it Playwright also has to download Chromium
(~150 MB), which is a one-shot — `playwright install` is idempotent.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_PATH = REPO_ROOT / "docs" / "screenshot.png"
APP_PATH = REPO_ROOT / "streamlit_app.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Streamlit health endpoint never responded: {url}")


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright is not installed. Run:\n"
            "  uv sync --extra screenshot\n"
            "  uv run playwright install chromium",
            file=sys.stderr,
        )
        return 1

    SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    print(f"starting streamlit on :{port}…")

    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_server(f"http://localhost:{port}/_stcore/health")
        print("streamlit ready, capturing…")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(
                viewport={"width": 1600, "height": 1300},
                device_scale_factor=2,  # retina
            )
            page = context.new_page()
            page.goto(
                f"http://localhost:{port}",
                wait_until="networkidle",
                timeout=60_000,
            )
            page.wait_for_selector(
                ".js-plotly-plot", state="visible", timeout=30_000
            )
            page.wait_for_selector(
                ".js-plotly-plot .scatter", state="attached", timeout=30_000
            )
            page.wait_for_timeout(3000)  # let animations settle
            page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
            browser.close()

        size = SCREENSHOT_PATH.stat().st_size
        print(f"wrote {SCREENSHOT_PATH.relative_to(REPO_ROOT)} ({size} bytes)")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
