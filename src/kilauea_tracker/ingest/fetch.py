"""HTTP fetch with conditional `If-Modified-Since` support.

USGS publishes a `Last-Modified` header on every PNG. We persist it next to the
cache and send it back on subsequent fetches so the server can return 304 when
nothing's changed — saves bandwidth and lets us tell the user "no new data."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

from .exceptions import FetchError

logger = logging.getLogger(__name__)

# Conservative timeout — USGS is usually fast but we don't want to hang the
# Streamlit container if it isn't.
DEFAULT_TIMEOUT_SECONDS = 30

# A polite user-agent so HVO can identify the traffic if they want to.
USER_AGENT = "kilauea-tracker/2.0 (+https://github.com/) python-requests"

# PNG magic bytes. HVO occasionally serves 200 OK with an empty or
# truncated body (observed periodically on month/week); we detect the
# malformed response by checking the magic signature and retry once.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Retry config for malformed responses. Total: at most 3 attempts with
# short exponential-ish backoffs; we want the caller to get a real error
# quickly rather than hang when USGS is systemically down.
_RETRY_ATTEMPTS = 2
_RETRY_BACKOFFS_SECONDS = [0.5, 2.0]


@dataclass
class FetchResult:
    """Outcome of a single PNG fetch."""

    body: Optional[bytes]            # the PNG bytes, or None on 304 Not Modified
    last_modified: Optional[str]     # the server's Last-Modified header
    changed: bool                    # True iff body is fresh; False on 304
    status_code: int


def fetch_tilt_png(
    url: str,
    cached_last_modified: Optional[str] = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> FetchResult:
    """GET a USGS tilt PNG, honoring `If-Modified-Since`.

    Args:
        url:                  The PNG URL (e.g. `config.USGS_TILT_URLS[...]`).
        cached_last_modified: The `Last-Modified` value we saw on the previous
                              successful fetch, if any. Forwarded as
                              `If-Modified-Since` so the server can short-circuit
                              with 304 when nothing has changed.
        timeout:              Request timeout in seconds.

    Returns:
        A `FetchResult`.

    Raises:
        FetchError: on any HTTP failure or network error. Carries the status
                    code when one is available so the UI can render an
                    informative banner.
    """
    headers = {"User-Agent": USER_AGENT}
    if cached_last_modified:
        headers["If-Modified-Since"] = cached_last_modified

    last_err: Optional[str] = None
    for attempt in range(_RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_err = f"network error: {e}"
            if attempt < _RETRY_ATTEMPTS:
                logger.warning(
                    "fetch %s attempt %d/%d failed (%s); retrying",
                    url, attempt + 1, _RETRY_ATTEMPTS + 1, last_err,
                )
                time.sleep(_RETRY_BACKOFFS_SECONDS[attempt])
                continue
            raise FetchError(f"network error fetching {url}: {e}") from e

        if response.status_code == 304:
            return FetchResult(
                body=None,
                last_modified=cached_last_modified,
                changed=False,
                status_code=304,
            )

        if response.status_code != 200:
            last_err = f"status {response.status_code}"
            if attempt < _RETRY_ATTEMPTS:
                logger.warning(
                    "fetch %s attempt %d/%d got %s; retrying",
                    url, attempt + 1, _RETRY_ATTEMPTS + 1, last_err,
                )
                time.sleep(_RETRY_BACKOFFS_SECONDS[attempt])
                continue
            raise FetchError(
                f"unexpected status {response.status_code} fetching {url}",
                status_code=response.status_code,
            )

        body = response.content
        # USGS occasionally serves 200 OK with an empty or truncated body.
        # That trips cv2.imdecode downstream with a cryptic "returned None"
        # error. Detect here via the PNG magic signature and retry — a
        # fresh request usually succeeds.
        if not body or not body.startswith(_PNG_MAGIC):
            last_err = f"body is not a valid PNG ({len(body)} bytes)"
            if attempt < _RETRY_ATTEMPTS:
                logger.warning(
                    "fetch %s attempt %d/%d: %s; retrying",
                    url, attempt + 1, _RETRY_ATTEMPTS + 1, last_err,
                )
                time.sleep(_RETRY_BACKOFFS_SECONDS[attempt])
                continue
            raise FetchError(
                f"USGS returned a non-PNG body after "
                f"{_RETRY_ATTEMPTS + 1} attempts: {last_err}",
                status_code=response.status_code,
            )

        return FetchResult(
            body=body,
            last_modified=response.headers.get("Last-Modified"),
            changed=True,
            status_code=200,
        )
    # Shouldn't reach here (the loop either returns or raises on last attempt)
    raise FetchError(f"fetch exhausted retries for {url}: {last_err}")
