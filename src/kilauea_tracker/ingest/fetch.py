"""HTTP fetch with conditional `If-Modified-Since` support.

USGS publishes a `Last-Modified` header on every PNG. We persist it next to the
cache and send it back on subsequent fetches so the server can return 304 when
nothing's changed — saves bandwidth and lets us tell the user "no new data."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from .exceptions import FetchError

# Conservative timeout — USGS is usually fast but we don't want to hang the
# Streamlit container if it isn't.
DEFAULT_TIMEOUT_SECONDS = 30

# A polite user-agent so HVO can identify the traffic if they want to.
USER_AGENT = "kilauea-tracker/2.0 (+https://github.com/) python-requests"


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

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise FetchError(f"network error fetching {url}: {e}") from e

    if response.status_code == 304:
        return FetchResult(
            body=None,
            last_modified=cached_last_modified,
            changed=False,
            status_code=304,
        )

    if response.status_code != 200:
        raise FetchError(
            f"unexpected status {response.status_code} fetching {url}",
            status_code=response.status_code,
        )

    return FetchResult(
        body=response.content,
        last_modified=response.headers.get("Last-Modified"),
        changed=True,
        status_code=200,
    )
