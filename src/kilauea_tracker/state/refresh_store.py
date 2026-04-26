"""Cross-thread coordinator for ingest-refresh state.

Why this exists: a daemon thread (the page-load background ingest)
has no ScriptRunContext and cannot write ``st.session_state``. The
file-locked JSON store + ``@st.cache_resource`` singleton give us a
state surface both the daemon and the main thread can read/write,
and that survives Streamlit Cloud container restarts.

Production wiring uses ``get_refresh_store()`` (cache_resource
factory). Tests instantiate ``RefreshStore`` directly with a tempfile
path — no Streamlit dependency in the constructor.

State transitions
-----------------

    idle  ──start(source)──▶  running ──advance(stage)──▶ running
                                  │                           │
                                  ▼                           ▼
                              complete()                  complete() / fail(err)
                                  │
                                  ▼
                                idle

Cooldown gating folds into ``start()``: if a previous refresh started
within ``cooldown_seconds`` and finished cleanly, ``start()`` returns
False. Concurrent ``start()`` calls are serialized under
``fcntl.flock``; only the first wins.

Stale-refresh recovery: if ``started_utc`` is more than
``stale_timeout_seconds`` older than now and ``finished_utc`` is
unset, ``snapshot()`` reports ``running=False``. Covers OS-level
process kills that bypass our ``try/finally``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import IO, TYPE_CHECKING

from ..config import DATA_DIR
from .snapshot import RefreshSnapshot, RefreshSource

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical persistence path. Replaces the legacy ``last_refresh.json``
# (which only carried ``last_refresh_utc``) with a richer schema that
# also captures finish time, current stage, source, and the most recent
# error. Old file is migrated on first read.
REFRESH_STATUS_FILE = DATA_DIR / "refresh_status.json"

# Cooldown between consecutive ``start()`` calls. Mirrors the previous
# ``USER_REFRESH_COOLDOWN_SECONDS`` from streamlit_app.py — 30s lets a
# real click feel responsive while still gating spam.
DEFAULT_COOLDOWN_SECONDS = 30

# How long after ``started_utc`` we treat an unfinished refresh as
# stale (i.e., the daemon must have crashed). 5 min comfortably covers
# the worst observed real ingest (~60s with all retries firing).
DEFAULT_STALE_TIMEOUT_SECONDS = 300


@dataclass
class _StoredState:
    """On-disk schema for ``refresh_status.json``.

    Mutable on purpose — only the in-memory copy held by the
    ``RefreshStore`` ever uses this directly; the public
    ``RefreshSnapshot`` is frozen and assembled from it.
    """

    started_utc: datetime | None = None
    finished_utc: datetime | None = None
    current_stage: str | None = None
    source: RefreshSource | None = None
    last_error: str | None = None

    def to_json(self) -> dict:
        return {
            "started_utc": _iso(self.started_utc),
            "finished_utc": _iso(self.finished_utc),
            "current_stage": self.current_stage,
            "source": self.source,
            "last_error": self.last_error,
        }

    @classmethod
    def from_json(cls, payload: dict) -> _StoredState:
        return cls(
            started_utc=_parse(payload.get("started_utc")),
            finished_utc=_parse(payload.get("finished_utc")),
            current_stage=payload.get("current_stage"),
            source=payload.get("source"),
            last_error=payload.get("last_error"),
        )


class RefreshStore:
    """File-locked cross-thread coordinator for ingest-refresh state.

    Singleton in production (one per Streamlit container) via
    ``@st.cache_resource``. Multiple threads/sessions share the same
    instance and serialize their writes through ``fcntl.flock`` on the
    persistence file.
    """

    def __init__(
        self,
        status_path: Path = REFRESH_STATUS_FILE,
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        stale_timeout_seconds: float = DEFAULT_STALE_TIMEOUT_SECONDS,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        self._path = status_path
        self._cooldown = cooldown_seconds
        self._stale_timeout = stale_timeout_seconds
        self._clock = clock
        # In-memory cache of the last on-disk read. Re-read when the
        # file's mtime is newer than what we hold. Process-local lock
        # protects the in-memory copy; cross-process safety is via
        # fcntl.flock on the file itself.
        self._lock = threading.Lock()
        self._cached: _StoredState = _StoredState()
        self._cached_mtime: float = 0.0

    # ──────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────

    def snapshot(self) -> RefreshSnapshot:
        """Return the current state. Re-reads from disk if mtime ticked."""
        state = self._read_through_cache()
        running = self._is_running(state)
        return RefreshSnapshot(
            running=running,
            current_stage=state.current_stage if running else None,
            started_utc=state.started_utc,
            finished_utc=state.finished_utc,
            last_error=state.last_error,
            source=state.source if running else None,
        )

    def _is_running(self, state: _StoredState) -> bool:
        """``running`` semantics: started but not finished, and not stale."""
        if state.started_utc is None:
            return False
        if state.finished_utc is not None and state.finished_utc >= state.started_utc:
            return False
        # Daemon-crash detector: if we've been "running" longer than
        # the stale timeout, treat as not-running. The next start()
        # will overwrite the stale state cleanly.
        elapsed = (self._clock() - state.started_utc).total_seconds()
        if elapsed > self._stale_timeout:
            logger.warning(
                "refresh state appears stale (started %.0fs ago, no finish); "
                "treating as not-running",
                elapsed,
            )
            return False
        return True

    # ──────────────────────────────────────────────────────────────
    # Write — all under fcntl.LOCK_EX
    # ──────────────────────────────────────────────────────────────

    def start(self, source: RefreshSource) -> bool:
        """Try to enter the running state. Returns False if blocked.

        Blocked when (a) another refresh is already running, or
        (b) a previous refresh finished within ``cooldown_seconds``.
        On success, persists ``started_utc`` and ``source``, and clears
        any prior ``current_stage`` / ``last_error``.
        """
        with self._with_locked_file() as (fh, state):
            now = self._clock()
            # Already running?
            if self._is_running(state):
                return False
            # Cooldown: most recent finish too recent?
            if (
                state.finished_utc is not None
                and (now - state.finished_utc).total_seconds() < self._cooldown
            ):
                return False
            new = _StoredState(
                started_utc=now,
                finished_utc=None,
                current_stage=None,
                source=source,
                last_error=None,
            )
            self._write(fh, new)
            return True

    def advance(self, stage: str) -> None:
        """Update the current-stage label. No-op if not running."""
        with self._with_locked_file() as (fh, state):
            if not self._is_running(state):
                return
            state.current_stage = stage
            self._write(fh, state)

    def complete(self) -> None:
        """Mark the running refresh as finished cleanly."""
        with self._with_locked_file() as (fh, state):
            now = self._clock()
            state.finished_utc = now
            state.current_stage = None
            state.last_error = None
            self._write(fh, state)

    def fail(self, error: str) -> None:
        """Mark the running refresh as finished with an error."""
        with self._with_locked_file() as (fh, state):
            now = self._clock()
            state.finished_utc = now
            state.current_stage = None
            state.last_error = error
            self._write(fh, state)

    # ──────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────

    def _read_through_cache(self) -> _StoredState:
        """Return a fresh copy of the on-disk state.

        Caches the parse result keyed on file mtime — re-reads only when
        the file has changed underneath us (e.g., another process or
        thread wrote to it since our last call).
        """
        with self._lock:
            try:
                mtime = self._path.stat().st_mtime
            except FileNotFoundError:
                # File doesn't exist yet — empty state. Don't update
                # cached_mtime; next existence check will refresh.
                self._cached = _StoredState()
                return self._cached
            if mtime != self._cached_mtime:
                try:
                    payload = json.loads(self._path.read_text() or "{}")
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        "could not parse %s, treating as empty: %s",
                        self._path, e,
                    )
                    payload = {}
                self._cached = _StoredState.from_json(payload)
                self._cached_mtime = mtime
            # Return a copy so the caller can mutate without poisoning
            # our cache. dataclass(frozen=False) so a shallow copy is fine.
            return _StoredState(**self._cached.__dict__)

    @contextlib.contextmanager
    def _with_locked_file(self) -> Iterator[tuple[IO[str], _StoredState]]:
        """Open the status file under an exclusive fcntl lock.

        Yields ``(open file handle, parsed state)``. Caller mutates state,
        then calls ``self._write(fh, state)`` to persist. The lock is
        held for the whole context so concurrent ``start()`` calls
        cannot race past the cooldown / running checks.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        with self._path.open("r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                raw = fh.read().strip()
                try:
                    payload = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    payload = {}
                state = _StoredState.from_json(payload)
                yield fh, state
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _write(self, fh: IO[str], state: _StoredState) -> None:
        """Persist ``state`` to the open file handle and refresh in-memory cache."""
        fh.seek(0)
        fh.truncate()
        json.dump(state.to_json(), fh)
        fh.flush()
        # Update in-memory cache so the next snapshot() doesn't pay
        # for a stat+re-read just to see the value we just wrote.
        with self._lock:
            self._cached = _StoredState(**state.__dict__)
            with contextlib.suppress(OSError):
                self._cached_mtime = self._path.stat().st_mtime


# ──────────────────────────────────────────────────────────────────────
# Production wiring
# ──────────────────────────────────────────────────────────────────────

# The cache_resource decorator is applied lazily — see get_refresh_store()
# below. We can't decorate at module scope because pytest collects this
# module without a Streamlit ScriptRunContext, and @st.cache_resource
# would be a no-op import-time error there.

_PROD_STORE: RefreshStore | None = None


def get_refresh_store() -> RefreshStore:
    """Return the singleton ``RefreshStore`` for the current Streamlit container.

    Wraps a module-level cache so non-Streamlit callers (e.g., tests
    importing the module without a ScriptRunContext) still get a
    consistent instance, and Streamlit's ``@st.cache_resource`` is
    applied only when the framework is actually present.
    """
    global _PROD_STORE
    try:
        import streamlit as st
    except ImportError:  # pragma: no cover — streamlit is a hard dep
        if _PROD_STORE is None:
            _PROD_STORE = RefreshStore()
        return _PROD_STORE

    @st.cache_resource
    def _factory() -> RefreshStore:
        return RefreshStore()

    return _factory()


# ──────────────────────────────────────────────────────────────────────
# datetime ↔ ISO helpers
# ──────────────────────────────────────────────────────────────────────


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _parse(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    s = str(value)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
