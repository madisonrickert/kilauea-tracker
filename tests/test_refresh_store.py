"""Tests for the cross-thread refresh-state coordinator.

These exercise the on-disk + in-memory hybrid behavior with tempfile
paths so no Streamlit context is required. Two store instances on the
same path are used to simulate the cross-process / cross-thread case
that ``@st.cache_resource`` doesn't actually defend against under
multi-worker deployments.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from kilauea_tracker.state.refresh_store import RefreshStore


@pytest.fixture
def tmp_status(tmp_path):
    return tmp_path / "refresh_status.json"


def _make_store(path, *, cooldown=30.0, stale=300.0, clock=None):
    return RefreshStore(
        path,
        cooldown_seconds=cooldown,
        stale_timeout_seconds=stale,
        clock=clock or (lambda: datetime.now(tz=UTC)),
    )


def test_initial_snapshot_is_idle(tmp_status):
    store = _make_store(tmp_status)
    snap = store.snapshot()
    assert snap.running is False
    assert snap.current_stage is None
    assert snap.started_utc is None
    assert snap.finished_utc is None
    assert snap.last_error is None
    assert snap.source is None


def test_start_persists_state(tmp_status):
    store = _make_store(tmp_status)
    assert store.start("manual") is True
    snap = store.snapshot()
    assert snap.running is True
    assert snap.source == "manual"
    assert snap.started_utc is not None
    assert snap.finished_utc is None


def test_advance_updates_stage(tmp_status):
    store = _make_store(tmp_status)
    store.start("manual")
    store.advance("Fetching three_month…")
    snap = store.snapshot()
    assert snap.running is True
    assert snap.current_stage == "Fetching three_month…"


def test_advance_is_noop_when_idle(tmp_status):
    store = _make_store(tmp_status)
    store.advance("ignored")
    snap = store.snapshot()
    assert snap.running is False
    assert snap.current_stage is None


def test_complete_clears_running(tmp_status):
    store = _make_store(tmp_status)
    store.start("manual")
    store.advance("doing things")
    store.complete()
    snap = store.snapshot()
    assert snap.running is False
    assert snap.current_stage is None
    assert snap.finished_utc is not None
    assert snap.last_error is None


def test_fail_records_error(tmp_status):
    store = _make_store(tmp_status)
    store.start("manual")
    store.fail("kaboom")
    snap = store.snapshot()
    assert snap.running is False
    assert snap.last_error == "kaboom"
    assert snap.finished_utc is not None


def test_concurrent_start_inside_one_store_only_one_wins(tmp_status):
    """Two start() calls before complete: second must return False."""
    store = _make_store(tmp_status)
    assert store.start("manual") is True
    assert store.start("background") is False


def test_cooldown_blocks_consecutive_starts(tmp_status):
    """A start within the cooldown window after a clean finish is blocked."""
    now = [datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)]

    def clock():
        return now[0]

    store = _make_store(tmp_status, cooldown=30.0, clock=clock)
    assert store.start("manual") is True
    store.complete()
    # 5 seconds later: still cooling down
    now[0] += timedelta(seconds=5)
    assert store.start("manual") is False
    # 31 seconds later: cooldown elapsed
    now[0] += timedelta(seconds=26)
    assert store.start("manual") is True


def test_stale_started_timeout_makes_running_false(tmp_status):
    """If started_utc is older than stale_timeout and never finished,
    snapshot reports running=False (covers daemon crashes)."""
    now = [datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)]

    def clock():
        return now[0]

    store = _make_store(tmp_status, stale=10.0, clock=clock)
    assert store.start("background") is True
    snap = store.snapshot()
    assert snap.running is True
    # Jump 11 seconds — past the stale timeout, still no complete()
    now[0] += timedelta(seconds=11)
    snap = store.snapshot()
    assert snap.running is False
    # And a fresh start() should now succeed (overrides stale state).
    assert store.start("manual") is True


def test_two_stores_share_state_via_file(tmp_status):
    """Store A advances; Store B with the same path observes the new stage.

    Mirrors the cross-process case: two Streamlit workers, two
    cache_resource singletons, one persistence file.
    """
    a = _make_store(tmp_status)
    b = _make_store(tmp_status)
    a.start("manual")
    a.advance("Fetching dec2024_to_now…")
    snap = b.snapshot()
    assert snap.running is True
    assert snap.current_stage == "Fetching dec2024_to_now…"
    a.complete()
    snap = b.snapshot()
    assert snap.running is False


def test_concurrent_starts_across_threads_one_wins(tmp_status):
    """Two threads racing on start(): exactly one returns True."""
    store = _make_store(tmp_status)
    results: list[bool] = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        results.append(store.start("manual"))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False, True]


def test_corrupt_file_is_recoverable(tmp_status):
    """A garbage status file is treated as empty, not a crash."""
    tmp_status.parent.mkdir(parents=True, exist_ok=True)
    tmp_status.write_text("not json {{{")
    store = _make_store(tmp_status)
    snap = store.snapshot()
    assert snap.running is False
    assert store.start("manual") is True


def test_snapshot_is_frozen(tmp_status):
    """RefreshSnapshot is a frozen dataclass — mutation raises."""
    from dataclasses import FrozenInstanceError

    store = _make_store(tmp_status)
    snap = store.snapshot()
    with pytest.raises(FrozenInstanceError):
        snap.running = True  # type: ignore[misc]
