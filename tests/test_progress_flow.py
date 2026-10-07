# Copyright (c) 2026 Dedalus Labs, Inc. and its contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anyio
from mcp import types
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext
import pytest

from dedalus_mcp.progress import (
    ProgressCloseEvent,
    ProgressConfig,
    ProgressEmitEvent,
    ProgressLifecycleEvent,
    ProgressTelemetry,
    progress,
)


@dataclass
class RecordedNotification:
    token: types.ProgressToken
    progress: float
    total: float | None
    message: str | None


class FakeSession:
    def __init__(self) -> None:
        self.notifications: list[RecordedNotification] = []

    async def send_progress_notification(
        self,
        progress_token: str | int,
        progress_value: float,
        *,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        self.notifications.append(RecordedNotification(progress_token, progress_value, total, message))


@asynccontextmanager
async def with_request_context(*, token: types.ProgressToken | None, session: FakeSession) -> AsyncIterator[None]:
    meta = types.RequestParams.Meta(progressToken=token) if token is not None else None
    ctx = RequestContext(request_id=1, meta=meta, session=session, lifespan_context=None)
    token_ctx = request_ctx.set(ctx)
    try:
        yield
    finally:
        request_ctx.reset(token_ctx)


@pytest.mark.anyio
async def test_progress_notifications_roundtrip() -> None:
    session = FakeSession()

    async with with_request_context(token="tok", session=session), progress(total=3) as tracker:
        await tracker.advance(1, "step1")
        await tracker.advance(2, "step3")

    assert session.notifications, "expected at least one notification"
    assert session.notifications[-1] == RecordedNotification("tok", 3, 3, "step3")
    values = [note.progress for note in session.notifications]
    assert values == sorted(values), "progress values must be monotonic"


@pytest.mark.anyio
async def test_progress_without_token_raises() -> None:
    session = FakeSession()

    async with with_request_context(token=None, session=session):
        with pytest.raises(ValueError):
            async with progress(total=1):
                await anyio.sleep(0)


@pytest.mark.anyio
async def test_progress_monotonicity_violation() -> None:
    session = FakeSession()

    async with with_request_context(token="tok", session=session), progress(total=5) as tracker:
        await tracker.set(2)
        with pytest.raises(ValueError):
            await tracker.set(1)


@pytest.mark.anyio
async def test_progress_telemetry_hooks_capture_events() -> None:
    session = FakeSession()
    starts: list[ProgressLifecycleEvent] = []
    emits: list[ProgressEmitEvent] = []
    closes: list[ProgressCloseEvent] = []

    telemetry = ProgressTelemetry(
        on_start=lambda evt: starts.append(evt),
        on_emit=lambda evt: emits.append(evt),
        on_close=lambda evt: closes.append(evt),
    )

    async with with_request_context(token="tok", session=session):
        async with progress(total=4, telemetry=telemetry, config=ProgressConfig(emit_hz=50)) as tracker:
            await tracker.advance(1, "one")
            await anyio.sleep(0.01)
            await tracker.advance(1, "two")
            await tracker.set(4, message="done")
            await anyio.sleep(0.01)

    assert len(starts) == 1 and starts[0].token == "tok"
    assert emits and any(evt.progress == 4 for evt in emits)
    assert closes and closes[-1].final_progress == 4


@pytest.mark.anyio
async def test_progress_throttle_event_emitted() -> None:
    session = FakeSession()
    throttled: list[int] = []

    telemetry = ProgressTelemetry(on_throttle=lambda evt: throttled.append(evt.pending_updates))

    async with with_request_context(token="tok", session=session):
        async with progress(total=2, telemetry=telemetry, config=ProgressConfig(emit_hz=1)) as tracker:
            await tracker.advance(1)
            await anyio.sleep(0.05)
            await tracker.advance(1)

    assert throttled, "expected throttle telemetry when emit_hz is low"
    assert session.notifications[-1].progress == 2


def _assert_monotonic(notifications: list[RecordedNotification]) -> None:
    values = [n.progress for n in notifications]
    assert values == sorted(values), f"progress went backwards: {values}"


@pytest.mark.anyio
async def test_close_during_throttled_send_does_not_regress() -> None:
    """Closing while a throttled update is in flight must not re-send an older value."""
    session = FakeSession()
    closes: list[ProgressCloseEvent] = []
    telemetry = ProgressTelemetry(on_close=lambda evt: closes.append(evt))

    async with with_request_context(token="tok", session=session):
        async with progress(total=4, telemetry=telemetry, config=ProgressConfig(emit_hz=20)) as tracker:
            await tracker.advance(1, "one")
            await anyio.sleep(0.005)  # first update delivered immediately
            await tracker.set(4, message="done")  # throttled
            await anyio.sleep(0.001)  # emitter picks it up; still in flight at close

    _assert_monotonic(session.notifications)
    assert session.notifications[-1].progress == 4
    assert closes
    assert closes[-1].final_progress == 4
    assert closes[-1].final_message == "done"


@pytest.mark.anyio
async def test_advance_uses_in_flight_state_as_base() -> None:
    """``advance`` must build on an update that is in flight, not the last delivered one."""
    session = FakeSession()

    async with with_request_context(token="tok", session=session):
        async with progress(total=10, config=ProgressConfig(emit_hz=20)) as tracker:
            await tracker.advance(1)
            await anyio.sleep(0.005)  # first update delivered immediately
            await tracker.set(3)  # throttled
            await anyio.sleep(0.001)  # emitter picks it up and waits out the throttle
            assert await tracker.advance(1) == 4

    _assert_monotonic(session.notifications)
    assert session.notifications[-1].progress == 4
