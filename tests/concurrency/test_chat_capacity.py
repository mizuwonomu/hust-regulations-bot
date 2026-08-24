"""Đo capacity của chat path — tách khỏi starvation probe (test b).

Phép đo này dùng pool 10 và
turn 4s (3-5 Groq calls + rerank) trên Supabase THẬT: RTT, TLS, Supavisor
pooler đều thật. LLM vẫn là stub: mô phỏng faithful TURN DURATION (biến quyết
định capacity) nhưng không mô phỏng Groq latency variance hay rate limiting
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid

import httpx
import pytest
from fastapi import FastAPI

from conftest import (
    CAPACITY_POOL_SIZE,
    CAPACITY_REQUESTS_EXCEED,
    CAPACITY_REQUESTS_FIT,
    CAPACITY_TIMEOUT_EXCEED,
    CAPACITY_TURN_DURATION,
)
from concurrency.probe import PoolProbe
from src.api.dependencies import get_rag_chain, get_sync_db_pool
from src.api.routes.chat import router as chat_router
from src.database.sync_connection import POOL_TIMEOUT_SECONDS


@pytest.fixture
async def capacity_exceed_client(make_sync_pool, slow_stub_chain):
    """Như capacity_client nhưng pool timeout = CAPACITY_TIMEOUT_EXCEED (synthetic).

    Production timeout (30s) không thể nổ qua cổng threadpool 40 thread (xem
    docstring module) — chỉ với timeout ngắn này pool mới trở thành ràng buộc
    và phép đo "pool bị cap" mới đo được thứ nó tuyên bố.
    """
    pool = make_sync_pool(CAPACITY_POOL_SIZE, CAPACITY_TIMEOUT_EXCEED)

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_sync_db_pool] = lambda: pool
    app.dependency_overrides[get_rag_chain] = lambda: slow_stub_chain

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        timeout=60.0,
    ) as client:
        yield client, pool


def _posts(client, n: int):
    """n chat POST vào các conversation id riêng biệt (uuid4)."""
    return [
        client.post(
            f"/conversations/{uuid.uuid4()}/messages",
            json={"question": f"capacity {i}"},
        )
        for i in range(n)
    ]


def _waves(n_requests: int) -> int:
    """Số wave khi R request chạy trên pool P (mỗi turn pin 1 conn)."""
    return math.ceil(n_requests / CAPACITY_POOL_SIZE)


def _is_success(result) -> bool:
    """Response 200 = thành công; exception (Starlette re-raise sau 500) hoặc
    non-200 = thất bại."""
    return isinstance(result, httpx.Response) and result.status_code == 200


@pytest.mark.slow
@pytest.mark.anyio
async def test_capacity_fits_inside_timeout(capacity_client):
    """R=20 trên pool 10, turn 4s → 2 waves ≈ 8s << 30s: TẤT CẢ thành công.

    Assert constants TRƯỚC (mirror assertion): nếu load không thật sự vừa
    timeout thì fail to chứ không âm thầm đo cái khác.
    """
    client, pool = capacity_client

    waves = _waves(CAPACITY_REQUESTS_FIT)
    assert (waves - 1) * CAPACITY_TURN_DURATION < POOL_TIMEOUT_SECONDS, (
        f"constants CAPACITY sai: {CAPACITY_REQUESTS_FIT} request → {waves} waves, "
        f"wave cuối chờ {(waves - 1) * CAPACITY_TURN_DURATION}s ≥ timeout "
        f"{POOL_TIMEOUT_SECONDS}s — load này KHÔNG vừa timeout, mirror assertion fail"
    )

    probe = PoolProbe(pool).start()
    t0 = time.monotonic()
    try:
        responses = await asyncio.gather(
            *_posts(client, CAPACITY_REQUESTS_FIT),
            return_exceptions=True,
        )
    finally:
        probe.stop()
    elapsed = time.monotonic() - t0

    failures = [r for r in responses if not _is_success(r)]
    assert not failures, (
        f"{len(failures)}/{CAPACITY_REQUESTS_FIT} request fail trên load vừa "
        f"timeout: {[repr(f)[:120] for f in failures[:3]]}"
    )

    stats = probe.get_stats()
    assert stats["saw_contention"], (
        "không request nào xếp hàng chờ connection — phép đo không đo cái nó "
        "tuyên bố (pool lớn hơn số request? turn quá ngắn?)"
    )

    print(
        f"[capacity-fit] N={CAPACITY_REQUESTS_FIT} pool={CAPACITY_POOL_SIZE} "
        f"turn={CAPACITY_TURN_DURATION}s timeout={POOL_TIMEOUT_SECONDS}s → "
        f"{len(responses) - len(failures)}/{len(responses)} ok, "
        f"elapsed={elapsed:.1f}s, peak requests_waiting="
        f"{stats['requests_waiting']['max']}"
    )


@pytest.mark.slow
@pytest.mark.anyio
async def test_capacity_exceeds_timeout(capacity_exceed_client):
    """R=90 trên pool 10, turn 4s → 9 waves; wave cuối chờ 8×4=32s > timeout
    5s (synthetic) → ít nhất một request PoolTimeout.

    Đọc kỹ failure trước khi diễn giải (xem docstring module): PoolTimeout =
    pool mình là ceiling (thứ đang đo); lỗi từ pooler = đo nhầm limit Supabase.
    """
    client, pool = capacity_exceed_client

    waves = _waves(CAPACITY_REQUESTS_EXCEED)
    assert (waves - 1) * CAPACITY_TURN_DURATION > CAPACITY_TIMEOUT_EXCEED, (
        f"constants CAPACITY sai: {CAPACITY_REQUESTS_EXCEED} request → {waves} "
        f"waves, wave cuối chờ {(waves - 1) * CAPACITY_TURN_DURATION}s ≤ timeout "
        f"{CAPACITY_TIMEOUT_EXCEED}s — load này KHÔNG vượt timeout, mirror assertion fail"
    )

    probe = PoolProbe(pool).start()
    t0 = time.monotonic()
    try:
        responses = await asyncio.gather(
            *_posts(client, CAPACITY_REQUESTS_EXCEED),
            return_exceptions=True,
        )
    finally:
        probe.stop()
    elapsed = time.monotonic() - t0

    successes = sum(1 for r in responses if _is_success(r))
    failures = len(responses) - successes
    assert failures >= 1, (
        f"TẤT CẢ {len(responses)} request thành công dù mirror assertion nói load "
        f"phải vượt timeout — pool không thật sự bị cap (sai constants? timeout "
        f"không áp dụng?)"
    )

    print(
        f"[capacity-over] N={CAPACITY_REQUESTS_EXCEED} pool={CAPACITY_POOL_SIZE} "
        f"turn={CAPACITY_TURN_DURATION}s timeout={CAPACITY_TIMEOUT_EXCEED}s "
        f"(synthetic — production 30s không nổ qua cổng threadpool, xem docstring) → "
        f"{successes} ok / {failures} fail, elapsed={elapsed:.1f}s, "
        f"peak requests_waiting={probe.get_stats()['requests_waiting']['max']}"
    )
