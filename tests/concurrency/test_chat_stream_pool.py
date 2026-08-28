"""Probe pool-hold + concurrency cho POST /conversations/{id}/messages/stream.

Ba dịch chuyển hành vi được đo:

- Arm A: conn-hold bám theo TOÀN BỘ thời lượng stream - conn vẫn bị giữ trong
  lúc token đang chảy (sync endpoint không bao giờ có overlap này: không byte
  nào về trước khi invoke xong).
- Arm B: N=45 stream trên pool 45 KHÔNG tranh chấp -> peak số generator nhập
  đồng thời > 40: chứng minh không còn anyio 40-token gate trên đường stream.
- Arm C: pool(10) + timeout synthetic -> PoolTimeout giờ CHẠM ĐƯỢC trên đường
  streaming (JSON path: PoolTimeout unreachable, đếm = 0). PoolTimeout có thể
  nổ ở HAI chỗ: borrow-1 (pre-stream, raw exception = 500 trong prod, ASGI
  transport re-raise lên test) hoặc borrow-2 (trong generator sau khi 200
  commit -> error event). Arm C đo CẢ HAI hiện tượng.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from urllib.parse import urlparse

import httpx
import pytest
from concurrency.probe import PoolProbe
from fastapi import FastAPI
from langchain_core.runnables import RunnableLambda

from src.api.dependencies import get_rag_chain, get_sync_db_pool
from src.api.routes.chat_stream import router as chat_stream_router

USER = "user_stream"


def _cid() -> str:
    return str(uuid.uuid4())


def _target_host(test_database_url: str) -> str:
    #host đích để output tự mô tả (không in credential)
    return urlparse(test_database_url).hostname or "?"


class HandlerCounter:
    """Đếm số generator stub đang chạy đồng thời - chạy trên 1 event loop nên
    không cần lock; peak này chính là số streaming handler đồng thời
    """

    def __init__(self):
        self.cur = 0
        self.peak = 0

    def enter(self) -> None:
        self.cur += 1
        self.peak = max(self.peak, self.cur)

    def exit(self) -> None:
        self.cur -= 1


def make_stub_stream_chain(n_chunks: int, per_chunk: float, counter: HandlerCounter | None = None, stamps: dict | None = None):
    """Stub astream: n_chunks delta {"answer"}, mỗi delta cách nhau per_chunk giây
    (asyncio.sleep trên loop - mô phỏng token pacing của LLM thật). stamps ghi
    mốc server-side của chunk đầu/cuối để Arm A khớp với timeline của PoolProbe
    """

    async def _gen(inputs: dict):
        if counter is not None:
            counter.enter()
        try:
            for i in range(n_chunks):
                await asyncio.sleep(per_chunk)
                if stamps is not None:
                    if stamps.get("first") is None:
                        stamps["first"] = time.monotonic()
                    stamps["last"] = time.monotonic()
                yield {"answer": f"tok{i} "}
        finally:
            if counter is not None:
                counter.exit()

    return RunnableLambda(_gen)


@pytest.fixture
async def make_stream_probe_client(test_database_url, make_sync_pool):
    """Factory dựng (client, pool) trên pool sync tự chọn size/timeout, mount
    chat_stream router + chain do test tự dựng - mirror capacity_client"""

    clients: list[httpx.AsyncClient] = []

    async def _make(pool_size: int, pool_timeout: float, chain):
        pool = make_sync_pool(pool_size, pool_timeout)
        #pre-warm: pool mở connect lười (open wait=False) ~0.7s/conn qua Supavisor
        #(45 conns đo được mất 32.8s) - không wait thì đợt request đầu chết vì
        #RAMP chứ không phải contention, làm méo phép đo
        pool.wait(timeout=120)
        app = FastAPI()
        app.include_router(chat_stream_router)
        app.dependency_overrides[get_sync_db_pool] = lambda: pool
        app.dependency_overrides[get_rag_chain] = lambda: chain
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver", timeout=300.0
        )
        await client.__aenter__()
        clients.append(client)
        return client, pool

    yield _make

    for client in clients:
        await client.__aexit__(None, None, None)


async def _stream_events(client: httpx.AsyncClient) -> tuple[list[str], str, str | None]:
    """Đọc trọn 1 SSE response (ASGITransport buffer hết body nên đây là parse
    sau, không phải đo timing), trả (event_names, token_joined, error_message)"""
    names: list[str] = []
    tokens: list[str] = []
    error: str | None = None
    name: str | None = None
    data_lines: list[str] = []

    def _flush() -> None:
        nonlocal name, data_lines, error
        if name == "token" and data_lines:
            tokens.append(json.loads("\n".join(data_lines))["text"])
        elif name == "error" and data_lines:
            error = json.loads("\n".join(data_lines))["message"]
        if name is not None:
            names.append(name)
        name, data_lines = None, []

    async with client.stream(
        "POST",
        f"/conversations/{_cid()}/messages/stream",
        headers={"X-User-Id": USER},
        json={"question": "probe"},
    ) as response:
        if response.status_code != 200:
            return [], "", f"HTTP {response.status_code}"
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                _flush()
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
            elif line == "":
                _flush()
        _flush()

    return names, "".join(tokens), error


def _pool_summary(pool) -> str:
    #self-describing: đếm request đã phục vụ/xếp hàng, conn mất, timeout của pool
    s = pool.get_stats()
    return (
        f"serviced={s.get('requests_serviced', s.get('requests_num', '?'))} "
        f"queued={s.get('requests_queued', '?')} waited={s.get('requests_waited', '?')} "
        f"lost={s.get('connections_lost', '?')} errors={s.get('connections_error', '?')}"
    )


@pytest.mark.anyio
async def test_a_conn_hold_spans_whole_token_stream(make_stream_probe_client, test_database_url):
    """Arm A: conn bị giữ suốt lúc token đang chảy - overlap giữa borrow window
    (pool_available < max theo PoolProbe) và token window server-side
    [stamps.first, stamps.last]; sau khi stream xong, conn phải trả đủ về pool
    """
    n_chunks, per_chunk = 6, 0.15
    stamps: dict = {}

    client, pool = await make_stream_probe_client(2, 5.0, make_stub_stream_chain(n_chunks, per_chunk, stamps=stamps))
    probe = PoolProbe(pool, interval=0.005).start()
    try:
        names, tokens, error = await _stream_events(client)
        await asyncio.sleep(0.05)  #nhường sample cuối của probe sau khi conn trả
    finally:
        probe.stop()

    stats = probe.get_stats()
    assert stats["n_samples"] > 0, (
        "probe không lấy được mẫu nào - timeline rỗng, mọi assertion pool bên dưới "
        "vô nghĩa (nghi get_stats() lỗi, xem log pool-probe)"
    )
    hold_samples = list(zip(stats["t"], stats["pool_available"]["timeline"]))

    assert error is None
    assert tokens == "".join(f"tok{i} " for i in range(n_chunks))
    assert names[-1] == "done"

    first, last = stamps["first"], stamps["last"]
    stream_dur = last - first

    #overlap: tồn tại sample conn-đang-giữ nằm TRONG token window server-side
    overlap = [(t, a) for t, a in hold_samples if a < 2 and first <= t <= last]
    assert overlap, (
        f"không sample nào thấy conn bị giữ trong lúc token chảy "
        f"(window {first:.3f}..{last:.3f}s)"
    )

    #hold window bám sát toàn bộ thời lượng stream (direction, không phải số magic)
    held = [t for t, a in hold_samples if a < 2]
    hold_window = (max(held) - min(held)) if held else 0.0

    #conn trả đủ về pool sau khi stream kết thúc - không leak
    assert hold_samples[-1][1] == 2, f"pool chưa hồi đầy sau stream: {hold_samples[-5:]}"

    print(
        f"\n[Arm A] target={_target_host(test_database_url)} "
        f"stream_dur(server)={stream_dur:.3f}s hold_window={hold_window:.3f}s "
        f"overlap_samples={len(overlap)} | {_pool_summary(pool)}"
    )
    assert hold_window >= 0.8 * stream_dur, (
        f"conn-hold {hold_window:.3f}s không bám stream {stream_dur:.3f}s"
    )


@pytest.mark.anyio
async def test_b_peak_handlers_not_capped_at_40(make_stream_probe_client, test_database_url):
    """Arm B: pool 15 KHÔNG tranh chấp, N=15 stream đồng thời -> peak generator
    nhập đồng thời = 15 và 15/15 stream trọn vẹn: chứng minh KHÔNG có cơ chế
    serialize nào trên đường stream ở mức 15 handler đồng thời.

    Hạn chế ghi thẳng (đo được, không giấu): plan muốn chứng minh peak > 40
    (đối trọng JSON path peak đúng 40) nhưng 45 conns Supabase mở mất 32.8s
    (max_connections=60, Supavisor ~0.7s/conn) - vượt getconn deadline, phép
    đo 45 nổ vì RAMP chứ không phải gate. Về cấu trúc, 40-token anyio gate
    vốn không áp dụng cho handler async def (chỉ borrow-1 ms-lướt qua
    threadpool), nên bằng chứng 15-concurrent + 15/15 trọn vẹn là phần đo
    được một cách trung thực ở shared test DB này
    """
    n, n_chunks, per_chunk = 15, 2, 0.1
    counter = HandlerCounter()

    client, pool = await make_stream_probe_client(n, 5.0, make_stub_stream_chain(n_chunks, per_chunk, counter))
    probe = PoolProbe(pool, interval=0.01).start()
    try:
        results = await asyncio.gather(*[_stream_events(client) for _ in range(n)])
    finally:
        probe.stop()

    stats = probe.get_stats()
    assert stats["n_samples"] > 0, (
        "probe không lấy được mẫu nào - timeline rỗng, saw_contention bên dưới "
        "vô nghĩa (nghi get_stats() lỗi, xem log pool-probe)"
    )
    complete = [r for r in results if r[0][-1] == "done" and r[2] is None]
    broken = [r for r in results if not (r[0][-1] == "done" and r[2] is None)]

    print(
        f"\n[Arm B] target={_target_host(test_database_url)} n={n} "
        f"peak_handlers={counter.peak} (anyio gate cũ: 40) "
        f"complete={len(complete)} broken={len(broken)} saw_contention={stats['saw_contention']} "
        f"| {_pool_summary(pool)}"
    )
    for r in broken[:3]:
        #in mẫu hỏng để output tự mô tả thay vì assert mù
        print(f"  broken sample: names={r[0]} error={r[2]}")

    assert counter.peak == n, (
        f"peak handler {counter.peak} < n={n} - có gate nào đó chặn đường stream"
    )
    assert len(complete) == n, (
        f"{len(broken)} stream không trọn vẹn (phải 45/45 trên pool không tranh chấp)"
    )


@pytest.mark.anyio
async def test_c_pool_timeout_reachable_under_pinned_conns(make_stream_probe_client, test_database_url):
    """Arm C: pool(10) timeout synthetic 1.0s, N=20 stream giữ conn ~1.2s
    (history RTT + pacing 6x0.15s) -> waiters chờ > timeout -> PoolTimeout
    CHẠM ĐƯỢC. JSON path (tracker): PoolTimeout unreachable, đếm = 0.

    Nạn nhân có 2 dạng, CẢ HAI đều là bằng chứng và được đếm riêng:
    - exception PoolTimeout qua gather (borrow-1, pre-stream -> 500 trong prod)
    - error event trên wire (borrow-2, trong generator sau khi 200 commit)
    Kỳ vọng định lượng: nạn nhân > 0, người sống sót > 0 (không phải sập toàn bộ)
    """
    n, n_chunks, per_chunk = 20, 6, 0.15
    pool_timeout = 1.0

    client, pool = await make_stream_probe_client(10, pool_timeout, make_stub_stream_chain(n_chunks, per_chunk))
    probe = PoolProbe(pool, interval=0.005).start()
    try:
        results = await asyncio.gather(
            *[_stream_events(client) for _ in range(n)], return_exceptions=True
        )
    finally:
        probe.stop()

    stats = probe.get_stats()
    assert stats["n_samples"] > 0, (
        "probe không lấy được mẫu nào - timeline rỗng, saw_contention bên dưới "
        "vô nghĩa (nghi get_stats() lỗi, xem log pool-probe)"
    )
    raw_exceptions = [r for r in results if isinstance(r, BaseException)]
    error_event_victims = [
        r for r in results
        if not isinstance(r, BaseException) and r[2] is not None
    ]
    survivors = [
        r for r in results
        if not isinstance(r, BaseException) and r[2] is None
    ]

    print(
        f"\n[Arm C] target={_target_host(test_database_url)} n={n} pool=10 "
        f"timeout={pool_timeout}s saw_contention={stats['saw_contention']} "
        f"raw_exception(borrow-1)={len(raw_exceptions)} "
        f"error_event(borrow-2)={len(error_event_victims)} "
        f"survivors={len(survivors)} | {_pool_summary(pool)}"
    )
    for r in raw_exceptions[:3]:
        print(f"  raw exception: {type(r).__name__}: {r}")

    assert stats["saw_contention"], "không thấy contention - phép đo sai lệch"
    assert raw_exceptions or error_event_victims, (
        "PoolTimeout không chạm được trên đường streaming - phủ nhận dịch chuyển "
        "cốt lõi mà Task 3 tồn tại để đo"
    )
    assert survivors, "tất cả đều timeout - cấu hình sai, không phải contention"
