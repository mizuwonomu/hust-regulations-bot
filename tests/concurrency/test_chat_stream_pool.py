"""Probe pool-hold + concurrency cho POST /conversations/{id}/messages/stream.

Hai dịch chuyển hành vi được đo (sau de-wrap - memory tách khỏi wrapper nên
astream KHÔNG giữ conn nữa):

- Arm B: pool 15 KHÔNG tranh chấp, N=15 stream đồng thời -> peak generator
  nhập đồng thời = 15 và 15/15 stream trọn vẹn: chứng minh không có cơ chế
  serialize nào trên đường stream.
- Arm D (payoff của de-wrap): pool(1) + N=3 stream, stub ngủ giữa các token -
  astream KHÔNG giữ conn nên 3 stream chồng lên nhau, tổng thời gian < 2x một
  lượt (thời conn-pin: 3 stream tuần tự, tổng ~3x). Borrow ngắn vẫn tranh
  chấp chốc lát trên pool 1 conn -> saw_contention > 0.

ĐÃ XOÁ cùng đợt de-wrap (đo hành vi conn-pin không còn tồn tại):
- Arm A (conn-hold bám toàn bộ stream) - overlap giữa conn-hold và token
  window giờ là rỗng, đó chính là kết quả de-wrap phải đạt được.
- Arm C (PoolTimeout chạm được nhờ conn bị giữ ~1.2s) - borrow giờ ngắn
  vài RTT nên PoolTimeout chỉ nổ khi pool bão hoà thật, không còn nằm trong
  thiết kế của đường stream.
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


def make_stub_stream_chain(n_chunks: int, per_chunk: float, counter: HandlerCounter | None = None):
    """Stub astream: n_chunks delta {"answer"}, mỗi delta cách nhau per_chunk giây
    (asyncio.sleep trên loop - mô phỏng token pacing của LLM thật)
    """

    async def _gen(inputs: dict):
        if counter is not None:
            counter.enter()
        try:
            for i in range(n_chunks):
                await asyncio.sleep(per_chunk)
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
async def test_d_astream_holds_no_conn_so_streams_overlap(make_stream_probe_client, test_database_url):
    """Arm D: astream giữ KHÔNG conn - conn chỉ được mượn trong 2 cửa sổ ngắn
    (đọc history + ghi turn), khoảng ngủ giữa các token (mô phỏng LLM chậm)
    không chiếm conn. Kết quả: N=3 stream trên pool 1 conn chồng lên nhau,
    tổng wall-clock < 2x một lượt. Trước de-wrap: conn-pin bám suốt astream ->
    3 stream tuần tự, tổng ~3x một lượt - phép đo này FAIL. Rule tracker: chỉ
    assert saw_contention (> 0), KHÔNG dùng peak requests_waiting làm depth
    """
    # Ngủ dài so với RTT borrow: phần không-conn (sleep) phải lấn át phần
    # overhead RTT tuần tự, nếu không phép đo thành "đo RTT Supabase"
    n_streams, n_chunks, per_chunk = 3, 6, 0.6
    turn_duration = n_chunks * per_chunk  # một lượt ~ pacing của stub + RTT borrow

    client, pool = await make_stream_probe_client(
        1, 5.0, make_stub_stream_chain(n_chunks, per_chunk)
    )
    probe = PoolProbe(pool, interval=0.005).start()
    try:
        start = time.monotonic()
        results = await asyncio.gather(
            *[_stream_events(client) for _ in range(n_streams)]
        )
        elapsed = time.monotonic() - start
    finally:
        probe.stop()

    stats = probe.get_stats()
    assert stats["n_samples"] > 0, (
        "probe không lấy được mẫu nào - timeline rỗng, mọi assertion pool bên dưới "
        "vô nghĩa (nghi get_stats() lỗi, xem log pool-probe)"
    )

    for names, tokens, error in results:
        assert error is None, f"stream lỗi giữa đường: {error}"
        assert names[-1] == "done", f"stream không trọn vẹn: {names}"
        assert tokens == "".join(f"tok{i} " for i in range(n_chunks))

    print(
        f"\n[Arm D] target={_target_host(test_database_url)} n={n_streams} pool=1 "
        f"turn~{turn_duration:.2f}s elapsed={elapsed:.3f}s "
        f"serial_would_be~{n_streams * turn_duration:.2f}s "
        f"saw_contention={stats['saw_contention']} | {_pool_summary(pool)}"
    )

    # 3 stream chồng lên nhau: tổng < 2x một lượt (tuần tự sẽ là ~3x)
    assert elapsed < 2 * turn_duration, (
        f"{n_streams} stream mất {elapsed:.3f}s >= 2x một lượt ({2 * turn_duration:.2f}s) "
        f"- nghi astream vẫn đang giữ conn (conn-pin chưa chết)"
    )

    # Borrow ngắn trên pool 1 conn vẫn tranh chấp chốc lát giữa các stream
    assert stats["saw_contention"], (
        "không thấy contention - 3 stream không hề chồng lên nhau, phép đo sai lệch"
    )
