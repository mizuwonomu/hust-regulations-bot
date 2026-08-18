"""Bộ test concurrency cho POST /conversations/{id}/messages (G3).

4 phép đo, mỗi phép gather N request rồi verify qua bảng chat_history:

(a) distinct sessions — 5 uuid4, gather 5 → mỗi session đúng 2 msg của chính nó
                        (bleed check: không lẫn msg giữa các session).
(b) starvation        — gather 5 với pool max_size=2 → tất cả 200, không
                        timeout/deadlock; probe xác nhận có lúc
                        requests_waiting > 0 (đã thật sự xếp hàng).
(c) disconnect        — cancel task giữa lúc stub đang sleep → ghi nhận hành vi
                        (msg có được ghi không) + pool_available trở về đủ
                        max_size (không leak connection).
(d) same-session race — 2 request cùng 1 conversation_id → chat_history đủ
                        4 msg (2 user + 2 ai), không mất msg.

Chạy: TEST_DATABASE_URL=<url DB test> uv run pytest tests/concurrency -v
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from conftest import N_CONCURRENT, POOL_MAX_SIZE, POOL_TIMEOUT, SLEEP
from concurrency.probe import PoolProbe


def _question(session_id: str, i: int) -> str:
    """Câu hỏi nhúng session id vào content để bleed check dò bằng substring."""
    return f"câu hỏi {i} thuộc session {session_id}"


async def _post(client, session_id: str, question: str):
    return await client.post(
        f"/conversations/{session_id}/messages", json={"question": question}
    )


@pytest.mark.anyio
async def test_a_distinct_sessions_no_bleed(client, fetch_messages):
    """5 session khác nhau gather cùng lúc → mỗi session đúng 2 msg của chính nó."""
    sessions = [str(uuid.uuid4()) for _ in range(N_CONCURRENT)]

    responses = await asyncio.gather(
        *[_post(client, sid, _question(sid, i)) for i, sid in enumerate(sessions)]
    )

    for sid, resp in zip(sessions, responses):
        assert resp.status_code == 200, f"{sid}: {resp.status_code} {resp.text}"

    for i, sid in enumerate(sessions):
        msgs = await fetch_messages(sid)
        assert len(msgs) == 2, (
            f"session {sid} có {len(msgs)} msg (lẽ ra 2 — bleed hoặc mất msg): {msgs}"
        )

        roles = [m["role"] for m in msgs]
        assert roles.count("user") == 1 and roles.count("ai") == 1, (
            f"session {sid} sai cấu trúc user/ai: {roles}"
        )

        # bleed check: mọi msg phải nhắc đúng session của chính nó
        for m in msgs:
            assert sid in m["content"], f"msg của session khác lẫn vào {sid}: {m}"
            for other in sessions:
                if other != sid:
                    assert other not in m["content"], (
                        f"BLEED: msg của {other} xuất hiện trong session {sid}: {m}"
                    )


@pytest.mark.anyio
async def test_b_starvation_all_200(client, test_pool, fetch_messages):
    """5 request cùng lúc trên pool max_size=2 → tất cả 200, không timeout/deadlock."""
    sessions = [str(uuid.uuid4()) for _ in range(N_CONCURRENT)]

    probe = PoolProbe(test_pool).start()
    try:
        responses = await asyncio.gather(
            *[_post(client, sid, _question(sid, i)) for i, sid in enumerate(sessions)]
        )
    finally:
        probe.stop()

    for sid, resp in zip(sessions, responses):
        assert resp.status_code == 200, (
            f"{sid}: {resp.status_code} {resp.text} — timeout pool phải > 3 × SLEEP "
            f"({POOL_TIMEOUT}s > {3 * SLEEP}s); nếu nhỏ hơn, request ăn timeout thay "
            f"vì xếp hàng → đo nhầm"
        )

    stats = probe.get_stats()
    # (option theo plan) xác nhận đã thật sự gây starvation: có lúc >= 1 request
    # xếp hàng chờ connection (5 request, 2 conn → tối thiểu 3 phải chờ)
    assert stats["saw_contention"], (
        "pool max_size=2 + 5 request đồng thời lẽ ra phải có lúc requests_waiting > 0; "
        "timeline rỗng có thể do probe ngừng trước khi request kịp xếp hàng"
    )
    assert stats["requests_waiting"]["max"] >= N_CONCURRENT - POOL_MAX_SIZE, stats

    # và mọi session vẫn ghi đủ 2 msg end-to-end (không ai bị bỏ dở)
    for sid in sessions:
        msgs = await fetch_messages(sid)
        assert len(msgs) == 2, f"session {sid} ghi thiếu msg: {msgs}"


@pytest.mark.anyio
async def test_c_disconnect_mid_sleep(client, test_pool, fetch_messages):
    """Cancel request giữa lúc stub đang sleep.

    Ghi nhận hành vi, không assert cứng theo plan:
    - msg có được ghi không? Starlette chạy handler sync trong threadpool — cancel
      asyncio task không kill thread, handler vẫn chạy hết và thường VẪN ghi msg.
    - pool_available có trở về đủ max_size không? (không leak connection) — đây là
      điều duy nhất assert cứng, vì leak chính là bug mà phép đo này sinh ra để bắt.
    """
    sid = str(uuid.uuid4())

    task = asyncio.create_task(
        _post(client, sid, "bỏ dở giữa chừng")
    )

    # chờ request kịp chiếm 1 connection và vào giữa sleep của stub
    await asyncio.sleep(SLEEP / 2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass  # cancel thành công, không quan tâm phản hồi

    # chờ handler chạy nốt trong threadpool và trả connection về pool
    deadline = time.monotonic() + POOL_TIMEOUT
    available = test_pool.get_stats()["pool_available"]
    while available < POOL_MAX_SIZE and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
        available = test_pool.get_stats()["pool_available"]

    # ghi nhận hành vi (hiện khi chạy -s hoặc khi test fail)
    msgs = await fetch_messages(sid)
    print(
        f"[disconnect] messages ghi được sau cancel: {len(msgs)} "
        f"(roles={[m['role'] for m in msgs]}) — handler chạy trong threadpool "
        f"nên thread vẫn chạy nốt dù asyncio task đã bị hủy"
    )
    print(f"[disconnect] pool_available sau cancel: {available}/{POOL_MAX_SIZE}")

    # không leak connection: pool phải trả về đủ max_size
    assert available >= POOL_MAX_SIZE, (
        f"pool_available={available}/{POOL_MAX_SIZE} — leak connection sau disconnect?"
    )


@pytest.mark.anyio
async def test_d_same_session_race_no_loss(client, fetch_messages):
    """2 request cùng 1 conversation_id → chat_history đủ 4 msg, không mất."""
    sid = str(uuid.uuid4())
    q1, q2 = _question(sid, 1), _question(sid, 2)

    r1, r2 = await asyncio.gather(
        _post(client, sid, q1),
        _post(client, sid, q2),
    )

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text

    msgs = await fetch_messages(sid)
    assert len(msgs) == 4, f"mất msg trong race same-session: {msgs}"

    roles = [m["role"] for m in msgs]
    assert roles.count("user") == 2 and roles.count("ai") == 2, roles

    contents = [m["content"] for m in msgs]
    # cả 2 câu hỏi lẫn 2 câu trả lời stub tương ứng đều phải có mặt
    for q in (q1, q2):
        assert q in contents, f"thiếu user msg {q!r}: {contents}"
        expected_answer = f"Stub trả lời: {q}"
        assert expected_answer in contents, f"thiếu ai msg {expected_answer!r}: {contents}"
