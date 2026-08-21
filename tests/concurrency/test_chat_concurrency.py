"""Bộ test concurrency cho POST /conversations/{id}/messages (G3).

5 phép đo, verify qua bảng chat_history và/hoặc cờ memory_persisted:

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
(e) db-side failure  — kill backend của pool (pg_terminate_backend) giữa lúc
                        stub đang sleep → VẪN 200 kèm answer nhưng
                        memory_persisted=false (lỗi ghi bị nuốt, không vứt bỏ
                        câu trả lời), không phantom write, pool tự thay conn chết.

Chạy: TEST_DATABASE_URL=<url DB test> uv run pytest tests/concurrency -v
"""

from __future__ import annotations

import asyncio
import time
import uuid

import psycopg
import pytest

from conftest import KILL_APP_NAME, N_CONCURRENT, POOL_MAX_SIZE, POOL_TIMEOUT, SLEEP
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


def _pool_pids(admin_conn) -> set[int]:
    """Tập backend pid của mọi conn thuộc test_pool (lọc theo application_name)."""
    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT pid FROM pg_stat_activity WHERE application_name = %s",
            (KILL_APP_NAME,),
        )
        return {row[0] for row in cur.fetchall()}


def _pool_activity(admin_conn) -> list[tuple]:
    """Toàn bộ conn của pool kèm state/query — debug khi timing lệch."""
    with admin_conn.cursor() as cur:
        cur.execute(
            """
            SELECT pid, state, left(query, 60)
            FROM pg_stat_activity
            WHERE application_name = %s
            ORDER BY pid
            """,
            (KILL_APP_NAME,),
        )
        return cur.fetchall()


def _require_terminate_privilege(admin_conn) -> None:
    """Skip test nếu user DB thiếu quyền pg_terminate_backend.

    Terminate pid 0 (không tồn tại → trả false, vô hại) để thử quyền; thiếu
    quyền → InsufficientPrivilege → skip thay vì fail khó hiểu.
    """
    try:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(0)")
            cur.fetchone()
    except psycopg.errors.InsufficientPrivilege as exc:
        pytest.skip(
            "user DB thiếu quyền pg_terminate_backend "
            f"(cần superuser hoặc pg_signal_backend): {exc}"
        )


@pytest.mark.anyio
async def test_e_db_side_failure_no_write_conn_replaced(
    client, test_pool, admin_conn, fetch_messages
):
    """Kill conn của pool giữa lúc stub đang sleep — DB-side failure (N=1).

    Mô phỏng server bóp chết backend đang giữ borrow-2 (restart, crash,
    terminate thủ công): transaction của PostgresChatMessageHistory đang mở,
    mọi INSERT chưa commit bị rollback trọn vẹn khi backend chết.

    Chứng minh 3 điều:
    1. Cú ghi thất bại không thành 500: theo contract mới, vẫn 200 kèm answer
       (không vứt bỏ câu trả lời hợp lệ đã tốn tiền LLM) nhưng cờ
       memory_persisted phải = false.
    2. Không phantom write: chat_history của session vẫn rỗng — turn bị kill
       không để lại nửa turn trong DB. (Không assert conversations: borrow-1
       đã commit row conversation trước khi kill.)
    3. Pool tự lành: conn chết bị discard + conn mới thế chỗ → sau deadline
       pool_available == POOL_MAX_SIZE và killed_pid biến mất khỏi
       pg_stat_activity.

    Skip nếu user DB thiếu quyền pg_terminate_backend (superuser / pg_signal_backend).
    """
    sid = str(uuid.uuid4())
    question = "câu hỏi bị kill giữa chừng bởi DB"

    # 0. Skip guard: terminate pid 0 (không tồn tại → trả false, vô hại) để thử
    #    quyền; thiếu quyền → InsufficientPrivilege → skip thay vì fail khó hiểu
    _require_terminate_privilege(admin_conn)

    # 1. Snapshot pid của pool trước khi chạy (để sau này chứng minh conn chết
    #    đã được thay bằng pid mới, chứ không phải reuse lại pid cũ)
    pids_before = _pool_pids(admin_conn)
    assert len(pids_before) == POOL_MAX_SIZE, (
        f"pool phải mở sẵn {POOL_MAX_SIZE} conn, thấy {len(pids_before)}: {pids_before}"
    )

    # 2-3. Tạo request, chờ request vào giữa cửa sổ stub-sleep: borrow-2 đang
    #      giữ conn, transaction còn mở (state = 'idle in transaction'),
    #      chưa tới khâu write
    task = asyncio.create_task(_post(client, sid, question))
    await asyncio.sleep(SLEEP / 2)

    # 4. Tìm backend đang bận của pool rồi terminate. state <> 'idle' bắt
    #    borrow-2 (idle in transaction trong lúc stub sleep); conn còn lại
    #    nằm idle trong pool nên không bị bắt nhầm
    with admin_conn.cursor() as cur:
        cur.execute(
            """
            SELECT pid, state, left(query, 60)
            FROM pg_stat_activity
            WHERE application_name = %s AND state <> 'idle'
            """,
            (KILL_APP_NAME,),
        )
        rows = cur.fetchall()

    assert rows, (
        f"không bắt được conn borrow-2 đang active/idle-in-transaction tại "
        f"SLEEP/2 — timing lệch hoặc transaction đã commit sớm. Toàn bộ conn "
        f"của pool lúc đó: {_pool_activity(admin_conn)}"
    )
    killed_pid, killed_state, killed_query = rows[0]
    print(
        f"[db-kill] terminate backend pid={killed_pid} "
        f"state={killed_state} query={killed_query!r}"
    )

    with admin_conn.cursor() as cur:
        cur.execute("SELECT pg_terminate_backend(%s)", (killed_pid,))
        assert cur.fetchone()[0] is True, f"pg_terminate_backend({killed_pid}) trả false"

    # 5. Chờ task kết thúc: theo contract mới, cú ghi thất bại bị
    #    CallbackManager nuốt (raise_error=False) → vẫn 200 kèm answer,
    #    nhưng cờ memory_persisted phải = false — client biết turn bị bỏ rơi
    try:
        resp = await task
    except Exception as exc:
        pytest.fail(
            f"task raise {type(exc).__name__}: {exc} — cú ghi thất bại phải bị "
            f"nuốt thành 200 + memory_persisted=false, không được ném lên handler"
        )
    assert resp.status_code == 200, (
        f"ghi memory thất bại vẫn phải trả 200 kèm answer (không vứt bỏ câu trả "
        f"lời hợp lệ), nhận {resp.status_code} {resp.text}"
    )
    body = resp.json()
    assert body["memory_persisted"] is False, (
        f"memory_persisted phải = false khi cú ghi thất bại, "
        f"nhận {body['memory_persisted']!r}"
    )
    assert body["answer"] == f"Stub trả lời: {question}", (
        f"answer phải còn nguyên vẹn khi ghi memory thất bại: {body['answer']!r}"
    )

    # 2. Không phantom write: turn bị kill không để lại gì trong chat_history
    msgs = await fetch_messages(sid)
    assert msgs == [], (
        f"PHANTOM WRITE: turn bị kill vẫn để lại msg trong chat_history: {msgs} — "
        f"transaction của PostgresChatMessageHistory không bị rollback trọn vẹn"
    )

    # 6. Pool tự lành là async (refill qua worker thread của pool) → poll deadline
    #    như test (c), đừng assert ngay sau kill
    deadline = time.monotonic() + POOL_TIMEOUT
    stats = test_pool.get_stats()
    live_pids = _pool_pids(admin_conn)
    while (
        (stats["pool_available"] < POOL_MAX_SIZE or killed_pid in live_pids)
        and time.monotonic() < deadline
    ):
        await asyncio.sleep(0.02)
        stats = test_pool.get_stats()
        live_pids = _pool_pids(admin_conn)

    print(
        f"[db-kill] pool sau lành: available={stats['pool_available']}/{POOL_MAX_SIZE} "
        f"pids={live_pids}"
    )

    # 3. Conn chết đã bị discard và được thay bằng conn mới
    assert stats["pool_available"] == POOL_MAX_SIZE, (
        f"pool không tự lành sau {POOL_TIMEOUT}s: "
        f"pool_available={stats['pool_available']}/{POOL_MAX_SIZE} — conn chết "
        f"không được discard/refill?"
    )
    assert killed_pid not in live_pids, (
        f"killed_pid={killed_pid} vẫn còn trong pg_stat_activity: {live_pids}"
    )
    new_pids = live_pids - pids_before
    assert new_pids, (
        f"không có conn mới thế chỗ conn chết: before={pids_before} now={live_pids}"
    )