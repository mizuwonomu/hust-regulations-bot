"""Fixtures dùng chung cho test suite.

Bộ test concurrency đo hành vi của POST /conversations/{id}/messages khi
nhiều request chạy đồng thời: pool sync bị thu nhỏ cố ý (max_size=2) để ép
starvation, LLM được thay bằng stub ngủ SLEEP giây — nếu không stub thì test
đo Groq rate limiter chứ không đo server. Không dùng LifespanManager: build
một FastAPI rỗng chỉ mount chat_router rồi override dependencies, nên không
load bge-m3 / reranker (test nhẹ, hermetic).

"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
import anyio
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.runnables import RunnableLambda
import psycopg
from psycopg.conninfo import make_conninfo
from psycopg_pool import ConnectionPool

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for _p in (ROOT, TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.api.dependencies import get_rag_chain, get_sync_db_pool  # noqa: E402
from src.api.routes.chat import router as chat_router  # noqa: E402
from src.database.conversation_queries import get_conversation_messages  # noqa: E402


# ── Hằng số dùng chung ──────────────────────────────────────────────────────
SLEEP = 0.3          # stub chain ngủ (giây), sync/blocking như chain thật trong threadpool
POOL_MAX_SIZE = 2    # thu nhỏ cố ý: N=5 trên pool 2 conn → ép starvation
POOL_TIMEOUT = 5.0   # phải > 3 × SLEEP (tổng thời gian 3 wave = 0.9s); nếu nhỏ hơn,
                     # phép (b) ăn timeout thay vì xếp hàng → đo nhầm
N_CONCURRENT = 5     # số request đồng thời cho mỗi phép đo
# ── Nhóm CAPACITY: mô phỏng THỜI LƯỢNG LƯỢT giống prod. Chậm → đánh dấu slow ─
# CAPACITY_TURN_DURATION xấp xỉ một lượt thật: 3-5 Groq call + rerank.
# CAPACITY_TIMEOUT_EXCEED - synthetic: timeout production không bao giờ nổ
CAPACITY_POOL_SIZE = 10
CAPACITY_TURN_DURATION = 4.0
CAPACITY_REQUESTS_FIT = 20        # 2 waves → wave cuối chờ 4s < 30s
CAPACITY_REQUESTS_EXCEED = 90     # 9 waves → wave cuối chờ 32s > 5s
CAPACITY_TIMEOUT_EXCEED = 5.0

KILL_APP_NAME = "hust-test-kill"  # application_name của test_pool — để admin_conn (test e)
                                  # bắn pg_terminate_backend đúng conn của pool, không bắn nhầm
TEST_SCHEMA = "regu_test_dsk"  # schema test trên Supabase — mọi pool test đặt search_path
                              # vào đây qua connection options; clean_db từ chối chạy nếu
                              # current_schema() không khớp (guard chống TRUNCATE nhầm public)


@pytest.fixture
def anyio_backend() -> str:
    """Ép @pytest.mark.anyio chạy trên asyncio (anyio có sẵn plugin pytest,
    không cần pytest-asyncio)."""
    return "asyncio"


@pytest.fixture
def test_database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip(
            "TEST_DATABASE_URL chưa được set — trỏ tới DB test (dbname chứa 'test') "
            "để chạy bộ concurrency test. Không dùng DATABASE_URL thật: clean_db "
            "sẽ TRUNCATE chat_history + conversations."
        )
    return url


def assert_test_schema(conn) -> None:
    """Guard runtime: từ chối mọi thao tác nguy hiểm nếu connection không trỏ schema test.

    Thay cho assert dbname chứa "test" (localhost-shaped, chết trên Supabase vì
    Supabase luôn đặt tên DB là "postgres"). Kiểm tra trạng thái THỰC của
    connection qua current_schema() — chặt hơn guard cũ: không thể bị đánh lừa
    bởi URL trông hợp lệ.

    Message nói rõ điều guard chặn: TRUNCATE nhầm dữ liệu conversation thật.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT current_schema()")
        schema = cur.fetchone()[0]

    assert schema == TEST_SCHEMA and schema != "public", (
        f"current_schema()={schema!r} — không phải schema test {TEST_SCHEMA!r}. "
        "Từ chối TRUNCATE: nếu chạy tiếp, có thể xoá dữ liệu conversation THẬT. "
        "Kiểm tra TEST_DATABASE_URL và connection options search_path."
    )


@pytest.fixture
def test_pool(test_database_url: str) -> Iterator[ConnectionPool]:
    """Pool sync thu nhỏ cố ý (max_size=2) để gây starvation ở N=5.

    timeout=POOL_TIMEOUT phải lớn hơn tổng thời gian 3 wave (3 × SLEEP),
    nếu không phép (b) sẽ ăn timeout thay vì xếp hàng → đo nhầm.
    kwargs={"prepare_threshold": None}: guard Supavisor 6543 (giống
    src/database/sync_connection.py), tránh lỗi prepared statement khi
    nhiều request chạy cùng lúc.

    options="-c search_path=regu_test_dsk": mọi conn của pool mở trong schema
    test — đặt qua connection options (không phải SET trong từng test) để
    không code path nào vô tình chạy trên public. Verified: search_path là
    GUC duy nhất sống sót qua Supavisor transaction pooler.

    application_name=KILL_APP_NAME: trên localhost, gán nhãn mọi conn của
    pool để test (e) nhắm đúng backend khi kill. Qua Supabase 6543 bị
    Supavisor tước (xem docs/testing-setup.md) — giữ lại cho localhost.
    """
    pool = ConnectionPool(
        conninfo=make_conninfo(
            test_database_url,
            application_name=KILL_APP_NAME,
            options=f"-c search_path={TEST_SCHEMA}",
        ),
        min_size=POOL_MAX_SIZE,
        max_size=POOL_MAX_SIZE,
        timeout=POOL_TIMEOUT,
        kwargs={"prepare_threshold": None},
        open=True,
    )
    try:
        yield pool
    finally:
        pool.close()


@pytest.fixture
def admin_conn(test_database_url):
    """Conn hành chính KHÔNG thuộc pool: query pg_stat_activity + pg_terminate_backend.

    psycopg.connect thẳng tới test_database_url — dùng test_pool thì chính nó
    làm nhiễu phép đo (chiếm/trả conn, lẫn vào application_name mà test đang
    theo dõi). autocommit=True để terminate không kẹt trong transaction.
    """
    conn = psycopg.connect(
        test_database_url,
        application_name="hust-test-admin",
        autocommit=True,
        prepare_threshold=None,
    )
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def stub_core_chain():
    """Stub core chain: sync + blocking (time.sleep) như chain thật chạy trong
    threadpool của Starlette. Nuốt key question/chat_history và nhả answer/context
    để bind_history (input_messages_key="question", output_messages_key="answer")
    wrap được."""
    def _invoke(inputs: dict) -> dict:
        time.sleep(SLEEP)
        question = inputs.get("question", "?")
        return {"answer": f"Stub trả lời: {question}", "context": []}

    return RunnableLambda(_invoke)


@pytest.fixture
async def client(test_pool, stub_core_chain):
    """FastAPI hermetic: chỉ mount chat_router, override pool + chain.

    Không dùng LifespanManager → lifespan không chạy → không load bge-m3 /
    reranker / không mở pool thật. Handler def vẫn chạy trong threadpool của
    Starlette y như production (thông qua ASGITransport).
    """
    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_sync_db_pool] = lambda: test_pool
    app.dependency_overrides[get_rag_chain] = lambda: stub_core_chain

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture(autouse=True)
def clean_db(test_pool, test_database_url):
    """TRUNCATE chat_history + conversations trước mỗi test.

    Guard runtime (assert_test_schema) kiểm tra current_schema() của connection
    thật trước khi TRUNCATE — chốt chặn an toàn để không bao giờ lỡ tay xoá
    nhầm dữ liệu thật (vd quên search_path, trỏ nhầm URL, schema bị drop).
    Chặt hơn guard dbname cũ: không thể bị đánh lừa bởi URL trông hợp lệ.
    """
    with test_pool.connection() as conn:
        assert_test_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE chat_history, conversations RESTART IDENTITY CASCADE"
            )
        conn.commit()
    yield


@pytest.fixture
def fetch_messages(test_pool):
    """Đọc chat_history của một session qua helper của chính project
    (chuẩn hoá về {role, content}, xem get_conversation_messages)."""
    async def _fetch(session_id: str) -> list[dict]:
        def _q():
            with test_pool.connection() as conn:
                return get_conversation_messages(conn, session_id)

        return await anyio.to_thread.run_sync(_q)
    return _fetch


# ── Fixtures cho read endpoints (async) ─────────────────────────────────────
from psycopg_pool import AsyncConnectionPool  # noqa: E402
from src.api.dependencies import get_db_pool  # noqa: E402
from src.api.routes.conversations import router as conversations_router  # noqa: E402


@pytest.fixture
async def async_test_pool(test_database_url):
    """Async pool cho read endpoints, cùng schema test với test_pool"""
    pool = AsyncConnectionPool(
        conninfo=make_conninfo(
            test_database_url,
            application_name=KILL_APP_NAME,
            options=f"-c search_path={TEST_SCHEMA}",
        ),
        min_size=2,
        max_size=5,
        timeout=POOL_TIMEOUT,
        kwargs={"prepare_threshold": None},
        open=False,
    )
    await pool.open()
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def api_client(test_pool, async_test_pool, stub_core_chain):
    """App hermetic mount CẢ chat_router (def, sync pool) LẪN conversations_router"""
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(conversations_router)
    app.dependency_overrides[get_sync_db_pool] = lambda: test_pool
    app.dependency_overrides[get_db_pool] = lambda: async_test_pool
    app.dependency_overrides[get_rag_chain] = lambda: stub_core_chain

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture
def seed_conversation(test_pool):
    """Chèn 1 conversation + n cặp message (human/ai) cho user chỉ định.

    Message ghi đúng envelope LangChain ({"type", "data": {"content"}}) để
    normalize_message thấy row đúng shape production, không phải shape bịa.
    """
    import json as _json

    def _seed(conversation_id: str, user_id: str, n_pairs: int = 1, title=None):
        with test_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (conversation_id, user_id, title) "
                    "VALUES (%s, %s, %s) ON CONFLICT (conversation_id) DO NOTHING",
                    (conversation_id, user_id, title),
                )
                for i in range(n_pairs):
                    for mtype, content in (
                        ("human", f"câu hỏi {i}"),
                        ("ai", f"trả lời {i}"),
                    ):
                        cur.execute(
                            "INSERT INTO chat_history (session_id, message) "
                            "VALUES (%s, %s)",
                            (
                                conversation_id,
                                _json.dumps({"type": mtype, "data": {"content": content}}),
                            ),
                        )
            conn.commit()

    return _seed


# ── Fixtures cho phép đo CAPACITY (slow) ────────────────────────────────────


@pytest.fixture
def make_sync_pool(test_database_url):
    """Factory dựng pool sync với size/timeout tuỳ ý, tự đóng khi test xong.

    test_pool giữ nguyên cấu hình starvation để 7 probe cũ không đổi hành vi;
    phép đo capacity dùng factory này để dựng pool giống prod.
    """
    pools: list[ConnectionPool] = []

    def _make(max_size: int, timeout: float) -> ConnectionPool:
        pool = ConnectionPool(
            conninfo=make_conninfo(
                test_database_url,
                application_name=KILL_APP_NAME,
                options=f"-c search_path={TEST_SCHEMA}",
            ),
            min_size=max_size,
            max_size=max_size,
            timeout=timeout,
            kwargs={"prepare_threshold": None},
            open=True,
        )
        pools.append(pool)
        return pool

    yield _make
    for pool in pools:
        pool.close()


@pytest.fixture
def slow_stub_chain():
    """Stub chain ngủ CAPACITY_TURN_DURATION — mô phỏng thời lượng một lượt thật."""

    def _invoke(inputs: dict) -> dict:
        time.sleep(CAPACITY_TURN_DURATION)
        return {"answer": f"Stub trả lời: {inputs.get('question', '?')}", "context": []}

    return RunnableLambda(_invoke)


@pytest.fixture
async def capacity_client(make_sync_pool, slow_stub_chain):
    """Client trên pool giống PRODUCTION (size 10 + timeout 30s) và stub chậm.

    httpx timeout phải lớn hơn hẳn pool timeout, nếu không client bỏ cuộc
    trước khi phép đo kết thúc và ta đo nhầm client thay vì pool.
    """
    from src.database.sync_connection import POOL_TIMEOUT_SECONDS

    pool = make_sync_pool(CAPACITY_POOL_SIZE, POOL_TIMEOUT_SECONDS)

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_sync_db_pool] = lambda: pool
    app.dependency_overrides[get_rag_chain] = lambda: slow_stub_chain

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", timeout=300.0
    ) as c:
        yield c, pool
