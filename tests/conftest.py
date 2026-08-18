"""Fixtures dùng chung cho test suite.

Bộ test concurrency đo hành vi của POST /conversations/{id}/messages khi
nhiều request chạy đồng thời: pool sync bị thu nhỏ cố ý (max_size=2) để ép
starvation, LLM được thay bằng stub ngủ SLEEP giây — nếu không stub thì test
đo Groq rate limiter chứ không đo server. Không dùng LifespanManager: build
một FastAPI rỗng chỉ mount chat_router rồi override dependencies, nên không
load bge-m3 / reranker (test nhẹ, hermetic).

Cấu hình chạy:
    TEST_DATABASE_URL=<url DB test, dbname phải chứa "test"> uv run pytest -v

clean_db (autouse) TRUNCATE chat_history + conversations trước mỗi test và
từ chối chạy nếu dbname không chứa "test" — không bao giờ để lỡ tay xoá DB thật.
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
from psycopg_pool import ConnectionPool

# App tự load .env (xem src/rag/qa_chain.py) — làm tương tự để TEST_DATABASE_URL
# có thể đặt trong .env thay vì export ở shell.
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


@pytest.fixture
def test_pool(test_database_url: str) -> Iterator[ConnectionPool]:
    """Pool sync thu nhỏ cố ý (max_size=2) để gây starvation ở N=5.

    timeout=POOL_TIMEOUT phải lớn hơn tổng thời gian 3 wave (3 × SLEEP),
    nếu không phép (b) sẽ ăn timeout thay vì xếp hàng → đo nhầm.
    kwargs={"prepare_threshold": None}: guard Supavisor 6543 (giống
    src/database/sync_connection.py), tránh lỗi prepared statement khi
    nhiều request chạy cùng lúc.
    """
    pool = ConnectionPool(
        conninfo=test_database_url,
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

    Assert tên DB chứa "test" trước khi TRUNCATE — chốt chặn an toàn để không
    bao giờ lỡ tay xoá nhầm dữ liệu DB thật (vd trỏ nhầm TEST_DATABASE_URL vào
    DATABASE_URL có dbname="postgres").
    """
    dbname = urlparse(test_database_url).path.lstrip("/")
    assert "test" in dbname, (
        f"TEST_DATABASE_URL phải trỏ tới DB test (dbname chứa 'test'), "
        f"nhận dbname={dbname!r} — từ chối TRUNCATE."
    )

    with test_pool.connection() as conn:
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
