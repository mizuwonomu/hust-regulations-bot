"""
    Tạo sync connection pool cho memory path (RunnableWithMessageHistory).

    Bản sync song song của async_connection.py. Sync pool phục vụ memory
    (chain chạy trong threadpool), async pool phục vụ title + SQL đọc.
    Hai pool độc lập trong app.state
"""

from __future__ import annotations

import os

from psycopg_pool import ConnectionPool

from .pool_config import SYNC_MIN_POOL, SYNC_MAX_POOL, POOL_TIMEOUT_SECONDS


def create_sync_pool() -> ConnectionPool:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for the sync database pool")

    return ConnectionPool(
        conninfo=database_url,
        min_size=SYNC_MIN_POOL,
        max_size=SYNC_MAX_POOL,
        timeout=POOL_TIMEOUT_SECONDS,
        #tắt prepared statement 
        kwargs={"prepare_threshold": None},
        check=ConnectionPool.check_connection,
        open=False,
    )
