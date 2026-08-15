"""
    Tạo sync connection pool cho memory path (RunnableWithMessageHistory).

    Bản sync song song của async_connection.py. Sync pool phục vụ memory
    (chain chạy trong threadpool), async pool phục vụ title + SQL đọc.
    Hai pool độc lập trong app.state
"""

from __future__ import annotations

import os

from psycopg_pool import ConnectionPool


def create_sync_pool() -> ConnectionPool:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for the sync database pool")

    return ConnectionPool(
        conninfo=database_url,
        min_size=2,
        max_size=10,
        #tắt prepared statement 
        kwargs={"prepare_threshold": None},
        check=ConnectionPool.check_connection,
        open=False,
    )
