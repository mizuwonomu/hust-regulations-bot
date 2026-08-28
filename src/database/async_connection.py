"""
    Tạo kết nối async tới database và tạo pool
"""

from __future__ import annotations
import os
from psycopg_pool import AsyncConnectionPool

from .pool_config import ASYNC_MIN_POOL, ASYNC_MAX_POOL, POOL_TIMEOUT_SECONDS


def create_async_pool() -> AsyncConnectionPool:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for the FastAPI database pool.")

    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=ASYNC_MIN_POOL,
        max_size=ASYNC_MAX_POOL,
        timeout=POOL_TIMEOUT_SECONDS,
        kwargs={"prepare_threshold": None},
        open=False,
    )
