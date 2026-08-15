"""
    Tạo kết nối async tới database và tạo pool
"""

from __future__ import annotations
import os
from psycopg_pool import AsyncConnectionPool


def create_async_pool() -> AsyncConnectionPool:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for the FastAPI database pool")

    return AsyncConnectionPool(
        conninfo=database_url,
        min_size=2,
        max_size=5, 
        open=False,
    )
