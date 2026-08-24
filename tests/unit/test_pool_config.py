"""Config guards cho cả 2 pool builders — KHÔNG cần kết nối DB thật.

Build pool với open=False rồi inspect tham số lưu sẵn, DATABASE_URL
monkeypatch thành dummy. Bảo vệ 2 divergence đã biết:

1. prepare_threshold=None phải có trên CẢ HAI pool. Async pool thiếu guard
   này — tracker.md: "The async pool lacks the prepare_threshold=None guard...
   will hit intermittent `prepared statement already exists` under any real
   concurrency over 6543". Task 3 vừa cho async pool consumer thứ 2 (read
   endpoints), nên guard phải về ngay — lỗi chỉ nổ dưới concurrency thật nên
   chỉ có test config này bắt được.
2. timeout phải tường minh = POOL_TIMEOUT_SECONDS ở cả hai, không dựa vào
   mặc định ngầm 30s của psycopg_pool. Task 8 (capacity measurement) định
   nghĩa theo con số này nên nó phải là một hằng số đọc từ một nơi.
"""

from __future__ import annotations

import asyncio

import pytest

import src.database.async_connection as async_mod
import src.database.sync_connection as sync_mod
from psycopg_pool import AsyncConnectionPool

DUMMY_URL = "postgresql://user:pass@localhost:5432/dummy"


def _close(pool) -> None:
    """Close pool đúng cách theo loại (async pool cần await)."""
    if isinstance(pool, AsyncConnectionPool):
        asyncio.run(pool.close())
    else:
        pool.close()


@pytest.fixture
def dummy_db_url(monkeypatch) -> str:
    monkeypatch.setenv("DATABASE_URL", DUMMY_URL)
    return DUMMY_URL


def test_sync_pool_disables_prepared_statements(dummy_db_url):
    pool = sync_mod.create_sync_pool()
    try:
        assert pool.kwargs == {"prepare_threshold": None}
    finally:
        _close(pool)


def test_async_pool_disables_prepared_statements(dummy_db_url):
    """Async pool THIẾU guard này (debt trong tracker.md) — fail cho tới khi
    được thêm. Supavisor transaction pooler swap connection giữa các
    transaction, nên prepared statement bị lẫn → intermittent
    'prepared statement already exists' chỉ nổ dưới concurrency thật."""
    pool = async_mod.create_async_pool()
    try:
        assert pool.kwargs == {"prepare_threshold": None}, (
            "async pool thiếu prepare_threshold=None — guard Supavisor 6543 "
            "(tracker.md: 'The async pool lacks the prepare_threshold=None guard'; "
            "đã được thêm vì read endpoints trở thành consumer thứ 2 của pool này)"
        )
    finally:
        _close(pool)


@pytest.mark.parametrize(
    ("builder", "name"),
    [(sync_mod.create_sync_pool, "sync"), (async_mod.create_async_pool, "async")],
    ids=["sync", "async"],
)
def test_pools_set_explicit_timeout(dummy_db_url, builder, name: str):
    """timeout tường minh = POOL_TIMEOUT_SECONDS — capacity measurement (Task 8)
    và production đọc cùng một con số từ một nơi."""
    pool = builder()
    try:
        assert pool.timeout == sync_mod.POOL_TIMEOUT_SECONDS, (
            f"{name} pool phải set timeout tường minh = POOL_TIMEOUT_SECONDS "
            f"({sync_mod.POOL_TIMEOUT_SECONDS}s), không dựa vào mặc định ngầm "
            f"30s — capacity measurement định nghĩa theo con số này"
        )
    finally:
        _close(pool)


@pytest.mark.parametrize(
    ("builder", "name"),
    [(sync_mod.create_sync_pool, "sync"), (async_mod.create_async_pool, "async")],
    ids=["sync", "async"],
)
def test_pools_not_opened_at_construction(dummy_db_url, builder, name: str):
    """Lifespan owns open/close — builder chỉ dựng pool, không mở."""
    pool = builder()
    try:
        assert pool._opened is False, (
            f"{name} pool không được open ở construction — lifespan owns "
            f"open/close (nếu builder mở sẵn, test concurrency sẽ lẫn conn "
            f"vào phép đo)"
        )
    finally:
        _close(pool)


def test_sync_builder_raises_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        sync_mod.create_sync_pool()


def test_async_builder_raises_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        async_mod.create_async_pool()
