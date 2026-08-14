"""
    Định nghĩa các reusable dependencies mà FastAPI cần inject
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Header, Request

from psycopg_pool import AsyncConnectionPool

DEFAULT_USER_ID = "user_vjp_pro_1"

def get_current_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str:
    """Lấy user id hiện tại"""
    return x_user_id or DEFAULT_USER_ID


def get_db_pool(request: Request) -> AsyncConnectionPool:
    """Lấy global DB pool đã được mở từ lifespan"""
    return request.app.state.db_pool


def get_rag_chain(request: Request):
    """Lấy RAG chain đã build sẵn trong lifespan (không build lại mỗi request)"""
    return request.app.state.rag_chain