"""
    Cấu hình HTTP dùng chung cho các frontend client gọi FastAPI backend
"""

from __future__ import annotations

import os

import httpx

DEFAULT_API_BASE_URL = "http://localhost:8000"
API_BASE_URL = os.environ.get("FASTAPI_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def auth_headers(user_id: str) -> dict[str, str]:
    """Header định danh owner"""
    return {"X-User-Id": user_id}


def chat_timeout() -> httpx.Timeout:
    """Timeout cho một chat turn, tách theo 4 pha:

    - connect ngắn: backend chết phải fail nhanh, không treo UI
    - read dài: pha bao trùm cả LLM chain chạy server-side
    - write ngắn: body chỉ {"question": ...}
    - pool ngắn: single-shot gần như không tranh chấp connection
    """
    return httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)
