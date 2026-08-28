"""
    Streamlit-side HTTP client cho 2 read endpoints:
    GET /conversations và GET /conversations/{id}/messages.

    Quy ước None vs []: None = request FAIL (backend chết, transport lỗi);
    [] = thành công nhưng chưa có gì
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "http://localhost:8000"
API_BASE_URL = os.environ.get("FASTAPI_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
REQUEST_TIMEOUT = 5.0


def _headers(user_id: str) -> dict[str, str]:
    return {"X-User-Id": user_id}


def list_conversations(user_id: str) -> list[tuple[str, str | None]] | None:
    """List (conversation_id, title) của user, mới nhất trước.

    None = request failed. Trả list tuple để sidebar giữ nguyên unpacking cũ
    (không đổi chỗ gọi). title có thể None - placeholder thuộc sidebar.
    """
    try:
        response = httpx.get(
            f"{API_BASE_URL}/conversations",
            headers=_headers(user_id),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            (item["conversation_id"], item.get("title"))
            for item in payload.get("conversations", [])
        ]
    except httpx.HTTPError:
        logger.exception("Failed to list conversations")
        return None


def fetch_messages(conversation_id: str, user_id: str) -> list[dict] | None:
    """Messages đã chuẩn hoá {role, content} của một conversation.

    404 xử lý RIÊNG: log warning + None - nó nghĩa là "không phải của mình
    hoặc không tồn tại" (endpoint cố ý không lộ sự tồn tại), không phải lỗi
    transport. Mọi httpx.HTTPError khác log exception + None.
    """
    try:
        response = httpx.get(
            f"{API_BASE_URL}/conversations/{conversation_id}/messages",
            headers=_headers(user_id),
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 404:
            logger.warning(
                "Conversation %s trả 404 cho user %s - không tồn tại hoặc "
                "không phải của user này",
                conversation_id,
                user_id,
            )
            return None
        response.raise_for_status()
        payload = response.json()
        return payload.get("messages", [])
    except httpx.HTTPError:
        logger.exception(
            "Failed to fetch messages for conversation %s", conversation_id
        )
        return None
