"""
    Streamlit-side HTTP client cho việc sinh title.

    Đóng gói các FastAPI request nhằm schedule sinh title ở backend và poll title state hiện tại,
    , giữ mọi thông tin HTTP trả về ra khỏi Streamlit workflow/components.

    Do chat_flow/sidebar component không cần raw URLs cũng như status handling.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "http://localhost:8000"
API_BASE_URL = os.environ.get("FASTAPI_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
REQUEST_TIMEOUT = 5.0


def _headers(user_id: str) -> dict[str, str]:
    return {"X-User-Id": user_id}


def schedule_title_generation(conversation_id: str, user_id: str) -> str | None:
    try:
        response = httpx.post(
            f"{API_BASE_URL}/conversations/{conversation_id}/title-generation",
            headers=_headers(user_id),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("status")
    except httpx.HTTPError:
        logger.exception("Failed to schedule title generation")
        return None


def get_title_status(conversation_id: str, user_id: str) -> dict[str, Any] | None:
    try:
        response = httpx.get(
            f"{API_BASE_URL}/conversations/{conversation_id}/title",
            headers=_headers(user_id),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        logger.exception("Failed to fetch title status")
        return None
